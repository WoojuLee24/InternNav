"""원본 3DGS PLY + 충돌 mesh → <name>_collision.usdz (GS .nurec + 보이지 않는 충돌 mesh.ply/mesh.usd 한 파일).

노은역·월드컵역·학습 데이터(data/GS_USDZ)와 같은 형식. sim_docker/third_party/usd2usdz_convert.sh 와 같은 결과를
usd2usdz 의 3DGRUT venv(16b7b19, ply_to_usd 축 패치, setup_3dgrut_venv.sh)만으로 만든다:
  ① 3DGRUT ply_to_usd            PLY → NuRec usdz (좌표 그대로)
  ② near_clip_distance 1e-8      16b7b19 템플릿 0.2 → 3DGRUT 최신 기본값 1e-8 (학습 데이터와 같음)
  ③ mesh.ply + mesh.usd 추가     충돌 켬·숨김, gauss.usda 에 over "mesh" + Volume proxy (3DGRUT add_mesh_to_usdz 와 같은 구성)
  ④ 64바이트 정렬 · ZIP32        OpenUSD 는 ZIP64 로컬 헤더를 못 읽는다 (4 GiB 넘으면 중단)

사용 (run_ply2usdz.sh 가 컨테이너·venv 를 준비한다):
  python ply2usdz.py <gs.ply> <mesh.obj|mesh.ply> <out.usdz> [--nurec-name NAME] [--near-clip 1e-8]
"""
import argparse
import gzip
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import msgpack
import numpy as np
from plyfile import PlyData, PlyElement
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

ALIGN = 64


def load_mesh(path: Path):
    """obj 또는 삼각형 ply → (V float32 (N,3), F int32 (M,3)). obj 다각형은 부채꼴로 나눈다."""
    if path.suffix.lower() == '.ply':
        ply = PlyData.read(str(path))
        v = ply['vertex']
        return (np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32),
                np.vstack(ply['face']['vertex_indices']).astype(np.int32))
    V, F = [], []
    for line in path.open(errors='replace'):
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            V.append(p[1:4])
        elif p[0] == 'f':
            idx = [int(t.split('/')[0]) for t in p[1:]]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            F += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return np.asarray(V, np.float32), np.asarray(F, np.int32)


def mesh_ply_bytes(V, F) -> bytes:
    vert = np.empty(len(V), [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vert['x'], vert['y'], vert['z'] = V.T
    face = np.empty(len(F), [('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = F
    buf = io.BytesIO()
    PlyData([PlyElement.describe(vert, 'vertex'),
             PlyElement.describe(face, 'face', len_types={'vertex_indices': 'u1'})]).write(buf)
    return buf.getvalue()


def mesh_usd_bytes(V, F, tmp: Path) -> bytes:
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, '/mesh')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel()))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set('none')
    mesh.MakeInvisible()
    stage.GetRootLayer().defaultPrim = 'mesh'
    stage.GetRootLayer().Export(str(tmp / 'mesh.usd'))
    return (tmp / 'mesh.usd').read_bytes()


def gauss_with_mesh(text: bytes, tmp: Path) -> bytes:
    p = tmp / 'gauss.usda'
    p.write_bytes(text)
    stage = Usd.Stage.Open(str(p))
    stage.OverridePrim('/World/mesh').GetReferences().AddReference(assetPath='mesh.usd')
    vol = next(x for x in Usd.PrimRange(stage.GetPrimAtPath('/World')) if x.GetTypeName() == 'Volume')
    (vol.GetRelationship('proxy') or vol.CreateRelationship('proxy', custom=True)).SetTargets([Sdf.Path('/World/mesh')])
    stage.Save()
    return p.read_bytes()


def write_aligned(out: Path, entries):
    """무압축·64바이트 정렬·ZIP32. 크기가 정확하므로 zipfile 의 ~2 GiB ZIP64 전환 기준을 4 GiB 로 올린다."""
    assert sum(len(d) + 200 for _, d in entries) < 2**32 - 2**20, 'USDZ 4 GiB 초과: OpenUSD 로 열 수 없음'
    zipfile.ZIP64_LIMIT = int((2**32 - 1) * 1.05) + 1
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED, allowZip64=False) as z:
        for name, data in entries:
            pad = (-(z.fp.tell() + 30 + len(name.encode()))) % ALIGN
            if 0 < pad < 4:  # extra 필드 헤더(4바이트)가 들어갈 자리가 없으면 한 주기 더
                pad += ALIGN
            zi = zipfile.ZipInfo(name)
            if pad:
                zi.extra = (0x1986).to_bytes(2, 'little') + (pad - 4).to_bytes(2, 'little') + b'\0' * (pad - 4)
            z.writestr(zi, data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('gs_ply')
    ap.add_argument('mesh', help='GS 와 같은 좌표계의 .obj 또는 삼각형 .ply')
    ap.add_argument('out_usdz')
    ap.add_argument('--nurec-name', default=None, help='usdz 안 .nurec 이름(확장자 제외). 기본: gs_ply 파일 이름')
    ap.add_argument('--near-clip', type=float, default=1e-8)
    a = ap.parse_args()
    gs, out = Path(a.gs_ply).resolve(), Path(a.out_usdz).resolve()
    name = a.nurec_name or gs.stem

    with tempfile.TemporaryDirectory(prefix='ply2usdz_') as t:
        tmp = Path(t)
        # ① .nurec 이름은 출력 파일 이름을 따른다
        subprocess.run([sys.executable, '-m', 'threedgrut.export.scripts.ply_to_usd', str(gs),
                        '--output_file', str(tmp / f'{name}.usdz')], check=True)
        with zipfile.ZipFile(tmp / f'{name}.usdz') as z:
            entries = [(n, z.read(n)) for n in z.namelist()]

        V, F = load_mesh(Path(a.mesh))
        print(f'[ply2usdz] mesh {len(V):,} verts, {len(F):,} faces')
        mesh = [('mesh.ply', mesh_ply_bytes(V, F)), ('mesh.usd', mesh_usd_bytes(V, F, tmp))]  # gauss.usda 가 참조하므로 먼저
        res = []
        for n, d in entries:
            if n.endswith('.nurec'):  # ② 3DGRUT exporter 와 같은 gzip(level 0)으로 다시 싼다
                payload = msgpack.unpackb(gzip.decompress(d), raw=False, strict_map_key=False)
                payload['nre_data']['config']['renderer']['culling']['near_clip_distance'] = a.near_clip
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=0) as f:
                    f.write(msgpack.packb(payload))
                d = buf.getvalue()
            elif n == 'gauss.usda':  # ③
                d = gauss_with_mesh(d, tmp)
            res.append((n, d))
        res += mesh
        out.parent.mkdir(parents=True, exist_ok=True)
        write_aligned(out, res)  # ④
    print(f'[ply2usdz] done: {out}')


if __name__ == '__main__':
    main()
