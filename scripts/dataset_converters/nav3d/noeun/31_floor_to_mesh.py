"""04b — 04 paths json 의 점별 바닥 z 를 navmesh 표면에서 **장면 mesh 바닥**으로 바꾼 사본을 만든다 (floor_mesh.py).

왜: 06b R4 FAIL (노은역 평지 카메라가 목표보다 중앙 3.2 cm 높음) 과 06v (vln_pe GT 의 H1 은 mesh 바닥을 딛는다:
카메라의 mesh 위 높이 1.656 ± 0.004 m, navmesh 위 높이는 에피소드마다 1.46~1.61 m) 의 공통 원인이 navmesh z 다.
06 코드는 그대로 두고 이 파일을 `06 --paths_json` 으로 준다.

출력 episode 마다: `trajectory` z 를 바꾸고, 원래 값은 `trajectory_navmesh_z`, 바꾼 양은 `floor_offset` 에 남긴다.

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/31_floor_to_mesh.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
sys.path.insert(0, str(HERE))
from floor_mesh import mesh_floor_z, stairs_mask_from_segments  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_json', default='scripts/dataset_converters/nav3d/logs/noeun/paths/noeun_station_paths.json')
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply', help='충돌 mesh (usdz 와 같은 것, z-up)')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/noeun/paths/noeun_station_paths_meshfloor.json')
    return ap.parse_args()


def main():
    import trimesh
    args = parse()
    d = json.loads(Path(args.paths_json).read_text())
    mesh = trimesh.load(args.mesh, force='mesh')
    offs, meas, fl = [], [], []
    for e in d['episodes']:
        T = np.asarray(e['trajectory'], dtype=np.float64)
        st = stairs_mask_from_segments(len(T), e['segments'])
        r = mesh_floor_z(T, st, mesh)
        e['trajectory_navmesh_z'] = T[:, 2].round(5).tolist()
        e['floor_offset'] = r['offset'].round(5).tolist()
        e['trajectory'] = np.c_[T[:, :2], r['z']].tolist()
        e['trajectory_z'] = r['z'].tolist()
        offs.append(r['offset'][~st])
        meas.append(r['measured'][~st].mean() if (~st).any() else 1.0)
        fl.append(e['kind'] == 'flat')
    d['floor_source'] = {'kind': 'mesh', 'mesh': args.mesh, 'from': args.paths_json, 'rule': 'floor_mesh.mesh_floor_z'}
    Path(args.out).write_text(json.dumps(d))
    o = np.concatenate(offs)
    print(f'[04b] {len(d["episodes"])} ep · 바닥 점 (mesh − navmesh) 5/50/95 % = {np.percentile(o, [5, 50, 95]).round(3)} m · '
          f'잰 비율 중앙 {np.median(meas):.3f} -> {args.out}')


if __name__ == '__main__':
    main()
