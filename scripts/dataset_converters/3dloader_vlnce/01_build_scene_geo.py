"""씬당 **지오메트리 전용 `.ply` 한 개**를 만든다 (타일링 대체).

## 왜 타일링을 버렸나 (실측 근거)
타일링은 "씬 전체를 worker에 올릴 수 없다"는 전제에서 나왔다. 그런데 실측하니 **무거운 것은 지오메트리가
아니라 텍스처**였다:

| | 크기 | 로드 | RSS |
|---|---|---|---|
| 씬 폴더(obj + 텍스처) | 38 ~ 367 MB | 1.0 ~ **11.6 s** | 0.7 ~ **8.5 GB** |
| **씬 전체 · 지오메트리 전용 ply** | **6.9 ~ 9.8 MB** | **0.09 ~ 0.15 s** | **0.08 ~ 0.10 GB** |
| 타일 12~15개 합 | 76 ~ 102 MB | 0.05 s/개 | 0.05 GB/개 |

씬 전체가 **타일 하나보다도 작다.** 168k 삼각형이 8.5 GB를 먹은 이유는 367 MB의 jpg 텍스처가 압축
해제되기 때문이고, **v1은 depth만 렌더하니 텍스처가 전부 낭비**다.

그리고 depth가 실제로 같은지 확인했다 — 파이프라인 유효 범위(5 m) 안에서 **1,628,548 픽셀 중 오차
>1 mm 인 것이 0개**. 전 범위로 넓히면 프레임당 9픽셀(0.0029%)이 다른데 그 픽셀의 depth가 11.9~14.4 m로
5 m clip 밖이다.

## 타일링을 버려서 없어지는 것
- `margin` 파라미터 튜닝 (그리고 그걸 재던 G4/R3의 margin 스윕 전체)
- 타일 경계 손실 가능성 → **원리적으로 불가능**해진다
- 프레임마다 타일 조회·교체(`tile_for_xy`, `clear_geometry` + `add_model`)
- 디스크 10배 (씬당 76~102 MB → 7~10 MB, 90씬 8 GB → 0.8 GB로 전부 RAM 캐시 가능)

## ⚠️ v2에서 재검토해야 한다
RGB를 렌더하기로 하면 텍스처가 다시 필요하고 8.5 GB 문제가 돌아온다. 그때는 타일링이 다시 의미를
가질 수 있다. 이 결정은 **"v1은 depth/BEV만"이라는 전제에 묶여 있다.**

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/01_build_scene_geo.py --scenes 17DRP5sb8fy,s8pcmisQ38h
      (--scenes 생략 시 mesh_root의 모든 씬)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))
from geometry_utils import find_scene_mesh  # noqa: E402

DEFAULT_OUT = 'data/embodiment_aug/scene_geo'


def geo_path(out_dir, scene):
    """규약을 한 곳에만 둔다. 로더(`embodiment_augment`)도 이 함수를 쓴다."""
    return Path(out_dir) / f'{scene}.ply'


def build_scene_geo(scene, mesh_root, out_dir, overwrite=False):
    """씬 mesh -> 지오메트리 전용 `.ply`. -> dict(정보) 또는 None(이미 있음).

    텍스처·UV를 **버린다**. depth 렌더에는 쓰이지 않고 RAM을 100배 먹는다(모듈 docstring).
    """
    import open3d as o3d
    f = geo_path(out_dir, scene)
    if f.exists() and not overwrite:
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    src = find_scene_mesh(Path(mesh_root), scene)
    t0 = time.perf_counter()
    m = o3d.io.read_triangle_mesh(str(src))
    t_load = time.perf_counter() - t0
    m.textures = []
    m.triangle_uvs = o3d.utility.Vector2dVector()
    o3d.io.write_triangle_mesh(str(f), m, write_ascii=False, compressed=False)
    ext = m.get_axis_aligned_bounding_box().get_extent()
    return dict(scene=scene, src_mb=src.stat().st_size / 1e6, out_mb=f.stat().st_size / 1e6,
                tris=len(m.triangles), load_s=t_load, extent=np.asarray(ext))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='', help='콤마열. 생략하면 mesh_root의 모든 씬')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_n1')
    ap.add_argument('--out_dir', default=DEFAULT_OUT)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    root = Path(args.mesh_root)
    scenes = ([s.strip() for s in args.scenes.split(',') if s.strip()]
              or sorted(p.name for p in root.iterdir() if p.is_dir()))
    made, skipped, failed, tot_src, tot_out = [], 0, [], 0.0, 0.0
    for sc in scenes:
        try:
            r = build_scene_geo(sc, root, args.out_dir, args.overwrite)
        except Exception as e:
            failed.append((sc, f'{type(e).__name__}: {e}')); continue
        if r is None:
            skipped += 1; continue
        made.append(r); tot_src += r['src_mb']; tot_out += r['out_mb']
        e = r['extent']
        print(f'  {sc:16s} {r["src_mb"]:6.1f} -> {r["out_mb"]:5.1f} MB · {r["tris"]/1e3:4.0f}k tri · '
              f'{e[0]:5.1f}x{e[1]:5.1f}x{e[2]:5.1f} m · 원본로드 {r["load_s"]:5.1f} s')
    print(f'\n생성 {len(made)} · 건너뜀(이미 있음) {skipped} · 실패 {len(failed)}')
    if made:
        print(f'  obj 합 {tot_src:.0f} MB -> ply 합 {tot_out:.0f} MB ({tot_out/max(tot_src,1e-9):.2f}배)')
    for sc, err in failed:
        print(f'  실패 {sc}: {err}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
