"""03h1 — vln_pe(H1) 용 navmesh: MP3D 씬 glb 로 habitat navmesh 를 **H1 몸 크기**로 다시 만든다.

공식 navmesh 는 vln_ce 로봇(반지름 0.10 m, 키 1.5 m)용이다. 그 위에서 계획한 sg 경로는 벽에 가까워
물리 H1 이 스치며 넘어진다 (06r: 평지 sg 넘어지지 않은 성공 12/19 vs GT 16/19).
만든 navmesh 를 `refine_paths.py --navmesh_dir` 로 주면 03 의 r_b 도 이 반지름을 따른다.

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--agent_radius` | navmesh 를 벽에서 깎는 반지름 = 03 의 r_b | 0.25 m — H1 다리 폭 ±0.2 m 보다 조금 크게. 노은역 로봇과 같은 값 |
| `--agent_height` | 머리 위 공간 | 1.5 m (habitat 기본) — 1.8 m 는 MP3D 낮은 곳에서 navmesh 가 끊긴다 (29hnd 섬 1 -> 9) |
| 나머지 | habitat 기본값 (02 E: 기본값 재빌드 = 공식본) | 반지름·키만 바꿔 비교가 그 효과만 보게 |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/vlnpe/tools/build_navmesh_h1.py --scenes 17DRP5sb8fy,1LXtFkjw3qL,29hnd4uzFmX,1pXnuDYAj8r
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
sys.path.insert(0, str(HERE))
import nav3d_utils as nu  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', required=True, help='쉼표 구분')
    ap.add_argument('--scene_root', default='data/InternData-N1-v0.5-mini/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--agent_radius', type=float, default=0.25)
    ap.add_argument('--agent_height', type=float, default=1.5, help='1.5 = habitat 기본. 1.8 (H1 키) 은 MP3D 의 낮은 곳(계단 밑·문틀)에서 navmesh 가 끊긴다 (29hnd 섬 1 -> 9) — H1 은 그 GT 경로를 실제로 걷는다')
    ap.add_argument('--out_dir', default=None, help='기본: logs/vlnpe/tools/build_navmesh_h1/r<반지름 cm>')
    return ap.parse_args()


def main():
    args = parse()
    out = Path(args.out_dir or f'scripts/dataset_converters/nav3d/logs/vlnpe/tools/build_navmesh_h1/r{int(round(args.agent_radius * 100)):03d}')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in args.scenes.split(','):
        off = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
        pf = nu.build_navmesh(f'{args.scene_root}/{s}/{s}.glb', {'agent_radius': args.agent_radius, 'agent_height': args.agent_height},
                              str(out / f'{s}.navmesh'))
        rows.append({'scene': s, 'official_area_m2': float(off.navigable_area), 'h1_area_m2': float(pf.navigable_area),
                     'official_islands': int(off.num_islands), 'h1_islands': int(pf.num_islands)})
        print(f"[03h1] {s}: 면적 {rows[-1]['official_area_m2']:.1f} -> {rows[-1]['h1_area_m2']:.1f} m² · 섬 {rows[-1]['official_islands']} -> {rows[-1]['h1_islands']}", flush=True)
    (out / 'summary.json').write_text(json.dumps({'args': vars(args), 'scenes': rows}, indent=1))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
