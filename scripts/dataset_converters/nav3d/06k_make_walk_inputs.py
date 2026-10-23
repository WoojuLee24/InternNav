"""06k 입력 만들기 — paths json 의 모든 episode 를 RBQ 걷기 입력(0.5 m 경유점 + 출발 yaw)으로 바꾼다.

전에는 손으로 만들던 첫 걷기 입력 `logs/06k_rbq_walk/eps_noeun_all_v2.json` 을 다시 만드는 단계다
(04c 출력 `noeun_station_paths_meshfloor_straight.json` 으로 200/200 그대로 재현, 2026-10-23 확인).
규칙은 04d `eps` · 04f · 04g 와 같다 (`walk_inputs.py`): 경로를 수평 0.5 m 간격으로 · 출발 yaw = R2R train 분포 (episode_id seed).

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--paths_json` | 04 형식 paths json (보통 04c 출력) | 06k 가 걸을 경로 |
| `--out` | 06k `--episodes_json` 으로 줄 파일 | 기본값 없음 — 기존 결과를 실수로 덮지 않게 |
| `--waypoint_decimals` | 경유점 좌표 반올림 자릿수 | 4 = 손으로 만든 eps_noeun_all_v2 와 같음 |
| `--start_seed` | 출발 yaw 를 뽑는 seed | 0 = 06 `--start_seed` 기본 · 04d/04f/04g 와 같음 |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/06k_make_walk_inputs.py --paths_json scripts/dataset_converters/nav3d/logs/04_export_paths/noeun_station/noeun_station_paths_meshfloor_straight.json --out scripts/dataset_converters/nav3d/logs/06k_rbq_walk/eps_noeun_all_v2.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from walk_inputs import r2r_initial, resample, start_yaw  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scene', default='noeun_station')
    ap.add_argument('--key_prefix', default='noeun', help='episode key = <prefix>:<episode_id>:<kind> (06k · 04d 가 이 형식으로 읽는다)')
    ap.add_argument('--background_usd', default='/ws/assets/GS_USDZ/noeun_station_collision.usdz', help='06k 가 불러올 씬 (isaac 컨테이너 경로)')
    ap.add_argument('--waypoint_decimals', type=int, default=4)
    ap.add_argument('--start_seed', type=int, default=0)
    ap.add_argument('--r2r_raw', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/train/train.json.gz')
    return ap.parse_args()


def main():
    args = parse()
    d = json.loads(Path(args.paths_json).read_text())
    init = r2r_initial(args.r2r_raw)
    E = []
    for ep in d['episodes']:
        T = np.asarray(ep['trajectory'], np.float64)
        E.append({'key': f"{args.key_prefix}:{ep['episode_id']}:{ep['kind']}",
                  'waypoints': resample(T).round(args.waypoint_decimals).tolist(),
                  'start_yaw': start_yaw(ep, T, args.start_seed, init)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'scene': args.scene, 'background_usd': args.background_usd, 'episodes': E}))
    print(f'[06k inputs] {len(E)} ep -> {args.out}')


if __name__ == '__main__':
    main()
