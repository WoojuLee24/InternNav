"""경로 계획 설정 프로파일 — 이름 붙인 파라미터 묶음.

왜 필요한가
-----------
경로 계획 설정이 **세 곳에 흩어져 있었고 어디에도 이름이 없었다.**

| 어디 | 무엇 |
|---|---|
| `esdf_utils.py` 상수 | `ROBOT_RADIUS_M=0.25` · `ASTAR_CELL_M=0.20` · `WAYPOINT_SPACING_M=0.8` · `H_NAV_M=0.10` |
| `03_sample_gt_paths.py` argparse 기본값 | `refine_radius=0.10` · `smooth='cubic'` · `downsample_mode='any'` |
| `reports.md` 권장 커맨드 문자열 | `--r_b 0.20 --h_nav_ratio 0.12 --refine_radius 0.20` |

셋이 **서로 달랐다.** 실제로 쓰던 설정은 코드 기본값이 아니라 md에 적힌 커맨드였고, 그래서
"이전 설정"을 부를 이름이 없었다. 스윕을 돌릴 때마다 폴더 이름(`r0.20`, `c3`)으로만 구분했다.

이 파일이 그 묶음에 이름을 준다. `camera_profiles.py`와 같은 규약이다(그쪽은 카메라, 이쪽은 경로).

프로파일
--------
`legacy`
    2026-09-09까지 쓴 설정. `reports.md`의 vln_pe 권장 커맨드 + `03` 기본값의 조합.
`reproduce_v1`  (**현재 baseline**, 2026-09-10 확정)
    vln_pe 45씬 · 2,274 에피소드 전수 비교로 고른 설정. 30여 개 조합을 재서 정했다.

    | | legacy | reproduce_v1 |
    |---|---|---|
    | chamfer 중앙값 | 0.191 m | **0.167 m** |
    | Fréchet 중앙값 | 0.553 m | **0.480 m** |
    | Fréchet p90 | 1.446 m | **1.295 m** |
    | 같은 루트(Fréchet<0.8m) | 73.1 % | **78.8 %** |
    | 하드 충돌 에피소드 | 18 | **0** |
    | r_b 침범 에피소드 | 427 | **113** |
    | `no_collision` 게이트 통과 씬 | 36/45 | **45/45** |
    | 쓸 수 있는 에피소드(충돌 0 + 같은 루트) | 1,649 | 1,700 |

    바뀐 값은 네 개다 — `refine_radius` 0.20→0.30, `waypoint_spacing_m` 0.8→0.2,
    `smooth` cubic→bezier, `downsample_mode` any→majority.

    측정에서 배운 것:
    - **`refine_radius`를 키우면 루트가 GT에 가까워진다** (사람은 벽을 스치지 않고 복도
      가운데로 걷는데 A*는 장애물 코너를 스친다). 단 0.5 이상에서 충돌이 폭증하고
      0.7 이상은 루트도 무너진다(1.0에서 38.5%).
    - **충돌은 refine이 아니라 스무딩이 만든다.** refine된 웨이포인트는 벽에서 떨어져
      있는데(최소 여유 0.071 m) 그 사이를 잇는 스플라인이 벽을 뚫는다(0.000 m).
      `waypoint_spacing_m`을 좁히면 곡선이 짧게 끊겨 충돌이 사라진다(18→0).
    - **`bezier`가 `cubic`보다 낫다.** 같은 refine에서 기준선 Fréchet을 0.438→0.399로
      낮춘다 — 곡선이 원래 꺾은선에 더 붙는다.
    - **A\* 격자를 굵게 하면 안 된다.** 0.25에서 충돌 159 에피소드, 0.30에서 405.
    - **한계**: 파라미터로는 78~79%가 끝이다. 시험한 30개 설정이 전부 73~79%에 있었다.
      남는 21%는 위상이 다른 경우(4.5%, GT가 36% 더 긴 길로 돌아감)와 "A*는 최단경로,
      사람은 아님"의 차이다(GT 회전량이 우리의 4배: 130 vs 34 °/m). 더 올리려면 비용
      함수를 바꿔야 한다.

커맨드
------
프로파일 목록과 값:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/path_profiles.py

두 프로파일의 차이만:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/path_profiles.py --diff legacy reproduce_v1
"""

from dataclasses import dataclass

import profiles

DEFAULT_PROFILE = 'reproduce_v1'


@dataclass(frozen=True)
class PathProfile(profiles.ProfileBase):
    """`03_sample_gt_paths.py`에 넘길 경로 계획 파라미터 묶음.

    필드 이름은 `03`의 CLI 인자 이름과 **정확히 같다** — 그래야 프로파일을 인자로 펼칠 때
    이름을 옮겨 적는 실수가 생기지 않는다.
    """

    name: str
    r_b: float
    h_nav_ratio: float
    refine_radius: float
    refine_mode: str
    astar_cell_m: float
    downsample_mode: str
    smooth: str
    smooth_step: float
    waypoint_spacing_m: float
    note: str = ''


PROFILES = {
    # 2026-09-09까지 쓴 설정. reports.md의 vln_pe 권장 커맨드(r_b 0.20 / h_nav_ratio 0.12 /
    # refine_radius 0.20) + 나머지는 03의 argparse 기본값.
    'legacy': PathProfile(
        name='legacy',
        r_b=0.20,
        h_nav_ratio=0.12,
        refine_radius=0.20,
        refine_mode='argmax',
        astar_cell_m=0.20,
        downsample_mode='any',
        smooth='cubic',
        smooth_step=0.05,
        waypoint_spacing_m=0.8,
        note='2026-09-09까지 쓴 설정 (reports.md 권장 커맨드 + 03 기본값). '
             'chamfer 0.191 / Fréchet 0.553 / 같은루트 73.1% / 충돌 18ep',
    ),
    # 2026-09-10 확정. vln_pe 45씬 2,274 에피소드로 30여 조합을 비교해 고름.
    # 스윕 원본: logs/gs-vlnpe/{refine_sweep,pareto,axes2,combo}/ (실험명 c3)
    'reproduce_v1': PathProfile(
        name='reproduce_v1',
        r_b=0.20,
        h_nav_ratio=0.12,
        refine_radius=0.30,          # legacy 0.20 — 키우면 루트가 GT에 가까워진다
        refine_mode='argmax',
        astar_cell_m=0.20,
        downsample_mode='majority',  # legacy any — any는 낙관적이라 좁은 틈을 지난다
        smooth='bezier',             # legacy cubic — 꺾은선에 더 붙어 기준선이 낮아진다
        smooth_step=0.05,
        waypoint_spacing_m=0.2,      # legacy 0.8 — 좁히면 스플라인이 벽을 못 뚫는다
        note='2026-09-10 확정 baseline. vln_pe 45씬 2,274ep 전수 비교. '
             'chamfer 0.167 / Fréchet 0.480 (p90 1.295) / 같은루트 78.8% / '
             '충돌 0ep / r_b 침범 113ep / 게이트 45/45씬',
    ),
}


def get(name: str = None) -> PathProfile:
    """이름으로 프로파일을 얻는다. `None`이면 현재 baseline."""
    return profiles.resolve(PROFILES, name, DEFAULT_PROFILE)


def names() -> list:
    return list(PROFILES)


def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--diff', nargs=2, metavar=('A', 'B'), default=None,
                    help='두 프로파일의 다른 값만 출력')
    ap.add_argument('--cli', default=None, metavar='NAME',
                    help='그 프로파일을 03에 넘길 인자 문자열로 출력')
    args = ap.parse_args()

    if args.cli:
        print(' '.join(get(args.cli).cli_args()))
        return 0

    if args.diff:
        profiles.print_diff(PROFILES, args.diff[0], args.diff[1], DEFAULT_PROFILE)
        return 0

    profiles.print_registry(PROFILES, DEFAULT_PROFILE, '경로 계획 프로파일')
    print('json:')
    print(json.dumps({n: p.to_dict() for n, p in PROFILES.items()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
