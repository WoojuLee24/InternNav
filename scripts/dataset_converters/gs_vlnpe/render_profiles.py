"""렌더 설정 프로파일 (조명·톤맵) — 이름 붙인 파라미터 묶음.

`04_render_obs_isaac.py` 안에 `MEASURED_RENDER_PRESET` dict로 박혀 있던 것을 꺼내
`path_profiles.py`와 **같은 규약**으로 옮긴 것이다. 공용 기반은 `profiles.py`.

프로파일
--------
`isaac_default`
    아무것도 건드리지 않는 상태. `--preset` 이전의 04 기본 동작과 같다
    (ambient 0 = raw `SimulationApp` 부팅에서 `/rtx/sceneDb/ambientLightIntensity`가 0,
    `film_iso` 100 = 톤맵 설정을 아예 안 만짐, 톤맵 op는 Isaac 기본 6=ACES).
    조명이 안 닿는 면이 어둡게 남아 GT와 크게 어긋난다 — 비교 기준선으로만 쓴다.

`vln_pe_measured`  ·  `vln_n1_measured`  (**현재 baseline**)
    126씬 전수 측정(2026-09-08)으로 고른 설정. 데이터셋별로 답이 **다르다**.

    | | vln_pe (61씬) | vln_n1 (65씬) |
    |---|---|---|
    | GT 대비 SSIM 중앙값 | 0.8232 → 0.8232 (유지) | 0.8384 → **0.8679** |
    | 톤맵 op | **6 (ACES) 유지** | **7 (Iray)** |
    | film_iso | **70 유지** | **90** |
    | rtx_ambient | 10.0 | 6.0 |

    왜 데이터셋별로 다른가:
    - **vln_n1은 op7이 65씬 중 62씬 우세**(+0.021)라 채택.
    - **vln_pe는 op7이 61씬 중 40씬 열세**(-0.013)라 6을 유지. iso 90도 중앙값 +0.0104지만
      11씬이 퇴보(최악 -0.041)하고 그 이득은 텍스처 효과(+0.25)의 4%라, 씬 간 톤 일관성을
      깨뜨릴 값어치가 없다고 판단해 70을 유지했다.

    **주의**: `tonemap_op`/`tonemap_crush`의 `None`은 "설정을 건드리지 않음"이라는 **유효한 값**이다
    (op 6 = Isaac 기본을 그대로 쓴다는 뜻). 0이나 빈 값이 아니다.

    이 측정 자체가 한 번 무효였다 — 처음엔 텍스처 심링크를 건너뛰어 **텍스처 없는 그림**을
    쟀다(SSIM 0.56 vs 0.83). 렌더 측정 스크립트는 반드시 파이프라인과 같은
    `load_scene_model()`을 거쳐야 한다.

커맨드
------
목록과 값:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/render_profiles.py

두 프로파일의 차이:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/render_profiles.py --diff vln_pe_measured vln_n1_measured
"""

from dataclasses import dataclass

import profiles

DEFAULT_PROFILE = 'vln_pe_measured'
# `04 --preset measured`가 데이터셋으로 프로파일을 고를 때 쓰는 표 (하위 호환).
MEASURED_BY_DATASET = {'vln_pe': 'vln_pe_measured', 'vln_n1': 'vln_n1_measured'}


@dataclass(frozen=True)
class RenderProfile(profiles.ProfileBase):
    """`04_render_obs_isaac.py`에 넘길 렌더 파라미터 묶음.

    필드 이름은 `04`의 CLI 인자 이름과 정확히 같다.
    """

    name: str
    light: str
    rtx_ambient: float
    film_iso: float
    tonemap_op: int = None      # None = /rtx/post/tonemap/op 을 건드리지 않음 (Isaac 기본 6=ACES)
    tonemap_crush: float = None  # None = crushBlacks 를 건드리지 않음. op 7에서만 읽힌다
    note: str = ''


PROFILES = {
    'isaac_default': RenderProfile(
        name='isaac_default',
        light='dome',
        rtx_ambient=0.0,
        film_iso=100.0,
        note='아무것도 안 건드린 상태(--preset 이전의 04 기본값). 비교 기준선 전용 — '
             'ambient 0이라 조명이 안 닿는 면이 어둡게 남는다',
    ),
    'vln_pe_measured': RenderProfile(
        name='vln_pe_measured',
        light='ambient_only',
        rtx_ambient=10.0,
        film_iso=70.0,
        tonemap_op=None,     # op7이 61씬 중 40씬 열세라 6(ACES) 유지
        tonemap_crush=None,
        note='126씬 전수 측정(2026-09-08). GT 대비 SSIM 중앙값 0.8232. '
             'iso 90은 중앙값 +0.0104지만 11씬 퇴보(최악 -0.041)라 70 유지',
    ),
    'vln_n1_measured': RenderProfile(
        name='vln_n1_measured',
        light='ambient_only',
        rtx_ambient=6.0,
        film_iso=90.0,
        tonemap_op=7,        # 65씬 중 62씬 우세 (+0.021)
        tonemap_crush=0.0,
        note='126씬 전수 측정(2026-09-08). GT 대비 SSIM 중앙값 0.8384 -> 0.8679. '
             'op7 Iray는 65씬 중 62씬 우세',
    ),
}


def get(name: str = None) -> RenderProfile:
    """이름으로 프로파일을 얻는다. `None`이면 현재 baseline."""
    return profiles.resolve(PROFILES, name, DEFAULT_PROFILE)


def for_dataset(dataset: str) -> RenderProfile:
    """`--preset measured` 하위 호환 — 데이터셋 이름으로 측정 프로파일을 고른다."""
    key = MEASURED_BY_DATASET.get(dataset)
    if key is None:
        raise KeyError(f'그 데이터셋의 측정 프로파일이 없다: {dataset!r} '
                       f'(있는 것: {", ".join(MEASURED_BY_DATASET)})')
    return PROFILES[key]


def names() -> list:
    return list(PROFILES)


def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--diff', nargs=2, metavar=('A', 'B'), default=None)
    ap.add_argument('--cli', default=None, metavar='NAME')
    args = ap.parse_args()

    if args.cli:
        print(' '.join(get(args.cli).cli_args()))
        return 0
    if args.diff:
        profiles.print_diff(PROFILES, args.diff[0], args.diff[1], DEFAULT_PROFILE)
        return 0
    profiles.print_registry(PROFILES, DEFAULT_PROFILE, '렌더 프로파일')
    print('json:')
    print(json.dumps({n: p.to_dict() for n, p in PROFILES.items()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
