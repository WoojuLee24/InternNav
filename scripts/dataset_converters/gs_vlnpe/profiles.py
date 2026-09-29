"""이름 붙인 설정 묶음의 **공용 기반** — 프로파일 규약을 한 곳에 둔다.

왜 필요한가
-----------
설정을 이름으로 고르는 장치를 스크립트마다 따로 만들었더니 **같은 로직이 복붙**됐다.
`03_sample_gt_paths.py`의 `apply_path_profile`과 `04_render_obs_isaac.py`의 `apply_preset`이
글자만 다르고 하는 일이 같았다 — "사용자가 명시하지 않은 인자만 프로파일 값으로 채운다".

이 파일이 그 규약을 한 번만 정의한다.

세 가지 영역
------------
| 영역 | 프로파일 정의 | 스위치 | 모양 |
|---|---|---|---|
| 경로 계획 | `path_profiles.py` | `--path_profile` | CLI 인자로 펼침 |
| 렌더 (조명·톤맵) | `render_profiles.py` | `--render_profile` / `--preset` | CLI 인자로 펼침 |
| 카메라 (해상도·intrinsic·depth 범위) | `camera_profiles.py` | `--camera` | **객체로 직접 씀** |

앞의 둘은 이 파일의 `ProfileBase`·`apply_to_args`를 쓴다. **카메라만 규약이 다르다** —
값이 CLI 인자에 일대일 대응하지 않고(`k`가 3x3 행렬) 렌더러가 객체째 받는다. 그리고
`apply_real/camera_profiles.py`에 사본이 따로 있어서(노은역 전용 프로파일) 합치면 그쪽이 깨진다.
그래서 카메라는 그대로 두고 여기 표로만 위치를 적어 둔다.

우선순위 (세 영역 공통)
-----------------------
```
개별 인자 (--refine_radius 0.4)   ← 가장 셈. 프로파일이 절대 덮지 않는다
프로파일  (--path_profile ...)
스크립트 argparse 기본값           ← 프로파일을 안 주면 이것 (종전 동작)
```
"""

from dataclasses import asdict, replace

SKIP_FIELDS = ('name', 'note')


class ProfileBase:
    """프로파일 dataclass가 함께 상속하는 공통 메서드.

    `@dataclass(frozen=True)`와 같이 쓴다. 필드 이름은 **그 스크립트의 CLI 인자 이름과
    정확히 같게** 두는 것이 규약이다 — 그래야 `cli_args()`/`apply_to_args()`가 이름을
    옮겨 적는 실수 없이 동작한다. `name`/`note`는 메타데이터라 인자에서 제외된다.
    """

    def to_dict(self) -> dict:
        return asdict(self)

    def values(self) -> dict:
        """CLI 인자에 대응하는 값만 (`name`/`note` 제외)."""
        return {k: v for k, v in asdict(self).items() if k not in SKIP_FIELDS}

    def cli_args(self) -> list:
        """`['--r_b', '0.2', ...]` 형태. `None` 값은 **넘기지 않는다** —
        None은 "건드리지 않음"을 뜻하는 유효한 값이라 문자열 'None'으로 넘기면 안 된다."""
        out = []
        for key, val in self.values().items():
            if val is None:
                continue
            out += [f'--{key}', str(val)]
        return out

    def with_(self, **kw):
        """일부 값만 바꾼 사본 (frozen이라 원본은 안 바뀐다). 스윕할 때 쓴다."""
        return replace(self, **kw)


def apply_to_args(parser, args, profile) -> list:
    """프로파일 값으로 **사용자가 명시하지 않은 인자만** 채운다.

    "명시하지 않았다"는 판정은 `parser.get_default(name)`과 현재 값이 같은지로 한다.
    argparse는 "기본값 그대로"와 "기본값과 같은 값을 직접 준 것"을 구분하지 못하는데,
    그 둘은 결과가 같으므로 실무상 문제가 없다.

    `parser`에 없는 필드는 조용히 건너뛴다 — 프로파일이 여러 스크립트에서 공유될 때
    일부 스크립트만 그 인자를 갖는 경우가 있다.

    반환: 실제로 채운 `(이름, 값)` 목록 (로그용).
    """
    filled = []
    for key, val in profile.values().items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, val)
            filled.append((key, val))
    return filled


def resolve(registry: dict, name: str, default: str):
    """이름으로 프로파일을 얻는다. 없으면 있는 목록을 알려주며 실패한다."""
    key = name or default
    if key not in registry:
        raise KeyError(f'그런 프로파일이 없다: {key!r} (있는 것: {", ".join(registry)})')
    return registry[key]


def print_registry(registry: dict, default: str, title: str) -> None:
    print(f'{title} — 현재 baseline: {default}\n')
    for p in registry.values():
        print(f'[{p.name}]')
        if getattr(p, 'note', ''):
            print(f'  {p.note}')
        for k, v in p.values().items():
            print(f'    {k:20s} {v!r}')
        print()


def print_diff(registry: dict, a_name: str, b_name: str, default: str) -> None:
    a, b = resolve(registry, a_name, default), resolve(registry, b_name, default)
    va, vb = a.values(), b.values()
    keys = [k for k in va if va[k] != vb[k]]
    print(f'{a.name} -> {b.name} : 다른 값 {len(keys)}개')
    for k in keys:
        print(f'  {k:20s} {va[k]!r:>12s} -> {vb[k]!r}')
