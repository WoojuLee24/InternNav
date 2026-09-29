"""파이프라인에 `--usd_override`로 먹일 덧칠 레이어(.usda) 목록 — **여기만 고치면 된다.**

`build.py`는 이 dict를 읽어서 `.usda`를 만들 뿐이다. 새 덧칠을 추가할 때 코드는 건드리지 않고
아래 표에 항목 하나만 더한다.

각 항목의 뜻
------------
kind        만드는 방식. `visibility` = 숨은 충돌 메시를 보이게, `textures` = 텍스처 절대경로를
            상대경로로 재지정. (`build.py`가 아는 종류만 쓸 수 있다)
base        덧칠 대상 원본. **이 파일은 절대 수정되지 않는다** — sublayer로 깔기만 한다.
out         만들어질 `.usda` 경로 (이 폴더 기준 상대). `{scene}`은 씬 ID로 치환된다.
doc         왜 필요한 덧칠인지 한 줄. `build.py --list`가 그대로 보여 준다.
"""

RECIPES = {
    # 노은역: 원본 usdz에 충돌용 메시가 visibility=invisible로 숨어 있어 렌더가 텅 빈다.
    # 파이프라인은 이걸 런타임 파이썬(`expose_collision_meshes_for_rendering`)으로 뒤집는데,
    # 그 함수는 **파일 이름이 `_collision.usdz`로 끝나는지**로 대상을 고른다. 이 레이어를
    # 쓰면 그 이름 규칙과 파이썬 개입 없이도 같은 결과가 나온다 (실측: mesh-anchor 동일).
    'noeun_collision_visible': {
        'kind': 'visibility',
        'base': 'data/GS_USDZ/Subway/noeun_station_collision.usdz',
        'out': 'noeun_station_collision.visible.usda',
        'doc': '노은역 usdz의 숨은 충돌 메시를 보이게 — 런타임 파이썬 뒤집기를 레이어로 대체',
    },
    # mp3d_pe: `isaacsim_*.usd`에 텍스처가 `/ssd/share/Matterport3D/...` 절대경로로 박혀 있다.
    # 그 경로는 이 컨테이너에 없어서 파이프라인이 `ensure_texture_symlink`로 컨테이너 밖에
    # 심링크를 만든다(부수효과). 이 레이어는 같은 텍스처를 **레포 안 상대경로**로 다시 가리켜
    # 심링크 없이도 텍스처가 붙게 한다 (실측: 90씬 중 심링크가 있는 건 2씬뿐).
    'mp3d_textures': {
        'kind': 'textures',
        'usd_root': 'data/scene_data/mp3d_pe',
        'variant': 'isaacsim',
        'out': 'mp3d_pe/{scene}.textures.usda',
        'doc': 'mp3d_pe 씬 텍스처를 레포 안 상대경로로 재지정 — /ssd/share 심링크 의존 제거',
    },
}
