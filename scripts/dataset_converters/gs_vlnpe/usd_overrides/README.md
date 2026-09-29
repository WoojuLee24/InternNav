# usd_overrides — 파이프라인에 먹이는 덧칠 레이어 모음

원본 USD를 **한 바이트도 고치지 않고** 값만 바꿔치는 `.usda` 레이어들을 여기 모아 둔다.
렌더 스크립트의 `--usd_override` 인자에 그대로 넣으면 된다.

```
--usd_override scripts/dataset_converters/gs_vlnpe/usd_overrides/noeun_station_collision.visible.usda
```

## 원리 (한 문단)

`.usda`가 원본을 `subLayers`로 깔고, 그 위에 `over`로 바꿀 값만 적는다. USD가 열 때 두 장을
겹쳐서 하나의 씬으로 합성하고, **위에 적힌 값이 이긴다.** 원본은 읽기만 하므로 sha256이
안 바뀐다 (빌더가 매번 확인해서 `원본 무변경 예`로 찍는다).

## `cp`로 옮기면 안 된다 — 반드시 빌더로 만들어라

`.usda` 안의 sublayer 경로는 **그 파일이 놓인 위치 기준 상대경로**다.

```
subLayers = [ @../../../../data/GS_USDZ/Subway/noeun_station_collision.usdz@ ]
```

다른 폴더에서 만든 `.usda`를 여기로 복사하면 이 경로가 어긋나 sublayer가 끊긴다. 그러면
에러가 아니라 **조용히 빈 씬**이 된다(합성 prim 1개). 그래서 항상 최종 위치에서 새로 만든다.

## 커맨드

목록과 현재 상태:

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --list`

전체 다시 만들기 (이미 만들어진 것만 갱신 — 씬을 새로 늘리지 않는다):

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --recipe all`

mp3d 텍스처 레이어를 전 씬(90개)에 대해 만들기:

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --recipe mp3d_textures --scene all`

한 씬만:

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --recipe mp3d_textures --scene 17DRP5sb8fy`

안 만들고 검사만 (sublayer가 끊겼는지 전수 확인):

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --verify_only`

Isaac Sim을 안 띄운다 — `pxr`만 쓴다.

## 지금 들어 있는 것

| 파일 | 무엇을 덮어쓰나 | 왜 |
|---|---|---|
| `noeun_station_collision.visible.usda` | 숨은 충돌 메시 1개를 `visibility=inherited` | 원본 usdz는 충돌용 메시가 `invisible`이라 렌더가 빈다. 파이프라인은 이걸 **런타임 파이썬**으로 뒤집는데, 그 함수는 파일 이름이 `_collision.usdz`로 끝나는지로 대상을 고른다. 레이어를 쓰면 이름 규칙과 파이썬 개입 없이 같은 결과가 나온다 |
| `mp3d_pe/<scene>.textures.usda` (90개) | 텍스처 `inputs:diffuse_texture`를 레포 안 상대경로로 | 원본 `isaacsim_*.usd`에 `/ssd/share/Matterport3D/...` 절대경로가 박혀 있다. 그 경로가 없어서 파이프라인이 컨테이너 밖에 심링크를 만든다(부수효과). 90씬 중 심링크가 있는 건 2씬뿐이라 나머지는 텍스처 없이 렌더된다 |

## 새 덧칠 추가하기

`recipes.py`의 `RECIPES` dict에 항목 하나를 더한다. `build.py`는 안 건드린다.
새로운 **방식**(kind)이 필요하면 `build.py`에 `build_<kind>()` 함수를 더하고
`main()`의 `if/elif/else`에 한 줄 붙인다.

## 레이어를 안 쓰는 게 기본이다

파이프라인의 기본 동작은 **그대로다** — `--usd_override`를 안 주면 종전 경로(런타임 파이썬 +
심링크)를 그대로 탄다. 이 폴더는 그 경로를 대체할 수 있음을 증명해 둔 것이고, 기본값 전환은
별도 결정 사항이다.

## 검증 기록

노은역: 원본 usdz + 런타임 파이썬 뒤집기 vs 이 레이어 — 두 렌더의 판정이 소수점까지 같다.

```
기준     전체 PASS · mesh-anchor median 0.00623 m / worst 0.00623 m
덧칠     전체 PASS · mesh-anchor median 0.00623 m / worst 0.00623 m
```

덧칠 쪽 로그에는 `enabled 1 hidden collision mesh(es)`(파이썬이 뒤집은 흔적)가 **없다.**
파이썬이 손을 안 댔는데 같은 데이터가 나왔다는 뜻이다.
