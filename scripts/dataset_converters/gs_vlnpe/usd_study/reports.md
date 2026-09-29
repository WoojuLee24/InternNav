# usd_study — USD 실습 5개 결과 (S1~S5)

USD 공부 5개 항목을 각각 **돌리면 결과가 나오는 스크립트 1개**로 만들어 돌린 기록.
공략 대상은 두 종류였다 — 파이프라인에 쌓여 있던 **문제 2개**(파일 4중 복제 · 컨테이너 밖
심링크)와, 코드·리포트에 **"원인을 못 좁혔다"고 적혀 있던 것 3개**(밝기 노브 무반응 ·
렌더 30분 · 남은 화질 차이의 정체).

작성 2026-09-04. 상세 경위는 `.claude/memory/260904_usd_study_result.md`.
기존 파이프라인 리포트는 `../reports.md`.

## 쉽게 요약 (이것만 읽어도 됨)

USD 파일을 **읽기만** 하던 걸 **직접 써보기**로 바꾼 작업이다. 그 과정에서 **화질이 좋아지는
설정을 발견**했다.

### 알아낸 것

**① 파일 4개의 정체를 알았다**
같은 방이 4가지 버전으로 저장돼 있는데 뭐가 다른지 아무도 몰랐다. 이제 안다.
하나는 예상과 **완전히 달랐다** — `_non_metric`은 "단위만 다른 것"이 아니라 **위아래 방향이
뒤집혀 있고(Y-up) 충돌 정보가 아예 없는** 변환 전 원본이다.

**② 레이어를 직접 써봤다 — 원래 목표**
원본은 안 건드리고 위에 얇은 파일(`.usda`) 한 장을 얹어서 값을 바꿨다. 포토샵에서 원본 사진 위에
보정 레이어를 올리는 것과 같다. 원본 지문(sha256)이 그대로인 것도 확인했다.

**③ 컨테이너 밖에 만드는 바로가기를 없앨 수 있다**
지금 코드는 `/ssd/share/...`라는 **컨테이너 밖 경로에 바로가기(심링크)를 만든다.** 환경이 바뀌면
깨진다. 레이어로 대체 가능함을 확인했다.

**④ 밝기 노브가 왜 안 먹었는지 알았다**
`crushBlacks`를 아무리 돌려도 그림이 안 변했다. 원인은 **톤맵 설정 번호가 6번(ACES)이고,
그 노브는 7번(Iray) 전용**이라는 것이다. Isaac Sim UI 소스로 확인했다.
기존 코드 주석에 "6번 = Iray"로 적혀 있었는데 그게 틀렸다.

**⑤ 그러다 화질이 좋아졌다 — 가장 큰 소득**
번호를 6에서 **7(Iray)**로 바꾸니 화질 점수(SSIM)가 **0.875 → 0.910**. 이전에 "아무리 해도
0.904가 한계"라고 적어둔 선을 넘었다. 이전 실험이 6번의 *하위 노브만* 돌리고 **번호 자체를
바꿔본 적이 없어서** 놓친 것이다.
**아직 채택 아님** — 씬 하나에서만 확인했고, 7로 가면 `crushBlacks`가 살아나므로 그 값도
같이 정해야 한다(기본 0.5 그대로면 오히려 0.705로 떨어진다).

### 못 푼 것 · 아닌 것

- **렌더가 30분 걸리던 문제** → 범인을 **4개로 좁혔다.** "렌더 다 끝날 때까지 기다려" 성격의
  설정 3개가 **느린 쪽에만 켜져** 있었다. 확정은 못 했다.
- **남은 화질 차이가 재질(머티리얼) 때문인가?** → **아니다.** 의심했던 값이 무해했다.
  다만 색공간이 **어디에도 안 적혀 있어서** 나중에 조용히 깨질 수 있다.

### 안전한가

원본 파일은 하나도 안 바꿨고(지문 대조), 기존 스크립트는 한 줄도 안 고쳤다.
기존 커맨드를 다시 돌려 **화질 0.876**이 그대로 나오는 것까지 확인했다.

아래는 근거·수치·재현 커맨드다.

---

## 용어 (모르는 단어가 나오면 여기부터)

USD 파일은 **3D 장면을 적어둔 문서**다. 파워포인트 파일처럼 안에 물건들이 들어 있고,
포토샵처럼 **여러 장을 겹쳐서** 하나의 결과를 만든다. 이 두 가지만 잡으면 나머지는 따라온다.

### 문서 구조

| 용어 | 쉬운 말 | 우리 데이터에서의 예 |
|---|---|---|
| **Stage** | 겹쳐서 **합쳐진 최종 결과**. "열어놓은 장면" | 파일 4개 중 하나를 열면 stage 하나 |
| **Prim** | 문서 안의 **물건 하나**. 폴더처럼 경로로 씀 | `/Root/geometry/chunk000_group000_sub002` = 벽 조각 하나 |
| **Attribute** | 그 물건의 **속성값** | `visibility`(보임/안보임), 텍스처 파일 경로 |
| **Mesh** | 삼각형으로 된 **실제 형상** | 이 씬은 prim 124개 중 **Mesh가 73개** |
| **defaultPrim** | 남이 이 파일을 가져다 쓸 때 **기본으로 가져올 물건** | `fixed.usd`는 `/Root`, `isaacsim_*.usd`는 `/isaacsim_<hash>` |
| **upAxis** | **어느 축이 위쪽**인가 | 대부분 `Z`, `_non_metric`만 `Y`(뒤집혀 있음) |
| **metersPerUnit** | 숫자 **1이 몇 미터**인가 | `0.01`=센티미터, `1.0`=미터 |
| **usdz** | USD와 텍스처를 **zip처럼 한 파일로 묶은 것** | 노은역 `noeun_station_collision.usdz` (2.2 GB) |

### 레이어 겹치기 (이 실습의 핵심)

| 용어 | 쉬운 말 |
|---|---|
| **Layer** | **파일 한 장.** 여러 장을 겹쳐 stage가 된다 |
| **Sublayer** | **아래에 깔아둔 파일.** 위 파일이 아래 파일을 덮어쓴다 |
| **Composition** | 여러 장을 **겹쳐 하나로 합치는 규칙** 전체 |
| **Composition arc** | **"이 물건의 값이 어느 파일에서 왔는가" 연결선.** 겹친 파일이 3장이면 arc 3개, **위에 있는 게 이김** |
| **Opinion** | 각 레이어가 내는 **"이 값은 이거야"라는 주장.** 위 레이어 주장이 이김 |
| **Override (`over`)** | 위 레이어에서 **값 덮어쓰기.** 원본은 안 바뀐다 |
| **Reference** | 다른 파일을 **가져와 붙이기** (sublayer와 달리 특정 위치에 끼워 넣음) |
| **Payload** | **"필요할 때 열기"** 방식 참조. 큰 파일을 미리 안 읽게 하는 용도 |
| **LIVRPS** | 겹쳤을 때 **누가 이기는지의 우선순위 규칙** 이름 (Local > Inherits > Variants > References > Payloads > Specializes) |

S1이 한 일을 이 말로 쓰면 이렇다:

> 원본 파일(`fixed.usd`)을 **sublayer로 깔고**, 그 위 빈 파일에 "저 벽 조각은 안 보이게"라고
> **override**를 적었다. 겹쳐 읽으니(`Stage`) 안 보이게 나왔고, **원본 파일은 그대로**였다.

### 규격 · 종류

| 용어 | 쉬운 말 |
|---|---|
| **Schema** | 그 물건이 **무슨 종류인가** 하는 규격 |
| **Typed schema** | 물건의 **종류 자체** (이건 Mesh다) |
| **Applied API schema** | 나중에 **덧붙인 기능** (이건 물리 충돌도 한다) |
| **PhysicsCollisionAPI** | "이 물건은 **물리 충돌 대상**" 표시. 로봇이 부딪히는 대상 |
| **Kind** | 이 물건이 **부품(component)인지 조립품(assembly)인지** 표시 |

### 파일 경로 찾기

| 용어 | 쉬운 말 |
|---|---|
| **Asset path** | USD 안에 적힌 **"텍스처 이미지가 어디 있다"는 문자열** |
| **Resolve** | 그 문자열로 **실제 파일을 찾아내는 것** |
| **Ar (resolver)** | 그 찾기를 담당하는 모듈 |
| **상대경로 / 절대경로** | `./textures/a.jpg`(파일 옆) / `/ssd/share/...`(고정 위치). 절대경로가 박히면 다른 환경에서 깨진다 |
| **Symlink (심링크)** | 파일시스템 **바로가기**. 지금 코드가 깨진 절대경로를 이걸로 우회하고 있다 |

### 표면 재질 · 렌더

| 용어 | 쉬운 말 |
|---|---|
| **Material / Shader** | **표면 재질** 설정 (색·거칠기·텍스처) |
| **MDL / OmniPBR** | NVIDIA의 재질 방식. 이 씬이 쓰는 것 |
| **UsdPreviewSurface** | USD 표준 재질 방식. **이 씬은 이게 아니다** |
| **colorSpace** | 이미지 **색을 어떻게 해석할지** (`sRGB` / `raw`). 틀리면 화면이 확 밝아진다 |
| **Tonemap op** | RTX가 **밝기 곡선을 어떤 방식으로 압축할지** 고르는 번호 (0~7). 사진의 "필름 종류" |
| **Render delegate** | 실제로 그림을 그리는 **렌더 엔진 종류** (RTX Real-Time / Path Tracing) |
| **carb settings** | Isaac Sim **내부 설정값 트리**. `/rtx/...` 같은 경로로 접근 |

### 측정 지표

| 용어 | 쉬운 말 |
|---|---|
| **SSIM** | 두 이미지가 **얼마나 닮았는지** 0~1 점수. 1이면 똑같다 |
| **sha256** | 파일 **지문**. 값이 같으면 파일이 안 바뀌었다는 증거 |
| **잔차 `1-SSIM`** | 아직 안 맞는 정도. `SSIM 0.904` → 잔차 `0.096` |

---

## 실행 환경 (측정해서 확인)

- `pxr` = **OpenUSD 0.26.5** (`/workspace/isaaclab/_isaac_sim/kit/python/.../pxr`)
- **`usdcat`/`usdchecker`/`usdview`/`usdzip` CLI는 이 컨테이너에 없다** → 전부 파이썬 API로
- `UsdUtils.ComplianceChecker` · `UsdUtils.CreateNewUsdzPackage` 사용 가능
- 실행은 항상 `/workspace/isaaclab/_isaac_sim/python.sh`
- **`pxr`을 `SimulationApp` 부팅 전에 import하면 Omniverse 확장이 전부 죽는다**(실측:
  `omni.usd`·`omni.UsdMdl`·`Semantics`가 "extension class wrapper ... has not been created yet").
  렌더가 필요한 S3/S5는 부팅을 먼저 하고 `pxr`을 함수 안에서 lazy import한다.


## 리포트 링크

| 실습 | Artifact | 로컬 |
|---|---|---|
| S1 레이어 겹쳐쓰기 (composition) | [링크](https://claude.ai/code/artifact/a743e42b-5920-40e0-b017-8ba8fdc0751e) | `logs/gs-vlnpe/usd_study/s1_layers/17DRP5sb8fy/report.html` |
| S2 asset resolution | [링크](https://claude.ai/code/artifact/a3befa4d-6ed2-46fa-b111-6f8f810f1b09) | `logs/gs-vlnpe/usd_study/s2_assets/17DRP5sb8fy/report.html` |
| S3 렌더러/톤매퍼 | [링크](https://claude.ai/code/artifact/3e05ea2a-ad9d-4973-ae58-4a75f00f3a93) | `logs/gs-vlnpe/usd_study/s3_renderer/17DRP5sb8fy_vln_n1/report.html` |
| S4 부팅 경로 carb diff | [링크](https://claude.ai/code/artifact/64b02d2d-f004-4263-9c05-f5c96c7a41b6) | `logs/gs-vlnpe/usd_study/s4_boot/report.html` |
| S5 머티리얼 | [링크](https://claude.ai/code/artifact/0ba95b10-e156-40c2-a9a8-01ac6b1f3585) | `logs/gs-vlnpe/usd_study/s5_materials/17DRP5sb8fy_vln_n1/report.html` |

---

## 한눈에 — 무엇이 해결됐나

| | 대상 | 결과 |
|:--:|---|---|
| 부채 1 | 같은 씬이 USD 파일 4개로 복제 | **측정 완료.** 추정이 틀렸던 부분 있음 (S1) |
| 부채 2 | `/ssd/share/...`에 시스템 심링크 생성 | **override 레이어로 대체 가능 확인** (S2) |
| 미해결 A | Iray `crushBlacks`/`burnHighlights` 무반응 | **원인 규명 완료 — 톤맵 op이 6이라서** (S3) |
| 미해결 B | `AppLauncher` 30분 vs raw 수십 초 | **후보 4개로 좁힘** (S4) |
| 미해결 C | 잔차 `1-SSIM=0.096`에 머티리얼 몫? | **없음 — 기존 결론 강화** (S5) |
| 보너스 | — | **톤맵 op 변경으로 SSIM 0.875 → 0.910 (+0.035)** (S3) |
| 적용 | 만든 레이어를 실제로 쓸 수 있나 | **된다 — 렌더 결과가 기존과 동일** (S6·S7) |
| 연결 | 파이프라인이 레이어를 집어가게 | **`--usd_override` 인자 추가 + 검증 완료** |

마지막 줄이 이번 실습의 가장 큰 소득이다. `../reports.md`가 기록한 "어떤 렌더 설정을 써도
최대 +0.034"라는 상한을 **op 하나로 사실상 다 회수한다**. 기존 스윕이 op 6의 *하위* 노브만
훑고 **op 자체를 바꿔본 적이 없어서** 놓친 것이다. → 아래 "후속" 참고.

---

# S1 · 레이어 겹쳐쓰기 (composition)

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s1_layers.py --scene 17DRP5sb8fy --log_dir logs/gs-vlnpe/usd_study
```

`--assets all`을 주면 노은역 usdz(2.2 GB)까지 포함한다(해싱·로드에 수 분 추가).
`--new_mpu`는 C2에서 덮어쓸 `metersPerUnit`(원본과 달라야 유효 — 기본 1.0).

## B · 복제본 4개는 정확히 뭐가 다른가 (측정값)

| | upAxis | metersPerUnit | defaultPrim | prim | Mesh | PhysicsCollisionAPI | 텍스처 경로 |
|---|:--:|:--:|---|--:|--:|:--:|---|
| `fixed.usd` | Z | 0.01 | `/Root` | 124 | 73 | 3 (+MeshCollision 2) | `./textures/...` **상대** |
| `fixed_docker.usd` | Z | 0.01 | `/Root` | 124 | 73 | 3 (+2) | `/isaac-sim/Matterport3D/...` |
| `isaacsim_<hash>.usd` | Z | **1.0** | `/isaacsim_<hash>` | 121 | 71 | 1 | `/ssd/share/Matterport3D/...` |
| `isaacsim_<hash>_non_metric.usd` | **Y** | 0.01 | `/World` | 120 | 71 | **0** | `./textures/...` **상대** |

**추정이 틀렸던 부분** (작업 시작 시 "`_docker`는 경로만, `_non_metric`은 단위만 다를 것"이라고 봤다):

- `_docker`는 **정말로 asset path만** 다르다 — prim 수·스키마·단위·defaultPrim 전부 동일. 추정 적중.
- `_non_metric`은 **단위만 다른 게 아니다.** `upAxis`가 **Y**(나머지는 Z), `defaultPrim`이 `/World`,
  그리고 **물리 충돌 스키마가 아예 없다**(0개). "non_metric"은 "미터로 안 바꾼 것"이 아니라
  **Z-up·미터·물리로 변환하기 전의 원본 상태**를 뜻한다.
- 미터 단위(`metersPerUnit=1.0`)인 것은 `isaacsim_<hash>.usd` **하나뿐**이고, 파이프라인이 쓰는 게
  바로 이 파일이다. 그리고 이 파일만 텍스처 절대경로가 박혀 있다 → 심링크의 존재 이유(S2).

모든 자산에서 `OmniPBR.mdl`이 **미해결 참조**로 뜬다. 이건 실제 문제가 아니다 → S5 참고.

## C · override 레이어 authoring (핵심)

`fixed.usd`를 sublayer로 얹는 `override.usda`를 만들어 두 가지를 덮어썼다. **원본은 안 건드린다.**

| | 원본 단독 | 레이어 얹은 직후 | 덮어쓴 뒤 합성값 | 원본 재확인 | 판정 |
|---|:--:|:--:|:--:|:--:|:--:|
| C1 `visibility` (prim opinion) | `inherited` | `inherited` | **`invisible`** | `inherited` | PASS |
| C2 `metersPerUnit` (layer metadata) | 0.01 | 0.01 | **1.0** | 0.01 | PASS |

- sublayer는 **상대경로**로 넣어서 레이어를 옮겨도 같이 움직인다.
- C2가 되므로 **`_non_metric` 류의 "단위/축만 다른 복제"는 레이어 한 장으로 표현 가능**하다.
- C1이 되므로 `usdz_scene_utils.expose_collision_meshes_for_rendering`이 런타임 stage에서
  코드로 하는 visibility 플립도 **레이어로 뺄 수 있다**.
- 원본 `fixed.usd` sha256 실행 전후 동일.

> 함정: 이 레이어를 **렌더에 쓰려면 `defaultPrim`을 반드시 레이어에 직접 써줘야 한다.**
> `UsdFileCfg`가 target prim 없이 reference로 스폰하므로 참조되는 레이어 자신의 `defaultPrim`이
> 필요하고, **sublayer의 `defaultPrim`은 참조 해석에 쓰이지 않는다.** 빼먹으면 씬이 빈 채로
> 렌더된다(S5에서 실제로 겪고 반영함).

로컬: `logs/gs-vlnpe/usd_study/s1_layers/17DRP5sb8fy/report.html`

---

# S2 · asset resolution (심링크를 없앨 수 있는가)

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s2_assets.py --scene 17DRP5sb8fy --log_dir logs/gs-vlnpe/usd_study
```

이 스크립트는 심링크를 **만들지도 지우지도 않는다** — 상태만 읽는다.

## A · 실태

자산마다 asset 속성 46개(텍스처 23 + MDL 23). 절대경로 개수: `fixed` 0 · `fixed_docker` 23 ·
`isaacsim` **23(전부 `/ssd/share/...`)** · `_non_metric` 0.

## B · 해결법 3개

| | 방법 | 결과 |
|:--:|---|---|
| ① | 심링크 (현재) | 지금 **살아 있음** — `17DRP5sb8fy`, `s8pcmisQ38h` 두 씬이 repo를 가리키는 심링크 |
| ② | `Ar` resolver search path | **경로 리다이렉트 안 됨** (resolve 결과가 `/ssd/share/...` 그대로) |
| ③ | override 레이어에서 상대경로 재지정 | **PASS** — 텍스처 23/23이 repo 로컬 경로로 resolve, 박힌 경로 0 |

**②에서 판정 기준을 한 번 틀렸다.** 처음엔 "resolve 되고 파일이 존재하면 성공"으로 재서
`works: True`가 나왔는데, 그건 **지금 심링크가 살아 있어서** 박힌 절대경로도 통과한 것이었다.
기준을 "**경로가 리다이렉트됐는가**"로 바꾸니 `redirected: False`. `ArDefaultResolver`의 search
path는 *상대* asset path에만 쓰이므로 박힌 절대경로는 원리적으로 못 고친다 — **기각도 결과**이고,
심링크를 없애려면 ③이 필요하다는 근거가 된다.

③에서 미해결로 남는 것은 `OmniPBR.mdl` 23개뿐이고, 그건 텍스처가 아니다(S5).

## C · ComplianceChecker

error 0 · warning 0 · **failed 24~25**(자산별). 전부 같은 종류다:

```
Shader <.../Looks/..._jpg/..._jpg> has invalid shader node. (fails 'ShaderPropertyTypeConformanceChecker')
```

`Ar`이 `OmniPBR.mdl`을 못 찾아 shader 노드를 검증할 수 없어서 나는 것이다 → S5에서 원인 확정.

로컬: `logs/gs-vlnpe/usd_study/s2_assets/17DRP5sb8fy/report.html`

---

# S3 · 렌더러/톤매퍼 — 미해결 A 규명 + 개선 후보 발견

```
timeout --signal=KILL 3000 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s3_renderer.py --scene 17DRP5sb8fy --dataset vln_n1 --n_frames 18 --log_dir logs/gs-vlnpe/usd_study
```

기존 `04d_tonemap_sweep.py`와 같은 방식(Isaac 1회 부팅 + `build_renderer` 1회 + 설정 순회).
18프레임은 원래 기록된 측정과 같은 프레임 수라서 값을 직접 비교할 수 있다.

## 결과 (씬1, vln_n1, 18프레임, ambient 6.0 / iso 70)

| 그룹 | 설정 | 밝기 | SSIM | 반응 |
|---|---|--:|--:|:--:|
| baseline | op 6 · iso 70 · crush 0.5 · RT | 115.85 | **0.8752** | (기준) |
| control | **filmIso 140** | 153.13 | 0.8681 | O |
| crush | crushBlacks 0.0 / 1.0 | 115.85 / 115.85 | 0.8755 / 0.8751 | **X** |
| burn | burnHighlights 0.0 / 2.0 | 115.85 / 115.85 | 0.8753 / 0.8748 | **X** |
| op | op 1 · crush 0 | 104.72 | **0.9077** | — |
| op | op 7 · crush 0 / 1 | 101.96 / 69.17 | **0.9099** / 0.7050 | **O** |
| op | op 8 · crush 0 / 1 | 101.95 / 69.18 | **0.9103** / 0.7046 | **O** |
| rendermode | PathTracing | 0.00 | 0.0008 | (아래 참고) |

**baseline SSIM 0.8752 vs `../reports.md` 기록 0.876** — 비교 유효.

## 미해결 A의 답

`crushBlacks`/`burnHighlights`는 **op 6에서 무반응이 재현**됐고, **op 7·8에서만 반응**한다
(crush 0→1이 밝기를 101.96 → 69.17로 내린다). 대조군 `filmIso`가 정상 반응하므로
"설정 적용 자체가 안 되는 것"은 아니다.

즉 원인은 이렇다: **`/rtx/post/tonemap/op = 6`은 `irayReinhard/*` 하위 파라미터를 읽지 않는 커브다.**
`04d_tonemap_sweep.py` docstring의 "Isaac 기본 톤매퍼는 `op = 6` (Iray)"라는 전제가 그 자체로는
맞지만(값은 6이 맞다), **그 op이 `irayReinhard/*`를 소비한다는 함의가 틀렸다.**
S4의 carb diff로 op이 두 부팅 경로 모두 6임을 먼저 확인해 "부팅 차이" 후보를 지운 뒤 좁힌 결과다.

### 톤맵 op 번호가 무엇인가 (출처 확인됨)

`/rtx/post/tonemap/op`은 RTX가 밝기 곡선을 어떤 방식으로 압축할지 고르는 번호다.
Isaac Sim의 RTX 설정 UI 소스에 목록이 그대로 있다
(`/isaac-sim/extscache/omni.rtx.settings.core-0.6.5+69cbf6ad/omni/rtx/settings/core/widgets/post_widgets.py:17-26`):

| 번호 | 이름 |
|:--:|---|
| 0 | Clamp |
| 1 | Linear (Off) |
| 2 | Reinhard |
| 3 | Modified Reinhard |
| 4 | HejlHableAlu |
| 5 | HableUc2 |
| **6** | **Aces** ← 현재 기본값 |
| **7** | **Iray** ← `irayReinhard/*`를 쓰는 것 |

같은 파일에 `if tonemapOpIdx == 7:  # Iray` 아래에서만 `crushBlacks`/`burnHighlights` 위젯을
노출한다 — **`irayReinhard/*`는 op 7 전용**이라는 것이 소스로 확인된다. 실측과 정확히 일치.

**따라서 `04d_tonemap_sweep.py` docstring의 "Isaac 기본 톤매퍼는 `op = 6` (Iray)"는 틀렸다.**
6은 **ACES**이고 Iray는 **7**이다. 노브가 무반응이던 이유가 이것이다.

**그리고 목록은 0~7까지 8개뿐이므로 op 8은 유효 범위 밖이다.** op 7과 8이 사실상 같은 값
(SSIM 0.9099 / 0.9103, crush 반응도 동일)으로 나온 것도 8이 범위를 벗어나 7과 같이 처리된
것으로 본다. **권장값은 op 7 (Iray)이다** — 아래 표에서 op 8 행은 참고용으로만 본다.

## 보너스 — 개선 후보 (아직 채택 아님)

**op 7(Iray) · crush 0.0에서 SSIM 0.9099** (baseline 0.8752 = ACES, **+0.035**).
`../reports.md`에 기록된 상한이 "프레임별 최적 gain/bias = 0.894, +시프트까지 = 0.904"였는데,
**op 변경 하나로 0.910**이다. 톤/노출 축의 잔차를 사실상 다 회수한다.

권장은 **op 7 (Iray)**이다. op 8은 유효 범위(0~7) 밖이라 값이 같게 나온 것이므로 쓰지 않는다.
차선은 op 1 (Linear, 0.9077).

채택 전 확인이 필요하다: ① 씬2(`s8pcmisQ38h`)와 `vln_pe`에서 재현 ② 새 op에서 `film_iso`
재튜닝 ③ 밝기가 115.85 → 101.96으로 내려가므로 GT 밝기(112.8)와의 정합을 다시 봐야 함
④ op 7로 가면 `crushBlacks`/`burnHighlights`가 **실제로 살아나므로**(기본 0.5/0.7)
그 두 값도 같이 튜닝 대상이 된다 — 지금 기본값 그대로면 crush 0.5가 걸려 SSIM 0.705로 떨어진다.

## PathTracing 결과는 결론이 아니다

`rendermode`를 `PathTracing`으로 바꾸면 프레임이 **완전히 검게**(밝기 0.00) 나왔고 프레임당
시간도 늘지 않았다(0.08s). 누적 샘플링이 필요한 경로인데 이 카메라/annotator 하네스가 그대로는
이미지를 못 뽑는다는 뜻으로 본다 — **"Path Tracing에서 노브가 먹는지"는 미측정**이고, 별도 설정
없이 이 하네스로는 확인할 수 없다는 것이 이번 결과다.

로컬: `logs/gs-vlnpe/usd_study/s3_renderer/17DRP5sb8fy_vln_n1/report.html`

---

# S4 · Isaac Sim 부팅 경로 carb 설정 diff — 미해결 B

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s4_boot.py --boot raw --log_dir logs/gs-vlnpe/usd_study
```
```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s4_boot.py --boot applauncher --log_dir logs/gs-vlnpe/usd_study
```
```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s4_boot.py --diff --log_dir logs/gs-vlnpe/usd_study
```

한 프로세스에서 두 번 부팅할 수 없어 3번 실행이다. GPU 렌더가 없어 가장 싸다(각 ~6초).

설정 개수: raw **5,409** · AppLauncher **4,897**. 값이 다른 키 **119** · raw에만 570 · AppLauncher에만 58.
**부팅 시간은 둘 다 ~5.7초로 차이 없다** — 30분은 부팅이 아니라 프레임 렌더에서 나는 것이 맞다.

## 기록돼 있던 추론 2개 — 둘 다 확정

| 키 | raw | AppLauncher |
|---|---|---|
| `/rtx/sceneDb/ambientLightIntensity` | **0.0** | **1.0** |
| `/isaaclab/cameras_enabled` | **(키 자체가 없음)** | **True** |

조명 문제의 근본 원인이 "raw 부팅에서 ambient가 0"이라는 기록,
그리고 `Camera` 센서가 `cameras_enabled` 없으면 죽는다는 기록이 **둘 다 데이터로 확인**됐다.

## 30분 미스터리 — 후보 4개

| 키 | raw | AppLauncher | 왜 후보인가 |
|---|---|---|---|
| `/app/renderer/waitIdle` | (없음) | **True** | 매 프레임 렌더러가 유휴 상태가 될 때까지 **블로킹** |
| `/app/hydraEngine/waitIdle` | False | **True** | 같은 성격 (Hydra 쪽) |
| `/app/updateOrder/checkForHydraRenderComplete` | −100 | **1000** | 렌더 완료 확인을 업데이트 순서 **뒤쪽**으로 밀어 매 프레임 대기 |
| `/rtx/hydra/mdlMaterialWarmup` | **True** | **False** | raw는 MDL 머티리얼을 미리 워밍업, AppLauncher는 안 함 |

앞 3개가 전부 "렌더가 다 끝날 때까지 기다려라" 계열이고 **AppLauncher에서만 켜져 있다**.
네 번째는 S5·S2와 교차한다 — 이 씬의 머티리얼이 MDL(`OmniPBR`)인데 워밍업이 꺼져 있으면
프레임마다 머티리얼 컴파일을 다시 만날 수 있다.

부수 후보: `/physics/fabricEnabled=True`(AppLauncher만) +
`/rtx/hydra/readTransformsFromFabricInRenderDelegate` False→True. 우리 렌더러는 카메라 pose를
USD로 쓰는데 render delegate가 Fabric에서 transform을 읽으면 어긋날 수 있다.

**이건 상관이지 인과가 아니다.** 확정하려면 raw 부팅에 위 키를 하나씩 켜서 30분이 재현되는지
봐야 하고, 그건 이번 범위 밖이다.

로컬: `logs/gs-vlnpe/usd_study/s4_boot/report.html` (+ `settings_raw.json`, `settings_applauncher.json`, `s4_diff.json`)

---

# S5 · 머티리얼 — 미해결 C

```
timeout --signal=KILL 2400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s5_materials.py --scene 17DRP5sb8fy --dataset vln_n1 --rtx_ambient 6.0 --film_iso 70 --n_frames 6 --log_dir logs/gs-vlnpe/usd_study
```

`--only`/`--tag`는 한 프로세스에서 여러 variant를 렌더할 때의 캐시 영향을 배제하려고
프로세스를 나눠 돌릴 때 쓴다(아래 교차검증에 사용).

## A · 실제 머티리얼은 무엇인가

```
implementationSource          : sourceAsset
sourceAsset(mdl)              : @OmniPBR.mdl@   subIdentifier: OmniPBR
diffuse_color_constant        : (0.6, 0.6, 0.6)
diffuse_texture               : @/ssd/share/....jpg@    colorSpace = ''  (미지정)
reflection_roughness_constant : (아예 저작돼 있지 않음)
```

`UsdPreviewSurface`가 아니라 **MDL `OmniPBR`**이다. shader 23개, 저작된 입력 이름 13종.

**S2/ComplianceChecker의 "미해결 `OmniPBR.mdl`"은 실제 문제가 아니다.** 실제 파일은
`/isaac-sim/kit/mdl/core/Base/OmniPBR.mdl`에 있고, carb 설정
`/renderer/mdl/searchPaths/templates`에 그 디렉토리가 등록돼 있다. `Ar`은 RTX의 MDL 검색경로를
모르므로 미해결로 보이지만 렌더는 정상이다 — **도구 관점의 산물**이다.

## B · override 레이어로 바꿔 렌더 비교 (6프레임, GT 밝기 112.8)

| 실험 | SSIM median | 렌더 밝기 | baseline 대비 |
|---|--:|--:|--:|
| baseline | 0.8690 | 113.2 | (기준) |
| `albedo` (`diffuse_color_constant` 0.6 → 1.0) | 0.8692 | 113.2 | +0.0002 |
| `colorspace_srgb` (`colorSpace` → `sRGB`) | 0.8692 | 113.2 | +0.0002 |
| **`colorspace_raw`** (`colorSpace` → `raw`) | **0.7284** | **179.2** | **−0.1406** |
| `roughness` (신규 저작 0.5) | 0.8689 | 113.2 | −0.0001 |

## 미해결 C의 답 — 머티리얼 몫은 없다

- **`diffuse_color_constant` 0.6은 무해하다.** "알베도가 60%로 눌려 있다"고 의심했는데
  1.0으로 올려도 그림이 안 변한다 — OmniPBR에서 `diffuse_texture`가 유효하면 이 상수를
  대체하기 때문이다. **가설 기각.** (교차검증: 프로세스를 나눠 baseline만/albedo만 따로
  렌더해도 0.8693 vs 0.8692로 동일 → in-process 캐시 아님)
- `roughness`도 무반응.
- 따라서 **머티리얼 노브로는 잔차를 줄일 수 없다.** `../reports.md`의 "남는 0.096은 메쉬/텍스처
  충실도"라는 결론이 **한 축 더 닫히면서 강화**된다.

## 그래도 하나 고칠 것 — `colorSpace`가 미지정이다

`colorSpace`를 `raw`로 바꾸면 SSIM이 0.869 → 0.728, 밝기가 113 → 179으로 **크게** 움직인다.
`sRGB`로 명시하면 baseline과 동일 → 지금 렌더러 기본값이 사실상 sRGB로 동작하고 있다는 뜻이다.

**즉 현재 파이프라인은 옳은 컬러스페이스를 쓰고 있지만 그것이 어디에도 명시돼 있지 않다.**
렌더러 기본값이 바뀌면 조용히 0.14가 날아간다. asset에 명시적으로 저작해 두는 것이 안전하다.

로컬: `logs/gs-vlnpe/usd_study/s5_materials/17DRP5sb8fy_vln_n1/report.html`
(`_p1`/`_p2` 폴더는 위 프로세스 분리 교차검증 산출물)

---

---

# S6 · 만든 레이어를 실제로 물려 쓴다

`s6_use_layers.py`. S1~S5는 "레이어를 만들 수 있나"까지였고, 여기서 **그 레이어로 실제 렌더**해
기존 방식과 픽셀로 비교한다.

## 판정에 대조군이 필수다

같은 프로세스에서 씬을 내렸다 다시 올려 렌더하면 **같은 원본끼리도 픽셀이 흔들린다**
(평균 차이 0.69~1.13). `../reports.md`의 "렌더러 결정론적"은 **새 프로세스에서 같은 커맨드를
다시 돌렸을 때** 얘기다. 그래서 `--control`(원본↔원본)을 먼저 재고 그 수준과 비교한다.
이 단계를 빼먹어 멀쩡한 것을 FAIL로 한 번 오판했다.

**한 프로세스에서 씬은 2번까지만 올릴 수 있다** — 3번째에
`AnnotatorRegistryError: Annotator rgb is not attached to any render products`로 죽는다.
그래서 대조군을 별도 실행으로 분리했다.

## A · vln_pe 텍스처 경로 (부채 2)

| 씬 | 심링크 | 판정 기준 | 결과 |
|---|:--:|---|---|
| `17DRP5sb8fy` | 있음 | 대조군 수준이어야 함 | 시험 **1.13** vs 대조군 **1.13** → PASS |
| `qoiz87JEwZ2` (90개 중 무작위) | **없음** | 레이어가 더 나아야 함 | SSIM **0.514 → 0.747** → PASS |

**무작위로 고른 덕분에 드러난 것**: `/ssd/share/...` 아래에는 **전에 렌더해 본 2개 씬만** 심링크가
있다. 나머지 88개는 원본 USD가 텍스처를 못 찾아 무늬 없이 렌더된다. 즉 심링크 방식은 **씬마다
부수효과를 한 번 실행해야** 쓸 수 있고, 레이어 방식은 그게 필요 없다.

## B · 노은역 visibility (런타임 파이썬 조작)

대상은 `/World/gauss/mesh` **하나**. 269 B짜리 레이어가 그것을 `invisible → inherited`로 바꾼다.

`--compare_to`로 두 가지를 본다.

| 모드 | 비교 대상 | 기대 | 결과 |
|---|---|---|---|
| `python` (기본) | 런타임 파이썬 뒤집기 | **같아야** 함 | 시험 **0.9202** = 대조군 **0.9202** · depth 0.945 양쪽 동일 → PASS |
| `nothing` | 아무도 안 뒤집은 상태 | **달라야** 함 | depth **0.000 → 0.945** → PASS |

**`--compare_to nothing`의 판정은 RGB로 하면 안 된다.** 이 씬은 **색이 NuRec 가우시안 볼륨**에서,
**깊이가 숨어 있던 충돌 메시**에서 나온다. 레이어가 살리는 건 깊이라서 RGB는 거의 안 변한다
(RGB 차이 1.33 vs 대조군 1.03 — 구별 안 됨). 처음 RGB로 재서 FAIL로 오판했고, **깊이가 나온 픽셀 비율**로 바꿔 PASS가 됐다.

대조 실험이 공짜로 붙어 있다 — `expose_collision_meshes_for_rendering`이 첫 줄에서
`if not name.endswith('_collision.usdz'): return 0`으로 빠져나가므로, 레이어를 열면
**파이썬 함수가 아무 일도 안 한다**(로그에 `뒤집기 0개`). 그런데도 그림이 같다.

## ⚠️ pose 규약 — 여기서 180° 뒤집혔다 (수정함)

파이프라인은 pose를 **두 단계**로 만든다.

```
action_poses = synthesize_action_poses(...)                            # ① raw action 포맷
poses_c2w = [action_to_c2w(a, 'cam2world_gl') for a in action_poses]   # ② OpenCV c2w
```
(`apply_real/04_render_obs_isaac.py:683`)

**S6·S7이 ②를 빠뜨리고 ①을 그대로 렌더러에 넣어 그림이 180° 뒤집혀 있었다.** 두 규약은 X축 기준
정확히 180° 다르다 — 실측: `right` 내적 **+1.0000**, `down`/`forward` 내적 **−1.0000**
(vln_n1·vln_pe 양쪽 동일).

**GT가 없는 씬에서 뒤집힘을 판정한 방법**: 정답 사진이 없으니 눈으로는 안 된다. 대신 픽셀을
depth+pose로 **3D로 되돌려 실제 높이(world Z)**를 쟀다 — "바닥은 카메라보다 아래"라는 사실이
기준이 된다.

| | 배열 위쪽 행 | 배열 아래쪽 행 | |
|---|--:|--:|---|
| 수정 전 | +0.009 m (바닥) | +2.890 m (천장) | **뒤집힘** |
| 수정 후 | +2.609 m (천장) | −0.081 m (바닥) | 정상 |

(카메라 Z = 0.996 m, 바닥 −0.05 m, 아래로 8.1° 기울임)

먼저 시도한 "위/아래 depth 비교"는 3.32 vs 3.45로 **판정이 안 됐다** — 이 씬은 천장이 낮아
거리가 비슷하다. **거리가 아니라 높이**로 바꿔야 2.9 m 차이가 나 명확해졌다.

**S6의 수치 결론은 유효하다** — 두 렌더가 똑같이 뒤집힌 상태로 비교됐으므로 "같다/다르다"
판정은 안 바뀐다. 리포트 이미지는 수정 후 다시 생성했다.

---

# S7 · Isaac Sim GUI로 RGB/depth 실시간 보기

`s7_live_view.py` + `s7_watch.py`. 결과를 파일로만 보던 것을 **창으로 본다.**

## 터미널 2개로 나눠야 한다

**`cv2.imshow`를 Isaac과 같은 프로세스에서 부르면 한 프레임 그린 뒤 세그폴트로 죽는다**(실측 —
`--headless`로 띄워도 같다). opencv의 Qt5와 Kit이 쥔 Qt/GL 충돌로 본다. 피할 방법이 없어 역할을 나눴다.

```
터미널 1  s7_live_view.py   렌더 -> 최신 프레임을 live_latest.jpg 로 흘려보낸다 (임시파일+rename)
터미널 2  s7_watch.py       그 파일을 읽어 창에 띄운다 (Isaac을 import하지 않는다)
```

`--headless`를 빼면 Isaac Sim 창도 같이 떠서 씬을 마우스로 돌려볼 수 있다.

## 이 실습의 관전 포인트

같은 커맨드에서 `--usd`만 바꿔 세 상태를 비교한다.

| `--usd` | 로그의 `visibility 뒤집기` | depth |
|---|:--:|:--:|
| `noeun_station_collision.usdz` (원본) | 1개 | ~0.9 |
| `s6_use_layers/b_noeun/override_visibility.usda` | **0개** | ~0.9 |
| `s6_use_layers/b_effect/no_override.usda` | 0개 | **0.000** |

두 번째가 결론이다 — **파이썬이 손을 하나도 안 댔는데** 화면이 원본과 같다.

## 실측으로 확인한 GUI 관련 사실

- Isaac GUI 부팅은 **~4초** (headless와 비슷)
- **GUI 모드에서는 `print`가 터미널에 안 나온다** — Kit이 stdout을 가져간다.
  그래서 진행 상황을 창에 얹고 `live_view.log`에도 남긴다
- **`04_render_obs_isaac.py`를 재사용할 수 없다** — import하는 순간 headless로 부팅하고,
  GUI로 먼저 띄운 뒤 import하면 **프로세스가 즉시 죽는다**(예외도 아님).
  그래서 씬 로드·조명·카메라를 s7이 직접 하되 **순서는 `build_renderer`와 동일하게** 맞췄다
- 끄는 법: **RGB/depth 창에서 ESC 또는 q.** Isaac 창을 X로 닫으면 Kit이 먼저 죽어 막을 수 없다

## 실행 중에 조명·텍스처 조건 바꾸기 (`control.json`)

뷰어를 띄운 채로 조건을 바꿔 **같은 pose에서 무엇이 어떻게 달라지는지** 본다.
`--hold --start_frame N`으로 카메라를 고정하면 조건 변화만 보인다.

### 왜 파일로 제어하나

창은 별도 프로세스(`s7_watch.py`)가 띄우므로 **키 입력을 렌더 프로세스로 보낼 수 없다.**
그래서 `<out_dir>/control.json`을 매 프레임 읽는다 — 터미널에서 `echo`로 고치면 다음 프레임에
반영된다. 편집 중 깨진 json을 읽으면 화면에 `control.json ERROR`를 띄우고 이전 상태를 유지한다.

### 바꿀 수 있는 것

| 키 | 값 | 하는 일 |
|---|---|---|
| `light` | `ambient_only` / `dome` / `three_light` / `dome_three` | 조명 레시피 (prim을 다시 만든다) |
| `ambient` · `iso` | 실수 | 전역 간접광 세기 · 노출 |
| `texture` | 이미지 경로 / `null` | **텍스처 이미지 교체** / 원본 복귀 |
| `texture_scale` · `texture_rotate` · `texture_translate` | 숫자 또는 `[u,v]` / `null` | UV 타일링·회전·이동 |
| `colorspace` | `sRGB` / `raw` / `null` | 텍스처 색 해석 |
| `albedo` · `roughness` | 0~1 / `null` | 알베도 상수(이 씬은 텍스처가 있어 무반응) · 거칠기 |
| `frame` · `hold` | 정수 · 참거짓 | 카메라 위치 고정·이동 |

시작 시 `<out_dir>/assets/checker.png`(체커보드)를 자동으로 만든다 — UV가 어디에 어떻게 붙는지
먼저 보는 용도다.

### 실측 — 조건별 화면 변화 (씬1 vln_pe, frame 20 고정, 왼쪽 RGB 영역)

| 조건 | 화면 밝기 | 기준과 차이 |
|---|--:|--:|
| `ambient_only` (기준) | 124.94 | — |
| `light: dome` | 141.18 | 12.70 |
| `light: three_light` | 135.66 | 9.18 |
| `iso: 130` | 141.48 | 12.47 |
| `colorspace: raw` | 155.05 | 21.56 |
| **`texture: checker.png`** | 146.56 | **39.54** |
| `texture` + `scale 4` + `rotate 30` | 144.35 | 38.37 |
| `texture: null` (원본 복귀) | 124.95 | **0.38** |

**텍스처 교체가 가장 크게 움직인다.** 복귀 후 0.38은 렌더러 자체 흔들림 수준이다.

### 쓰는 법 (경로를 그대로 적는다 — 변수를 쓰면 이스케이프가 꼬인다)

렌더러 `--out_dir`, 창 `--dir`, `echo` 대상이 **셋 다 같아야** 한다. 아래는 `s7_run`으로 통일한 것.

```
echo '{"light":"ambient_only","ambient":10.0,"iso":70.0,"albedo":null,"roughness":null,"colorspace":null,"frame":20,"hold":true,"fps":4.0,"texture":"logs/gs-vlnpe/usd_study/s7_run/assets/checker.png","texture_scale":4,"texture_rotate":30,"texture_translate":null}' > logs/gs-vlnpe/usd_study/s7_run/control.json
```

`texture`를 `null`로 두면 원본으로 복귀한다. 전체 순서와 나머지 예시는 `TUTORIAL.md` 4단계.

### 반영이 안 될 때 — 원인 3개 (전부 실제로 겪음)

| | 증상 | 원인 | 확인 |
|:--:|---|---|---|
| ① | `light`는 먹는데 `texture`는 무시 | **렌더러가 옛 코드로 돌고 있다** — 파이썬 프로세스는 시작 시점 코드를 들고 간다. 스크립트를 고쳐도 이미 돌던 프로세스엔 반영 안 됨 | `grep -c "재질/텍스처 적용" <out_dir>/live_view.log` 이 0이면 옛 코드(구버전은 `재질 적용:`) → 재시작 |
| ② | 아무 변화 없음 | **창이 다른 폴더를 보고 있다** — 여러 번 띄우면 갈린다 | `ls -la --time-style=+%H:%M:%S logs/gs-vlnpe/usd_study/*/live_latest.jpg` 로 살아 있는 폴더 확인 |
| ③ | 중간 상태가 안 보임 | **여러 줄을 연달아 넣었다** — 프레임마다 한 번 읽으므로 마지막 것만 남는다 | 한 줄씩 넣고 10초 대기 · `tail -f <out_dir>/live_view.log` 로 `제어 변경` 확인 |

②③은 도구 문제가 아니라 **절차 문제**다. ①은 실시간 편집 도구를 만들 때 반드시 겪는 것이라
확인 커맨드를 같이 적어 둔다.

### 재질 변경은 런타임 스테이지에만 쓴다

S5의 override 레이어와 같은 효과를 **파일 저장 없이** 보는 것이다. 원본 USD는 안 건드린다.
조명 레시피 상수(dome 2M · distant 1000 · disk 5000 · raise 0.2 · dome_three 500K)는
`04_render_obs_isaac._add_lights`와 같은 값을 옮겨 적었다 — 그 모듈은 import하면 headless로
또 부팅해 죽어서 재사용이 안 된다. **값을 바꿀 일이 생기면 두 곳을 같이 고쳐야 한다.**

### 여기서 잡은 버그 3개

**① `null`이 "원래대로"가 아니라 "무시"였다.** 처음엔 값이 있을 때만 적용했더니, 한 번 준
`texture_scale`이 계속 남아 원본 복귀가 안 됐다(복귀 후에도 원본과 평균차 **33**).
→ 시작 시 재질 원래값을 전부 스냅샷해 두고, `null`이면 그 값으로 되돌린다
(원래 저작돼 있지 않던 입력은 삭제해 MDL 기본값으로).

**② `sh.GetInput(name)`은 없는 입력에 `None`이 아니라 「무효 객체」를 준다.**
`if inp is not None:`이 통과해버리고 그 뒤
`RuntimeError: Accessed invalid attribute '' on null prim`으로 죽었다.
**pxr 게터 전반이 이렇다 — `is not None`이 아니라 유효성으로 검사해야 한다.**
`valid_input()` 헬퍼로 전부 교체했다.

**③ `try/finally`가 예외를 삼켰다.** 종료 시 세그폴트를 피하려고 `finally: os._exit(0)`으로
감쌌더니, 예외가 나도 트레이스백 없이 조용히 끝났다 — 프로세스는 살아 있고 로그만 멈춰서
**"멈춤"으로 보였다.** py-spy는 ptrace 권한이 없어 스택도 못 떴다.
→ `except BaseException`으로 트레이스백을 로그·stderr에 남긴 뒤 종료하도록 바꿨고,
그러자마자 ②의 원인이 한 번에 나왔다. **종료 편의를 위한 예외 삼킴은 디버깅을 막는다.**

---

# 파이프라인에 붙인 것 — `--usd_override`

만든 레이어를 실제 파이프라인이 쓰게 하려면 인자 하나가 필요했다. 렌더러가 원본 USD를
**glob으로 직접 찾기** 때문에(`find_scene_usd()`), 레이어를 옆에 둬도 무시된다.

**`04_render_obs_isaac.py`에 추가** (이 세션에서 유일하게 손댄 기존 파일):

- `--usd_override` 인자 1개 (기본 `None`)
- `resolve_scene_usd(args)` 함수 1개 — guard clause 위임
- 호출부 2곳을 `load_scene_model(...)` → `resolve_scene_usd(args)`로

총 25줄 추가·2줄 변경. **기본값이 `None`이라 인자를 안 주면 이전과 같은 경로를 탄다.**

## 검증

```
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --scene 17DRP5sb8fy --mode gt_replay --light ambient_only --rtx_ambient 6.0 --film_iso 70 --out_dir scripts/dataset_converters/gs_vlnpe/logs
```

| | SSIM |
|---|---|
| 인자 없음 (기존 경로) | **median 0.876 / min 0.756** ← 기록값과 일치 |
| `--usd_override <S2가 만든 텍스처 레이어>` | **median 0.876 / min 0.756** ← 소수점까지 같음 |

즉 **레이어를 물려도 기존과 동일한 결과**가 나온다. 심링크를 레이어로 바꿀 수 있다는 뜻이다.

---

# 실습 도구 — `usd_peek.py` · `TUTORIAL.md`

이 컨테이너에 `usdview`가 없어서 USD를 눈으로 볼 수단이 없었다. `usd_peek.py`가 그 대용이다.

```
... usd_peek.py <파일>                                 파일이 뭐라고 선언하나 + 물건 수
... usd_peek.py <파일> --tree --depth 4                물건 계층
... usd_peek.py <파일> --prim <경로> --attr visibility  값 하나가 **어느 파일에서 왔는지**
```

마지막이 핵심이다 — 덧칠이 이겼는지를 눈으로 확인하는 유일한 방법이다.

**깨진 sublayer를 도구가 크게 알린다.** 경로를 틀려도 USD는 경고 한 줄만 내고 넘어가는데,
**내 덧칠 파일에 적어둔 값은 자기 파일에서 오므로 그대로 읽힌다** — 그래서 `--prim`으로 값만
물어보면 멀쩡해 보인다. 실제로 이걸로 한 번 헷갈렸다. 물건 수가 0인지로 판단한다.

`TUTORIAL.md`에 0~3단계 실습 순서가 있다. 1단계에서 손으로 쓸 덧칠 파일은
`logs/gs-vlnpe/usd_study/my/first.usda`에 미리 만들어 동작을 확인해 뒀다.

## 규칙 하나 — 「파일 전체 설정」은 sublayer에서 안 올라온다

물건(prim)과 값은 올라오지만 **파일 전체 설정은 안 올라온다**(실측).

| | 원본이 적어둔 값 | 덧칠 파일이 아무것도 안 적었을 때 |
|---|:--:|:--:|
| 물건 수 | 121 | **121** (올라옴) |
| `upAxis` | Z | **Y** (USD 기본값) |
| `metersPerUnit` | 1.0 | **0.01** (USD 기본값) |
| `defaultPrim` | 있음 | **없음** |

물건 수가 멀쩡해서 잘 된 줄 알기 쉽다. `defaultPrim`을 빼먹으면 **렌더할 때 빈 화면**이 나온다.
`new_override_layer()`가 이 셋을 원본에서 복사해 준다.


# 원본 보존 증거

계획의 "기존 코드·데이터 보존" 제약을 코드로 확인했다.

- 5개 스크립트 전부가 대상 자산의 **sha256을 실행 전후로 비교**해 리포트에 찍는다
  (`usd_study_utils.PreservationGuard`). 전 실행 **ALL UNCHANGED**.
- 모든 authoring은 **override 레이어(새 `.usda`)**로만 했다. 원본 `.usd`/`.usdz`는 열기만 했다.
- S2는 심링크를 만들지도 지우지도 않고 상태만 읽는다.
- 기존 스크립트는 **한 줄도 수정하지 않았다.** `04_render_obs_isaac.py`는 `importlib`로 로드해
  `build_renderer`/`render_along`/`RENDER_W/H`를 그대로 재사용한다.
- 기존 파이프라인 재현: `../reports.md`의 권장 커맨드를 **그대로 재실행**해 확인했다.

  ```
  timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --scene 17DRP5sb8fy --mode gt_replay --light ambient_only --rtx_ambient 6.0 --film_iso 70 --out_dir scripts/dataset_converters/gs_vlnpe/logs
  ```

  결과 **SSIM median 0.876 / min 0.756** — `../reports.md` 기록값과 정확히 일치.
  depth median 0.0027~0.0051 m. 즉 기본 경로는 이번 작업으로 **변하지 않았다.**
  판정은 문서대로 **`report.html` 파일과 그 안의 수치**로 했다. 실제로 이 실행은
  `simulation_app.close()`에서 900초 timeout에 걸려 SIGKILL됐고 **stdout은 버퍼째 날아갔다** —
  `../reports.md`가 경고하는 바로 그 경우다. 렌더·리포트 저장은 그 전에 끝나 있었다
  (`report.html` mtime이 실행 중 시각으로 갱신됨, 1.9 MB).
- `git status`에 기존 파일 수정은 `.claude/memory/MEMORY.md`(색인, 규약상 갱신 대상) 하나뿐이고
  나머지는 전부 신규 파일이다.

**한 가지 우회를 호출자 쪽에서 했다**: `build_renderer`는 `/World/Scene`만 지우고
`/World/RenderCamera`는 남기므로 한 프로세스에서 두 번 부르면
`ValueError: A prim already exists at path: '/World/RenderCamera'`로 죽는다. 기존 파일을 고치지
않기 위해 **S5가 호출 전에 카메라 prim을 정리**한다(`_clear_render_prims`).

---

# 후속 후보 (이번 범위 밖)

ROI 순.

1. **톤맵 op 7(Iray) 채택 검토** — SSIM +0.035. 씬2·`vln_pe` 재현, `film_iso` 재튜닝,
   GT 밝기 정합 재확인, 그리고 **`crushBlacks`/`burnHighlights` 값 결정**(op 7에서는 살아난다)이
   선행 조건. 이게 이번 실습의 최대 소득이다. op 8은 범위 밖이라 쓰지 않는다.
2. **`colorSpace`를 asset에 명시** — 지금 옳게 동작하지만 명시돼 있지 않아 조용히 깨질 수 있다.
3. ~~심링크 → override 레이어 전환~~ — **가능함을 확인했고 `--usd_override`로 연결까지 끝났다**
   (S6-A: 심링크 없는 무작위 씬에서 레이어만 정상 · SSIM 0.514 → 0.747).
   남은 것은 파이프라인 기본값을 레이어 쪽으로 바꿀지 결정하는 일이다.
4. **30분 미스터리 확정** — raw 부팅에 S4의 후보 키를 하나씩 켜서 재현 여부 확인.
5. **복제 USD 4개를 레이어로 정리** — S1 C가 가능함을 보였다. 단 `_non_metric`은 축·물리까지
   다르므로 "단위 override 한 장"으로는 안 되고 별도 판단이 필요하다.
6. **VariantSet으로 카메라 프로파일 표현**(`d455_nominal`/`d455_30m`) — 현재는 파이썬 dict.
7. **노은역 usdz를 payload wrapper로 참조** — 2.2 GB 로드 시간 측정. S1에 `--assets all`로 자산
   구조는 이미 볼 수 있다.

**하지 않은 것**: 복제 USD 파일 삭제(불필요해지는지만 보였다), 원본 mutate, Kit 익스텐션 개발.
