# 260904 · USD 실습 5개 (usd_study S1~S5) 결과

## 쉽게 요약

USD 파일을 **읽기만** 하던 걸 **직접 써보기**로 바꿨다. 원본 위에 얇은 파일(`.usda`) 한 장을
얹어 값을 덮어쓰는 방식(포토샵 레이어와 같음)을 처음으로 써봤고, 원본은 하나도 안 바꿨다.

그 과정에서 **화질이 좋아지는 설정을 발견**했다 — 톤맵 번호를 **6(ACES) → 7(Iray)**로 바꾸니
SSIM 0.875 → 0.910. 이전에 "아무리 해도 0.904가 한계"라고 적어둔 선을 넘는다.
이전 실험이 6번의 *하위 노브만* 돌리고 **번호 자체를 바꿔본 적이 없어서** 놓쳤다.
아직 채택 아님(씬 하나 · op 7에서는 `crushBlacks`가 살아나므로 그 값도 같이 정해야 함).

같이 알아낸 것: 파일 4개의 실제 차이(하나는 예상과 완전히 달랐다) · 밝기 노브가 안 먹은 이유
(6번이 그 노브를 안 읽는다) · 컨테이너 밖 심링크를 레이어로 대체 가능 · 렌더 30분 범인 4개로 좁힘 ·
남은 화질 차이는 재질 때문이 **아니다**.

아래는 근거와 수치. 재현 커맨드는
`scripts/dataset_converters/gs_vlnpe/usd_study/reports.md`.

---

## 무엇을 했나

`gs_vlnpe` 파이프라인이 USD를 **읽기만** 하고 composition을 전혀 쓰지 않는 상태였다
(`Usd.Stage.Open` 한 장 + `UsdFileCfg` reference 한 개). 그래서 생긴 부채 2개와, 코드·리포트에
"원인을 못 좁혔다"고 명시돼 있던 미해결 실측 3개를 **돌리면 결과가 나오는 스크립트 5개**로 공략했다.

신규 파일 (전부 `scripts/dataset_converters/gs_vlnpe/usd_study/`, 기존 파일 무수정):

| 파일 | 역할 |
|---|---|
| `usd_study_utils.py` | 자산 경로 해석, `PreservationGuard`(sha256 전후 비교), 표 HTML |
| `s1_layers.py` | composition 읽기 + 복제본 4개 diff + **override 레이어 authoring** |
| `s2_assets.py` | asset resolution 3가지 해결법 비교 + ComplianceChecker |
| `s3_renderer.py` | 톤맵 op / Iray 노브 / rendermode 스윕 |
| `s4_boot.py` | 두 부팅 경로 carb 설정 덤프 + diff (GPU 없음) |
| `s5_materials.py` | 머티리얼 덤프 + override 레이어로 바꿔 렌더 비교 |
| `reports.md` | 통합 리포트 (수치·커맨드·판정 전부) |

리포트 HTML은 기존 `viz_utils.save_gallery`/`blink_widget_html`을 그대로 재사용했다.

## 결과 요약

| | 대상 | 결과 |
|:--:|---|---|
| 부채 1 | 씬이 USD 파일 4개로 복제 | 측정 완료 — **추정이 부분적으로 틀렸다** |
| 부채 2 | `/ssd/share/...`에 시스템 심링크 | **override 레이어로 대체 가능** |
| 미해결 A | Iray `crushBlacks` 무반응 | **원인 규명 — op 6이 그 파라미터를 안 읽는다** |
| 미해결 B | `AppLauncher` 30분 | **후보 4개로 좁힘** (확정은 못 함) |
| 미해결 C | 잔차 0.096에 머티리얼 몫? | **없음 — 기존 결론 강화** |
| 보너스 | — | **톤맵 op 변경으로 SSIM 0.875 → 0.910** |

수치·표·커맨드 전체는 `scripts/dataset_converters/gs_vlnpe/usd_study/reports.md`.

## 기존 기록을 정정해야 하는 것 3개

### 1. `_non_metric`은 "단위만 다른 복제"가 아니다

작업 시작 시 파일명으로 추정했던 것과 실제 측정값이 다르다.

- `fixed_docker.usd`는 **정말로 asset path만** 다르다 (prim/스키마/단위/defaultPrim 전부 동일).
- `isaacsim_<hash>_non_metric.usd`는 **`upAxis`가 Y**(나머지 Z), `defaultPrim`이 `/World`,
  그리고 **물리 충돌 스키마가 0개**다. "미터로 안 바꾼 것"이 아니라 **Z-up·미터·물리 변환 이전의
  원본 상태**를 뜻한다.
- `metersPerUnit=1.0`인 것은 `isaacsim_<hash>.usd` 하나뿐이고 파이프라인이 쓰는 게 그 파일이다.
  그리고 **그 파일만** 텍스처 절대경로가 박혀 있다 → 심링크의 존재 이유가 데이터로 증명됨.

### 2. `04d_tonemap_sweep.py` docstring의 함의가 틀렸다

거기 적힌 "Isaac 기본 톤매퍼는 `/rtx/post/tonemap/op = 6` (Iray)"에서 **op 값이 6인 것은 맞지만
그것을 Iray라고 부른 것이 틀렸다.** Isaac Sim RTX 설정 UI 소스에 열거형이 그대로 있다
(`/isaac-sim/extscache/omni.rtx.settings.core-0.6.5+69cbf6ad/omni/rtx/settings/core/widgets/post_widgets.py:17-26`):

    0 Clamp · 1 Linear(Off) · 2 Reinhard · 3 Modified Reinhard ·
    4 HejlHableAlu · 5 HableUc2 · **6 Aces** · **7 Iray**

즉 **6은 ACES이고 Iray는 7**이다. 같은 파일이 `if tonemapOpIdx == 7:  # Iray` 아래에서만
`crushBlacks`/`burnHighlights` 위젯을 노출한다 — **`irayReinhard/*`는 op 7 전용**임이 소스로 확정된다.

실측도 정확히 일치했다: `crushBlacks` 0→1이 op 6에서는 밝기를 전혀 안 바꾸고(115.85 → 115.85),
op 7에서 바꾼다(101.96 → 69.17). 대조군 `filmIso`는 정상 반응하므로 설정 적용 메커니즘의
문제는 아니다. → **미해결 A 해결.**

**목록이 0~7까지 8개뿐이므로 op 8은 유효 범위 밖이다.** 스윕에서 op 8이 op 7과 사실상 같은 값
(SSIM 0.9103 / 0.9099, crush 반응 동일)으로 나온 것은 8이 범위를 벗어나 7과 같이 처리된
것으로 본다. **권장값은 op 7이다** — 처음에 "op 8이 최고"라고 적었던 것을 정정한다.

### 3. `OmniPBR.mdl` "미해결 참조"는 실제 문제가 아니다

`UsdUtils`/ComplianceChecker가 자산 4개 전부에서 `OmniPBR.mdl`을 미해결로 보고하고, 그래서
`ShaderPropertyTypeConformanceChecker` 실패가 24~25건 뜬다. 실제 파일은
`/isaac-sim/kit/mdl/core/Base/OmniPBR.mdl`에 있고 carb 설정 `/renderer/mdl/searchPaths/templates`에
그 디렉토리가 등록돼 있다. **`Ar`이 RTX의 MDL 검색경로를 모르기 때문에 생기는 도구 관점의
산물**이고 렌더는 정상이다.

## 새로 얻은 것

### 톤맵 op — 잔차 회수 후보 (가장 큰 소득)

씬1 `vln_n1` 18프레임(기록된 측정과 같은 프레임 수), ambient 6.0 / iso 70:

| 설정 | SSIM | 밝기 |
|---|--:|--:|
| baseline (op 6) | 0.8752 | 115.85 |
| op 1 | 0.9077 | 104.72 |
| **op 7 · crush 0 (Iray) ← 권장** | **0.9099** | 101.96 |
| op 8 · crush 0 (범위 밖, 참고용) | 0.9103 | 101.95 |

`260809_gs_vlnpe_render_residual_result.md`가 기록한 상한은 "프레임별 최적 gain/bias 0.894,
+시프트까지 0.904"였다. **op 변경 하나로 0.910**이므로 톤/노출 축 잔차를 사실상 다 회수한다.
기존 스윕이 op 6의 *하위* 노브만 훑고 **op 자체를 바꿔본 적이 없어서** 놓친 것이다.

채택 전 확인 필요: 씬2·`vln_pe` 재현 / 새 op에서 `film_iso` 재튜닝 / 밝기 101.96 vs GT 112.8
정합 재확인 / **op 7에서는 `crushBlacks`·`burnHighlights`가 실제로 동작하므로 그 두 값도 같이
정해야 한다** — 기본값 0.5/0.7을 그대로 두면 SSIM이 0.705로 오히려 떨어진다.

### 30분 미스터리 후보 4개 (S4)

부팅 시간은 raw 5.75s vs AppLauncher 5.71s로 **차이 없다** — 30분은 부팅이 아니라 프레임 렌더에서
난다는 것이 확인됐다. 값이 다른 키 119개 중 "렌더 끝날 때까지 대기" 계열이 AppLauncher에서만 켜져 있다:

- `/app/renderer/waitIdle` (없음 → True)
- `/app/hydraEngine/waitIdle` (False → True)
- `/app/updateOrder/checkForHydraRenderComplete` (−100 → 1000)
- `/rtx/hydra/mdlMaterialWarmup` (True → **False**) ← 이 씬 머티리얼이 MDL이므로 교차한다

**상관이지 인과가 아니다.** 확정하려면 raw 부팅에 하나씩 켜서 재현 여부를 봐야 한다.

동시에 기록돼 있던 추론 2개가 확정됐다: `/rtx/sceneDb/ambientLightIntensity`가 raw에서 **0.0**,
AppLauncher에서 **1.0** / `/isaaclab/cameras_enabled`는 raw에 **키 자체가 없고** AppLauncher만 True.

### `colorSpace`가 미지정이라는 잠재 취약점 (S5)

`diffuse_texture`의 `colorSpace`가 `''`(미지정)이다. `raw`로 명시하면 SSIM 0.869 → 0.728,
밝기 113 → 179으로 크게 어긋나고, `sRGB`로 명시하면 baseline과 동일하다. **즉 지금 옳은
컬러스페이스로 동작하지만 그게 어디에도 명시돼 있지 않다** — 렌더러 기본값이 바뀌면 조용히
0.14가 날아간다.

## 판정 기준을 틀렸다가 고친 것 2개 (같은 종류의 실수)

두 번 다 **"성공했나"를 결과의 존재로 재서 착시에 걸린 것**이다.

1. **S2 ②** `Ar` search path를 "resolve되고 파일이 존재하면 성공"으로 재서 `works: True`가
   나왔다. 그런데 그건 **지금 심링크가 살아 있어서** 박힌 절대경로가 통과한 것이었다. 기준을
   "**경로가 리다이렉트됐는가**"로 바꾸니 `redirected: False`. → 올바른 결론: `ArDefaultResolver`의
   search path는 *상대* asset path에만 쓰이므로 박힌 절대경로는 원리적으로 못 고친다.
2. **S1 C2** `--new_mpu` 기본값이 0.01이었고 원본도 0.01이라, "안 바뀐 것을 안 바뀌었다고
   확인"하는 무의미한 PASS였다. 기본값을 1.0으로 바꾸고 `c2_is_meaningful` 플래그를 추가해
   원본과 다른 값을 요청했는지 자체를 판정에 넣었다.

일반원리: **판정 기준에 "다른 경로로도 같은 결과가 나올 수 있는가"를 반드시 넣는다.**
대조군/무효 조건을 스크립트가 스스로 검사하게 만들어야 한다(S3의 `filmIso` 대조군이 그 예 —
그게 없었으면 "op 6 무반응"을 "설정 적용 실패"와 구별할 수 없었다).

## 개발 중 부딪힌 환경 제약 (재현 시 필요)

1. **`pxr`을 `SimulationApp` 부팅 전에 import하면 Omniverse 확장이 전부 죽는다.**
   `omni.usd`, `omni.UsdMdl`, `Semantics` 등이
   `RuntimeError: extension class wrapper for base class ... has not been created yet`로 실패한다.
   `/opt/USD`의 standalone pxr이 먼저 로드되기 때문. → 렌더가 필요한 S3/S5는 부팅을 먼저 하고
   `pxr`을 **함수 안에서 lazy import**한다. (렌더 없는 S1/S2는 standalone pxr로 잘 동작한다.)
2. **`usdcat`/`usdchecker`/`usdview`/`usdzip` CLI가 이 컨테이너에 없다.** 전부 파이썬 API로 해야 한다.
   `pxr` 버전은 **OpenUSD 0.26.5**, `UsdUtils.ComplianceChecker`·`CreateNewUsdzPackage`는 있다.
3. **`carb.settings`의 `get_settings_dictionary('/')`는 python dict가 아니라
   `carb.dictionary.Item`을 준다** (그리고 `IDictionary`에 `get_dict`가 없다). `s.get('/')`가
   진짜 dict(최상위 56키 → 평탄화 5,409키)를 준다.
4. **`build_renderer`를 한 프로세스에서 두 번 부르면 죽는다** —
   `/World/Scene`은 지우지만 `/World/RenderCamera`는 남겨서
   `ValueError: A prim already exists at path: '/World/RenderCamera'`. 기존 파일을 고치지 않기 위해
   **호출자 쪽에서** 카메라 prim을 먼저 정리했다(`s5_materials._clear_render_prims`).
5. **override 레이어를 렌더에 쓰려면 `defaultPrim`을 레이어에 직접 써야 한다.** `UsdFileCfg`가
   target prim 없이 reference로 스폰하므로 참조되는 레이어 자신의 `defaultPrim`이 필요하고,
   **sublayer의 `defaultPrim`은 참조 해석에 쓰이지 않는다.** 빼먹으면 씬이 빈 채로 렌더된다.

## 보존 제약 준수 증거

- 5개 스크립트 전부 `PreservationGuard`로 대상 자산 **sha256 전후 비교** → 전 실행 ALL UNCHANGED
- 모든 authoring은 **새 `.usda` override 레이어**로만. 원본 `.usd`/`.usdz`는 열기만 함
- S2는 심링크를 만들지도 지우지도 않고 상태만 읽음
- 기존 스크립트 **한 줄도 수정 안 함** — `04_render_obs_isaac.py`는 `importlib`로 로드해
  `build_renderer`/`render_along`/`RENDER_W/H` 재사용
- 기존 파이프라인 재현 확인: `../reports.md` 권장 커맨드를 그대로 재실행 →
  **SSIM median 0.876 / min 0.756**(기록값과 정확히 일치), depth median 0.0027~0.0051 m.
  교차로 S3 baseline 18프레임도 0.8752로 같은 대역
- 리포트 Artifact 5개: [S1](https://claude.ai/code/artifact/a743e42b-5920-40e0-b017-8ba8fdc0751e) ·
  [S2](https://claude.ai/code/artifact/a3befa4d-6ed2-46fa-b111-6f8f810f1b09) ·
  [S3](https://claude.ai/code/artifact/3e05ea2a-ad9d-4973-ae58-4a75f00f3a93) ·
  [S4](https://claude.ai/code/artifact/64b02d2d-f004-4263-9c05-f5c96c7a41b6) ·
  [S5](https://claude.ai/code/artifact/0ba95b10-e156-40c2-a9a8-01ac6b1f3585)

---

## 후속 (2026-09-06~07) — S6·S7 추가 + 파이프라인 연결

S1~S5는 "알아보기"였고, 여기서 **만든 레이어를 실제로 렌더에 물려** 쓸 수 있는지까지 확인한 뒤
파이프라인에 인자로 연결했다.

### 새 파일

| 파일 | 역할 |
|---|---|
| `s6_use_layers.py` | 만든 레이어로 실제 렌더해 기존 방식과 픽셀 비교 (A: 텍스처 경로 / B: 노은역 visibility) |
| `s7_live_view.py` | Isaac GUI + RGB/depth 실시간 스트리밍 |
| `s7_watch.py` | 그 스트림을 창에 띄우는 별도 프로세스 (Isaac을 import하지 않는다) |
| `usd_peek.py` | USD 안을 들여다보는 실습용 도구 (`usdview`가 없어서) |
| `TUTORIAL.md` | 0~3단계 실습 순서 |

### 기존 파일 수정 — 이번 세션에서 유일

`04_render_obs_isaac.py`에 **`--usd_override`** 추가 (25줄 추가·2줄 변경).
인자 1개 + `resolve_scene_usd()` 함수 1개 + 호출부 2곳 교체. 기본값 `None`이면 기존 경로.

검증: 인자 없음 **SSIM median 0.876 / min 0.756**(기록값 일치) ·
`--usd_override <S2 텍스처 레이어>` **median 0.876 / min 0.756**(소수점까지 동일).

### 결과

| | 결과 |
|---|---|
| S6-A 심링크 있는 씬 | 시험 1.13 = 대조군 1.13 → 레이어가 심링크와 같은 결과 |
| S6-A 무작위 씬(`qoiz87JEwZ2`) | **원본이 깨짐** SSIM 0.514 vs 레이어 0.747 |
| S6-B `--compare_to python` | 시험 0.9202 = 대조군 0.9202, depth 0.945 양쪽 동일 |
| S6-B `--compare_to nothing` | depth **0.000 → 0.945** |

**무작위 씬이 드러낸 것**: `/ssd/share/...`에는 **전에 렌더해 본 2개 씬만** 심링크가 있다.
나머지 88개는 원본이 텍스처를 못 찾는다 → 심링크 방식은 씬마다 부수효과 실행이 필요하고,
레이어 방식은 필요 없다.

### ⚠️ 정정 — pose 규약 180° (내 버그)

파이프라인은 `synthesize_action_poses()` → **`action_to_c2w(a, 'cam2world_gl')`** 두 단계를 거친다
(`apply_real/04_render_obs_isaac.py:683`). **S6·S7이 두 번째를 빠뜨려 렌더가 180° 뒤집혀 있었다.**
실측: `right` 내적 +1.0000, `down`/`forward` 내적 −1.0000 (vln_n1·vln_pe 양쪽 동일).
**실제 파이프라인과 노은역 데이터셋(8,378 프레임)은 정상** — 그쪽은 두 단계를 다 거친다.

수정 후 S6-B를 재생성했고 **수치 결론은 안 바뀌었다**(두 렌더가 똑같이 뒤집힌 상태로 비교됐으므로).

### 판정 실수 4건에서 나온 공통 원리

이번 작업에서 판정 기준을 네 번 틀렸다. 전부 같은 종류다 — **"결과가 나왔으니 맞다"로 잰 것.**

1. `Ar` search path가 "성공"으로 보였다 → 실은 **심링크가 받아준 것**
2. `--new_mpu` 기본값이 원본과 같아 **무의미한 PASS**
3. "픽셀 동일"을 기준으로 삼았다 → 같은 원본끼리도 **1.13만큼 흔들린다**(대조군 필요)
4. 덧칠 효과를 RGB로 쟀다 → 이 씬은 **깊이가 바뀌는 것**이었다

**원리**: 판정에 반드시 넣을 것 — ① **대조군**(안 바꿨을 때 얼마나 흔들리나) ② **다른 경로로도
같은 결과가 나올 수 있는가** ③ **재는 대상이 실제로 변하는 것인가**.
그리고 **비교 대상이 독립적인지** 확인할 것 — s7이 s6과 같다는 것은 옳음의 증거가 아니었다
(둘 다 내 코드, 같은 버그).

### GT 없는 씬에서 뒤집힘을 판정한 방법 (재사용 가치 있음)

정답 사진이 없으면 눈으로 판정이 안 된다. 대신 **픽셀을 depth+pose로 3D로 되돌려 실제 높이를
재고** "바닥은 카메라보다 아래"라는 사실로 판정한다.

| | 배열 위쪽 | 배열 아래쪽 | |
|---|--:|--:|---|
| 수정 전 | +0.009 m (바닥) | +2.890 m (천장) | 뒤집힘 |
| 수정 후 | +2.609 m (천장) | −0.081 m (바닥) | 정상 |

먼저 시도한 "위/아래 depth 비교"는 3.32 vs 3.45로 판정 불가였다(천장이 낮은 씬).
**거리가 아니라 높이**로 바꿔야 2.9 m 차이가 나 명확해졌다.

### 새로 알게 된 환경 제약

6. **`cv2.imshow`를 Isaac과 같은 프로세스에서 부르면 한 프레임 그린 뒤 세그폴트**
   (`--headless`로도 같다). Qt 충돌. → 렌더와 표시를 프로세스로 분리(`s7_live_view` + `s7_watch`).
7. **GUI 모드에서는 `print`가 터미널에 안 나온다** — Kit이 stdout을 가져간다. 로그 파일 필수.
8. **GUI로 띄운 뒤 `04_render_obs_isaac.py`를 import하면 프로세스가 즉시 죽는다**(예외 아님) —
   그 모듈이 import 시점에 headless로 또 부팅해서. GUI 뷰어는 독립 파일이어야 한다.
9. **Isaac 정리 단계(atexit)에서 세그폴트가 난다.** 결과는 이미 저장된 뒤라 무해하지만 실패처럼
   보인다 → `exit_skipping_isaac_teardown()`(`os._exit`)로 모든 종료 경로를 끊는다.
   단 **Isaac 창을 X로 닫는 경우는 못 막는다**(Kit이 먼저 죽는다).
10. **「파일 전체 설정」은 sublayer에서 안 올라온다** — `defaultPrim`·`upAxis`·`metersPerUnit`.
    물건 수는 121로 멀쩡한데 축·단위는 USD 기본값(Y, 0.01)으로 돌아간다.
    `new_override_layer()`가 셋을 원본에서 복사한다.

### S7 확장 (2026-09-07) — 실행 중 조명·텍스처 조건 변경

뷰어를 띄운 채 `<out_dir>/control.json`을 `echo`로 고치면 다음 프레임에 반영된다.
바꿀 수 있는 것: `light`(4종) · `ambient` · `iso` · **`texture`(이미지 교체)** ·
`texture_scale/rotate/translate` · `colorspace` · `albedo` · `roughness` · `frame/hold`.
시작 시 체커보드(`assets/checker.png`)를 자동 생성해 UV 확인용으로 쓴다.

실측 변화폭(씬1 vln_pe, frame 20 고정): 텍스처 교체 **39.5** > colorspace raw 21.6 >
dome 12.7 ≈ iso130 12.5 > three_light 9.2. 원본 복귀는 0.38(노이즈 수준).

**왜 파일 제어인가**: 창이 별도 프로세스라 키 입력을 렌더 쪽으로 못 보낸다.
**재질은 런타임 스테이지에만 쓴다** — 원본 USD 무변경, S5 override 레이어와 같은 효과를 저장 없이.
조명 상수는 `04_render_obs_isaac._add_lights`와 같은 값을 옮겨 적었으므로 **두 곳을 같이 고쳐야 한다.**

### 반영이 안 되는 원인 3개 (도구가 아니라 절차 문제 2개 포함)

실시간 편집을 실제로 써보니 "안 먹는다"의 원인이 셋이었다.

1. **렌더러가 옛 코드로 돌고 있다** — 파이썬 프로세스는 시작 시점 코드를 들고 간다.
   스크립트를 고쳐도 이미 돌던 프로세스엔 반영 안 됨. `light`는 먹는데 `texture`는 무시되면
   거의 이 경우. 확인: `grep -c "재질/텍스처 적용" <out_dir>/live_view.log` == 0 → 재시작.
   **실시간 편집 도구를 만들 때 반드시 겪는다 — 버전 확인 방법을 문서에 같이 적어야 한다.**
2. **창이 다른 폴더를 본다** — 렌더러 `--out_dir` / 창 `--dir` / `echo` 대상이 셋 다 같아야 한다.
   테스트로 여러 개 띄우면 갈린다(실제로 `s7_live_view`/`s7_cond1`/`s7_tex`/`s7_tex4` 4개가 떠 있었다).
3. **여러 줄을 연달아 넣었다** — 프레임마다 한 번 읽으므로 마지막 것만 남고 중간 상태는 스쳐 간다.

문서의 커맨드도 셸 변수(`$B`/`$D`)를 쓰지 않고 **경로를 그대로 적는 형태로 바꿨다** —
변수를 쓰면 따옴표 이스케이프가 꼬이고, 위 2번 실수를 유발한다.

### 여기서 잡은 버그 3개 — 전부 재발하기 쉬운 종류

**① `null`이 "원래대로"가 아니라 "무시"였다.** 값이 있을 때만 적용하니 한 번 준
`texture_scale`이 남아 원본 복귀가 안 됐다(복귀 후에도 평균차 33). → 시작 시 재질 원래값을
스냅샷하고 `null`이면 되돌린다(원래 없던 입력은 삭제).
**원리: "되돌리기"는 별도 경로가 아니라 적용 경로와 같은 표로 다뤄야 한다.**

**② `sh.GetInput(name)`은 없는 입력에 `None`이 아니라 무효 객체를 준다.**
`is not None` 검사가 통과해 `RuntimeError: Accessed invalid attribute '' on null prim`.
**pxr 게터 전반이 이렇다 — `is not None`이 아니라 유효성(`valid_input()`)으로 검사할 것.**
같은 함정이 `Usd.Stage.GetPrimAtPath`·`GetAttribute` 등에도 있다.

**③ `try/finally`로 종료를 감싸 예외를 삼켰다.** 세그폴트 회피용
`finally: os._exit(0)` 때문에 예외가 트레이스백 없이 사라져, 프로세스는 살아 있고 로그만 멈춰
**"멈춤"으로 오진**했다(py-spy는 ptrace 권한 없어 스택도 못 뜸). `except BaseException`으로
트레이스백을 남기게 바꾸자 ②가 한 번에 나왔다.
**원리: 종료 편의를 위한 예외 삼킴은 디버깅을 막는다. 반드시 남기고 나서 끊을 것.**

## 후속 후보 (ROI 순)

1. ~~톤맵 op 8 채택 검토~~ — **op 8은 유효 범위(0~7) 밖이라 오기.** 그리고 이 항목의 근거였던
   전수 측정은 무효였다(아래 「측정을 세 번 틀린 기록」). 텍스처를 붙여 재측정 중
2. `colorSpace`를 asset에 **명시**
3. 심링크 → override 레이어 전환 (S2 ③)
4. 30분 미스터리 확정 — raw 부팅에 S4 후보 키 하나씩 켜기
5. 복제 USD 4개를 레이어로 정리 (`_non_metric`은 축·물리도 달라 별도 판단 필요)
6. 카메라 프로파일을 VariantSet으로
7. 노은역 usdz payload wrapper + 로드 시간 측정

---

## 측정을 세 번 틀린 기록 (2026-09-08)

톤맵 전수 측정(126씬)을 돌리고 결과를 보고했는데, **세 번 연속 틀린 조건으로 재고 있었다.**
세 번 다 같은 실수다 — **격자나 조건이 최적점을 감싸는지 확인하지 않고 숫자를 믿었다.**

### ① 텍스처가 안 붙은 그림을 쟀다 (가장 컸다)

`s8`/`s9`가 `mp3d_usd_variants()`로 USD 파일을 **직접 열어서**, 파이프라인이 쓰는
`load_scene_model()` -> `ensure_texture_symlink()`를 건너뛰었다. mp3d USD에는 텍스처가
`/ssd/share/Matterport3D/...` **절대경로로 박혀** 있어 그 심링크 없이는 resolve가 안 된다.

측정된 씬 61개 중 심링크가 있던 건 **2개**(`17DRP5sb8fy`, `s8pcmisQ38h`)뿐이었고,
**그 2개가 SSIM 상위 3위**였다 — 그게 단서였다.

| | SSIM 중앙값 (30씬) |
|---|---|
| 텍스처 없음 (무효) | 0.546 |
| 텍스처 있음 | **0.821** |

**+0.254 (최대 +0.449) · 29씬 상승 · 하락 0.**
안 변한 유일한 씬이 원래 심링크가 있던 `17DRP5sb8fy`(0.781 -> 0.781)로, 대조군이 정확히 맞았다.
얻으려던 톤맵 이득(+0.036)의 **7배**다.

**고침**: `s8`/`s9`에 `--textures {symlink,raw}` 추가(기본 `symlink` = 파이프라인과 동일).

### ② `crushBlacks`를 0으로 고정해 놓고 iso만 훑었다

`build_configs`가 op7 스윕에서 `crush=0.0`을 고정하고, `crush=0.5`는 iso 70에서 한 번만 쟀다.
텍스처를 붙여 재보니 `crush 0.25`가 최적이었다 — **격자 밖이 아니라 격자 사이에 있었다.**

| `D7G3Y4RVNrH` | crush 0 | 0.25 | 0.5 |
|---|---|---|---|
| iso 40 | 0.686 | **0.698** | 0.677 |
| iso 50 | 0.675 | **0.701** | 0.697 |

iso가 올라가면 최적 crush도 올라간다 — **한 축만 훑으면 원리적으로 못 찾는 모양.**
"op만 바꾸고 crush를 두면 손해"라던 기존 결론도 텍스처 없을 때의 착시였다.

**고침**: `--crushes`를 축으로 추가해 iso × crush 2차원으로 훑는다.

### ③ iso 범위가 씬 전체를 못 덮었다 (양쪽 경계 다 틀림)

- 1차: `50,70,90,110` -> 최적이 **120/126씬에서 하한 50**. 진짜 최적은 50 아래.
- 2차: `30,40,50,70`으로 내렸더니 `17DRP5sb8fy`가 **상한 70에 걸려** op7이 −0.0078로 뒤집혔다
  (원래 이 씬의 최적 iso는 110).

원인은 **GT 밝기가 씬마다 72~201로 넓다**는 것이다. 어두운 씬은 낮은 iso, 밝은 씬은 높은 iso를
원한다. 한 격자가 양쪽을 다 덮어야 한다.

**고침**: `30,50,70,90,110` 전 구간. 넓힌 뒤 그 씬의 op7이 −0.0078 -> **+0.0006**으로 복구됐다.

### 부수 확인 — SSIM 최적점은 밝기가 GT와 맞는 곳

`D7G3Y4RVNrH`: GT 밝기 87.8, 최적 설정의 렌더 밝기 **86.5 (−1.3)**.
밝기 일치가 SSIM 최적과 겹친다 -> 격자 검증에 쓸 수 있는 값싼 지표.

### 일반 원리

1. **최적값이 격자 경계에 있으면 그 측정은 끝난 게 아니다.** 경계 hit는 "더 가야 한다"는 신호다.
   전수를 돌리기 전에 1씬으로 봉우리가 **안쪽**에 생기는지 먼저 확인한다.
2. **파이프라인이 쓰는 loader를 우회하면 조건이 달라진다.** 측정 스크립트는 파일을 직접 열지 말고
   파이프라인과 같은 함수를 쓴다(부수효과까지 포함해서 그게 조건이다).
3. **파라미터가 상호작용하면 한 축씩 훑는 건 틀린다.** 다른 축을 기본값에 고정하는 것도 선택이다.

### 새로 만든 것

- `s10_global_pick.py` — 씬별 최적이 아니라 **전 씬 공통으로 쓸 설정 하나**를 고른다.
  파이프라인은 `--film_iso`를 모든 씬에 같이 쓰므로 이 값이 실제로 필요한 답이고,
  "씬별 튜닝의 추가 이득"도 같이 나온다.
- `usd_overrides/` — `.usda` 덧칠 레이어 모음 + 빌더. **텍스처 문제와는 별개**다
  (심링크를 컨테이너 밖 `/ssd/share`에 쓰는 부수효과를 없애려는 것).
  측정만 할 거면 심링크로 충분하고 이 레이어는 필요 없다.
