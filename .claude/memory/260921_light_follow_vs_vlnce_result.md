# 추종 조명 vs VLN-CE 무조명 렌더링 — 실측 비교

작성 2026-09-21 · 브랜치 `feature/embaug_v0.1`
Artifact: https://claude.ai/artifact/7mraFYXQDy8hfsxXcK2bLt

## 질문

"렌더링할 때 **조명**이 궤적/로봇을 따라가게 하면 더 현실감 있다"는 주장을
기존 VLN-CE 렌더 방식과 비교해라.

## 답

이 레포 자산에서는 **아니다**. 추종 조명은 GT에서 더 멀어지게 하고,
최종 확정 설정(`ambient_only`)은 **VLN-CE가 하는 무조명 렌더와 사실상 같은 방식**이다.

## 레포 안의 세 가지 조명 배치

| | A VLN-CE (Habitat) | B VLN-PE (InternUtopia) | C gs_vlnpe 오프라인 |
|---|---|---|---|
| 조명 | **없음** | DistantLight 1000 + up/down DiskLight 5000 (반지름 50 m) | 같은 3-light를 **매 프레임 카메라로 이동** |
| 갱신 | — | `post_reset`에서 `start_position`으로 **에피소드당 1회** | 프레임마다 카메라 ±0.2 m |
| 근거 | habitat-sim `SimulatorConfiguration.h:90` = `Esp.h:120 NO_LIGHT_KEY "no_lights"`, InternNav·habitat-lab 모두 미오버라이드 | `vln_eval_task.py:31,64,103` | `04_render_obs_isaac.py:265,322` |

**중요**: InternUtopia 평가에서 조명은 에피소드당 1회만 갱신된다 →
"조명이 로봇을 따라간다"는 **에피소드 단위로만** 참, 스텝 단위로는 거짓.

## 실측 1 — 레포가 이미 한 9종 비교 (`04c_light_explorer.py`, GT 대비 mean_err, 낮을수록 좋음)

| config | mean_err |
|---|---|
| camera_light (RTX 헤드램프, 카메라 완전 추종) | ~52 (수치 최저, 육안 "너무 강함"으로 기각) |
| dome_2M (정적) | ~63 |
| gui_default_distant3000 (정적) | ~68 |
| **three_light raise 0.1~0.5 (카메라 추적)** | **~64~74 (최악)** |

최종(2026-08-10): **`--light ambient_only --film_iso 70`** — 라이트 프림 0개.
SSIM 0.648(dome) → **0.876**. ⚠️ mean_err와 SSIM은 **지표·조건이 달라 같은 축에 못 올린다**
(9종 비교는 톤맵 튜닝 이전). 방향만 일관: **조명을 뺄수록 GT에 가까워졌다.**

## 실측 2 — "안 따라가서 생기는 손해"는 잡음 바닥 이하 (오늘 측정)

vln_pe는 조명이 시작점 고정 → 시작점 거리 = 광원 거리. 대조군은 조명이 **하나도 없는** vln_ce.

| 지표 | vln_pe (조명 있음) | vln_ce (조명 없음·대조군) |
|---|---|---|
| 거리–휘도 상관 median | −0.230 | −0.137 |
| 거리 기울기 median | −1.02 DN/m | −0.35 DN/m |
| 구간 평균 **최대 이탈** | **−2.0 %** (8–10 m) | **−4.4 %** (4–6 m) |
| 에피소드 내 휘도 범위 median | 57.6 DN | 49.0 DN |
| 프레임간 \|Δ휘도\| median | 3.86 DN | 3.82 DN |
| 씬별 평균 휘도 편차 | 25.3 DN (161.3–186.6) | 25.6 DN (110.5–136.1) |
| 표본 | 8씬 96ep 8,483 frames | 8씬 96ep 6,242 frames |

**조명이 있는 쪽이 없는 쪽보다 덜 흔들린다** → 거리 감쇠 효과가 장면 내용 잡음에 묻혀 있다.
이유: disk light 반지름이 **50 m**라 거의 평행광. 몇 m 걸어도 조도가 안 변한다.

## 왜

MP3D 텍스처는 실사진이라 그림자·창빛이 **픽셀에 이미 구워져 있다**. 조명을 얹으면 조도 이중 계산,
그 조명이 카메라를 따라오면 **사진 위 손전등** — 움직이는 밝기 그라디언트가 사진 속 고정 그림자와 모순.
측정 순위(추종 > 정적 > 무조명 순으로 나쁨)와 일치. `ambient_only`는 Habitat `no_lights`를 RTX에서 흉내 낸 것 —
**두 파이프라인이 독립적으로 같은 답에 도달**.

추종 조명이 맞는 경우(이 레포 미측정): baked lighting 없는 PBR/CAD 씬 · 광원 하나로 못 덮는 거대 씬 ·
실제 로봇 헤드램프를 재현해야 할 때. 노은역 NuRec GS도 radiance가 구워지므로 MP3D와 같은 논리.

## ⚠️ 확인 필요 — up disk light 부호 불일치

- 프로덕션 `vln_eval_task.py:69`는 up light를 `(x, −y, −z−1)`로 **y·z 부호를 뒤집는다**.
- 오프라인 포트 `04_render_obs_isaac.py:326`은 `(x, y, z+0.2)`로 **안 뒤집는다**.
- 원인은 `AddRotateXYZOp(180°)`/`AddTranslateOp` 합성 순서로 보이나 **미확인**.
- 현재 기본값이 `ambient_only`라 프로덕션 영향 없음. 단 **"VLN-PE 원본 조명 재현"** 목적이면 먼저 맞춰야 한다.

## 미측정

- 9종 비교는 톤맵 튜닝 이전 → **현재 확정 조건에서 `three_light`를 다시 잰 값 없음** (아래 커맨드).
- 노은역 GS 씬 조명 옵션 비교 안 함.
- 휘도는 **프레임 전체 평균**만 — 추종 조명이 만드는 **화면 내 밝기 그라디언트**는 안 잡힌다.

## 다음 실험 (`--light`만 다름)

```
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode gt_replay --light ambient_only --rtx_ambient 10.0 --film_iso 70 --out_dir scripts/dataset_converters/gs_vlnpe/logs
```
```
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode gt_replay --light three_light --film_iso 70 --out_dir scripts/dataset_converters/gs_vlnpe/logs
```
두 `report.html`의 `ssim median` 비교. 성공 판정은 report.html 존재로.
