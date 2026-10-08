# P-B1 리뷰 대응 (R1/R2/R3) — 2026-08-15

사용자 지적 3건. 조사 결과 **내 이전 보고에 과장·버그가 있었다** — 아래에 정정을 남긴다.

## 발견한 버그 (P-B1 결과에 영향)
**`synthesize_action_poses` 인자 순서 오류**. 시그니처는 `(xy_world, floor_z, h_b, pitch_down_deg)`
(`geometry_utils.py:253`)인데 `(traj, h_b, pitch, floor_z)`로 호출했다.
- `embodiment_augment.py:138`: 카메라 z가 1.25(절대)로 고정 — 실제로는 `floor_z+1.25`여야 함(층이 다르면 오차).
- `99_bench_onthefly_deprecated.py:77`: 카메라가 **z=7.0 m**(천장 위)에 있었다 → **빈 화면을 렌더** → G3 시간이 낙관적.
→ 둘 다 수정. **이전 G3 수치(depth 9.9 ms/frame, T=5 53 ms)는 무효.**

## R3 — 타일 무손실 (이전 G4 정정)
이전 G4가 불충분했던 두 이유:
1. **조건이 자명**: 17DRP 씬은 16.4×8.3 m인데 타일 박스가 22×22 m(half=radius+margin=11) → 타일이 **씬 전체 포함**.
2. **지표가 실패를 못 잡음**: 마스크가 `isfinite(depth_tile)`라 **타일에서 잘려나간 픽셀을 분모에서 제외**.
   crop 손실이 일어난 픽셀을 빼고 median을 재니 항상 ~0. 구조적으로 실패 감지 불가.

**재검증**(`02_verify_tiling.py`, 큰 씬 s8 + 타일 소유영역 경계 카메라 + `missing_frac`을 분자로):
| margin | 유효 depth 5 m(실제 파이프라인) | 유효 depth 10 m(렌더러 far) |
|---|---|---|
| 5 m (현 설정) | **PASS** (missing 0.000%, diff 0.0001 m) | **FAIL** (missing 최대 100%, 일부 뷰 4%) |
| 8 m | PASS | PASS |

**공식**: `tile_for_xy`가 L∞ 최근접이라 소유영역은 center±radius/2 → **보장 반경 = margin + radius/2**.
radius=6·margin=5 → 8 m 보장. 학습 depth는 5.0 m clip(`preprocess_depth_image_v2`)이고 BEV range도 5.0 m
→ **현 설정은 실제 파이프라인에서 무손실**. 다만 10 m까지 쓰려면 margin ≥ 7 m 필요.

## R2 — 속도·병렬성 (이전 G3 정정)
**프로파일 실측**(s8, 올바른 pose):
| 구성 | 시간 | 비고 |
|---|---|---|
| align(에피소드당) | **14.8 s** | 대부분 씬 mesh(199 MB) 로드 → mesh 캐시 추가. **offline 배치로 미리 계산해야 함** |
| replan(snap 포함) | **18 ms** | 문제 없음 |
| depth 렌더 | **13.9 ms/frame** | 올바른 pose(지오메트리 99% 화면). 버그 pose는 10.3 ms(23%만 보임) |
| ~~depth→BEV (CPU)~~ | ~~165 ms/frame~~ | ⚠️ **측정 오류. 2026-08-17 정정 — 아래 참고** |

> ### ⚠️ 2026-08-17 정정 — "depth→BEV 165 ms/frame"은 BEV 비용이 아니라 **torch 스레드 설정**이었다
> 이 항목만 `08_bench_parallel.py:168-174`에서 **메인 프로세스**로 잰다. 부모의 torch intra-op 스레드 수가
> **코어 수와 같으면(24 == `os.cpu_count()`)** OpenMP full-subscription spin-wait로 무너진다.
> `torch.set_num_threads(1)` 한 줄만 넣고 같은 코드·같은 입력으로 재측정:
>
> | torch threads | CPU BEV ×12 | frame당 |
> |---|---|---|
> | 24 (= 코어 수) | 2164 ms | **180.4 ms** |
> | 1 | 21 ms | **1.8 ms** |
>
> **100배**. 2dloader에서 스레드 수를 훑으면 코어 수에서만 절벽이 생긴다 —
> 1→1.41 ms, 8→1.14 ms, 16→1.05 ms, 23→1.09 ms, **24→402 ms**.
> 즉 "작은 텐서에 멀티스레드가 오버헤드"가 아니라 full subscription 문제다.
> (상세: `260817_2dloader_vlnce_result.md` §5, `2dloader_vlnce/local_map.py:limit_torch_threads`)
>
> **worker throughput 표는 영향 없음**(재측정으로 확인: 3.88/6.73/10.23/13.55 — 아래 원래 값과 동일).
> `__getitem__`은 numpy/scipy와 Open3D만 쓰고 torch를 안 쓰며, DataLoader worker는 원래 1스레드로 뜬다.

**핵심 설계 정정**: **augmenter가 BEV를 만들 필요가 없다.** S1 BEV는 dataloader가 아니라 모델 하류
(`internvla_n1_unified_provider`)가 `traj_depths`에서 **GPU로** 계산한다(초기 탐색에서 확인).
→ `render_bev_along(..., with_bev=False)`를 학습 경로 기본으로.
**단, 이유는 "느려서"가 아니라 중복이기 때문이다**(위 정정). 스레드만 제한하면 CPU BEV는 sample당
21 ms로 켜도 무방한 수준이다.
- 수정 후 학습 경로 sample 비용 ≈ replan 18 ms + 렌더 12×13.9 ≈ **~190 ms**(단일 프로세스, 타일 캐시 시).
- 추가 수정: 타일 모델 **메모리 캐시**(이전엔 타일 전환마다 `.ply` 디스크 재읽기 105 ms + add 66 ms).
**병렬성 실측(`08_bench_parallel.py`, epoch 200 samples)**:
| num_workers | samples/sec | 배속 |
|---|---|---|
| 0 | 3.84 | 1.0× |
| 2 | 6.61 | 1.7× |
| 4 | **10.23** | 2.7× |
| 8 | 13.31 | 3.5× |

batch_size 4 vs 8(workers=4): 10.76 vs 10.34 → **batch 영향 거의 없음**(sample 단위 병렬이라 당연).
single-sample: replan 42 ms + depth 렌더 ×12 349 ms = **391 ms**(단일 프로세스).
~~CPU BEV 켜면 +2022 ms~~ → **+21 ms**(torch 1스레드). 2026-08-17 재측정: replan 37 + 렌더 347 = 385 ms.

**⚠️ 병렬화의 결정적 제약(실측)**: **부모 프로세스가 Open3D 렌더러를 만든 적이 있으면 fork한 worker가
데드락**한다(무한 대기). 원인: 살아있는 EGL/Filament 컨텍스트를 fork로 상속. 증거:
- 부모가 깨끗하면 workers=2/4/8 정상. 부모가 `num_workers=0`을 먼저 돌리면(=부모에서 렌더) 이후 조합 전부 hang.
- `mp.Process` fork 단독 테스트는 성공(0.72 s), `spawn`은 실패(exitcode 1).
→ **P-B2 이식 규칙**: dataset `__init__`(부모)에서 augmenter/렌더러를 만들지 말고 **worker의 첫
`__getitem__`에서 lazy 생성**. 부모는 offline 정합 캐시만 읽는다. `num_workers=0`은 지원 불가로 문서화.

**벤치마킹 함정(정정)**: 몇 배치만 재면 prefetch 큐에서 꺼내는 시간만 재서 **53,747 samples/s** 같은
허수가 나온다. epoch 전체(≥200 samples, worker×prefetch×batch보다 충분히 크게)를 소비해야 한다.
또 조합마다 **별도 subprocess**로 재야 부모 오염이 없다(`--one W,B` 모드).

## R1 — 리포트 색 범례
`summary.html`에 범례가 **아예 없었고**(grep 0건), body엔 "하늘/노랑/빨강" **텍스트만**(색 견본 0개),
게다가 **infeasible인 r_b에도 색을 배정**해 오해를 줬다. 또 r_b를 4개 이상 주면 범례에서 `IndexError`.
→ `chip(rgb,label)`로 **실제 색 견본** 생성, **그린 경로만** 범례에 넣고 infeasible은 "계획 불가(선 없음)"로
명시, start/goal도 칩으로. summary·body 양쪽에 배치. 5색까지 확장.

## 남은 것
- ~~R2 worker/batch 측정 완료 → 리포트 발행.~~ 완료. 2026-08-17 스레드 정정 반영해 재발행
  (`logs/embodiment_augment/perf/`, artifact "R2 Speed Correction").
- `08_bench_parallel.py`에 `limit_torch_threads()` 추가됨 — 다른 3dloader 스크립트에서 CPU BEV를 잴 때도
  같은 함정에 빠지지 않도록 주의.
- align을 offline 배치로 미리 계산해 캐시(P-B2 이식 시 필수).
- 그 후 P-B2(기존 dataloader 이식).

---

## 추가 질문 대응 — "r_b만 바꿨는데 왜 depth/BEV가 바뀌나?" (2026-08-15)

**답**: occupancy dilation이 depth를 직접 바꾸는 게 아니라 **경로를 통해 간접적으로** 바꾼다.
`r_b ↑ → dilation → navigable 축소 → A* 경로 변경 → 카메라를 그 경로 위에 재배치 → 다른 위치에서 렌더`.
실측(17DRP ep0): r_b 0.15 vs 0.30에서 같은 순번 프레임 카메라가 **최대 1.31 m** 떨어짐
(대표 프레임 (-7.07,-0.29) vs (-6.74,-0.24)), 그래서 depth median 1.87 vs 1.98 m.

### 그로 인해 드러난 설계 문제 (논문 C3)
현재 방식은 `e`가 **입력(관측)과 정답(GT path)을 동시에** 바꾼다 → "다른 위치 × 다른 정답"이라
모델이 `e`를 무시하고 "위치가 달라서"로 설명 가능 → **C3 paired counterfactual의 gradient 논리가 약해짐**.
(C3 논리: 같은 장면에 e별로 다른 정답이 와야 그 차이를 설명할 변수가 e밖에 없다.)

### 사용자 결정: **두 모드 모두 구현(모드 선택)**
| | (A) follow | (B) fixed |
|---|---|---|
| 관측 | r_b마다 다름(카메라가 새 경로 따라감) | **동일**(원본 프레임 고정) |
| GT path | 다름 | 다름 |
| e를 관측에 넣는 법 | 위치 변화로 간접 | **BEV robot-radius dilation**(논문 C1의 빠진 조각) |
| C3 | 약함 | **성립** |
| 비용 | 경로마다 재렌더 | 원본 depth 재사용 가능 |

**구현**(`embodiment_augment.py`):
- `dilate_bev_occupancy(bev, r_b, bev_range)` — 원형 구조요소로 점유셀 팽창(C-space 변환).
  BEV 4.46 cm/px → r_b 0.25 m = 5.6 px. 기존 `depth_rgb_to_bev_torch`엔 dilation이 **전무**했다.
- `render_at_poses(scene, poses_abs_c2w, r_b=...)` — 모드 (B): 주어진 원본 pose에서 렌더 + BEV dilation.
- `render_bev_along(...)` — 모드 (A) 유지.
- 검증 `06b_validate_modes.py` → `logs/embodiment_augment/modes/` + Artifact.
  실측: (B)에서 카메라 고정(-8.85,-1.24), BEV occupied **10.5% → 12.8%**(r_b 0.15→0.30). depth는 동일.

---

## 끝점 보호 규칙 (사용자 지시, 2026-08-15)

**지시**: "r_b 변화로 생성된 path가 기존 r_b의 gt path의 end point와 달라지면 기각하고 기존 r_b로 만든 path를 써야 됨."

**왜 필요한가**: `_snap_navigable`이 dilation으로 지워진 start/goal을 최근접 navigable 셀로 **옮긴다**.
그러면 도착지가 바뀌어 instruction("…화장실에서 멈춰라")이 **거짓 라벨**이 된다.
실측(미적용): goal 최대 **37.8 cm** 이동(17DRP ep0 r_b=0.5), start 최대 **157 cm** 이동.

**구현**(`embodiment_augment.py`):
- `GOAL_TOL_M = 0.10`(격자 5cm × 2칸 — 양자화 오차는 허용, 실제 goal 이동은 기각),
  `BASELINE_R_B = 0.15`("기존 r_b").
- `replan(..., goal_tol_m)` — 계획 후 `||traj[-1] - goal_xy|| > tol`이면 **None(기각)**.
- `replan_or_fallback(...)` → `(traj, used_r_b, status)`, status ∈ `ok` / `fallback`(기각·실패 시
  baseline r_b 경로) / `failed`(baseline도 실패).
- `06_validate_augment.py` / `06b_validate_modes.py`가 이 API를 쓰고, 기각된 r_b는 **범례에 색을 배정하지 않고**
  표에 "기각 → 기존 r_b 경로 사용"으로 표기.

**검증**(2씬 × 2에피소드 × r_b 4값 = 16): ok 7 / 기각→fallback 9 / 실패 0.
채택된 경로는 **전부 끝점 오차 ≤3.1 cm**(격자 양자화 수준) → goal이 이동하지 않음이 보장된다.
큰 r_b일수록 기각률↑ = 좁은 통로를 큰 로봇이 못 지나간다는 물리적 사실의 반영.

---

## pixel-goal 보존형 재계획 (사용자 재정의, 2026-08-16)

**지시**: 기본 path·원본 이미지·기존 pixel goal은 유지하고, (1) 현재 프레임 기준 r_b가 바뀌어도
pixel goal에 도달하는 경로를 생성, (2) pixel goal에서 충돌 시 goal 위치를 조정.

### 조사로 확정 (실측)
- pixel goal = frame `i+rel+1`의 **카메라 아래 바닥점**을 frame `i` **룩다운 카메라**에 투영.
- 저장 규약 **`[u,v] = [col,row]`**, **640×480 정수**. (`habitat_vln_evaluator.py:777` cv2.circle은 swap 버그—시각화만)
- **intrinsics가 rig마다 다름**: 125cm hfov 79°(fx 388.19), 60cm hfov≈68.7°(fx≈465.8).
- 라벨은 **S2의 텍스트 출력** `f'{u} {v}'`(`:1183`) → goal 조정은 곧 **언어 라벨 변경**.
- goal 거리 최대 3.25 m → 재계획이 국소라 저렴. `traj_poses`는 같은 구간의 다른 표현(끝점이 pixel goal).
- 3D goal은 샘플의 `pose[-1]`에서 **정확히** 얻는다(언프로젝션 불필요).

### V1에서 잡은 버그 (중요)
바닥을 `z=0` 고정으로 두면 **계단·층 이동 에피소드에서 깨진다**(카메라 z가 −0.61까지 내려감).
→ 바닥을 프레임마다 `cam_z − rig_height`로 **유도**. 고정 시 22.7%가 5px 초과(최악 19798px),
유도 후 **MAE u=1.22 v=0.94 px, >5px 0.00% (N=1701, 3 rig × 2 씬)** — V1 PASS.

### 구현 (`pixel_goal_utils.py`, `07_validate_pixel_goal.py`)
- `goal_world_from_poses` / `project_to_pixel` / `check_visible`(전방·화면안·**비가림**) / `adjust_goal` / `intrinsics_for_rig`.
- 조정 두 모드(사용자: 둘 다 구현):
  - **retreat**: 원본 경로를 뒤에서부터 훑어 clearance≥r_b인 **가장 먼** 점. 방향·경로 유지, **단조**.
  - **nearest**: goal 주변 최근접 안전점. 거리 유지, 옆으로 밀림.
- 조정 후 화면 밖·가림이면 **기각 → 기존 r_b 사용**(사용자 결정).

### 검증 결과 (17DRP ep0, 6프레임 × r_b 4값)
| 게이트 | 결과 |
|---|---|
| V1 투영 재현 | max 2.25 px (정수 절삭 수준) |
| V2 단조성 | **retreat=True(12쌍)**, nearest=False(18쌍) |
| V3 도달성 | 새 경로 끝점 오차 max **2.5 cm** |

동작: r_b 0.15 unchanged → 0.25/0.35 **adjusted**(문틀 앞에서 뒤로 물러남) → 0.50 retreat는 기각,
nearest는 살려냄. **trade-off**: retreat는 단조·semantics 보존이지만 기각률↑, nearest는 샘플 보존이지만 단조성 없음.
Artifact: retreat / nearest 각각 발행.

### 추가 지적 2건 대응 (2026-08-16)
1. **이미지 색 설명 누락** — 리포트에 원본/조정 goal 색 규약이 없었고, 심지어 `ADJ_COLOR`(초록) 칩을
   범례에 넣었지만 **실제로는 r_b별 `RB_COLORS`로 그리고 있었다**(범례≠그림). 또 `unchanged`인 r_b는
   아무것도 안 그리는데 그 설명도 없었다. → 프레임마다 **실제로 그린 것만** 색 칩으로 표시:
   원본 goal=빈 원(테두리), 조정 goal=r_b별 채운 원, 조정 없으면 "채운 원 없음" 명시.
2. **baseline r_b에서 원본 재현 여부 미검증** — 언급조차 안 했었다. **V0 identity gate 신설**.
   - **baseline r_b는 0.15가 아니라 `0.10`**: vln_ce는 habitat navmesh 기반이고
     `scripts/eval/configs/vln_r2r_mini.yaml`에 agent radius 미지정 → `AgentConfig().radius = 0.1`(확인).
     `BASELINE_R_B`를 0.10으로 수정.
   - 결과: **goal 유지 26/26 프레임, 픽셀 최대 변화 2.26 px**(정수 절삭 수준) → **PASS**.
     baseline에서 원본을 재현하므로, r_b를 키웠을 때의 변화를 embodiment 효과라고 말할 수 있다.
3. 버그 수정: `u0, v0, Z0 = project_to_pixel(...)`가 V0 결과 dict `v0`를 **가려서** 요약 생성이 실패했다.
   투영 변수명을 `u_chk, v_chk`로 변경.

### shift 모드 추가 (사용자 요청, 2026-08-17)
`retreat`(거리↓·방위 유지) 외에 **`shift`(거리 유지·방위↕)** 를 구현했다.
- 방법: 로봇→goal **거리 `rng`를 고정**하고 방위 `θ0`에서 좌/우로 `shift_step_deg`씩 번갈아 넓혀가며
  `clearance ≥ r_b`인 **가장 적게 튼** 지점을 채택(최대 `max_shift_deg=60°`).
- 의미: "같은 거리만큼 나아가되 옆으로 비켜선다" → **이미지에선 goal이 가로(u)로 이동**(행 v는 거의 유지).
  실측 이미지에서 문틀 옆 원본 goal이 r_b가 커질수록 **왼쪽 열린 바닥 쪽으로** 이동함을 확인.
- self-check: 거리 보존을 assert(1.050 m 유지, 방위만 변경).

**3종 비교** (17DRP ep0, 6프레임 × r_b 4값 = 24):
| 모드 | unchanged | adjusted | rejected | V2 단조 | 성격 |
|---|---|---|---|---|---|
| retreat | 10 | 8 | **6** | True | 거리↓·방위 유지. 경로 semantics 최강, 기각 많음 |
| **shift** | 10 | **13** | **1** | True | 거리 유지·방위↕. 샘플 보존 좋고 단조도 유지 |
| nearest | 10 | 14 | 0 | **False** | 최소 변위. 보존 최대지만 단조성 없음 |

→ `shift`가 **보존율과 단조성을 동시에** 만족해 기본값 후보로 가장 균형적이다.

### 좌우 시야 이탈 차단 (사용자 지시, 2026-08-17)
**문제**: 증강 경로의 3~8%가 화면 **좌우로** 이탈했다(원본 GT는 0%). S1은 현재 관측으로 궤적을 예측하는데
옆으로 안 보이는 공간을 지나면 근거 없는 라벨이 된다.
**해결(사후 기각이 아니라 생성 단계 차단)**: `EmbodimentAugmenter.fov_mask` + `replan_in_fov` 신설.
- 격자 셀을 현재 룩다운 카메라에 투영해 **수평 시야 안(0≤u<640)** 인 셀만 True → A* 입력 navigable에 AND.
- **세로는 제한 안 함**: 룩다운은 발밑 근거리가 이미지 아래로 빠지는데(원본 GT도 31%) 이는 데이터 고유
  성질이고 로봇 자신의 발밑이라 무해하다. 막는 건 좌우뿐.
- 함정 2개(둘 다 실측으로 발견·수정):
  1) 카메라 원점(로봇 자신)은 Z≈0이라 마스크에서 빠진다 → `near_radius_m=1.0` 근거리 예외.
  2) `plan_episode`는 내부에서 start/goal을 스냅하는데 `replan_in_fov`가 그걸 빠뜨려 **12개 중 1개만
     성공**했다 → `snap_to_grid`로 마스크된 navigable 위로 스냅 추가.
**결과**(17DRP ep0, r_b 0.35/0.50):
| 모드 | FOV=OFF 좌우 | FOV=ON 좌우 | 경로 수 | inside |
|---|---|---|---|---|
| retreat | 7.8% | **0.0%** | 6→6 | 82→89% |
| shift | 4.3% | **0.0%** | 11→11 | 91→95% |
| nearest | 7.5% | **0.0%** | 12→12 | 88→95% |
경로 수 손실 없이 좌우 이탈만 제거됐고 화면 내 비율도 올라갔다.

### shift를 lateral 정렬로 수정 (사용자 지적, 2026-08-17)
**지적**: 초기 `shift`는 로봇→goal **방위를 회전**시켜서 "목적지를 유지한 채 통과"가 아니라
**옆으로 돌아가**게 만들었다. 원인은 pixel goal 생성 시 **그 지점의 pose 방향(heading)을 안 쓴 것**.
**수정**: `adjust_goal(mode='shift')`를 **goal heading에 수직인 lateral 방향 이동만**으로 변경
(`_heading_at_end`로 경로 끝 tangent 추정, `±max_lateral_m` 5cm 스텝 최소 탐색).
→ 좁은 틈의 **중앙에 정렬**돼 원래 목적지를 유지하며 통과. **lateral로도 불가하면 통과 자체가 안 되므로
경로를 따라 뒤로** 물린다(status `adjusted_retreat`).
- self-check: 진행방향(along) 성분 **정확히 0**, lateral만 이동. 막히면 폴백 확인.
- 실측(17DRP ep0): lateral 통과 4건(변위 median **5 cm**), retreat 폴백 4건.
- 리포트 표시: lateral 통과=채운 원, retreat 폴백=테두리 원으로 구분.
**V2 지표도 수정**: 단조성을 직선거리로 재면 lateral이 거리를 소폭 늘려 깨진 것처럼 보인다 →
**진행 방향 전진량(heading 투영)** 으로 변경. 두 모드 모두 V2 True.

---

## 리포트 3건 정정 (사용자 지적, 2026-08-18)

### 1. G2에 원본 GT 궤적 추가
기존 G2는 r_b별 재계획 경로만 겹쳐 보여줘 **"baseline이 원본을 재현하는가"를 확인할 수 없었다**.
→ `05b_g2_report.py`에 `dataset_utils.load_gt_episode`로 **원본 GT(body_xyz)를 굵은 흰 선**으로 오버레이하고,
r_b별 **원본 GT와의 평균 거리** 표를 추가.
**결과**: r_b=0.10 → 31.7 cm, **r_b=0.25 → 19.7 cm(최소)**, r_b=0.40 → 21.5 cm.
→ `esdf_utils.ROBOT_RADIUS_M = 0.25`("논문 r_b, InternNav agent_radius 기본값")와 **일치**.
**중요**: **vln_n1의 baseline은 0.25, vln_ce는 0.1**(habitat 기본 agent radius)로 **서로 다르다**.
pixel-goal 작업(vln_ce)은 0.1을 쓰고, G2(vln_n1)는 0.25가 기준이다.

### 2. R3 이미지가 BEV인지 FPV인지 불명확
→ figtype 이름을 **"FPV depth 3분할 (BEV 아님)"** 으로 바꾸고 "카메라 시점 224×224 렌더, 위에서 본 BEV가
아니다"를 명시. 파일명 인코딩(`m50`=margin, `d10`=유효depth, 끝 숫자=yaw)도 함께.

### 3. R2의 "CPU BEV 165 ms/frame 병목"은 **오측** — 삭제
사용자 지적: "원래 GPU에서 BEV 만드는 코드가 있었는데, path를 만들려면 loader에서 CPU BEV를 만들어야
하는 상황인가?" → **아니다.** 확인 결과:
- **3dloader의 경로 계획은 BEV를 안 쓴다** — 캐시된 **3D occupancy**에서 `derive_obstacle_2d`로 2D를 뽑는다.
- S1 BEV는 모델이 **GPU**에서 `traj_depths`로부터 계산한다.
- 즉 이 loader에 **BEV가 필요한 곳이 없다**. `with_bev=False`의 이유는 "느려서"가 아니라 **"중복이라서"**.
- **165 ms/frame은 torch 스레드 경합**이었다(재측정: 24스레드 162 ms → **1스레드 1.7 ms**, 약 100배).
  2dloader 메모가 먼저 지적한 것과 동일 현상. **replan도 49→24 ms**로 영향받았다.
- ⚠️ **2dloader는 BEV가 planning 격자**라 loader가 BEV를 만들어야 하고, 거기서는 스레드 설정이 직접 영향.

**정정 후 유효 R2 수치**(worker당 `torch.set_num_threads(1)` 적용):
replan 44 ms + **Open3D depth 렌더 350 ms(29 ms/frame, 지배 비용)** = sample 393 ms.
workers 0/2/4/8 → 3.83/6.57/10.39/13.78 samples/s → batch 16 준비 4.18/2.44/**1.54**/1.16 s.
batch 크기 영향 없음. **GPU step 시간 미측정 → 병목 여부 미결**(남은 숙제).
bench에 `torch.set_num_threads(1)`을 worker 초기화에 추가했다.
