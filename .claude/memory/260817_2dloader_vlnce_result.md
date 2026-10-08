# 2dloader_vlnce — 시야각 내 RGB/depth만으로 on-the-fly embodiment augmentation (결과)

2026-08-17. 근거 문서: `/ws/src/wiki/VLN/paper writing/VLN collision.md` §"해결 방법 3"(L149–165)의
세 구현안 중 **"시야각 내의 이미지, depth만 활용"**. 계획: `/root/.claude/plans/rosy-shimmying-scroll.md`

## 이전과 무엇이 달라졌나

| | `3dloader_vlnce` (기존) | **`2dloader_vlnce` (신규)** |
|---|---|---|
| 필요한 것 | 씬 mesh(199 MB) + 3D occupancy 캐시 + Open3D 렌더러 + 에피소드 정합(14.8 s) | **그 프레임의 RGB/depth 한 장뿐** |
| 장애물 맵 | mesh → 3D occ → `derive_obstacle_2d` | depth → **학습이 쓰는 그 BEV 함수**(`depth_to_bev_occ_ros2`) |
| 관측 변경 | 새 경로에서 mesh를 다시 렌더 | RGB·depth·BEV를 **합성 장애물 하나로 동시에** 변경 |
| 프레임당 비용 | 391 ms | **18 ms** |
| 실데이터 확장 | 불가(mesh 필요) | **가능** — 논문 L165의 근거 |
| 한계 | 없음(전역 occ) | 관측 범위 밖 우회 불가 (L158). W1이 수치화 |

## 어느 rig를 쓰는가 (자주 헷갈리는 지점)

학습 preset은 `(height, pitch_1, pitch_2)` 3인자이고 (`r2r_125cm_0_30` → 125 cm, pitch_1=0°, pitch_2=30°),
**pose·depth·pixel goal 라벨은 전부 pitch_2(룩다운) rig 기준**이다
(`internvla_n1_lerobot_dataset.py:891` `setting = f'{height}cm_{pitch_2}deg'`).

| 무엇 | 어느 rig | 근거 |
|---|---|---|
| `pose.<rig>` (4×4 cam2world) | **pitch_2 룩다운** | `get_annotations_from_lerobot_data(data_path, setting)` |
| depth PNG | **pitch_2 룩다운** | `__getitem__`이 `_{pitch_1}deg`→`_{pitch_2}deg` 치환 후 `rgb`→`depth` |
| `goal.<rig>` = pixel goal 라벨 | **pitch_2 룩다운, 640×480 정수 [u,v]** | 답변 문자열이 룩다운 `<image>` 다음 턴 |
| `traj_images` (S1 입력) | **pitch_2 룩다운** | `traj_images.append(lookdown_image)` |
| BEV / planning 격자 | **pitch_2 룩다운** (depth가 여기서 나오므로) | 이 폴더 전체 |
| FPV RGB (pitch_1) | S2가 보는 정면 이미지 | 여기엔 **장애물 합성만** 한다 |

두 rig는 **카메라 중심이 같고 pitch만 다르다**(실측 `‖t_fpv − t_ld‖ = 0.000000 m`). 그래서 같은 3D
박스를 각자의 pitch로 투영하면 두 이미지가 자동으로 정합된다.

## 코드 (`scripts/dataset_converters/2dloader_vlnce/`)

기존 코드는 **import만**, 수정 없음.

| 파일 | 역할 |
|---|---|
| `episode_io.py` | parquet(두 rig pose/goal) + 640×480 jpg/png 로더. 학습의 depth 전처리 재현 |
| `local_map.py` | depth → planning 격자. 좌표계 3개 변환, `build_bev`, `carve_free_from_floor`, `build_local_map`, `plan_path`, `limit_torch_threads` |
| `obstacle_synth.py` | 3D 박스 → ray-box 교차 depth 합성 + flat Lambert RGB. **`kind` 분기 = 확장 seam** |
| `augment2d.py` | 오케스트레이터. `Embodiment`, `Aug2DCfg`, `Augmenter2D.augment/decide_goal` |
| `viz2d.py`, `report_common.py` | 그리기 + **범례/용어표/파라미터표**(모든 리포트가 공유) |
| `validate_w0..w6.py` | 게이트별 검증 + HTML 리포트 |
| `command_2dloader.md` | 전체 명령어 |

## 검증 결과 (17DRP5sb8fy ep0 + s8pcmisQ38h ep0, 각 6프레임 × r_b 4값)

| 게이트 | 17DRP5sb8fy | s8pcmisQ38h |
|---|---|---|
| **W0** 좌표·항등 | **PASS** — 중력정렬 4.0e-5, 학습 GT 프레임과 **3.6e-8 m** 일치, pixel goal 재현 ≤ 2.25 px, GT가 점유셀 밟음 0 | **PASS** |
| **W1** 3D 오라클 대비 | IoU 0.594, 미관측 장애물 87.4% | IoU 0.473, 미관측 97.0% |
| **W2** 장애물 정합 | **PASS** — footprint 안 100%, 표면오차 0.00 mm, 교차 rig ≥ 99.8% | **PASS** |
| **W3** e→경로 | **PASS** — feasibility 단조 6/6, 원본 GT 통행 불가 12/24, baseline 대비 이탈>10 cm 8/24 | **PASS** — 11/24, 이탈 1/24 |
| **W4** pixel goal | **PASS** — V0/V2/V3. unchanged 16 / adjusted 8 / 기각 0 | **PASS** — 14 / 5 / 5 |
| **W5** 예산 | **20.7 ms/frame — 목표 20 ms 미달성**. 같은 프레임 이미지 읽기(7.8 ms)의 2.7배라 IO에 묻히지 않는다 | — |
| **W6** paired counterfactual | 유효 16/16 | 유효 13/16 |

리포트에는 **모든 색·기호·상태값·표 열의 뜻**이 표로 들어 있다(`report_common.py`가 단일 출처).
로컬: `logs/embodiment_augment2d/{w0..w6, s2_*}/{*.jpg, report.html, artifact.html}`
발행 주소 전체 목록: `scripts/dataset_converters/2dloader_vlnce/reports.md`

## 개발 중 발견해서 고친 것 (전부 실측 근거)

### 1. "goal이 시야각 밖"은 **틀린 진단이었다** — BEV free 생성 방식의 한계였다
처음엔 baseline `r_b=0.10`인데도 goal 라벨이 371 px 튀는 프레임이 있었고, 이를 "goal이 시야각 밖"으로
설명했다. **틀렸다.** 실측하면 그 goal들은 전부 **이미지 안이고 가려지지도 않았다**:

| 프레임 | pixel goal (u,v) | 가시성 | Z | BEV 셀 |
|---|---|---|---|---|
| s8pcm f11 | (222, 182) | ok | 3.34 m | **unknown** |
| s8pcm f14 | (320, 178) | ok | 3.44 m | **unknown** |

진짜 원인: `depth_to_bev_occ_ros2`의 raycast는 **점유 셀로 향하는 광선 위에만** free를 찍는다.
점유는 높이 밴드 `(h_nav, h_b]`에 반환이 있어야 생기므로, **그 방향에 밴드 안 물체가 없으면
(=바닥만 보이면) 광선 자체가 없어 화면에 뻔히 보이는 바닥이 unknown으로 남는다.**

→ **해결: `carve_free_from_floor`** — 같은 depth 한 장에서 바닥 반환(`z ≤ h_nav`)으로 free를 추가로 판다.
바닥이 보인다는 것은 "그 방향 그 거리까지 통행 가능"의 직접 증거이고, 같은 이미지 안의 정보이므로
정보 누수가 아니다. **관측 BEV(`m['bev']`)는 그대로 두고 planning용 `free`/`observed`만 보강한다.**
결과: 두 씬 12프레임 전부 goal이 관측된 free 칸이 되고 `unobserved` 0, 관측 면적 5.9%→7.5%(f12).

### 2. 미관측 셀 정책
`block`(보수적)은 시야각 원뿔의 측면 경계가 장애물이 되어 GT clearance가 0.00–0.10 m로 붕괴 → 계획 불가.
`free`는 clearance는 정상이지만 **A\*가 벽 뒤 미관측 영역으로 우회**(BEV 그림에서 발견).
최종 **`nontraversable`**: ESDF(여유)는 실제 관측된 물체까지만, navigable은 관측된 free 칸으로 제한.
+ `robot_free_m=0.5`(룩다운이 못 보는 발밑) 없으면 A* 시작 셀이 통행 불가.

### 3. goal 라벨과 계획 목표 분리 (`Augmenter2D.decide_goal`)
- 관측 free + 여유 ≥ r_b → `unchanged` (라벨 유지)
- 관측 free + 여유 < r_b → `adjusted` (**라벨도 갱신** — 이 로봇에겐 진짜 못 가는 곳)
- 관측 불가 → `unobserved` (**라벨 유지**, 계획 목표만 물림)

### 4. goal 조정 모드가 3D와 반대
`retreat` 기각 0 · V2 단조 True(채택) / `shift` 기각 4 / `nearest` V2 **False**.
`shift`는 거리를 유지한 채 방위만 트는데 관측 부채꼴이 좁아 그 방위가 미관측으로 나간다.

### 5. ⚠️ torch 스레드 절벽 — **이전 설명도 정정**, 그리고 3dloader R2의 근거가 무효다
처음엔 "224² 텐서에 멀티스레드가 순수 오버헤드"라고 썼다. **틀렸다.** 스레드 수를 훑으면
**코어 수와 같을 때(24 == `os.cpu_count()`)만** 무너진다:

```
threads   1 -> 1.41 ms      threads   8 -> 1.14 ms      threads  23 -> 1.09 ms
threads   2 -> 1.73 ms      threads  16 -> 1.05 ms      threads  24 -> 402.30 ms   ← 코어 수
```
23개만 돼도 정상이고 24에서만 **370배** 느려진다 — full subscription에서 OpenMP spin-wait 경합이다.

**3dloader R2의 "depth→BEV 165 ms/frame"도 같은 원인이다** (직접 재측정,
`EmbodimentAugmenter.render_bev_along(T=12, with_bev=True)`, s8pcmisQ38h):

| threads | CPU BEV ×12 | frame당 |
|---|---|---|
| 24 (= 코어 수) | 2164 ms | **180.4 ms** |
| 1 | 21 ms | **1.8 ms** |

(`08_bench_parallel.py`를 고쳐 같은 스레드 설정에서 짝지어 median으로 잰 값. worker throughput 표는
영향 없음 — 재측정 3.88/6.73/10.23/13.55로 원래 값과 동일하다.)

R2가 잰 곳은 DataLoader worker가 아니라 **메인 프로세스**(`08_bench_parallel.py:168-174`)라 torch 기본
24스레드였다. → **R2의 결론 "CPU BEV가 병목이므로 `with_bev=False`로 가야 한다"는 근거가 무효다.**
`limit_torch_threads()` 한 줄로 **100배** 빨라진다.
→ **3dloader R2 리포트·메모 갱신 완료**(`08_bench_parallel.py`에 스레드 제한 + 정정 절 추가,
`logs/embodiment_augment/perf/` 재발행, `260815_review_r1r2r3_result.md` §R2 정정).

### 5b. ⚠️ 경로가 점유·미관측 칸을 지나고 있었다 (사용자 지적 → 수정)
W3 그림에서 경로가 검은(미관측)·빨간(점유) 칸을 지난다는 지적을 받고 실측: **점유칸 최대 18%,
미관측칸 최대 39%**. 세 원인이 겹쳐 있었다.

| 원인 | 근거 | 수정 |
|---|---|---|
| `thin_waypoints(0.8 m)` + `cubic` spline이 코너를 가로지름 | 격자를 4.46 cm까지 낮춰도 위반이 남음 | spacing **0.4 m** + **bezier**(볼록껍질을 벗어나지 않음) → 점유칸 통과 **0%** |
| `greedy_refine`이 원본 `esdf`를 봐서 미관측 쪽으로 밀어냄 | 미관측은 장애물이 아니라 clearance가 큼 | refine을 **`esdf_nav`**(미관측=0)로 |
| coarse A*가 거의 미관측인 칸을 통과 | `downsample('any')`는 16칸 중 1칸만 통행 가능해도 통과 | `navigable`은 `any`, **`free`는 `all`**로 따로 축약 → 위반 0, 도달률 유지(29/48) |

+ **fine 격자 재검사**(`path_violations`)를 계획 끝에 넣어 위반 시 실패로 돌린다.
결과: 두 씬 × 장애물 유/무 × r_b 4값에서 **채택된 경로 96개 중 위반 0개**.

### 5c. 발밑 blind 반경을 rig 기하에서 계산 (사용자 지적 → 수정)
`ROBOT_FREE_M=0.5` 상수로는 0.5~0.67 m 고리가 unknown으로 남아 **로봇 바로 앞에서 경로가 미관측
칸을 지났다**(f0에서 경로의 20%). 이 rig가 원리적으로 볼 수 있는 최소 거리는
`cam_height / tan(pitch + vfov/2)` = **0.67 m**(125cm_30deg, 224px)이므로 `near_blind_radius`로
계산해 쓴다(여유 1.15배 → 0.773 m). 적용 후 **원본 GT의 위반이 0%**가 되어 기준선이 생겼다.

### 5d. goal 후퇴를 clearance가 아니라 **도달 가능성**으로 (사용자 지적 → 수정)
clearance만 보면 큰 로봇은 goal이 당겨져 쉽게 성공하고 작은 로봇은 먼 goal 그대로 실패하는
**뒤집힌 결과**가 나왔다(17DRP f29: r_b 0.10/0.20 기각, 0.35/0.50 성공). `_plan_to_farthest`가
GT 경로를 따라 물러나며 **도달 가능한 가장 먼 지점**을 찾도록 바꾸고, 하한
`min_goal_m=0.5`(로봇 자기 자리까지 당겨지는 것 방지)를 두었다. 3dloader의 baseline fallback은
같은 비단조를 만들어 제거했다. 결과: **비단조 프레임 0**.

### 5e. 통과 가능한 장애물 배치 (사용자 요청)
경로 위에 그냥 놓으면 모든 embodiment가 똑같이 막혀 counterfactual이 안 된다. 넣기 전 지도의 여유
`c0`를 읽어 박스를 `d = c0 − 반폭 − 2g`만큼 밀어 **clearance가 g인 통로**를 남긴다
(`ObstacleCfg.gap_m=(0.08, 0.55)`, sweep하는 r_b 범위를 걸치게). 결과(두 씬 12프레임):

| r_b | 통과 | 우회 | 기각 |
|---|---|---|---|
| 0.10 | 10 | 0 | 2 |
| 0.20 | 9 | 1 | 2 |
| 0.35 | 4 | 1 | 7 |
| 0.50 | 0 | 0 | 12 |

### 5f. 시각화가 **계획에 쓴 격자**를 그리도록 (사용자 지적 → 수정)
"지도를 실제로 활용해 path를 만드는가? 그렇다면 그 지도를 시각화해야 한다"는 지적.
확인해 보니 **아니었다**: A*는 `plan_path`가 만든 **17.9 cm 조립 격자**(`nav_coarse`)에서 도는데,
그림은 4.46 cm 세밀 격자를 그리고 있었다. 두 격자는 다르다 —
`nav_coarse = downsample(navigable,'any') & downsample(free,'all')`.
실측 차이(17DRP f0): 통행 가능 칸 A*격자 4192 vs 세밀 4657.

→ `plan_path`가 이미 반환하던 `nav_coarse`를 `_plan_to_farthest`/`plan_and_label`이 위로 전달하고,
`viz2d.bev_rgb_planning(m, nav_coarse)`가 **그 격자를 그린다**(4×4 확대라 네모가 굵게 보인다).
계획이 실패한 경우에도 **A*에게 준 격자**를 그려 "이 지도에서 못 찾았다"가 보이게 했다.
경로 다듬기는 세밀 `esdf_nav`로, 최종 검사는 세밀 격자로 한다는 점도 범례에 적었다.

### 6. 장애물 합성 범위
640×480 전체 ray-box는 rig당 40 ms. 박스 8꼭짓점 투영의 bbox 안에서만 계산하면 **3 ms**
(볼록체 실루엣 ⊂ 꼭짓점 볼록껍질이므로 결과 동일, self-check로 확인).

### 7. W2 교차 rig 지표의 거짓 실패
바닥에 붙은 낮은 박스가 가까우면 **수평 FPV(pitch_1=0)의 화각 아래**로 내려가 안 보인다. 빈 마스크를
precision 0으로 세면 게이트가 거짓 실패 → `nan`으로 제외하고 별도 카운트. S2는 룩다운 이미지도 함께
받으므로 학습 문제는 아니다.

### 8. 정합 rig
W1의 `T_sf2mesh`는 **FPV(pitch_1) rig**로 잰다. 룩다운은 바닥을 스치듯 보아 residual이 커진다
(0.0033 m vs 0.0156 m). T는 rig와 무관하므로 좋은 쪽을 쓴다.

## 논문에 그대로 쓸 수 있는 수치

- **"구현 3안" 비교의 비용**: 시야각 내 depth만 쓰면 3D 전역 occ 대비 관측영역 IoU 0.47–0.59,
  **장애물의 87–97%를 못 본다**(±5 m 정사각 기준).
- **`e`가 GT를 실제로 바꾼다**: 같은 관측에서 `r_b` 0.10→0.35이면 원본 GT가 통행 불가로 바뀌는
  프레임이 **6/6**(두 씬). baseline 재계획 경로 대비 최대 이탈 최대 1.85 m.
- **비용**: 프레임당 **20.7 ms** (CPU 1스레드). 같은 프레임 이미지 읽기가 7.8 ms이므로 **IO에 묻히지
  않는다** — worker 수로 흡수하거나 바닥 carving(3.3 ms)·장애물 합성(4.1 ms)을 더 줄여야 한다.
- **저장 비용 논변**: vln_ce는 5개 리그를 미리 렌더해 346 GB인데 `r_b` 축은 아예 없다.
  이 방법은 저장 0으로 `r_b`·`h_nav`·`h_b`를 **연속**으로 서비스한다.

## 남은 것 / 한계

1. **학습 dataloader 연결은 이번 범위 밖**(사용자 확정). 함정: pixel goal 라벨이 chat 문자열
   (`f'{action[0]} {action[1]}'`, `..._lerobot_dataset.py:1183`)로 **토크나이즈 전에** 확정되므로
   `__getitem__` 말미 후처리로는 못 바꾼다 → 전체 override 또는 base에 훅 추가 필요.
2. **큰 `r_b`에서 기각률이 높다** — 관측 범위 안에 틈이 없으면 만들 GT가 없다(L158). `fallback` 또는 폐기.
3. **장애물 사실성** — v1은 flat-shaded 단색 직육면체. 텍스처·복잡 형태·dataset 유사 object는
   `render_obstacle`의 `kind` 분기 한 곳만 늘리면 되고 검증 스크립트도 그대로 재사용된다.
4. **씬 의존성** — s8pcmisQ38h는 경로 이탈>10 cm가 1/24로 17DRP(8/24)보다 훨씬 적다. 넓은 복도에서는
   `e`를 키워도 경로가 안 바뀐다 — 논문의 "좁은 통로 sub-episode 분리 평가"와 같은 얘기.
5. **하드코딩 intrinsics 불일치**(기존 코드 이슈) — 학습은 `BEVProcessor(fx=388.19,...)`를 rig와 무관하게
   쓴다(`..._lerobot_dataset.py:1026`). 125 cm rig에선 같지만 60 cm rig(fx≈465.8)에선 틀리다.
   이 폴더는 `intrinsics_for_rig`로 올바르게 계산한다 — 이식 시 함께 고쳐야 train==eval이 유지된다.
