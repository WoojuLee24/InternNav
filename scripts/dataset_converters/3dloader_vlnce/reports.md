# 리포트 목록 (로컬 경로 + Artifact 주소)

on-the-fly embodiment augmentation 작업의 발행 리포트.
**파일명 앞 번호가 실행 순서**다(`gs_vlnpe`와 같은 규칙). 번호 없는 `*_utils.py`·`embodiment_augment.py`·
`navmesh_grid.py`·`vlnce_align.py`·`publish_artifact_report.py`는 라이브러리라 순서가 없다.

각 리포트는 **로컬**(`logs/embodiment_augment/<stage>/`)과 **Artifact**(발행본) 양쪽에 있다.
로컬 파일은 두 개다 — `summary.html`(브라우저로 바로 열기용) / `artifact.html`(이미지 base64 내장, 발행본).

- 계획: `.claude/plans/ws-src-wiki-vln-paper-writing-v0-2-wiggly-dongarra.md`
- 실행순서·명령: `command_embodiment_augment.md` (같은 폴더)
- 단계별 상세: repo `.claude/memory/260814_*.md`, `260815_review_r1r2r3_result.md`,
  `260819_map_align_rb_ladder_result.md`

---

## 1. 3dloader_vlnce — "3D occ 전부 활용" (이 폴더)

> **조건 통일(2026-08-20)**: 모든 단계의 `--rig` 기본값을 `125cm_0deg`로 맞췄다.
> 두 리그는 **pitch만 다르고 카메라 위치가 완전히 동일**하므로(실측 median/max **0.00 mm**)
> `03`~`05`는 위치·바닥높이만 쓰고 렌더를 하지 않아 **수치가 바뀌지 않는다** — 재실행으로 확인했다.

### 00~02 · 전제 검증 — 실패하면 뒤 작업이 무의미
| # | 리포트 | 실행 스크립트 | 로컬 경로 | 무엇을 물었나 | 결과 | Artifact |
|---|---|---|---|---|---|---|
| 00 | 📐 **카메라를 집 3D 좌표에 놓을 수 있나** (구 G1) | `00_verify_pose_mesh.py` | `logs/embodiment_augment/g1/` | vln_ce 카메라를 씬 mesh 좌표에 정확히 놓을 수 있나 | s8·ep0·125cm_0deg(#02와 동일 조건): 정합 **0.0003 m**, 렌더 vs 저장 depth **median 0.0005 m** · p90 0.0009 · 꼬리 **0.04%**. 17DRP도 median 0.0005 m → **씬 비의존** | https://claude.ai/code/artifact/e9ff4aa3-7fc3-4618-93d6-026b08b026ef |
| 01 | 🧱 **집 3D를 지오메트리 전용 ply로** | `01_build_scene_geo.py` | (산출물 `data/embodiment_aug/scene_geo/`) | 씬 전체를 worker에 올릴 수 있나 | 텍스처를 버리면 **7~10 MB · 로드 0.1 s · RAM 0.08 GB** — 타일 하나(6 MB)보다 작다. 무거운 건 지오메트리가 아니라 텍스처(367 MB jpg → RAM 8.5 GB). **타일링 폐기** | — |
| 02 | 🔬 **depth 세 소스 비교 — 저장 / 우리 / habitat** (구 G1c) | `02_verify_depth_sources.py` | `logs/embodiment_augment/depth_sources/` | 저장 / 우리(Open3D·obj) / habitat(glb) depth가 범위별로 얼마나 다른가 | **우리↔habitat median 0.00~0.01 mm** (5·10·15·20 m 전 범위, p90 0.03~0.05) — 에셋·렌더러가 달라도 동일. 저장과의 0.50 mm는 **저장 포맷의 버림**(부호 99.3~100% 양수) → **G1의 0.5 mm는 우리 오차가 아니다** | https://claude.ai/code/artifact/da0cb778-1510-4c61-824b-fec94734ee65 |
| 03 | 🗺️ **데이터셋이 쓴 지도에 두 경로를 올린다** ← **기준점** (구 S1) | `03_verify_navmesh_gt.py` | `logs/embodiment_augment/s1_navmesh/` | 원본 정답이 데이터셋의 지도에서 벽을 뚫나 · waypoint를 이어 만든 경로는 원본을 얼마나 재현하나 | 배포 navmesh 설정이 **habitat 기본값 그대로** → "원본 파라미터"의 자유도 0. waypoint **474/474 · 531/531**, leg `find_path` ≤0.5 m **69/69 · 47/47**, 5 cm로 채운 경로 점 ⓐ **100% / 99.43%** · ⓑ **100% / 100%**. **"벽 뚫기"는 재현되지 않는다** — J1 밖 **0 / 14점**(최대 깊이 **3.7 cm** < 한 칸), J1통과·J2밖 **0 / 58점 전부 깊이 1칸** = 격자 반올림. ⚠️ **길이비 ≈ 1은 "같은 경로"의 증거가 아니다** (s8 median 56.4 cm인데 길이비 1.002 — 측지 최단선이 장애물 반대편으로 돈다). `recompute_navmesh` 재현 오차 **0.02%/0.15%** | https://claude.ai/code/artifact/330c32e4-4b25-4f51-96c6-5fc54f1756a0 |
| 04 | 🧭 **우리 지도가 habitat 지도와 같은 판정을 내리나** (구 S2) | `04_calibrate_map.py` | `logs/embodiment_augment/s2_mapcal/` | 우리 지도가 기준과 같은 판정을 내리나 · **왜 다른가** · 그 위에서 정답이 성립하나 | 좌표 정합 **1.0000**. **원인을 처음 분해**: s8 거짓 승인의 **66.6%가 "이 높이에 바닥 없음"**(17DRP는 6.0%, 대신 경계 1칸 80%) → 다층 씬 문제. 한 줄 수정으로 s8 일치율 **75.83→90.69%**지만 **거짓 기각 164→1437**로 채택 불가 → default는 navmesh 래스터화 유지. **pathfollower**: 재현 오차 median s8 **56.4→8.8 cm**인데 분포가 두 덩어리(5개 5.4 cm / **4개 72.8 cm = 경로 갈라짐**). GT가 우리 지도를 벗어난 점 **0**. **recast-like v2**(칸마다 바닥+despike+시작점 연결성): 17DRP 96.23% · s8 **92.30%(모든 후보 중 최고)** · GT 이탈 0/0 — 단 s8 잔여 F 745로 G1 실패 → default는 navmesh 래스터화 유지, v2는 navmesh 없는 씬용 후보. ⚠️ 게이트 C·D·G1(s8) 실패 | https://claude.ai/code/artifact/fc3a8d19-e399-45b1-86cd-0389371a86ad |

### 종합 리포트 (여러 stage를 가로지르는 정리)
| 리포트 | 무엇 | Artifact |
|---|---|---|
| ⚖️ **지도 3종 비교 — navmesh vs 밴드 vs recast v2** | #03·#04·벤치의 비교표 종합: 질문 구조 · 정확도(일치율/F·R·S/게이트/ablation) · GT 생존 · **속도 벤치**(밴드 1.7~4.2 ms · navmesh 웜 37~73 ms · recast v2 17~41 ms) · r_b 실험 가능성 · 시나리오 추천 | https://claude.ai/code/artifact/aea56605-e2ef-4868-af5b-69f4939a988e |

### 05 · 정답 경로 생성 — **현재 방식**
GT를 기본으로 따라가고 못 지나갈 때만 보정. 지도는 `--map_source occ|navmesh`로 고른다(기본 `occ`).

| # | 리포트 | 실행 스크립트 | 로컬 경로 | 조건 | 결과 | Artifact |
|---|---|---|---|---|---|---|
| 05a | 📍 17DRP · `occ` ← **기본 조합** | `05_validate_waypoints.py --map_source occ` | `logs/embodiment_augment/waypoints_17drp_occ/` | 밴드 h_nav 0.20 / h_obs 1.50 (habitat 직독값) | **W5 PASS** `none` 69/69 · 보정 **0** · GT 거리 **0.00 cm** · **거짓 승인 2**/276 (r_b=0.30) · 거짓 기각 47 | https://claude.ai/code/artifact/003a26b8-2335-4cdb-bde5-bc0ce6d3c057 |
| 05b | 📍 s8 · `occ` | `05_validate_waypoints.py --map_source occ` | `logs/embodiment_augment/waypoints_occ/` | 동일 | **W5 PASS** `none` 47/47 · 보정 **0** · GT 거리 **0.00 cm** · **거짓 승인 0**/188 · 거짓 기각 45 | https://claude.ai/code/artifact/e6fd96c9-f60a-446d-b0a8-1f17db0ee1e6 |
| 05c | 📍 17DRP · `navmesh` | `05_validate_waypoints.py --map_source navmesh` | `logs/embodiment_augment/waypoints_17drp/` | navmesh 래스터화 | **W5 PASS** `none` 69/69 · GT **0.00 cm** · **거짓 승인 0**/276 · 거짓 기각 58 · 보정은 r_b=0.2 nudge 2, r_b=0.3 corridor 5 | https://claude.ai/code/artifact/2948897e-949b-48c6-a302-c0cb36d66026 |
| 05d | 📍 s8 · `navmesh` | `05_validate_waypoints.py --map_source navmesh` | `logs/embodiment_augment/waypoints/` | navmesh 래스터화 | W5 **CHECK** `none` 37/42 · 보정 2 · GT 0.58 cm (실패 5건 전부 `hard`=1셀 오차) · **거짓 승인 0**/188 · 거짓 기각 58 | https://claude.ai/code/artifact/a2f7fbc8-d644-4855-9576-6c2b16fe44fa |
| 05e | 🛣️ **로봇 크기가 정말 경로를 바꾸나** (구 G2) | `05b_g2_report.py` | `logs/embodiment_augment/g2/` | 로봇 반경만 바꿔도 경로가 달라지나 · 생성 경로가 거짓 라벨은 아닌가 | 계획성공이 `r_b`에 단조 감소(0.1→6/7, 0.35→0/7). **장애물 관통 경로 4/7 발견 → 원인 제거(fine 격자 A*) + 4중 기각** | https://claude.ai/code/artifact/01c694fc-7762-4f8b-a99b-47f51e15a077 |

### 06~07 · 관측·라벨
| # | 리포트 | 실행 스크립트 | 로컬 경로 | 무엇을 물었나 | 결과 | Artifact |
|---|---|---|---|---|---|---|
| 06b | 🔀 **로봇 크기를 관측에 넣는 두 방식** | `06b_validate_modes.py` | `logs/embodiment_augment/modes/` | `e`를 관측에 넣는 두 방식 중 어느 것이 C3에 맞나 | **(B) 카메라 고정 + BEV dilation** 채택 (BEV 점유 7.6→10.2%) | https://claude.ai/code/artifact/0742d328-6554-4f42-8ed2-efdfcc1f672d |
| 07a | ↔️ **화면 목표점 조정 — shift** ← **채택** | `07_validate_pixel_goal.py --goal_adjust shift` | `logs/embodiment_augment/pixelgoal_shift/` | goal의 **진행 방향 수직**으로만 최소 이동 → 목적지 유지하며 통과 | 단조 / 도달 2.6 cm | https://claude.ai/code/artifact/60cfa34a-fd40-4ab8-9b9c-a307ad9b6f8e |
| 07b | ⬅️ **화면 목표점 조정 — retreat** | `07_validate_pixel_goal.py --goal_adjust retreat` | `logs/embodiment_augment/pixelgoal_retreat/` | 경로를 따라 **뒤로 물러남**. shift의 폴백 | 단조 / 4.6 cm | https://claude.ai/code/artifact/e2dc0817-96c5-45ea-8747-b037ca696ab5 |
| 07c | 🎯 **화면 목표점 조정 — nearest** (대조군) | `07_validate_pixel_goal.py --goal_adjust nearest` | `logs/embodiment_augment/pixelgoal_nearest/` | 방향 무관 **최소 변위** — 보존율 최고지만 단조성 없음 | **비단조** / 7.1 cm | https://claude.ai/code/artifact/e56f43ec-65c1-465b-b65f-5b74435f7e4c |
| ~~06~~ | 🎬 **P-B1** augment 시각 검증 | `06_validate_augment.py` | 없음 — **리포트 폐기** | 같은 프레임 × r_b 4값에서 관측·경로가 달라지나 | **폐기: 모드 (A) follow 자체가 폐기**됐다(→ #06b에서 (B) 채택). 코드는 다른 스크립트에 헬퍼를 제공하려 유지 | — |

공통 게이트(07a~c): **V0 원본 재현 0.00 px (26/26)** · V1 2.25 px · **좌우 시야 이탈 0.0%**

### 08 · 속도
| # | 리포트 | 실행 스크립트 | 로컬 경로 | 무엇을 물었나 | 결과 | Artifact |
|---|---|---|---|---|---|---|
| 08 | ⚡ **학습을 느리게 만들지 않나** (구 R2) | `08_bench_parallel.py` | `logs/embodiment_augment/perf/` | augment가 **GPU를 기다리게 만들지 않나** (↓아래 상세) | sample 1개 **390 ms**(지배: Open3D 렌더 339) · 4 workers **15.8 samples/s** = batch 16 준비 **1.01 s** · ⚠️ **GPU step 미측정이라 병목 여부는 미결** | https://claude.ai/code/artifact/88c35ccc-8740-413b-8d59-7e2abd0e3e22 |

### 09 · waypoint-only 재현 (GT를 입력으로 쓰지 않는 실험)
| # | 리포트 | 실행 스크립트 | 로컬 경로 | 무엇을 물었나 | 결과 | Artifact |
|---|---|---|---|---|---|---|
| 09 | 🧭 **waypoint만 주면 GT를 얼마나 재현하나** — navmesh vs recast v2 | `09_waypoint_replan.py` | `logs/embodiment_augment/waypoint_replan/` | GT 궤적 없이 waypoint 좌표만으로 두 맵에서 **동일 planner**로 생성한 경로가 GT에 얼마나 가깝나 | 루트가 유일한 leg는 두 맵·두 씬 모두 median **8.1~9.2 cm**(플로어 2.7 cm) — planner는 지배 변수가 아니다. 남는 오차는 전부 **route-split**(waypoint 사이 장애물 섬, s8 navmesh 4/47 leg·median 128 cm)이고 이는 instruction에만 있는 정보라 **waypoint-only의 구조적 한계**. recast v2는 거짓 승인 shortcut으로 split +2(s8 6/47) → G3·G4 FAIL. GT 생존율 navmesh 100/99.18% · recast 100/100% | https://claude.ai/code/artifact/b510cb8e-fb12-40a0-9869-4ba8c89042dd |

### 폐기 (인용하지 말 것) — 로컬 산출물도 삭제함
| 폐기 리포트 | 왜 | 대체 |
|---|---|---|
| **G3 예산 초판** (`g3/`) | `synthesize_action_poses` 인자 순서 오류로 카메라가 천장 위(z=7 m)에서 **빈 화면**을 렌더 → 시간이 실제보다 짧게 나왔다 | **#08 R2** |
| **타일링 전체 (G4 초판 · R3 재판)** | 타일링 자체가 **불필요**했다 — 무거운 건 지오메트리가 아니라 텍스처였고(367 MB jpg → RAM 8.5 GB), 텍스처를 버리면 씬 전체가 7~10 MB로 타일 하나보다 작다. 게다가 G4 초판은 타일 박스가 씬 전체를 덮는 자명한 조건이었고, R3 재판은 "worst case"라던 합성 카메라 **30개 중 14개가 벽·가구 속**이었다 | **#01 씬 지오메트리 빌드** + **#02 depth 세 소스 비교** |
| **P-B1 augmenter 검증** (`pb/`) | **모드 (A) 전용**이라 사용자 재정의(카메라 고정)로 폐기된 방식이고, 내용이 **#6 modes의 부분집합**이다(`06b_validate_modes.py`가 같은 헬퍼를 import해 A·B를 함께 그린다). 코드 `06_validate_augment.py`는 헬퍼 제공용으로 유지 | **#7 modes** |
| **pixel-goal 초판** (`pixelgoal/`) | 경로의 좌우 시야 이탈을 막기 전 버전 | **#07** |
| **G2 vln_n1판** | 학습에 쓰는 데이터셋은 vln_ce인데 `03_sample_gt_paths.py`가 vln_n1만 지원해 엉뚱한 데이터셋으로 쟀다(baseline `r_b`도 다르다) | **#2 vln_ce판** |

> 위 Artifact URL들은 **플래너를 고친 뒤(fine 격자 A* + 4중 기각) 전부 재실행·재발행**했다.
> `g1`/`depth_sources`는 정합·렌더만 다뤄 플래너와 무관하므로 재실행 대상이 아니다.

---

## 🧱 #01+#02 — 타일링을 버리고 씬 전체 지오메트리로 (2026-08-20)

`01_build_scene_geo.py` · `02_verify_depth_sources.py` · `logs/embodiment_augment/depth_sources/`

### 왜 버렸나 — 무거운 건 지오메트리가 아니라 텍스처였다

타일링은 "씬 전체를 worker에 올릴 수 없다"는 전제에서 나왔다. 그 전제를 실측했다
(독립 프로세스, `/proc/self/statm` 기준 — `ru_maxrss`는 누적 최대라 델타가 무의미하다):

| 무엇 | 크기 | 로드 | RSS |
|---|---|---|---|
| 씬 폴더 (obj + **텍스처**) | 38 ~ **367 MB** | 1.0 ~ **11.6 s** | 0.7 ~ **8.5 GB** |
| `.obj` 파일만 | median **18 MB** | | |
| **씬 전체 · 지오메트리 전용 ply** | **6.9 ~ 9.8 MB** | **0.09 ~ 0.15 s** | **0.08 ~ 0.10 GB** |
| 타일 12~15개 합 | 76 ~ 102 MB | 0.05 s/개 | 0.05 GB/개 |

**씬 전체(7.4 MB)가 타일 하나(6.0 MB)보다 작다.** 168k 삼각형이 8.5 GB를 먹은 이유는 367 MB의 jpg
텍스처가 압축 해제되기 때문이고, **v1은 depth만 렌더하니 텍스처가 전부 낭비**다.

depth가 실제로 같은지 확인했다 — 파이프라인 유효 범위(5 m) 안에서 **1,628,548 픽셀 중 오차 >1 mm 인
것이 0개**. 전 범위로 넓히면 프레임당 9픽셀(0.0029%)이 다른데 그 픽셀의 depth가 11.9~14.4 m로 clip 밖이다.

### 없어진 것
`margin` 파라미터 튜닝 · 타일 경계 손실 가능성(**원리적으로 불가능**해짐) · 프레임마다 타일 조회·교체 ·
디스크 10배(씬당 76~102 MB → 7~10 MB, 90씬 8 GB → **0.8 GB**로 전부 RAM 캐시 가능).

### 실험결과 — depth 세 소스, 범위별 (단위 mm, median / p90)

| 비교 | ≤5 m | ≤10 m | ≤15 m | ≤20 m |
|---|---|---|---|---|
| ① 저장 ↔ ② 우리(Open3D · obj) | 0.50 / 0.91 | 0.51 / 0.92 | 0.52 / 0.93 | 0.52 / 0.93 |
| ① 저장 ↔ ③ habitat(glb, 지금 렌더) | 0.49 / 0.89 | 0.49 / 0.90 | 0.50 / 0.91 | 0.50 / 0.91 |
| **② 우리 ↔ ③ habitat** | **0.00 / 0.03** | **0.01 / 0.04** | **0.01 / 0.05** | **0.01 / 0.05** |

**우리 렌더와 habitat 렌더가 사실상 동일하다** — 에셋(obj vs glb)도 렌더러도 다른데 지오메트리는 같다.
그리고 **범위를 5→20 m로 넓혀도 median이 커지지 않는다.** 이것이 미뤄뒀던 **G1c**의 답이다.
①↔③이 0.49 mm인 것은 **우리 habitat 설정이 맞다는 검증**이기도 하다(데이터셋이 habitat 산출물이므로).

### 부수 발견 — G1이 보고한 "0.5 mm"는 우리 오차가 아니다

두 렌더 모두 저장 depth와 0.50 mm 차이인데, **부호를 봤다**:

| 지표 | 값 |
|---|---|
| median | **+0.50 mm** (항상 양수 방향) |
| 양수 비율 | **99.3 ~ 100%** |
| 분포 | 0 ~ +1 mm **균일** (p5 +0.05, p95 +0.96) |

반올림이면 ±0.5 mm로 **대칭**이어야 한다. 즉 데이터셋이 depth를 **버림(truncation)** 으로 저장했다
(`uint16(depth*1000)`). mm로 반올림해 비교하면 ~50%가 정확히 0, 나머지가 정확히 1.000 mm가 되는 것도
같은 이야기다. **이전에 이 0.5 mm를 "정합 잔차 + 양자화"로 애매하게 설명했는데 정정한다** —
저장 포맷 탓이고 우리 파이프라인의 오차 하한은 0.01 mm다.

### 한계
- 씬 1채 · 에피소드 1개 · 6프레임. 씬마다 에셋 품질이 다를 수 있다.
- **RGB는 비교하지 않았다.** v1은 depth/BEV만 바꾼다.
- ⚠️ **v2에서 RGB를 렌더하면 텍스처가 다시 필요하고 8.5 GB 문제가 돌아온다** → 타일링 결정을 재검토해야
  한다. 이 결정은 "v1은 depth/BEV만"이라는 전제에 묶여 있다.
- max가 수 m로 튀는 픽셀이 남는다(①↔② 최대 8.4 m). G1이 꼬리로 보고한 **실루엣·거울/창** 픽셀로
  보이나 **원인을 특정하지 않았다**. 저장과의 마스크 불일치 1.8~2.0%도 원인을 나누지 않았다.

---

## 🗺️ #03 · 데이터셋이 쓴 지도에 두 경로를 올린다

`03_verify_navmesh_gt.py` · `logs/embodiment_augment/s1_navmesh/`

### 왜 했나
이 폴더는 **로봇 크기(`r_b`)를 바꿨을 때 정답 경로를 다시 그려주는 코드**다. 그러면 순서가 정해진다 —
**큰 로봇용을 새로 그리기 전에, 원래 크기에서 원래 정답을 그대로 뽑아낼 수 있는지 먼저 확인해야 한다.**
게다가 "우리 지도에서 원본 정답이 벽을 뚫는다"는 증상이 보고돼 있었다. 그래서 **데이터셋이 실제로 쓴
지도**(배포된 `<scan>.navmesh`)에 두 경로를 올려 점 단위로 판정한다.

| | 경로 | 색 | 무엇인가 |
|---|---|---|---|
| ⓐ | **원본 정답** | 노랑 | 저장된 프레임별 위치. 학습이 실제로 먹는 라벨 |
| ⓑ | **waypoint를 이어 만든 경로** | 주황 | leg별 `find_path`. VLN-CE가 데이터셋을 만든 절차 그 자체 |

`r_b = 0.10`은 우리가 고른 값이 아니다 — 논문이 *"1.5 m tall cylinder of diameter of 0.2 m"*로 못박았고
배포 navmesh를 직독하면 정말 그 값이다. **"원본 파라미터를 따른다" = 바꿀 값이 0개**이고 그것이 게이트 A다.

### 실험결과 1 — 게이트 4개 (두 씬 전부 통과)

| 게이트 | 묻는 것 | 17DRP5sb8fy | s8pcmisQ38h |
|---|---|---|---|
| A 파라미터 | 배포 navmesh가 habitat 기본값인가 | 불일치 **없음** | 불일치 **없음** |
| B 벽 뚫기 | 5 cm로 채운 점이 `is_navigable` 통과하나 | ⓐ **100.00%** · ⓑ **100.00%** | ⓐ **99.43%** · ⓑ **100.00%** |
| C leg 도달 | leg `find_path`가 다음 waypoint 0.5 m 안에 닿나 | **69/69** | **47/47** |
| D 캐시 | `recompute_navmesh`가 배포본을 재현하나 | 52.04 vs **52.03** (+0.02%) | 184.01 vs **183.73** (−0.15%) |

C는 **VLN-CE 논문 자신의 navigability 기준**(*"within 0.5 m of the next waypoint"*)이고 100% 재현된다.
직독값: `agent_radius` 0.10 · `agent_height` 1.50 · `agent_max_climb` **0.20** · `cell_size` 0.05 ·
`cell_height` 0.20 · `edge_max_error` 1.3셀 · `filter_low_hanging_obstacles` True.
waypoint 자체도 **474/474 · 531/531** navigable, `snap_point` 이동 **0.0000 m**.

### 실험결과 2 — **"벽 뚫기"는 데이터셋의 지도에서 재현되지 않는다**

원본 정답 경로를 5 cm 간격으로 채워 두 심판에게 물었다. **J1** = `is_navigable`(habitat 직접 답 = 진실),
**J2** = `get_topdown_view(0.05, fz)` 래스터(**우리가 그리고 계획할 때 보는 것** — `navmesh_grid`가
이걸 샘플링한다).

| 씬 | ⓐ 점 수 | J1 밖 (진짜 뚫음) | 그 점의 snap 거리 max | J1통과·J2밖 | 깊이 1칸 | 2칸 | ≥3칸 |
|---|---|---|---|---|---|---|---|
| 17DRP | 2465 | **0** (0.00%) | 0.0000 m | **0**/2465 | 0 | 0 | 0 |
| s8 | 2450 | **14** (0.57%) | **0.0371 m** | **58**/2436 | **58** | **0** | **0** |

**반증 가능한 술어로 냈다**: "J1 통과·J2 밖인 점이 전부 깊이 1칸이면 원인은 격자 반올림이다." 2칸 이상이
하나라도 나오면 반증인데 **나오지 않았다**. J1 밖인 14점도 **최대 3.7 cm = 한 칸(5 cm) 미만**이라
관통이 아니라 경계에 걸친 이산화 오차다.

→ **원 증상("우리 맵에서 GT가 `clearance ≥ r_b` 위반")은 데이터셋의 지도 문제가 아니다.** 그 숫자는
`Z_OFFSET_M = 0.20` 정렬 버그가 있던 때, 그리고 데이터셋이 쓴 적 없는 `r_b = 0.2`에서 나온 것이다.
버그 정정 후 #05의 W5 identity는 69/69 · 47/47 PASS · GT 거리 0.00 cm다.

### 실험결과 3 — ⚠️ **길이비 ≈ 1은 "같은 경로"의 증거가 아니다**

ⓑ vs ⓐ, 호길이 등간격 리샘플 대응점 거리:

| 씬 | 에피소드 | 대응점 거리 median | max | 길이비 (ⓑ/ⓐ) |
|---|---|---|---|---|
| 17DRP | 14 | **12.9 cm** | 112.6 cm | **1.001** |
| s8 | 9 | **56.4 cm** | 195.9 cm | **1.002** |

어긋남에 **두 종류**가 섞여 있고 길이비는 둘을 구분하지 못한다:

1. **코너 절단·궤적 여유** — 원본 GT는 `0.25 m 전진 / 15° 회전` 이산 액션으로 저장돼 넓게 돌고,
   `find_path`는 측지 최단이라 자른다. 17DRP가 이 경우(12.9 cm).
2. **장애물의 반대편으로 갈라짐** — 두 우회로 길이가 비슷하면 측지 최단선이 GT와 **다른 쪽**으로 돈다.
   s8의 큰 값(56.4 cm)이 이 경우고 **길이비는 그래도 1.002**다. 그림에서 노랑과 주황이 어두운 섬을
   위/아래로 갈라져 지나간다(`s8pcmisQ38h_ep008.jpg`).

→ **회랑 중심선은 `find_path`가 아니라 GT 서브궤적이어야 한다.** 반대편으로 갈라지는 경우가 있으니
이건 선택이 아니라 **필수**다(현 `follow_waypoints`가 이미 그렇게 한다 — #03이 그 근거).

### 실험결과 4 — r_b 스윕 + 캐시

`agent_radius`만 바꿔 재계산한 navmesh에서 GT waypoint가 살아남는 비율:

| r_b | 17DRP area | waypoint navigable | s8 area | waypoint navigable |
|---|---|---|---|---|
| **0.10** | 52.03 m² | **474/474 (100%)** | 183.73 m² | **531/531 (100%)** |
| 0.15 | 44.72 | 456/474 (96.2%) | 162.53 | 471/531 (88.7%) |
| 0.20 | 38.29 | 441/474 (93.0%) | 146.99 | 471/531 (88.7%) |
| 0.30 | 26.94 | 426/474 (89.9%) | 115.63 | 420/531 (79.1%) |
| 0.45 | 14.09 | 267/474 (56.3%) | 77.26 | 147/531 (27.7%) |

r_b별 navmesh는 `data/embodiment_aug/navmesh/<scene>_rb{r_b}.navmesh`에 캐시(21 KB/개) —
**#04의 default 맵이자 #05의 habitat 오라클(W6)이 이걸 읽는다**(없으면 assert).

### 이번 판에서 지운 것 — `clearance` 분석 전부

옛 게이트 D·D2, `clearance_html`, `_dense3`, `distance_to_closest_obstacle` 호출.
**navmesh는 이미 `agent_radius` 0.10만큼 깎인 configuration space**라 물어야 할 것은 "칸 안이냐"뿐이고,
거기에 `clearance ≥ r_b`를 또 요구하면 **반경을 두 번 센다**(`navmesh_grid.esdf_from_mask`의 `+ r_b`
트릭이 그 증거). 이 분석에서 나온 헤드라인 결론은 이미 한 번 철회됐다 —
~~"GT는 navmesh 경계를 여유 0으로 스친다"~~는 `clearance min`(최악의 한 점)을 전형값으로 읽은 오류였다.
게이트 8개(A/B/C/D/D2/E/F/G) → **4개**. ⚠️ **줄 수는 줄지 않았다**(447 → 593: docstring 50 ·
리포트 HTML 191 · 분석 코드 352). `clearance`를 지운 만큼 3중 판정(`judge` / `off_depth` / 깊이 술어)과
skip 버킷이 늘었다. 줄어든 것은 **읽어야 할 개념 수**다 — 게이트 절반, 철회된 분석 삭제, 분포 표 하나를
반증 가능한 술어로 대체.

### 알아둘 것
- `T_sf2mesh`는 **전부 해석적**이라(`vlnce_align.fit_sf2mesh`) 이 검증에 씬 mesh·Open3D가
  전혀 필요 없다. **우리 플래너와 우리 occ 맵은 이 리포트 범위 밖**이다(#04 / #05).
- `get_topdown_view(mpp, h)` 규약 **실측 확정**: `tv[iz, ix]`, `ix=(x−lo[0])/mpp`, `iz=(z−lo[2])/mpp`
  (`lo=get_bounds()[0]`). **단일 높이 절단**이라 다층 씬은 에피소드별 floor로 잘라야 하고
  **계단 에피소드는 제외**한다(`--max_z_spread 0.30`, s8은 14개 중 5개 제외).
- `is_navigable`은 **수평 1 cm · 수직 0.5 m 맹점**이 있다. 게이트 B는 "지도를 벗어나지 않는다"가 아니라
  "평면상 navmesh 폴리곤 안, ±1 cm"다.
- 게이트 D는 자기가 **쓰는** r_b=0.10 지도만 검증한다. r_b>0.10 지도가 옳은지는 #05의 W6이 본다.
- `chain_find_path`에 `per_leg=True`를 추가했다 — 기본값(`False`)은 실패한 leg를 버리고 이어붙여서
  **떨어진 두 구간 사이에 벽을 통과하는 가짜 직선**을 만든다. 점 단위 판정에는 반드시 `per_leg=True`.

---

## 🧭 #04 · 우리 지도가 habitat 지도와 같은 판정을 내리나

`04_calibrate_map.py` · `logs/embodiment_augment/s2_mapcal/`

### 왜 했나
학습 중에 정답을 즉석에서 다시 그리려면 **habitat 없이 도는 지도**가 필요하다(Isaac/VLN-PE도 같은 지도를
쓴다). #03이 데이터셋의 지도를 기준으로 세웠으니, 우리 지도가 그 기준과 **같은 판정을 내리는지**, 그리고
**그 지도 위에서 정답 경로가 성립하는지** 본다.

**밴드 높이 8조합 스윕을 없앴다.** 조합 간 일치율 차이가 0.08~0.23%p로 판별력이 없음이 이미 증명됐다.
값을 고르는 대신 **배포 navmesh에서 직접 읽는다** — `h_nav` 0.20(`agent_max_climb`) ·
`h_obs` 1.50(`agent_height`) · `cell` 0.05 · `r_b` 0.10. **자유 파라미터 0개.**

두 경로를 올린다: **ⓐ 원본 정답**(노랑) · **ⓑ waypoint를 habitat pathfollower로 이은 경로**(초록,
0.25 m 전진 / 15° 회전). ⓑ가 새 요소다 — VLN-CE가 저장한 것이 *"shortest path following the waypoints
via low-level actions"*이므로 **pathfollower가 곧 데이터셋 생성기 그 자체**다.
`GreedyGeodesicFollower`를 **헤드리스로** 돌린다(Simulator·GPU·glb 불필요, `.navmesh` 파일만).

### 실험결과 1 — 게이트 (2개 통과 · 2개 실패)

| 게이트 | 묻는 것 | 17DRP5sb8fy | s8pcmisQ38h |
|---|---|---|---|
| A 좌표 정합 | 우리 격자와 habitat 좌표가 맞물리나 (≥0.98) | **1.0000** ✅ | **1.0000** ✅ |
| B follower 도달 | leg 끝점이 ≤0.5 m에 닿나 (100%) | **69/69** ✅ | **47/47** ✅ |
| C 불일치 분해 | F+R이 거짓 승인의 ≥90%를 설명하나 | 86.1% ❌ | 84.4% ❌ |
| D 수정 채택 | 부작용 없이 채택 가능한가 | ✅ (기각 179→179) | ❌ (기각 164→**1437**) |

**게이트 기준을 데이터 보기 전에 정했고, 실패를 통과로 만들지 않았다.**

### 실험결과 2 — **왜 다른가를 처음으로 분해했다**

이전 판의 한계 항목이 스스로 "**왜** 다른지를 분해하지 않았다"고 적어 두었던 부분이다.

**가설**: `derive_obstacle_2d`는 "바닥 위 밴드에 장애물이 있나"만 묻고 **"이 높이에 바닥이 있나"를 묻지
않는다.** 그래서 다층 집에서 **다른 층에 속한 칸**이 통행 가능으로 새어 들어온다.
`coverage`(열 전체에 점유가 하나라도 있나)는 위층 슬래브 때문에 이걸 못 걸러낸다.

| 씬 | 거짓 승인 | F 바닥 없음 | R 경계 1칸 | S 나머지 | F+R 설명률 | F 비중 |
|---|---|---|---|---|---|---|
| 17DRP | 9,137 | 546 | **7,319** | 1,272 | 86.1% | **6.0%** |
| s8 | 44,219 | **29,452** | 7,848 | 6,919 | 84.4% | **66.6%** |

**예측대로 갈렸다** — s8(3층)은 F가 지배하고, 17DRP(사실상 1층)는 R(경계 1칸)이 80%다.
버킷은 배타적이고(F → R → S 순) 합이 전체다. 진단 두께는 `agent_max_climb` 0.20으로 **고정**했다 —
수정 두께와 묶으면 튜닝할 때 설명률이 따라 움직여 지표가 못 된다.

⚠️ **"수정하면 거짓 승인이 정확히 F만큼 준다"는 항등식이라 게이트가 못 된다** — F가 바로 그 집합이므로.
정보가 있는 것은 **F의 비중**(가설의 설명력)과 **부작용의 크기**(게이트 D)다.
#03에서도 같은 함정에 빠졌었다(`none` rung이 GT를 복사하므로 "일치"가 항상 성립).

### 실험결과 3 — 수정은 크게 도움이 되지만 **채택할 수 없다**

| 지도 | 씬 | habitat 일치율 | IoU | 거짓 승인 | 거짓 기각 |
|---|---|---|---|---|---|
| 밴드 투영 | 17DRP | 95.56% | 0.9232 | 9,137 | 179 |
| **+ `floor_exists`** | 17DRP | 95.82% | 0.9273 | 8,591 | **179** |
| 밴드 투영 | s8 | 75.83% | 0.6456 | 44,219 | 164 |
| **+ `floor_exists`** | s8 | **90.69%** | **0.8222** | 15,474 | **1,437** |

s8 일치율 **+14.9%p**인데 거짓 기각이 **9배**로 늘었다 — 거짓 승인을 줄이려고 거짓 기각을 만드는 거래다.
**거짓 승인은 못 갈 길을 정답으로 가르치는 것이라 되돌릴 수 없고, 거짓 기각은 샘플을 버리는 것뿐**이지만
9배는 과하다.

두께 스윕(`--floor_belows 0.05,0.10,0.15,0.25,0.40`): 두꺼울수록 바닥을 잘 찾아 거짓 기각이 줄지만
다른 층도 통과시켜 거짓 승인이 다시 늘어난다. **두 열이 반대로 움직여 이 파라미터 하나로는 둘을 동시에
만족시킬 수 없다.**

| `floor_below` | s8 일치율 | 거짓 승인 | 거짓 기각 |
|---|---|---|---|
| 0.05 | 55.26% | 1,408 | 79,535 |
| 0.10 | 60.36% | 5,215 | 64,636 |
| 0.15 | 87.15% | 14,302 | 7,329 |
| **0.25** ← 채택 | **90.69%** | 15,474 | 1,437 |
| 0.40 | 81.51% | 33,224 | 594 |

**0.25는 끼워맞춘 값이 아니다** — `agent_max_climb` 0.20 + 격자 1셀이고, 스윕에서도 최적이다.
17DRP는 0.15 이상에서 전부 동일하다(바닥이 바로 거기 있다) → **이 축은 다층 씬에서만 판별력이 있다.**

### 실험결과 4 — pathfollower는 **코너는 고치지만 갈림은 못 고친다**

| 씬 | 에피소드 | median | max | 길이비 | ≤20 cm | >20 cm | #03 `find_path` |
|---|---|---|---|---|---|---|---|
| 17DRP | 14 | **10.7 cm** | 61.7 cm | 0.974 | 13개 (10.6 cm) | 1개 (27.2 cm) | 12.9 cm |
| s8 | 9 | **8.8 cm** | 199.0 cm | 0.973 | **5개 (5.4 cm)** | **4개 (72.8 cm)** | 56.4 cm |

s8 median이 **56.4 → 8.8 cm**로 6.4배 좋아졌다 — 이산 액션이 코너 처리를 확실히 개선한다.
⚠️ **그런데 median만 보면 안 된다 — 분포가 두 덩어리다.** s8 9개 중 4개는 여전히 50~82 cm이고, 원인은
**경로가 장애물의 반대편으로 갈라지는 것**이다(#03에서 `find_path`에 대해 발견한 것과 **같은 현상**).
follower는 측지 최단선을 따라가므로 **두 우회로 비용이 비슷할 때의 갈림을 그대로 물려받는다.**
→ 회랑 중심선은 여전히 **GT 서브궤적이어야 한다**(`follow_waypoints`의 현재 선택이 옳다).

### 실험결과 5 — **결과가 예상과 반대다**

| 경로 | 씬 | 우리 지도 밖 | 깊이 max | navmesh 래스터 밖 | 깊이 max |
|---|---|---|---|---|---|
| ⓐ GT | 17DRP | **0**/9,179 | 0.000 m | 0/9,179 | 0.000 m |
| ⓐ GT | s8 | **0**/8,459 | 0.000 m | **66**/8,459 | 0.050 m |
| ⓑ follower | 17DRP | 12/8,915 | 0.050 m | 82/8,915 | 0.050 m |
| ⓑ follower | s8 | 11/8,178 | 0.050 m | **221**/8,178 | 0.100 m |

**GT가 우리 지도를 벗어난 점은 두 씬 모두 0**이다. 오히려 navmesh를 래스터화한 지도가 더 많이 자른다 —
우리 밴드 맵이 더 관대하기 때문이고, 잘리는 깊이가 1~2셀이라 원인은 #03이 술어로 확인한
**격자 반올림 + 단일 높이 절단**이다.
→ **"정답이 우리 지도를 뚫는다"는 원 증상은 #04에서도 재현되지 않았다.**

### 실험결과 6 — 질문 자체를 바꿔봤다: 칸마다 바닥을 찾는 recast-like (`recast_like.py`)

recast의 span 논리를 occ 격자 위에 흉내 냈다. 칸마다 ① `[floor_y−0.5, +0.2]` 창에서 실제 바닥 voxel
(없으면 차단) ② 머리 공간 1.5 m를 **자기 바닥에 앵커** ③ 이웃 단차 >0.2 m면 절벽 경계(ledge)
④ 장애물에서만 침식 ⑤ 1 m² 미만 섬 삭제. habitat navmesh 표면 y 실측이 근거다 — 한 층 안에서도
2.13~3.95 m로 울퉁불퉁하고 계단 위에도 navmesh가 있다(칸마다 다른 바닥을 max_climb으로 연결).

**v1은 다층 씬에서 실패했다.** GT 차단의 **100%가 ledge**(1,183점 — 스캔 구멍의 kf가 −0.36 m 아래
면으로 새어 가짜 seam이 방을 가로지름), F 버킷의 98%가 **진짜 navmesh가 걸을 수 없다고 판정한
"바닥처럼 생긴 면"**. 이 진단에서 v2의 수정 두 개가 나왔다:

- **despike**: kf 스파이크(바닥 voxel 빠짐)를 이웃 중앙값으로 보간 → s8 거짓 기각 11,233→**895** ·
  GT 차단 1,183→**0**
- **시작점 연결성**: 시작점이 속한 연결 성분만(recast 도달 가능성 근사) → "바닥처럼 생긴 면" 탈락,
  s8 F 14,559→**745** · 일치율 84.64→**92.30%**

| 씬 | 밴드 | +floor_exists | recast v1 | **recast v2** | v2 게이트 |
|---|---|---|---|---|---|
| 17DRP | 95.56% | 95.82% | 96.65% | **96.23%** · GT 0 · fol 58 @ ≤1칸 | **전부 통과** |
| s8 | 75.83% | 90.69% | 81.40% (GT 1,183 차단) | **92.30%** · GT **0** · fol 15 @ ≤1칸 · FR 895 | G2·G3 통과 · **G1 실패**(F 745 > 200) |

**v2가 밴드 계열 후보 중 최선이다** — s8에서 모든 후보 중 최고이고 GT·follower가 안전하다.
G1은 잔여 F 745로 실패인데 기준을 낮춰 통과시키지 않는다(29,452→745 = 97.5% 감소가 사실).

### 결정
**default 맵은 navmesh 래스터화로 그대로 둔다** — 정의상 100%가 캐시로 이미 있는데 92%를 쓸 이유가
없다. `& floor_exists`(게이트 D 실패)는 폐기 수준, **recast-like v2는 navmesh가 없는 씬**(Isaac 신규 씬
등)에서 habitat 없이 만들 수 있는 지도 중 **최선의 후보**로 남긴다(GT·follower 안전 확인됨).
남은 개선 축은 `.house` region별 바닥높이(잔여 F 745와 R의 일부)다.

### 줄 수는 줄지 않았다 (492 → 783)
docstring 58 · 리포트 HTML 241 · 분석 코드 484. 스윕을 지운 만큼 **분해·follower·두께 스윕 표**가 늘었다.
같은 스타일의 `00`은 217줄, `02`는 264줄이라 04가 여전히 가장 크다.
**줄어든 것은 읽어야 할 개념 수다** — 지도 후보 9→1개, 게이트가 실제로 무엇을 묻는지 명시,
항등식이던 게이트 제거.

### 관련 종합 리포트
지도 3종(navmesh / 밴드 / recast v2)의 정확도·속도·제약·시나리오를 한 페이지로 비교한 리포트:
<https://claude.ai/code/artifact/aea56605-e2ef-4868-af5b-69f4939a988e> (기록: `.claude/memory/260821_map_comparison_report.md`)

### ⚠️ 이전 판과 숫자를 직접 비교하면 안 된다 — **지표를 바꿨다**
이전 판의 "s8 88.7%"는 **GT 궤적 점**에 대한 판정 일치율을 r_b 5값에 걸쳐 평균한 값이다. GT 점은 두 지도
모두에서 거의 항상 통행 가능이므로 **그 지표는 쉬운 문제만 채점한다**(그래서 높게 나온다).
새 판의 "s8 75.83%"는 **GT 주변 2 m 안의 모든 셀**에 대한 셀 단위 일치율이다 — 훨씬 엄격하고,
거짓 승인이 어디서 얼마나 생기는지를 실제로 반영한다.
**수치가 낮아진 것은 지도가 나빠진 게 아니라 채점이 정직해진 것이다.**

### 알아둘 것
- **`move_filter_fn`을 반드시 넣어야 한다.** `agent.controls.move_filter_fn = pf.try_step`이 없으면
  헤드리스 롤아웃이 **벽을 통과한다**(실측 5점). `Simulator`가 있을 때만 habitat이 자동으로 채워준다.
- `habitat_sim` 기본 action_space는 0.25 m / **10°**다 — VLN-CE의 15°로 덮어써야 한다.
- habitat-lab의 `ShortestPathFollower`는 `HabitatSim` 래퍼를 요구해 헤드리스로 못 쓴다. 그 내부도 결국
  `GreedyGeodesicFollower`를 부르므로 후자를 직접 쓴다.
- follower `goal_radius`를 0.25 m로 두었으나 **데이터셋 생성 시 쓴 값은 알 수 없다**(스윕 안 함).
- **판정 기준이 여전히 habitat이다.** 목표는 habitat 없이 도는 지도인데 정확도를 habitat으로 재고 있어
  **상한이 habitat**이다.
- diff 그림은 **GT 주변 2 m 밖을 어둡게** 깐다 — 표의 숫자가 그 범위에서만 세어지므로 전체를 똑같이
  칠하면 그림과 숫자가 어긋나 보인다(처음에 이 실수를 했다).

---

## 🛣️ #05b · G2 — 다운스트림을 바꾼 두 가지 실측

**1. 구분해야 하는 두 개의 `r_b`.**

| | 값 | 출처 | 성격 |
|---|---|---|---|
| vln_ce GT의 **명목** 반경 | **0.1 m** | 배포된 `<scan>.navmesh` **직독** | 데이터셋 속성 — **측정 불필요** |
| 우리 플래너의 **캘리브레이션 `r_b*`** | **0.20 m** (잠정) | 원본 GT가 우리 occupancy 맵에서 확보한 ESDF | 우리 쪽 값 — **측정 외 방법 없음** |

명목값은 데이터에서 바로 읽힌다: `PathFinder().load_nav_mesh('<scan>.navmesh').nav_mesh_settings`
→ `agent_radius=0.1, cell_size=0.05, agent_height=1.5, max_climb=0.2, max_slope=45`.
`radius`의 뜻: recast는 로봇을 **수직 원기둥**(반지름 r, 높이 h)으로 보고, 원기둥 중심축을 놓을 수 있는
영역만 남기려고 걷기 가능 표면을 `ceil(r/cell_size)`셀 깎는다(0.1/0.05 = 정확히 2셀).
⚠️ **GT의 반경 근거는 `.navmesh`이지 config가 아니다.** `reference_path`가 그 navmesh 위를 걸어 생성됐고,
데이터셋 json에는 radius 필드가 아예 없다(키: `episode_id, trajectory_id, scene_id, start_position,
start_rotation, info, goals, instruction, reference_path`). 이전 판에서 `vln_r2r_mini.yaml`을 근거로 든 것은
잘못이었다 — 그 파일은 **eval용**이고 main에도 없다(이 브랜치 `f2e551f`에서 추가).

**⚠️ config의 `radius`는 무해하지 않다 — habitat-sim이 navmesh를 자동 재계산한다.**
`habitat-sim/src/esp/sim/Simulator.cpp:215-224`가 로드된 navmesh 설정과 요청 설정이 **다르면
`recomputeNavMesh`를 호출**한다. `habitat_simulator.py:361`이 `navmesh_settings.agent_radius =
agent_config.radius`를 넣고 `default_agent_navmesh` 기본값은 **True**다.
r2r에서 배포본이 그대로 쓰이는 이유는 "재계산 코드가 없어서"가 아니라 **값이 정확히 같아서**다 —
`AgentConfig` 기본값(radius 0.1 / height 1.5 / climb 0.2 / slope 45)이 배포 navmesh와 완전 일치한다.
반대로 main의 `objectnav_hm3d.yaml`(radius **0.18**, height **1.25**)은 **런타임에 navmesh를 다시 굽는다**.
→ **P-C1 eval에서 `r_b`를 바꾸려면 yaml에 `radius:` 한 줄만 쓰면 된다**(새 코드 불필요). 단 `reference_path`는
radius 0.1로 만든 것이라 SPL/NE가 새 로봇이 못 따라갈 경로를 기준으로 삼는 문제는 남는다(계획서 한계 항목).

`r_b*`가 명목 0.1보다 큰 이유는 두 요인이 섞여 있다 — (a) recast 자체의 추가 보수성
(`edge_max_error`=1.3 voxel=6.5 cm 폴리곤 단순화, `region_merge_size`=20), (b) **우리 맵이 recast와 다르게
장애물을 만든다**(height slab 투영 vs recast의 span walkability). 즉 `r_b*`는 vln_ce의 속성이 아니라
**우리 파이프라인의 캘리브레이션 상수**다.
→ 게이트 판정과 **P-B2의 augment 범위 하단**은 `r_b*`로 잡아야 baseline 샘플이 원본 GT와 일치한다.
⚠️ 현재 `r_b*`=0.20은 **씬 1개·9 에피소드**만의 잠정치(`Z_OFFSET_M` 정정 후 재측정. 정정 전 0.18) → 논문/구현 확정 전 전 씬 재측정 필요
(렌더링 불필요, occupancy+ESDF+정합만).

<details>
<summary><b>troubleshooting — 틀렸던 측정과 버그 이력 (펼치기)</b></summary>

**2. 생성 경로가 장애물을 통과하고 있었다 (거짓 라벨) — 원인 제거 + 3중 기각.**

r_b=0.1에서 재계획 경로 **4/7이 장애물 voxel 내부를 통과**했다. **원본 GT는 0/7**이라 데이터가 아니라
**우리 플래너 문제**였다.

**원인 특정** — `check_path_navigable`의 `points_min`(웨이포인트)과 `segments_min`(점 사이 선분)을 나눠 봤다:
`A* 웨이포인트 자체가 장애물 안`(`points_min = 0.000`)이었다. 선분이 코너를 자른 것도, 스플라인도 아니다.
기본값 `downsample_factor=4, mode='any'`가 20 cm coarse 셀을 "16개 fine 셀 중 **하나만** navigable이면
통과 가능"으로 보고, 웨이포인트를 그 **셀 중심**에 놓기 때문이다.

⚠️ **작은 r_b에서 더 심한 것은 로봇이 작아서가 아니다** (직관과 반대로 보이는 지점). 하드 충돌은
`clearance == 0`, 즉 **로봇 크기와 무관한 절대 조건**(벽 관통)이다. r_b가 작으면
`truncate_navigable(esdf, r_b)`가 벽에 붙어 커지므로 **거의 벽인 coarse 셀도 자격을 얻고**, 그 셀 중심이
벽 안이다. r_b가 크면 마스크가 벽에서 물러나 그런 셀이 애초에 안 생긴다. 즉 실패는 **coarse 격자
집계 방식**의 문제다.

**설정 비교** (s8, 7 에피소드):

| 설정 | r_b=0.10 | r_b=0.18 | 시간 |
|---|---|---|---|
| factor=4 `any` (기존 기본값) | 7/7 · **하드 4** · 몸통 5 | 7/7 · 하드 0 · 몸통 3 | 9 ms |
| factor=4 `majority` | 7/7 · 하드 1 · 몸통 1 | 3/7 · 0 · 0 | 10 ms |
| factor=4 `all` | 3/7(좁은 문 막힘) · 0 · 0 | 3/7 · 0 · 0 | 8 ms |
| **factor=1 (채택)** | **7/7 · 하드 0** · 몸통 1 | 4/7 · **0 · 0** | 17–24 ms |

**조치 1 — 원인 제거**: A*를 **fine 격자**에서 돈다(`downsample_factor=1`). +10 ms는 sample 예산 393 ms
(Open3D 렌더 350 ms 지배, #08 R2)에서 무시할 수준이다.

**조치 2 — 생성 경로 라벨 유효성 4중 기각** (`EmbodimentAugmenter._path_ok`, `replan`·`replan_in_fov` 공통).
**원본 GT에는 적용하지 않는다** — 생성 경로 전용이다:

| 조건 | 왜 |
|---|---|
| 하드 충돌 `clearance == 0` | 장애물 voxel 내부. 질점 로봇도 통과 불가 |
| 몸통 침범 `clearance < r_b` | 로봇 몸통이 장애물과 겹침 |
| **미관측(unknown) 통과** | 스캔 안 된 곳은 ESDF가 크게 나와 **장애물 검사로 안 잡힌다** → `compute_scan_coverage_mask`로 따로 막음. 실측 0/7이지만 검사가 없으면 보장이 없다 |
| **target 도달** | 끝점이 요청된 goal에서 `GOAL_TOL_M`=10 cm 안. 도착지가 밀리면 instruction이 거짓 라벨이 된다. 끄는 우회로(`goal_tol_m=None`)를 없앴다 |

몸통 침범을 **원본 GT 판정에는 쓰지 않는 이유**: 원본 GT가 우리 맵에서 이걸 위반한다(r_b=0.2에서 4/7).
생성 경로에 대해서는 "A*가 fine 격자에서 이미 보장한 것을 refine·스무딩이 깨지 않았는지"를 확인하는
**자기 정합성 검사**라 정당하다.
⚠️ **정정(2026-08-19)**: 그 이유를 "우리 맵이 recast보다 보수적이라서"라고 적었는데 **방향이 반대**였다
(→ #04). ⚠️ **재정정(2026-08-20)**: ~~"GT가 navmesh 경계를 여유 0으로 스치기 때문"~~도 철회됐다 —
#03 재작성 결과 GT는 배포 navmesh에서 벽을 뚫지 않고(위반 0 / 0.57%, 깊이 최대 3.7 cm < 한 칸),
원 증상은 `Z_OFFSET_M = 0.20` 정렬 버그 + 데이터셋이 쓴 적 없는 `r_b = 0.2`에서 나온 것이었다.
이제 default 맵이 navmesh 래스터화라 이 항목 자체가 해소됐다 —
`none` rung이 GT를 그대로 통과시키고(#05 W5), 마스크 밖이면 정직하게 `blocked`가 된다.

**target 도달에서 추가로 고친 것**: `replan_in_fov`는 도달 여부를 **스냅된** goal(`sg[1]`)과 비교했다.
`snap_to_grid`는 dilation으로 지워진 goal을 최대 **1.5 m**까지 옮기므로, 스냅된 goal과 비교하면 목표가
밀려도 통과한다. 요청된 `goal_xy` 기준으로 바꿨다(실측으로는 발동한 적 없음 — V3 도달 최대 2.6 cm — 이지만
보장이 없었다).

**self-check**: `/usr/bin/python scripts/dataset_converters/3dloader_vlnce/embodiment_augment.py` →
합성 격자(폭 40 cm 문 + 미관측 영역)로 4개 조건이 각각 발동하는지 6/6 확인. 씬·렌더러 불필요.

**왜 안 잡혔나**: `plan_episode`가 `check_trajectory`를 이미 계산하고 문서에 "0이 아니면 무조건 실패로
봐야 한다"고 써놨는데 `replan()`이 읽지 않았고, `replan_in_fov()`는 `check_path_navigable`을
**import만 하고 호출하지 않았다**.

**영향 범위**: 학습되는 양(pixel-goal 국소 ≤3.25 m 재계획)은 **결과 완전 불변** — 4중 기각을 다 켠 뒤
재검증해도 V0 0.00 px / V1 2.25 px / V2 단조 / V3 2.6 cm, status 패턴 동일. 결함은 두 함수에 다 있었지만
**전체 에피소드 재계획 + 작은 r_b**에서만 발현했다. 즉 모든 가드가 학습 경로에서는 비용 0이다.

⚠️ **자체 정정**: 가드 직후 "계획 성공률이 r_b*에서 정점을 이룬다"고 보고했는데 **그것도 coarse
다운샘플 artifact**였다. factor=1에서는 성공률이 r_b에 **단조 감소**한다(0.1→6/7, 0.18→4/7, 0.25→4/7,
0.35→0/7) — 물리적으로 그게 맞다. 게이트 판정은 "공통 집합에서 r_b*가 명목 0.1보다 원본 GT에
가까운가"로 바꿨다(단일 argmin은 n이 작아 신뢰하지 않는다).

**3. 계단·층간 에피소드는 제외해야 한다.** `floor_z = median(cam_z) − rig_height`는 카메라 높이가
1.8~2.1 m 변하는 에피소드(s8 ep4/5/6)에서 깨지고, 그 결과 원본 GT가 우리 2D 맵에서 **clearance 0
(= 벽 속)** 으로 나온다. slab projection의 알려진 한계이므로 `--max_z_spread 0.30`으로 걸러낸다.
같은 실패 모드를 pixel-goal의 `goal_world_from_poses`에서도 겪었고(거기선 프레임별 floor를 계산해 해결).

**4. G2가 학습되는 양을 재지 않는다는 점** — 학습 샘플은 `internvla_n1_lerobot_dataset.py:942-956`에서
`poses[start_frame_id : start_frame_id + goal_len + 1]`, 즉 **현재 프레임에서 pixel goal까지의 국소
구간(`goal_len`≤13 스텝 ≈ ≤3.25 m)**이다. 에피소드 전체 경로는 어떤 학습 타깃도 아니다. G2가 전체
에피소드를 재계획하는 것은 **전제 확인**(그 길이여야 homotopy 변화가 관측될 여지가 있다) 목적이고,
학습되는 양 자체는 pixel-goal 리포트 **#8/#9**가 측정한다.

**5. 측정 위생 3건** — (a) `r_b`마다 계획 성공 집합이 달라서 "성공한 것만 평균"을 비교하면 **생존 편향**
(큰 `r_b`에서 살아남는 건 원래 쉬운 에피소드)이 생긴다 → 공통 집합 평균 + 에피소드별 argmin 집계로 판정.
(b) 정합 residual > 0.02 m 에피소드 제외(원본 GT를 mesh에 잘못 놓으면 비교 자체가 무의미).
(c) **시야(FOV) 제약을 걸지 않는다** — 원본 GT가 12프레임 동안 회전하며 이동하므로 한 프레임의 시야 콘에
갇힐 이유가 없다. 이걸 걸었을 때 계획 성공이 2/9로 떨어졌던 것은 방법론 오류였다.
(FOV 제약은 카메라를 고정하는 pixel-goal 국소 재계획 #8/#9의 조건이다.)


</details>

---

## 📍 #05 · W — GT를 기본으로 따라가고, 못 지나갈 때만 보정 (현재 방식)

`05_validate_waypoints.py` · default 맵 = **VLN-CE navmesh 래스터화**(#04 S2 결론)

ladder 기본값 **`('none', 'nudge', 'corridor')`** — 자유도 낮은 쪽부터 첫 성공에서 멈춘다.
밑단 `none`이 **GT를 그대로** 쓰므로(refine·thin·spline 전부 생략) 통과 가능한 leg는 이탈이 **정의상 0**이다.

### 실험결과 1 — r_b별 leg 결과 (두 씬)

**17DRP5sb8fy** (14 에피소드 / 69 leg)

| r_b | 성공 mode<br>**none** / nudge / corridor | **보정 필요 leg** | ok | blocked | GT와 거리 | 쓸 수 있는 프레임 | W6 habitat 대조<br>둘다통과 / **거짓승인** / 거짓기각 / 둘다막힘 |
|---|---|---|---|---|---|---|---|
| **0.1** (VLN-CE 정의값) | **69** / 0 / 0 | **0/69 (0%)** | 69 | 0 | **0.0 cm** | **515/515 (100%)** | 69 / **0** / 0 / 0 |
| 0.2 | 55 / **2** / 0 | 2/60 (3%) | 57 | 3 | 0.3 cm | 404/515 (78%) | 57 / **0** / 12 / 0 |
| 0.3 | 40 / 0 / **5** | 5/52 (10%) | 45 | 7 | 0.8 cm | 320/515 (62%) | 45 / **0** / 19 / 5 |
| 0.45 | 15 / 0 / 0 | 0/28 (0%) | 15 | 13 | 0.0 cm | 86/515 (17%) | 15 / **0** / 27 / 27 |

**s8pcmisQ38h** (9 에피소드 / 42 leg)

| r_b | 성공 mode<br>**none** / nudge / corridor | **보정 필요 leg** | ok | blocked | GT와 거리 | 쓸 수 있는 프레임 | W6 habitat 대조 |
|---|---|---|---|---|---|---|---|
| **0.1** (VLN-CE 정의값) | 37 / **2** / 0 | 2/42 (5%) | 39 | 3 | 0.6 cm | 282/360 (78%) | 39 / **0** / 8 / 0 |
| 0.2 | 23 / 0 / **3** | 3/33 (9%) | 26 | 7 | 3.3 cm | 177/360 (49%) | 26 / **0** / 16 / 5 |
| 0.3 | 5 / 0 / **3** | 3/17 (18%) | 8 | 9 | 10.3 cm | 62/360 (17%) | 8 / **0** / 29 / 10 |
| 0.45 | 0 / 0 / 0 | 0/9 (0%) | 0 | 9 | — | 0/360 (0%) | 0 / **0** / 5 / 42 |

네 가지를 읽어야 한다:

1. **"보정 필요 leg"가 이 리포트의 핵심 숫자다** — "몇 개 leg가 실제로 GT를 손대야 했나". 17DRP는
   baseline에서 **0/69**, GT 거리 **0.0 cm**. 요청 사항("통과 가능한데 일부러 밀어낼 필요 없다")이
   수치로 성립한다. **이전 판에는 이 열이 없어 `nudge 47/47`을 "47개를 밀어냈다"로 오독했다**
   (실제 이동량은 median 0.000 m였다).
2. **보정은 정말 필요할 때만 걸린다** — 17DRP r_b=0.2에서 `nudge` 2개, r_b=0.3에서 `corridor` 5개.
   `corridor` 열이 0이 아니면 그 leg는 점별 밀어내기로는 실패했고 **회랑 A*로 재라우팅해서** 통과했다.
3. **W6: 거짓 승인이 전 r_b에서 0** (17DRP 276 leg, s8 188 leg 대조). habitat이 막았다는 leg를 우리가
   승인한 적이 없다 — "갈 수 없는 길"을 정답으로 가르치지 않는다는 **독립 검증**이다.
4. **거짓 기각은 있다**(17DRP 최대 27, s8 29) = 우리가 더 엄격하다. 두 원인이 있고 둘 다 의도적이다:
   (a) leg 시작점은 `GOAL_TOL_M`(0.10 m) 이상 옮길 수 없다(아래 troubleshooting), (b) 우리는 첫 blocked에서
   **경로를 끊는데** habitat은 leg를 독립으로 판정한다. 안전한 방향이라 수치로만 보고한다.

### 실험결과 2 — **두 지도 옵션 비교** (`--map_source`)

지도를 두 방식으로 만들 수 있고 **config로 고른다**(`map_source`). 같은 게이트로 나란히 재봤다.

| 옵션 | 만드는 법 | 장점 | 단점 |
|---|---|---|---|
| **`occ`** ← **기본값** | 3D 스캔을 `h_nav=0.20`~`h_obs=1.50` m 밴드로 투영. 두 값은 habitat의 `agent_max_climb` / `agent_height` **직독값** | habitat 비의존(Isaac에도 그대로) · 원본 GT를 안 건드림 · 샘플을 덜 버림 | habitat과 **75.8~95.6%**만 일치(#04, GT 주변 2 m 셀 단위) → **거짓 승인이 생긴다**. 원인의 66.6%가 다층 씬의 "바닥 없음" |
| `navmesh` | 캐시된 r_b별 VLN-CE navmesh를 우리 격자에 래스터화 | habitat 판정과 **100% 일치** → 거짓 승인 0 | habitat에 묶임 · 래스터 1셀 오차로 s8에서 GT 5개 구간을 건드림 |

| 씬 | 옵션 | W5 identity (@r_b=0.1) | **거짓 승인** | 거짓 기각 | Artifact |
|---|---|---|---|---|---|
| 17DRP | `occ` | **PASS** 69/69 · **0.00 cm** | **2** (r_b=0.30) | 47 | https://claude.ai/code/artifact/003a26b8-2335-4cdb-bde5-bc0ce6d3c057 |
| 17DRP | `navmesh` | **PASS** 69/69 · **0.00 cm** | **0** | 58 | https://claude.ai/code/artifact/2948897e-949b-48c6-a302-c0cb36d66026 |
| s8 | `occ` | **PASS** 47/47 · **0.00 cm** | **0** | 45 | https://claude.ai/code/artifact/e6fd96c9-f60a-446d-b0a8-1f17db0ee1e6 |
| s8 | `navmesh` | CHECK 37/42 · 0.58 cm | **0** | 58 | https://claude.ai/code/artifact/a2f7fbc8-d644-4855-9576-6c2b16fe44fa |

**읽는 법**: `occ`는 **두 씬 모두 원본 GT를 전혀 건드리지 않고**(identity PASS, GT 거리 0.00 cm) 버리는
샘플도 적다(거짓 기각 45~47 vs 58). 대가는 17DRP r_b=0.30의 **거짓 승인 2건**(276 leg 중) —
habitat이 막았다는 구간을 우리가 승인한 것이다. `navmesh`는 그 2건을 없애지만 s8에서 GT 5개 구간을
건드리고(1셀 오차) 샘플을 더 버린다.

**기본값을 `occ`로 정한 이유**(사용자 결정 2026-08-20): 8조합의 habitat 일치율 차이가 **0.08~0.23%p**뿐이라
실측 최적점(0.10/1.25)을 고를 근거가 약하고, 그렇다면 **원리 있는 값**(habitat 설정 직독값 0.20/1.50)을
쓰는 게 낫다. 거짓 승인 2건은 W6이 잡아내므로 감시 가능하다. `navmesh`는 옵션으로 남긴다 —
"거짓 승인 0"이 반드시 필요한 실험에서 쓴다.

### 실험결과 3 — W5 identity (baseline에서 GT를 건드리지 않나)

| 씬 | `none` | 보정한 leg | GT와 거리 | 판정 | 실패 사유 |
|---|---|---|---|---|---|
| 17DRP5sb8fy | **69/69** | **0** | **0.00 cm** | **PASS** | — |
| s8pcmisQ38h | 37/42 | 2 | 0.58 cm | CHECK | `hard` → blocked ×3 · `hard` → nudge ×2 |

s8의 실패 5건은 **전부 `hard`** = GT가 navmesh 마스크 밖이다. S2가 이미 정량화한 것과 정확히 같다 —
**66/8459점(0.78%)이 마스크를 벗어나고 최대 깊이가 정확히 1셀(0.050 m)**. 원인은 **5 cm 격자 반올림**이다
(#03이 술어로 확인: J1 통과·J2 밖인 58점이 **전부 깊이 1칸**, 2칸 이상 0). 2건은 `nudge`가 흡수했다
(GT 거리 0.58 cm).
**기준을 느슨하게 해서 100%로 만들지 않았다** — 1셀 오차의 출처를 아는 것이 더 낫다.

### 실험결과 4 — mode별 GT 충실도 (@ r_b=0.1)

| mode | 방법 | GT와 거리 |
|---|---|---|
| **`none`** | GT 서브궤적을 **그대로**. refine·thin·spline 전부 생략 | **0.0 cm** (정의상) |
| `nudge` | `refine_min_move` 점별 국소 밀어내기. 창 밖으로 못 나가 **재라우팅 불가** | ~3 cm |
| `corridor` | GT 서브궤적 주변 **회랑(0.75 m) 안에서만** A*. 가구 우회 O, 다른 방 X | ~18 cm |
| `free` | 전체 navigable에서 A*. waypoint 순서만 제약 → 루트 이탈 위험 | ~26 cm |

### 실험결과 5 — spine 앵커 품질 (W1)

| 지표 | s8 | 17DRP |
|---|---|---|
| waypoint→최근접 GT 프레임 XY 거리 (mean) | **0.094 m** | **0.091 m** |
| 프레임 순서 단조성 · spine 채택률 | 9/9 | 14/14 |

<details class="trouble"><summary>troubleshooting — 이 단계에서 잡은 버그</summary>

**1. ladder에 밑단이 없어 통과 가능한 leg도 전부 `nudge`를 거쳤다.**
증상: mode 열이 `nudge 47/47`로 나와 "47개를 밀어냈다"로 읽힘. 실제 이동량은 median **0.000 m**.
`refine_min_move`는 여유가 충분한 점을 **손대지 않으므로** `nudge`는 "성공한 방법의 이름"일 뿐이었고,
"몇 개가 실제로 보정을 필요로 했나"는 **아예 측정되지 않았다.**
조치: `none` rung 추가 + **보정 필요 leg 열** + W5 identity 게이트.

**2. `snap_to_grid`가 leg 시작점을 순간이동시키는데 끝점만 검사했다** (결과가 **비단조**로 나옴).
증상: 17DRP ep0에서 leg0이 r_b=0.20/0.30에서 blocked인데 **r_b=0.45에서 통과**. 반경이 커졌는데
통과가 늘어나는 건 물리적으로 불가능하다.
원인: `corridor`/`free`가 끝점을 마스크로 스냅하는데(`snap_to_grid`, 최대 1.5 m), `_path_ok`는
**target만** 검사하고 시작점은 안 봤다. 실측 — 시작점 이동 r_b=0.2에서 **0.79 m**(A* 실패),
r_b=0.45에서 **1.68 m**(A* **성공**). 즉 더 큰 로봇이 더 멀리 순간이동해서 "성공"한 것이다.
(같은 종류의 버그를 goal에서 한 번 고쳤는데 start는 빠져 있었다.)
조치: 스냅된 시작점이 `GOAL_TOL_M`(0.10 m)을 넘으면 기각. 시작점이 그 embodiment로 점유 불가면
그 leg는 **진짜 blocked**다 — 로봇은 물리적으로 거기 있다. 수정 후 프레임 커버리지가 단조로 회복됐다
(ep0 40→0→0→0, ep1 40→40→40→27, ep2 28→28→28→0).
검증: 마스크 자체는 완전히 단조임을 먼저 확인했다(`mask(r₂) ⊆ mask(r₁)` 위반 셀 **0**) — 맵이 아니라
플래너 문제임을 분리한 것이 진단의 핵심이었다.

**3. navmesh 마스크를 기존 clearance 소비자에 붙이는 방법.**
navmesh는 이미 `r_b`로 깎인 configuration space라 "clearance ≥ r_b"가 아니라 "마스크 안"이 맞는 질문인데,
`refine_min_move`/`truncate_navigable`/`check_path_navigable`/`_path_ok`는 전부 clearance 규약을 쓴다.
조치: `esdf = (마스크 안) ? 경계까지거리 + r_b : 0` — 두 규약이 정확히 같아져 **하류를 하나도 안 고쳤다**
(`navmesh_grid.py`, self-check 7/7).
알아둘 것: `compute_esdf_2d`가 경계 마스크 셀에 정확히 1셀(0.05)을 주므로 `PLAN_MARGIN_M=0.05`로는
경계가 안 깎인다(`>=`). 1셀을 실제로 비우려면 0.10이 필요하다.

**4. 이전 판의 잘못된 수치는 폐기했다.** #6a/#6b의 옛 표(GT 거리 3.0/2.3 cm, 프레임 커버리지
100→61→8%)는 **옛 ladder + 옛 밴드 맵**으로 잰 값이다. 위 표로 대체.
</details>

### 방법 — 왜 순수 A*를 버렸나

`start→goal` 순수 A*는 **instruction을 모른다**. R2R GT는 사람이 고른 경유지를 지나도록 만들어졌는데
A*는 기하적 최단만 찾으므로 다른 방으로 돌아버리고, 그러면 instruction이 거짓 라벨이 된다.
정량적 근거: `reference_path` 길이가 `geodesic_distance`보다 **긴** 에피소드가 절반 가까이다
(17DRP ep95: 5.70 vs 4.94 m) — 그 차이가 instruction을 따르느라 돌아가는 부분이고 A*는 그걸 잘라낸다.

**해결 = 데이터셋 생성 절차를 그대로 재적용**했다. 출처를 찾아보니 이미 VLN-CE가 한 것이었다:

| | 출처 | 내용 |
|---|---|---|
| waypoint의 정체 | [R2R, Anderson et al. CVPR 2018](https://arxiv.org/abs/1711.07280) | Matterport3D **파노라마 뷰포인트**. 엣지는 mesh ray-trace 후 5 m 초과 제거 + 수동 검증. **다른 방** start/goal, "5 m 미만 / 엣지 4~6개 밖" 제외 → 7,189 경로. 경로마다 **AMT 3명**이 3D fly-through 보고 지시문 작성 → 21,567개 |
| 연속화 | [VLN-CE, Krantz et al. ECCV 2020](https://arxiv.org/abs/2004.02857) | 노드를 *"a ground-based agent represented by a **1.5m tall cylinder of diameter of 0.2m**"* 가 점유 가능한 점으로 투영(98.3% 성공) → *"waypoint locations"* |
| **leg별 계획** | 같은 논문 | *"We run this algorithm **between each waypoint in a trajectory to the next** … navigable if … the shortest path to **within 0.5 m** of the next waypoint"* → R2R 궤적의 **77%**만 통과 |

즉 **`r_b`=0.1 · `h`=1.5 · `leg_tol`=0.5 m는 우리가 고른 값이 아니라 데이터셋의 정의값**이다
(배포 `.navmesh`의 `agent_radius=0.1, agent_height=1.5` 직독값과 일치).

### waypoint 표현 — 두 학습 모드를 재생성 없이 지원

```
spine (embodiment 무관)              per_e (r_b별)
  waypoints_mesh   4~7개              legs[i].status ∈ ok | detour | blocked
  anchor_frame     [f_0..f_T]         path_xy          성공 구간 이어붙인 경로
  anchor_dist_m    품질                frames_covered   GT 비교 시 반드시 이 구간으로 자를 것
  legs             프레임 구간          reach_ok[f]      프레임별 0|1|2  ← 로더가 실제로 쓰는 것
```

학습 샘플은 `poses[start_frame_id : start_frame_id+goal_len+1]`(≤3.25 m 국소 창)이므로 창이 걸치는
프레임 플래그만 O(1)로 보면 된다 — **제외 모드**는 창에 2가 있으면 버리고, **도달불가 학습 모드**는
2가 처음 나오는 프레임을 stop 라벨로 쓴다(기존 `stop_list` 경로 재사용).
**leg 하나가 막혀도 앞쪽 leg 프레임은 전부 살아남는다.**

<details>
<summary><b>troubleshooting — 틀렸던 측정과 버그 이력 (펼치기)</b></summary>

| 증상 | 원인 | 조치 |
|---|---|---|
| r_b=0.45에서도 **blocked가 0**, 모든 leg 성공 | `_plan_leg`가 **끝점 거리만** 보고 `_path_ok` 4조건을 빼먹었다. `nudge`는 실패를 반환하지 않고 점을 조금 밀기만 하므로 끝점이 거의 항상 GT 끝점이라 `leg_tol`(0.5 m)을 통과 → **벽을 지나는 경로가 전부 성공으로 집계** | leg마다 4조건(하드 충돌·몸통 침범·미관측 통과·target 도달) 적용 |
| GT와의 거리가 **225 cm / 409 cm** | blocked에서 끊긴 **부분 경로를 전체 GT와 비교**했다. 호길이 리샘플이라 길이가 다르면 대응점이 전부 어긋난다 | `frames_covered`로 GT를 잘라 비교 → 5.5 / 8.4 cm |
| `corridor`와 `free`의 GT 거리가 **동일(20.9 cm)** | 회랑 반경 기본값 2.0 m가 leg 길이(median 1.79 m)보다 넓어 **제약이 전혀 안 됐다** | 0.75 m로 낮춤 → 17.8 vs 25.9 cm로 분리 |
| 리포트 그림에 waypoint status를 색(초록/노랑/빨강)으로 칠했다 | leg 통과·우회·막힘은 **색 선이 어디서 끊기고 얼마나 벌어지는지로 이미 읽힌다** — 중복 인코딩이라 범례만 복잡해짐 | waypoint를 단색 하나로 |

**구조적 교훈**: 위 1·2번은 둘 다 "성공/실패를 판정하는 자리에서 검증을 빼먹으면 지표가 조용히
좋아진다"는 같은 실패다. 생성 라벨을 다루는 코드에서는 **기각 조건을 한 곳(`_path_ok`)에 모아 모든
경로가 반드시 통과**하게 두는 것이 맞다.

</details>

---

## ⚡ #08 · R2 — 무엇을 물었고 무엇이 문제인가

**물음**: augment를 켜면 **GPU가 데이터 준비를 기다리게 되는가?**

**유효 측정**(s8 씬, T=12프레임/sample = 학습 `max_len`, epoch 200 samples, worker당 torch 스레드 1개):

| 항목 | 값 |
|---|---|
| replan (occ→2D→**fine 격자** A*+refine+spline) | **51 ms** |
| **Open3D depth 렌더 ×12** | **339 ms** (≈28 ms/frame) ← **지배 비용** |
| **sample 1개 합** | **390 ms** (단일 프로세스) |

| num_workers | samples/sec | batch 16 준비 |
|---|---|---|
| 0 | 5.90 | 2.71 s |
| 2 | 10.10 | 1.58 s |
| **4** (학습 기본값) | **15.77** | **1.01 s** |

batch_size 4 vs 8(workers=4): 16.08 vs 15.33 samples/s → **batch 크기는 영향 없음**(sample 단위 병렬).
(fine 격자 A*로 replan이 44→51 ms 늘었지만 렌더가 지배해 총량 변화는 없다.)

**병목인가 — 아직 단정할 수 없다.** batch 16 준비가 **1.01 s**다:
- GPU step이 1.01 s보다 **느리면** → augment는 뒤에 숨어 사실상 공짜
- GPU step이 1.01 s보다 **빠르면** → **augment가 병목**

**GPU step 시간을 아직 안 재봤다.** 그게 남은 숙제다 — 다만 P-B2 이식 후 `embodiment_aug` off/on으로
학습 steps/s를 직접 비교하면 되므로 **별도 마이크로벤치는 불필요**하다. 병목으로 드러나도 대응 수단은
있다 — workers 8, 또는 렌더 프레임 수 T 축소.

### 삭제한 잘못된 측정
- ~~"우리 occupancy 맵이 recast navmesh보다 **과대 장애물**이라서 원본 GT가 `clearance ≥ r_b`를 위반한다"~~
  → **방향이 반대였다.** S2 실측(#04): 우리 밴드 맵은 recast보다 일관되게 **더 관대**하다
  (s8 r_b=0.2에서 마스크 밖 우리 **178** vs habitat **925**).
- ~~"GT가 위반하는 이유는 **navmesh 경계를 여유 0으로 스치기 때문**"~~ → **이것도 철회(2026-08-20, #03 재작성).**
  GT는 배포 navmesh에서 벽을 뚫지 않는다(17DRP 0/2465 · s8 14/2450, 깊이 최대 3.7 cm < 한 칸).
  근거로 쓴 `clearance min = 0`은 **최악의 한 점**이었고, 애초에 `clearance ≥ r_b`는 이미 반경만큼 깎인
  지도 위에서 **반경을 두 번 세는** 잘못된 질문이었다. 원 증상은 `Z_OFFSET_M = 0.20` 정렬 버그 +
  데이터셋이 쓴 적 없는 `r_b = 0.2`에서 나온 숫자다.
  이 오해 때문에 `CLEARANCE_OFFSET_M = 0.10`(GT 판정 반경을 깎아주는 보정)을
  도입하려 했는데, 그건 **관대한 맵을 더 관대하게** 만드는 것이라 정반대로 위험했다. 폐기하고 맵 자체를
  navmesh 래스터화로 교체했다.
- ~~"`agent_max_climb=0.20`을 맞추면(h_nav 0.15→0.20) 문턱 문제가 풀린다"~~ → **기각.** 일치율이
  88.67%→88.65%로 사실상 불변. 차이의 원인이 밴드 하단이 아니라 recast의 region/ledge/도달가능성 필터다.
- ~~"CPU BEV 165 ms/frame이 최대 병목"~~ → **오측이었다.** torch 스레드가 코어 수와 같아(24=24) 생긴
  경합 현상이고, **스레드 1개면 1.0 ms/frame**(약 100배 차). 게다가 **이 loader에는 BEV가 필요한 곳이 없다**:
  S1 BEV는 모델이 **GPU**에서 `traj_depths`로부터 계산하고, **경로 계획은 캐시된 3D occupancy**를 쓴다
  (`derive_obstacle_2d`). → `with_bev=False`는 유지하되 이유는 "느려서"가 아니라 **"중복이라서"**.
  ⚠️ BEV를 planning 격자로 쓰는 것은 **2dloader 설계**이고, 거기서는 loader가 BEV를 만들어야 한다.
- ~~prefetch만 재서 나온 53,747 samples/s~~ → 측정 방법 오류. epoch 전체를 소비하고 조합마다 별도
  subprocess로 재도록 고쳤다(위 표는 그 방식으로 측정).

### 유지되는 제약 (검증됨)
**부모 프로세스가 Open3D 렌더러를 만든 뒤 fork하면 worker가 데드락**한다(무한 대기).
→ dataset `__init__`에서 렌더러를 만들지 말고 worker의 첫 `__getitem__`에서 생성. `num_workers=0`은 미지원.
아울러 **worker당 `torch.set_num_threads(1)`** 이 필요하다(위 스레드 경합 때문).

---

## 🧭 #09 · waypoint만 주면 GT를 얼마나 재현하나 — navmesh vs recast v2 (2026-08-21)

`09_waypoint_replan.py` · `logs/embodiment_augment/waypoint_replan/` ·
Artifact https://claude.ai/code/artifact/b510cb8e-fb12-40a0-9869-4ba8c89042dd

계기: "navmesh 맵과 recast v2 맵이 주어짐. 각각 확인하고 정답 경로에 가까워지도록 경로 생성해야 함.
GT 경로는 입력 금지(waypoint만), r_b=0.10 고정, 양자화는 일단 제외."

### 왜 했나
#05의 ladder는 **GT를 입력으로 쓴다**(통과 가능하면 verbatim, 회랑 중심선=GT). 이 리포트는 반대 질문이다 —
**GT 없이 waypoint 좌표만 주면** 두 맵 위에서 GT를 얼마나 재현하나. 이게 되면 GT가 없는 상황(새 씬,
합성 에피소드)의 라벨 생성 가능성이 열리고, 안 되면 "corridor 중심선=GT" 설계가 선택이 아니라
**필요조건**임이 실측으로 확정된다. 맵은 **bool 마스크만 공급**하고 마스크→(esdf, navigable) 변환·계획·
후처리는 단일 코드 경로다(map별 생성 분기 없음 — 기존 파일 수정 0, 신규 1파일).

### GT는 어떻게 만들어졌나 (논문·코드로 확인 — 추정 아님)
- VLN-CE(Krantz et al., ECCV 2020, arXiv:2004.02857): 로봇 = *"1.5m tall cylinder of diameter of 0.2m"*.
  R2R nav-graph 노드를 navmesh에 스냅(2 m 하향 레이캐스트, 수평 변위 ≤ 0.5 m, 초과분 수동 수정).
  waypoint 사이마다 *"an A\*-based heuristic search algorithm to compute an approximate shortest path"*,
  *"within 0.5 m of the next waypoint"* 면 navigable(77% 전이 성공).
- `gt.json.gz`의 locations/actions = 그 최단경로를 구식 `ShortestPathFollowerCompat`(0.25 m 전진/15° 회전,
  매 스텝 `get_straight_shortest_path_points` 재계획)로 실행한 흔적 — VLN-CE repo
  `habitat_extensions/shortest_path_follower.py`가 "데이터셋 생성 오라클과의 호환"용으로 보존한 코드다.
- → **GT = "waypoint 경유 최단경로 + 이산 실행"**. 같은 절차라면 waypoint만으로 근사 재현이 가능해야 한다.
  GT 자신이 양자화 흔적이라 매끈한 경로의 gt_dist에는 **노이즈 플로어**(GT vs smooth GT, 실측 median
  **2.6~2.7 cm**)가 있다. 양자화 출력은 이번 범위에서 제외(추후 적용 가능).

### 실험 설계 (사전 고정)
변형 사다리(두 맵 동일): **P0** waypoint 체인 A*(8-이웃, 후처리 없음) → **P1** +솎기 0.8 m·cubic spline →
**P2(w)** +A* `clearance_weight` 스윕 {0.5, 1, 2} (GT가 순수 최단선보다 벽에서 먼 스타일이라 —
clearance는 0.30 m로 캡해 개활지 비용 퇴화를 막음). 평가: 호길이 대응점 거리(#03 방식), leg 분해
(split = leg 대응거리 > 0.20 m), len비, SPL-if-followed(l = 같은 맵 start→goal 최단거리).

### 실험결과 1 — 게이트 (17DRP 5/5 · s8 3/5)

| 게이트 | 묻는 것 | 17DRP5sb8fy | s8pcmisQ38h |
|---|---|---|---|
| G1 GT 생존율 ≥99% | 맵 자체가 GT를 살리나 | navmesh **100%** · recast **100%** ✅ | navmesh **99.18%** · recast **100%** ✅ |
| G2 체인 A* 전 leg 연결 | waypoint가 이어지나 | 실패 0 ✅ | 실패 0 ✅ |
| G3 최선 gt_med ≤15 cm | GT에 가깝나 | 9.4 / 10.9 cm ✅ | navmesh 10.9 ✅ · **recast 17.5 ❌** |
| G4 두 맵 차 ≤5 cm | planner가 지배 변수인가 | 1.5 cm ✅ | **6.6 cm ❌** |
| G5 len비 ≤1.05 & SPL 손실 ≤0.02 | 길이 효율 | ✅ | ✅ |

**게이트 기준을 데이터 보기 전에 정했고, 실패를 통과로 만들지 않았다.**

### 실험결과 2 — 변형별 수치 (gt_dist median, 에피소드 median)

| 맵 | 변형 | 17DRP [cm] | s8 [cm] | s8 split leg | len비(s8) |
|---|---|---|---|---|---|
| navmesh | P0 | 10.1 | 13.5 | 4/47 | 1.034 |
| navmesh | P1 | 12.2 | **10.9** | 4/47 | 0.999 |
| navmesh | **P2 w=0.5** | **9.4** | 10.9 | 4/47 | 1.002 |
| recast | P0 | 11.6 | 21.2 | 7/47 | 0.992 |
| recast | **P2 w=0.5** | **10.9** | **17.5** | 6/47 | 0.964 |

clearance_weight는 17DRP에서 P0 대비 −0.7 cm(9.4 vs 10.1)의 미미한 개선이고 w=2부터는 악화 —
**스무딩·가중치로는 더 못 줄인다**(아래 분해가 이유).

### 실험결과 3 — 오차는 어디서 오나: **전부 route-split이다**

split leg(대응거리 > 0.20 m)를 빼면 남는 leg는 두 맵·두 씬 모두 **median 8.1~9.2 cm**로 균일하다
(플로어 2.7 cm + 셀 5 cm 위에서 planner 잔차 ~4 cm 수준).

| (최선 변형) | 비-split leg median | split leg 수 | split leg median |
|---|---|---|---|
| 17DRP navmesh | 8.1 cm | 7/69 | 22.1 cm |
| 17DRP recast | 8.7 cm | 6/69 | 23.2 cm |
| s8 navmesh | 9.0 cm | 4/47 | **127.9 cm** |
| s8 recast | 9.2 cm | 6/47 | 70.5 cm |

- **navmesh의 split 4/47(s8)**: ep9 그림이 전형 — GT는 중앙 장애물 섬을 **위로**, 우리 경로는 **아래로**
  돈다. waypoint가 섬 양끝에 있어 leg 안에서 어느 쪽으로 돌지는 waypoint에 없는 정보다(instruction에만
  있다). #04의 habitat follower도 같은 씬에서 4/9 leg가 갈라졌다(72.8 cm) — **생성기의 문제가 아니라
  waypoint 표현의 정보 한계**다.
- **recast의 추가 split +2**: ep1 그림이 전형 — recast가 중앙 구역을 통행 가능으로 **잘못 열어**(#04의
  잔여 거짓 승인 F=745) A*가 waypoint 사이를 직진하고, navmesh 맵에서는 같은 구역이 막혀 GT를 따라간다.
  **거짓 승인은 shortcut이 되어 GT 재현을 직접 훼손한다** — "거짓 승인은 못 갈 길을 정답으로 가르친다"
  (#04)의 경로 버전.

### 결론
1. **waypoint만으로 루트가 유일한 구간은 GT를 8~9 cm로 재현한다** — planner(체인 A*+스무딩)는 충분하다.
2. **route-split은 waypoint-only의 구조적 한계다** — 어느 쪽으로 돌지는 instruction 정보. GT에 최대한
   가까운 라벨이 목표면 **corridor 중심선=GT(#05 ladder 설계)가 필요조건**임이 실측으로 확정됐다.
3. **맵은 navmesh 래스터화가 안전하다** — recast v2는 GT 생존은 완벽(100%)하지만 거짓 승인이 split을
   만든다. #04의 "default는 navmesh" 결론이 경로 생성 축에서도 재확인됐다.

### 남은 것
- 2씬·계단 에피소드 제외·r_b=0.10만. habitat find_path 직접 대조는 #04 수치 인용(재측정 안 함).
- 양자화(0.25 m/15°) 출력 미실시 — 생성 path를 새 GT로 쓸 때 필요해질 수 있음(사용자 결정으로 보류).
- split leg를 waypoint 없이 고치는 방법(instruction 파싱, GT 힌트)은 이 리포트 범위 밖.
- 에피소드 중복(같은 instruction 재등장, 17DRP ep0=ep10 등)을 제거하지 않고 그대로 집계했다.

---

## 2. 2dloader_vlnce — "시야각 내 이미지·depth만 활용" (별도 폴더/세션)
| 리포트 | Artifact |
|---|---|
| W0 Frame Gate | https://claude.ai/code/artifact/45f4346b-a5bc-474d-bfe6-a2a671fdabd1 |
| W1 Oracle Gap | https://claude.ai/code/artifact/b685b1e2-757f-4678-a62b-9768a4857c0a |
| W2 Obstacle Alignment | https://claude.ai/code/artifact/a41460be-6a99-4021-987f-0b991433655d |
| W3 Embodiment Effect | https://claude.ai/code/artifact/a73835fb-8974-4b03-bcea-5b839c5d9e54 |
| W4 Pixel Goal Relabel | https://claude.ai/code/artifact/a7871393-1a70-4b9f-8107-96836878bc5e |
| W5 Budget | https://claude.ai/code/artifact/b9af0fb2-4f26-468e-a8d9-f20680f17b9e |
| W6 Paired Counterfactual | https://claude.ai/code/artifact/4f372536-f207-4bd0-9f5f-d5ccb125048f |
| R2 Speed Correction | https://claude.ai/code/artifact/55795bd9-9db1-4ad8-b08d-9a558bc7c800 |

> 2dloader가 제기한 **"3dloader R2의 165 ms/frame은 torch 스레드 문제"**는 이번에 **재측정으로 확인**했다
> (24스레드 162 ms/frame → 1스레드 1.7 ms/frame). #08 R2에서 해당 측정을 삭제·정정했다.
> ⚠️ 단, **2dloader는 BEV가 planning 격자**여서 loader가 BEV를 만들어야 하고, 거기서는 이 스레드 설정이
> 성능에 직접 영향을 준다(3dloader는 3D occupancy로 계획하므로 BEV가 불필요).

---

## 3. gs_vlnpe 파이프라인 (이 작업 이전, 재사용 대상)
| 리포트 | Artifact |
|---|---|
| 00 inspect_vln_n1 | https://claude.ai/code/artifact/de729add-5777-444e-8491-d4ae1dc966f4 |
| 01 prepare_scene | https://claude.ai/code/artifact/e7f57d22-bfe9-449f-bd82-ce416c245013 |
| 02 build_freemap_esdf | https://claude.ai/code/artifact/7c1bc45b-7fd0-4813-b666-6438fce6b9d7 |
| 03 sample_gt_paths | https://claude.ai/code/artifact/1bea4e9a-997c-4f43-9fdf-2920f5596a8f |
| 04 render_obs (Open3D) | https://claude.ai/code/artifact/0621a082-76fa-423f-a48d-8ae3e0309a15 |
| 04 render_obs_isaac (최신판) | https://claude.ai/code/artifact/1c9330d2-a7d4-438e-ab3d-efcb97e5ffc7 |
| 04d tonemap_sweep — vln_n1 | https://claude.ai/code/artifact/e9167e2b-1d53-45f1-9a3b-ce55ef07bfc0 |
| 04d tonemap_sweep — vln_pe | https://claude.ai/code/artifact/2dd470b3-2a55-4fa9-86a6-66cd4eea31b3 |

---

## 발행 규약 (실제로 겪은 실수에서 나온 것)
1. **`<title>` 필수** — 없으면 갤러리에 전부 `artifact`로 떠서 구분이 안 된다. `--title`이 넣는다.
2. **목적 2블록 자동** — *전체 파이프라인의 목적*(발행기 상수 `PIPELINE_INTRO`) + *이 리포트의 목적*(`--purpose`).
3. **그림 설명은 1회만** — `--legend depth,bev,floorplan`(공용 표기) + `--figtypes "pat|이름|설명"`(종류별)로
   **범례 카드에 한 번** 쓰고, 각 그림엔 파일명 + 짧은 이름만. 처음엔 g1이 같은 캡션을 20번,
   pixelgoal이 프레임마다 색 규약을 재출력해 읽기 어려웠다.
4. **`body.html`을 실어야** 그림과 설명이 짝지어 간다. `summary.html`+jpg만 발행하면 캡션이 통째로 빠진다.
5. **이미지는 base64 인라인** — Artifact CSP가 상대경로를 막는다.
6. **같은 파일 경로로 재발행하면 URL 유지.** 다른 대화에서는 `url` 인자로 기존 주소를 넘긴다.
