# 3dloader_vlnce — 로봇 크기에 맞춰 정답 경로를 다시 만든다

## 한 줄 요약

VLN-CE 학습 데이터의 정답 경로는 **지름 20 cm 작은 로봇** 하나만을 위해 만들어져 있다.
이 폴더는 **로봇 크기를 바꿨을 때 정답 경로를 다시 만들어 주는 코드**다.

---

## 왜 필요한가

로봇에게 이런 문장을 주고 따라가게 학습시킨다:

> "부엌으로 가서 식탁 옆에 서라"

학습 데이터 = **문장 + 그 문장대로 가는 정답 경로**.

문제는 그 정답 경로가 **작은 로봇 하나**(지름 20 cm)용이라는 것이다. 큰 로봇은 좁은 문을 못 지나가니
**정답도 달라져야 한다.** 그런데 지금 데이터셋에는 로봇 크기 축이 아예 없다.

```
  작은 로봇  ●→→→→→ 문 통과 ─→ 부엌            정답 = 이 경로
  큰 로봇    ●→→→→→ 문에 막힘                  정답 = ??? ← 이걸 만든다
```

## 무엇을 어떻게 바꾸나

바꾸는 것은 **로봇 반경(`r_b`)** 하나다. 그러면 두 가지가 함께 바뀐다.

| 바뀌는 것 | 어떻게 |
|---|---|
| **정답 경로** | 못 지나가는 구간에서만 밀어내거나 우회, 그것도 안 되면 멈춤 |
| **로봇이 보는 것** | 큰 로봇에게는 좁은 틈이 "막힌 것"으로 보여야 한다 (BEV 장애물을 `r_b`만큼 부풀림) |

**핵심 원칙: 지나갈 수 있으면 원본을 절대 건드리지 않는다.**
정답 경로는 사람이 쓴 문장과 짝이므로, 이유 없이 흔들면 문장이 거짓 라벨이 된다.
정답 instruction에서 크게 벗어나지 않으면서 로봇이 통과할 수 있도록 경로를 만들어야 한다. 

경로를 만들 때 아래 순서로 시도하고 **처음 성공한 데서 멈춘다**:

```
0. none      원본 그대로 되나?      → 되면 원본 그대로 (이탈 0)
1. nudge     살짝 밀면 되나?        → 이탈 ~3 cm
2. corridor  원본 근처로 우회하면?  → 이탈 ~18 cm (다른 방으로는 못 감)
3. blocked   안 되면 → 멈춤 (그 앞 구간까지는 학습에 쓴다)
```

---

## 파일 — **앞 번호가 실행 순서** (`gs_vlnpe`와 같은 규칙)

| 파일 | 하는 일 | 상태 | 리포트 (로컬 · `logs/embodiment_augment/`) | Artifact |
|---|---|---|---|---|
| `00_verify_pose_mesh.py` | 카메라를 집 3D 좌표에 정확히 놓을 수 있나 ← **기초. 틀리면 전부 무의미** | ✅ 정합 0.3 mm · 렌더 0.5 mm | `g1/summary.html` | [열기](https://claude.ai/code/artifact/e9ff4aa3-7fc3-4618-93d6-026b08b026ef) |
| `01_build_scene_geo.py` | 집 3D를 **지오메트리 전용 ply**로 저장 (텍스처 버림) | ✅ 7~10 MB · 로드 0.1 s · RAM 0.08 GB | — | — |
| `02_verify_depth_sources.py` | depth 세 소스 비교: 저장 / 우리(Open3D) / **habitat 직접 렌더** · 5·10·15·20 m | ✅ 우리↔habitat **0.01 mm** | `depth_sources/summary.html` | [열기](https://claude.ai/code/artifact/da0cb778-1510-4c61-824b-fec94734ee65) |
| `03_verify_navmesh_gt.py` | **데이터셋이 쓴 지도**에 원본 정답과 우리 경로를 올려 "벽 뚫기" 판정 + 크기별 지도 캐시 | ✅ 게이트 A/B/C/D 통과 · 뚫기 미재현 | `s1_navmesh/report.html` | [열기](https://claude.ai/code/artifact/330c32e4-4b25-4f51-96c6-5fc54f1756a0) |
| `04_calibrate_map.py` | **우리 지도**가 habitat 지도와 같은 판정을 내리나 + **왜 다른가 분해** + pathfollower | ⚠️ 일치 75.8~95.6% · 게이트 C/D 실패 | `s2_mapcal/report.html` | [열기](https://claude.ai/code/artifact/fc3a8d19-e399-45b1-86cd-0389371a86ad) |
| `05_validate_waypoints.py` | **정답 경로 생성 검증** ← 현재 방식 | ✅ 원본과 0.00 cm | `waypoints_17drp_occ/` ← 기본<br>`waypoints_occ/` · `waypoints_17drp/` · `waypoints/` | [17DRP occ](https://claude.ai/code/artifact/003a26b8-2335-4cdb-bde5-bc0ce6d3c057) · [s8 occ](https://claude.ai/code/artifact/e6fd96c9-f60a-446d-b0a8-1f17db0ee1e6) · [17DRP nav](https://claude.ai/code/artifact/2948897e-949b-48c6-a302-c0cb36d66026) · [s8 nav](https://claude.ai/code/artifact/a2f7fbc8-d644-4855-9576-6c2b16fe44fa) |
| `05b_g2_report.py` | 로봇 크기가 정말 경로를 바꾸나 | ✅ 바뀜 | `g2/summary.html` | [열기](https://claude.ai/code/artifact/01c694fc-7762-4f8b-a99b-47f51e15a077) |
| ~~`06_validate_augment.py`~~ | (폐기 — 모드 A 자체가 폐기. 코드는 헬퍼용 유지) | ❌ | 없음 (`pb/` 삭제됨) | — |
| `06b_validate_modes.py` | 로봇 크기를 관측에 넣는 두 방식 비교 | ✅ (B) 채택 | `modes/summary.html` | [열기](https://claude.ai/code/artifact/0742d328-6554-4f42-8ed2-efdfcc1f672d) |
| `07_validate_pixel_goal.py` | 화면 목표점을 어떻게 옮기나 | ✅ shift 채택 | `pixelgoal_shift/` ← 채택<br>`pixelgoal_retreat/` · `pixelgoal_nearest/` | [shift](https://claude.ai/code/artifact/60cfa34a-fd40-4ab8-9b9c-a307ad9b6f8e) · [retreat](https://claude.ai/code/artifact/e2dc0817-96c5-45ea-8747-b037ca696ab5) · [nearest](https://claude.ai/code/artifact/e56f43ec-65c1-465b-b65f-5b74435f7e4c) |
| `08_bench_parallel.py` | 학습을 느리게 만들지 않나 | ⚠️ 미결 (아래) | `perf/summary.html` | [열기](https://claude.ai/code/artifact/88c35ccc-8740-413b-8d59-7e2abd0e3e22) |

리포트 경로는 **repo 루트 기준** `logs/embodiment_augment/<위 경로>`다. 각 폴더에 파일이 두 개 있다:
`summary.html`(브라우저로 바로 열기 — 이미지는 옆 jpg 참조) / `artifact.html`(이미지 base64 내장, 발행본).
⚠️ `logs/`는 **gitignore 대상**이라 clone에는 없다 — 직접 돌려서 만들거나 Artifact 링크를 본다.

**번호 없는 파일 = 라이브러리** (실행 순서 없음)

| 파일 | 역할 |
|---|---|
| `embodiment_augment.py` | 핵심 클래스. 경로 생성 ladder + 라벨 유효성 4중 검사 |
| `vlnce_align.py` | 에피소드를 집 좌표에 맞추는 변환 (**수식 하나로 계산**, mesh 로드 불필요) |
| `waypoint_spine.py` | 사람이 찍은 중간지점을 프레임에 붙이고 구간을 나눔 |
| `corridor_utils.py` | 원본 경로 주변 "우회 허용 범위" 만들기 + 이탈량 재기 |
| `navmesh_grid.py` | habitat 지도를 우리 격자로 변환 (하류 코드 무수정) |
| `recast_like.py` | **칸마다 바닥을 찾는** 미니-recast 지도 (#04 후보 v2 — s8 92.3%로 밴드 계열 최고 · GT 안전 · navmesh 없는 씬용) |
| `habitat_render.py` | habitat에서 임의 pose depth 렌더 (렌더러 gap 측정용, offline 전용) |
| `pixel_goal_utils.py` | 화면 목표점 투영·가시성·조정 |
| `publish_artifact_report.py` | 시각화 → 공유용 HTML **(매 단계 후 실행)** |

---

## 실행

```bash
# 0) self-check — 씬·habitat 불필요, 몇 초. 코드를 건드린 뒤엔 항상 먼저
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/embodiment_augment.py
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/navmesh_grid.py
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/waypoint_spine.py
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/corridor_utils.py
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/04_calibrate_map.py --selfcheck
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/recast_like.py
```

```bash
# 1) 씬 지오메트리 빌드 — 한 번만. 텍스처를 버려 7~10 MB / 로드 0.1 s 로 만든다
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/01_build_scene_geo.py --scenes 17DRP5sb8fy,s8pcmisQ38h
```

```bash
# 2) depth 세 소스 비교 (저장 / 우리 Open3D / habitat) — 범위 5·10·15·20 m
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/02_verify_depth_sources.py --scene s8pcmisQ38h --ranges 5,10,15,20
```

```bash
# 3) 데이터셋의 지도에 두 경로 올리기 + 크기별 지도 캐시 (04/05가 이 캐시를 읽는다)
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/03_verify_navmesh_gt.py --scenes 17DRP5sb8fy,s8pcmisQ38h --r_bs 0.10,0.15,0.20,0.30,0.45 --episodes 14 --out_dir logs/embodiment_augment/s1_navmesh
```

```bash
# 4) 우리 지도 vs habitat 지도 + 원인 분해 + pathfollower 재현
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/04_calibrate_map.py --scenes 17DRP5sb8fy,s8pcmisQ38h --episodes 14 --out_dir logs/embodiment_augment/s2_mapcal
```

```bash
# 5) 정답 경로 생성 검증 (기본 조합)
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/05_validate_waypoints.py --scene 17DRP5sb8fy --episodes 14 --corridor_m 0.75 --map_source occ --out_dir logs/embodiment_augment/waypoints_17drp_occ
```

```bash
# 6) 리포트 발행 (매 단계 후)
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/publish_artifact_report.py --stage_dir logs/embodiment_augment/waypoints_17drp_occ --title "W 17DRP occ"
```

| 인자 | 의미 |
|---|---|
| `--map_source occ` / `navmesh` | **어느 지도로 판정할지.** 아래 §지도 두 개 참고 |
| `--r_bs` | 시험할 로봇 반경 목록 [m]. `0.10`이 원본 데이터셋 값 |
| `--corridor_m 0.75` | 우회 허용 반경 [m]. 구간 길이가 median 1.79 m라 2.0은 제약이 안 된다 |
| `--episodes` | 씬당 에피소드 수 |
| `--out_dir` | 로컬 산출물 (`summary.html` = 바로 열기용, 그림 jpg) |

전체 명령·인자는 `command_embodiment_augment.md`, 결과 표·링크는 `reports.md`.

---

## 지도 두 개 — `--map_source`로 고른다

"이 로봇이 여기 서 있을 수 있나"를 판정할 지도가 두 가지 있다.

| 옵션 | 만드는 법 | 장점 | 단점 |
|---|---|---|---|
| **`occ`** ← 기본 | 집 3D 스캔을 **바닥 위 0.20~1.50 m 밴드**로 잘라 투영 | habitat 없이 동작 (Isaac에도 그대로) · 원본 경로를 안 건드림 · 버리는 샘플 적음 | habitat과 **75.8~95.6%**만 일치(GT 주변 2 m, 셀 단위) → **못 갈 길을 승인**. 다층 씬에서 원인의 **66.6%가 "이 높이에 바닥 없음"**(#04) |
| `navmesh` | habitat이 쓰는 지도를 우리 격자에 찍음 | habitat 판정과 **100% 일치** | habitat에 묶임 · 격자 1칸 오차로 원본 경로를 살짝 건드림 |

`0.20`과 `1.50`은 우리가 고른 값이 아니라 **habitat 설정을 그대로 읽은 값**이다
(`agent_max_climb` = 20 cm까지 밟고 넘어감, `agent_height` = 로봇 키).

실측 비교:

| 집 | 옵션 | 원본 경로 안 건드림? | **위험**(못 갈 길 승인) | 손실(갈 수 있는데 버림) |
|---|---|---|---|---|
| 17DRP | **occ** | ✅ 69/69 · **0.00 cm** | **2** / 276 | 47 |
| 17DRP | navmesh | ✅ 69/69 · 0.00 cm | 0 | 58 |
| s8 | **occ** | ✅ **47/47 · 0.00 cm** | **0** / 188 | 45 |
| s8 | navmesh | ⚠️ 37/42 · 0.58 cm | 0 | 58 |

---

## 확인된 사실 (인용 가능)

| 사실 | 근거 |
|---|---|
| VLN-CE 정답 경로는 **habitat 기본 설정** 그대로 만들어졌다 (`agent_radius` 0.10 · `agent_height` 1.50) | 배포된 `.navmesh` 파일 직접 읽음 |
| 정답 경로의 중간지점 **1,005개 전부** habitat 지도에서 유효 | `is_navigable` 100% |
| **정답 경로는 데이터셋의 지도에서 벽을 뚫지 않는다** | 17DRP 위반 **0**/2465점 · s8 **14**/2450점이고 그 깊이 최대 **3.7 cm < 한 칸(5 cm)** |
| 우리가 보는 5 cm 래스터가 막는 점도 **전부 깊이 1칸** = 격자 반올림 | s8 58점 전부 1칸, 2칸 이상 **0**. 2칸 이상이 나오면 반증인데 안 나옴 |
| ↑ 그래서 원 증상("우리 맵에서 정답이 위반")은 **우리 쪽 문제**였다 | 정렬 버그(`Z_OFFSET_M` 0.20→0.0) 정정 후 #05 identity 69/69 PASS |
| habitat의 최단경로는 **경로는 재현하지만 궤적은 재현하지 못한다** | median 12.9 / 56.4 cm 차 |
| ⚠️ **길이비 ≈ 1은 "같은 경로"의 증거가 아니다** | s8은 길이비 **1.002**인데 최단경로가 **장애물 반대편**으로 돈다 (`s8pcmisQ38h_ep008.jpg`) |
| → 그래서 우회 범위의 중심선은 **원본 궤적**을 쓴다 (선택이 아니라 필수) | `05_validate_waypoints.py` |
| **우리 지도가 habitat과 다른 주된 이유**: 밴드 투영이 "이 높이에 **바닥이 있는지**"를 안 묻는다 | #04 분해 — s8 거짓 승인의 **66.6%**가 "바닥 없음"(다층 집에서 다른 층 공간이 새어 들어온다). 17DRP는 6.0%뿐이고 대신 경계 1칸이 80% |
| 그 수정(`& floor_exists`)은 **크게 도움 되지만 채택 불가** | s8 일치율 **75.8→90.7%**인데 거짓 기각이 **164→1437**(9배). 두께를 흔들면 두 오류가 반대로 움직여 하나로는 둘을 못 만족 |
| **pathfollower는 코너는 고치지만 경로 갈림은 못 고친다** | s8 재현 오차 median **56.4→8.8 cm**인데 분포가 두 덩어리 — 5개 5.4 cm / **4개 72.8 cm(반대편으로 갈라짐)** |
| ⚠️ 헤드리스 follower는 **`move_filter_fn`을 반드시 넣어야** 한다 | 없으면 롤아웃이 **벽을 통과한다**(실측 5점). `Simulator`가 있을 때만 habitat이 자동으로 채워준다 |
| **타일링은 필요 없다** — 씬 전체 지오메트리(7~10 MB)가 타일 하나(6 MB)보다 작다 | 무거운 건 지오메트리가 아니라 **텍스처**(367 MB jpg → RAM 8.5 GB). v1은 depth만 렌더 |
| **우리 렌더 = habitat 렌더** (median 0.01 mm, 5~20 m 전 범위) | 에셋(obj vs glb)·렌더러가 달라도 지오메트리는 같다 |
| 저장 depth와의 0.5 mm 차이는 **우리 오차가 아니다** | 데이터셋이 depth를 **버림(truncation)** 으로 저장 — 부호가 99.3~100% 양수, 분포 0~+1 mm 균일 |

## 아직 안 된 것 (정직하게)

| | 상태 |
|---|---|
| **실제 학습 코드에 안 붙었다** | 전부 이 폴더에서만 돌아간다. 붙이는 게 다음 큰 작업 |
| **모델이 좋아지는지 한 번도 안 봤다** | 지금까지는 전부 "데이터가 올바른가" 검증 |
| GPU가 얼마나 걸리는지 안 재봤다 | 그래서 "느려지지 않는다"고 말할 근거가 없다 |
| 집 61채 중 **2채만** 봤다 | 계단 있는 에피소드는 아예 제외 (큰 집에서 14개 중 5개 = 36%) |
| 구간마다 **따로** 계획해서 이음새가 생긴다 | 갈 수 있는 구간도 최대 47개 버린다 |
| 화면 목표점 쪽은 아직 옛 방식 | `07_validate_pixel_goal.py`는 사람 주석을 안 쓴다 |
| 컬러 영상은 안 바꾼다 | v1 한계. depth/BEV만 |

---

## 코드를 만질 때 주의할 것

**1. 요약 숫자 하나가 결함을 가린다.** 지금까지 나온 버그 4건이 전부 이 이유였다.

| 버그 | 무엇이 가렸나 |
|---|---|
| 카메라가 20 cm 밀려 있었다 | **평균**만 봤다 (바닥·천장만 틀려서 평균은 정상) |
| 지도 판정 방향이 반대였다 | 표 **이름을 거꾸로** 출력했다 |
| "47개를 밀어냈다"고 오해 | **"몇 개 고쳤나"를 아예 안 셌다** |
| 시작점이 1.7 m 순간이동 | **끝점만** 검사했다 |

→ 지표를 하나 추가할 때마다 물어라: **"이 숫자가 좋게 나오면서도 틀릴 수 있나?"**

**2. 파일명이 숫자로 시작하면 `import X`가 안 된다.** `importlib.import_module('03_verify_navmesh_gt')`로
쓴다(`gs_vlnpe`와 같은 방식). 이미 그렇게 되어 있으니 새로 만들 때만 주의.

**3. 새 코드는 이 폴더에만.** `gs_vlnpe/`·`internnav/`는 **import만** 한다. 시각 검증이 끝난 뒤에 이식한다.

**4. `--map_source`처럼 동작을 바꾸는 것은 반드시 인자로.** 하드코딩 금지, 기본값은 기존 동작 유지.

**5. 결과가 `r_b`에 대해 단조인지 항상 확인.** 로봇이 커졌는데 통과가 늘어나면 버그다
(실제로 이걸로 시작점 순간이동 버그를 잡았다).

**6. 오차는 반드시 "기준선"과 나란히 보여라.** "타일 오차 0.003 cm"만 쓰면 큰지 작은지 모른다 —
같은 프레임의 정합 오차 0.49 mm와 나란히 놓아야 "기준선의 1/16"이라고 말할 수 있다
(`02_verify_depth_sources.py`의 5패널이 이 목적이다).

**7. 여러 그림을 나란히 놓을 때는 색범위를 고정해라.** `geometry_utils.colorize_depth`는 **이미지마다**
자기 min/max로 정규화한다 — 값이 0.5 mm만 달라도 두 패널이 전혀 다른 색으로 보인다. 비교용이면
`00_verify_pose_mesh.colorize_fixed`를 써라. 그리고 그림은 **통과해도 저장**해라(실패 시에만 저장하면
통과한 run에는 근거가 없다).

---

## 관련 문서

| 파일 | 내용 |
|---|---|
| `reports.md` | 리포트 목록 — **로컬 경로 + Artifact 주소** + 결과 표 + troubleshooting |
| `command_embodiment_augment.md` | 전체 실행 명령·인자 의미·이식 시 주의사항 |
| repo `.claude/memory/260819_map_align_rb_ladder_result.md` | 지도 맞추기 3단계 결과 상세 |
| repo `.claude/memory/260814_*.md` | 초기 게이트(G1~G4) 결과 |
| `../gs_vlnpe/` | 재사용하는 기하·계획 라이브러리 (`esdf_utils`, `geometry_utils`, `viz_utils`) |
