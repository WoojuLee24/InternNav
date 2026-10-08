# 3dloader_vlnce — On-the-fly embodiment augmentation (계획 요약 · 실행 순서)

**목표**: 학습 중 embodiment `e`(로봇 반경 `r_b`, 지면높이 `h_nav`, 카메라 높이/pitch)를 바꿔
**관측(depth/BEV)과 GT path를 on-the-fly로 재생성**한다(논문 §해결 v0.2 C1). occupancy만으론 렌더
불가 → 씬 mesh를 **지오메트리 전용 ply**(텍스처 버림, 7~10 MB)로 저장해두고 worker가 그걸 올려
Open3D로 depth 렌더. (타일링은 2026-08-20 폐기 — 무거운 건 지오메트리가 아니라 텍스처였다.)

**코드 배치**: 이 작업 신규 코드는 전부 이 폴더(`scripts/dataset_converters/3dloader_vlnce/`).
기존 코드(`gs_vlnpe/*`, `internnav/*`)는 **import만**(수정 금지). 시각화 검증 완료 후 기존 코드에 이식.

**환경**: 이 컨테이너 = habitat. 모든 스크립트 **`/usr/bin/python`**으로 실행
(habitat_sim/habitat 0.3.3 + magnum + open3d 0.19 EGL headless. Isaac/pxr 없음).

**규약**: 각 단계 산출물은 `logs/embodiment_augment/<stage>/`에 저장 + `publish_artifact_report.py`로 발행.

---

## 전체 그림 (왜 이걸 하는가)

논문 주장(C1): **embodiment `e`가 바뀌면 정답 경로도 바뀌어야 한다.** 기존 VLN 데이터는 리그(카메라
높이/pitch)별로 **미리 렌더해 저장**하는 방식이라 `r_b`(로봇 반경) 축이 아예 없고, 리그 추가마다
수십 GB가 든다. → **씬의 3D를 저장해두고 학습 중 `e`에 맞춰 관측·정답을 만들어내자**가 이 작업이다.

```
[offline 1회]  씬 mesh ──> 지오메트리 ply  +  occupancy(npz)  +  에피소드 정합(T_sf2mesh)
                                   │
[학습 중 worker] e=(r_b,h_nav) 샘플 ──┼──> occ→2D→dilation(r_b)→A*→spline = 새 GT path
                                   └──> 씬 ply에서 depth 렌더 ─(모델 GPU)→ BEV
```

**모드 2가지** (`e`가 관측·정답을 바꾸는 방식 — 둘 다 구현):
| | (A) follow | (B) fixed |
|---|---|---|
| 관측 | 카메라가 **새 경로**를 따라감 → r_b마다 다름 | **원본 프레임 고정** → 동일 |
| GT path | r_b로 재계획 | r_b로 재계획 |
| `e`를 관측에 넣는 법 | 위치 변화로 **간접** | **BEV robot-radius dilation** (논문 C1의 빠진 조각) |
| 용도 | 데이터 다양성(augmentation) | **paired counterfactual (C3)** — 같은 장면×다른 e |
| 비용 | 경로마다 재렌더 | 원본 depth 재사용 가능 |

> (A)만 쓰면 `e`가 입력·정답을 동시에 바꿔 "다른 위치라서 정답이 다르다"로 설명 가능 → C3의
> gradient 논리가 약해진다. (B)는 위치를 고정해 **정답 차이를 설명할 변수가 `e`뿐**이게 만든다.

### 끝점 보호 규칙 (필수)
새 `r_b` 경로의 **끝점이 원본 GT goal에서 10 cm 넘게 벗어나면 기각**하고 **기존 `r_b`(0.15) 경로를 쓴다**
(`replan_or_fallback`). 이유: `r_b` dilation이 goal 셀을 지우면 `_snap_navigable`이 도착지를 옮기는데,
그러면 instruction("…에서 멈춰라")이 **거짓 라벨**이 된다.
- 실측(미적용): goal이 최대 **37.8 cm** 이동, start는 최대 157 cm 이동.
- 적용 후: 채택된 경로는 전부 끝점 오차 **≤3.1 cm**(격자 양자화 수준). 16조합 중 ok 7 / 기각→fallback 9 / 실패 0.
- 큰 `r_b`일수록 기각률이 높다 = 좁은 통로를 큰 로봇이 못 지나가는 물리적 사실의 반영.

---

## Gate가 하는 일 (왜 이 순서인가)

augmenter를 짓기 전에, augmenter가 성립할 **전제**들을 값싸게 검증한다. 실패하면 뒤 작업이 무의미.

- **G1 — pose↔mesh 정합**: vln_ce 데이터를 3D mesh에 정확히 붙일 수 있는가?
  augmenter는 mesh에서 depth를 렌더하고 occupancy로 경로를 짜므로, 에피소드 카메라 pose가
  mesh 좌표에 정확히 놓여야 한다. (vln_ce pose는 에피소드 start-relative라 절대정합 필요 → `vlnce_align`.)
- **G2 — `e`가 GT path를 바꾸는가**: 로봇 반경 `r_b`를 바꿨을 때 계획 경로가 실제로 달라지는가?
  안 달라지면 embodiment-conditional 학습의 전제가 깨진다. (같은 start/goal, r_b sweep → 경로 비교.)
- **R2 — on-the-fly 예산**: augment 1회(씬 ply 로드+occ→2D→A*+spline+depth 렌더)가 학습 dataloader
  worker에서 감당할 만큼 빠른가? 목표 **<30 ms/sample**. (느리면 학습이 IO/CPU에 막힘.)
- **#02 — depth 세 소스 일치**: 저장 depth / 우리 Open3D 렌더 / habitat 직접 렌더가 범위별
  (5·10·15·20 m)로 얼마나 다른가? 에셋도 렌더러도 다르므로 둘 다 저장에 가까우면서 서로 다를 수
  있다 — 실측 결과 **우리↔habitat 0.01 mm**.

---

## 파이프라인 — **파일명 번호가 실행 순서** (`gs_vlnpe`와 같은 규칙)

| 순서 | 파일 | 하는 일 | 상태 |
|---|---|---|---|
| **00** | `00_verify_pose_mesh.py` | G1 — vln_ce pose→mesh 정합 + 렌더 depth vs 저장 depth | ✅ 0.0004 / 0.0005 m |
| **01** | `01_build_scene_geo.py` | 씬 전체를 **지오메트리 전용 ply**로 (텍스처 버림) | ✅ 7~10 MB · 로드 0.1 s · RAM 0.08 GB |
| **02** | `02_verify_depth_sources.py` | depth 세 소스 비교: 저장 / 우리(Open3D) / **habitat 직접 렌더** · 5·10·15·20 m | ✅ 우리↔habitat **0.01 mm** |
| **03** | `03_verify_navmesh_gt.py` | 데이터셋이 쓴 지도에 두 경로를 올려 "벽 뚫기" 판정 + r_b별 navmesh 캐시 | ✅ 게이트 A/B/C/D · 뚫기 미재현 |
| **04** | `04_calibrate_map.py` | 우리 지도 vs navmesh 일치율 + **원인 분해** + pathfollower 재현 | ⚠️ 일치 75.8~95.6% · 게이트 C/D 실패 (default는 navmesh 유지) |
| **05** | `05_validate_waypoints.py` | W — GT 우선 + 못 지나갈 때만 보정. `--map_source occ\|navmesh` | ✅ W5 GT 0.00 cm · W6 거짓승인 0~2 |
| **05b** | `05b_g2_report.py` (+ `gs_vlnpe/03_sample_gt_paths.py`) | G2 — r_b sweep 경로 오버레이 + path shift/feasibility | ✅ 16.9 cm, 15/15→0/15 |
| ~~**06**~~ | `06_validate_augment.py` | P-B1 — augmenter **모드 A** 시각화 | ❌ 리포트 폐기(모드 A 폐기). 코드는 헬퍼 제공용 유지 |
| **06b** | `06b_validate_modes.py` | (A) follow vs (B) fixed+BEV dilation | ✅ **(B) 채택**, BEV occ 7.6→10.2% |
| **07** | `07_validate_pixel_goal.py` | pixel goal 보존형 재계획 (`shift`/`retreat`/`nearest`) | ✅ **shift 채택**, V0 0.00 px |
| **08** | `08_bench_parallel.py` | R2 — 속도·병렬성(T=12, num_workers, batch) | ✅ 4 workers **15.8 samples/s** |

### 번호 없는 파일 = 라이브러리 (실행 순서 없음)
| 파일 | 역할 |
|---|---|
| `vlnce_align.py` | GT `start_position`+`start_rotation` → `T_sf2mesh` (**해석적**, mesh 로드 불필요) |
| `embodiment_augment.py` | 핵심 클래스 `EmbodimentAugmenter` — leg 계획 ladder, `_path_ok` 4중 기각 |
| `waypoint_spine.py` | 사람 주석 waypoint를 GT 프레임에 앵커 + 프레임 플래그 |
| `corridor_utils.py` | GT 주변 회랑 마스크 + 이탈량 측정 |
| `navmesh_grid.py` | 캐시된 navmesh → 우리 격자 esdf (하류 무수정용 `+r_b` 트릭) |
| `recast_like.py` | 칸마다 바닥을 찾는 미니-recast 지도 (#04 후보 v2 — navmesh 없는 씬용 최선) |
| `pixel_goal_utils.py` | pixel goal 투영·가시성·조정 |
| `publish_artifact_report.py` | stage 시각화 → self-contained Artifact HTML (**매 단계 후 실행**) |

**self-check** (씬·habitat 불필요, 총 47개 assert): `embodiment_augment.py` · `navmesh_grid.py` ·
`04_calibrate_map.py --selfcheck` · `recast_like.py` · `waypoint_spine.py` · `corridor_utils.py` — 각각 직접 실행.

### 최종 성능 (R2, s8 · T=12 · epoch 200)
| num_workers | samples/sec |
|---|---|
| 0 | 3.84 |
| 2 | 6.61 |
| **4** (학습 기본값) | **10.23** |
| 8 | 13.31 |

sample 1개 = replan 42 ms + depth 렌더 ×12 349 ms ≈ **391 ms**(단일). batch 크기는 영향 미미.

### P-B2 이식 시 반드시 지킬 3가지 (실측 근거)
1. **렌더러는 worker에서 lazy 생성**. 부모가 Open3D 렌더러를 만든 뒤 fork하면 **자식이 데드락**한다
   (EGL/Filament 컨텍스트 상속). → dataset `__init__`에서 만들지 말 것. `num_workers=0`은 미지원.
2. **BEV는 dataloader에서 만들지 말 것**. S1 BEV는 모델 하류가 `traj_depths`에서 GPU로 계산한다.
   CPU BEV는 165 ms/frame(+2022 ms/sample)로 최대 병목. (모드 B의 dilation만 예외적으로 필요)
3. **정합(align)은 offline 배치로 미리 계산해 캐시**. 에피소드당 14.8 s(대부분 199 MB mesh 로드).

### 재검증에서 정정된 사항 (2026-08-15)
- **pose 인자 버그**: `synthesize_action_poses(xy, floor_z, h_b, pitch)` 순서를 틀리게 호출해 카메라가
  엉뚱한 높이(bench는 z=7 m)에 있었다. 두 파일 모두 수정. → 이전 G3 수치는 무효(빈 화면 렌더).
- **타일링 폐기(2026-08-20)**: 무거운 건 지오메트리가 아니라 **텍스처**였다(367 MB jpg → RAM
  8.5 GB · 로드 11.6 s). 텍스처를 버리면 씬 전체가 7~10 MB / 로드 0.1 s / RAM 0.08 GB로 **타일
  하나(6 MB)보다 작다** → margin 튜닝·타일 경계 손실·프레임별 타일 조회가 전부 사라졌다.
  depth 동일성: 5 m 안에서 1,628,548 픽셀 중 오차 >1 mm **0개**. ⚠️ v2 RGB에서 재검토 필요.
