# 실행 계획 — VLN-N1 검증 → VLN-PE 전환 → 실제 GS-map 확장

> InternNav 저장소 기준으로 경로·코드·데이터 구조를 실측 대조해 작성한 실행 계획.
> 모든 경로는 저장소 루트(`/ws/src/InternNav`) 기준 상대경로이며, 파일/클래스명은 현재 코드베이스에서 확인된 것이다.
>
> - 참고 논문: NavDP, InternVLA-N1, DualVLN
> - 관련 근거: VLN-PE (arXiv 2507.13019)
> - 상태: 초안 (미검토)

## Context

최종 목적은 **실제 취득한 GS-map에서 VLN-N1식 GT path를 생성해 VLN-PE 포맷 dataset을 만드는 것**이다.
바로 GS-map에 뛰어들지 않고 **3단계로 de-risking** 한다:

1. **Stage 1 (M1) — VLN-N1 생성·검증**: 표준 오픈소스 씬(mesh 보유)에서 `rgb·depth·gt-path·pose`를 생성하고, **로컬에 릴리스된 InternData-N1 `vln_n1`과 포맷·품질을 대조**해 생성 파이프라인 자체를 검증한다. ← **지금 상세화하는 마일스톤.**
2. **Stage 2 (M2) — VLN-PE 전환**: 물리 USD 씬(InternUtopia) + H1 locomotion controller + VLN-PE LeRobot 포맷 + FR/StR로 확장.
3. **Stage 3 (M3) — 실제 GS-map 확장**: 검증된 파이프라인의 **RGB 소스를 mesh 렌더 → GS splat 렌더로 교체**하고 취득 GS-map(splat+collision mesh)에 적용.

**핵심 설계**: Stage 1은 **mesh 기반 렌더**로 생성 알고리즘(freemap/ESDF/A\*/waypoint-opt/smoothing/render/pose)의 정확성만 검증한다. GS 렌더는 Stage 3에서만 끼워 넣는다 → 관심사 분리.
**정답지 원칙**: 포맷을 발명하지 않고 **로컬 `vln_n1` 실제 파일 스키마를 그대로 복제**한다.
환경(InternNav + IsaacSim 4.2.0 / IsaacLab + InternData-N1-v0.5-mini)은 준비 완료 전제.

---

## 재사용할 기존 자산 (새로 만들지 않음)

- **정답지/레퍼런스 데이터 (로컬)**: `data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i/<scene>/`
  - 하위 구조: `data/chunk-000/episode_*.parquet` + `meta/` + `videos/chunk-000/`
  - LeRobot **v2.1**, fps 30(단, mp4 컨테이너 자체는 10fps로 실측됨). RGB/Depth는 `videos/chunk-000/{observation.video.rgb, observation.images.rgb, observation.video.depth, observation.images.depth}/` 로 저장 — **`observation.images.*`(프레임별 jpg/png)가 1차 소스**이고, `observation.video.*`(에피소드별 mp4)는 8bit 손실 재인코딩본(depth는 uint8 그레이스케일로 뭉개짐)이라 메트릭 계산에는 쓸 수 없다.
- **LeRobot 변환 베이스**: `scripts/dataset_converters/vlnce2lerobot.py` 의 `NavDataset(LeRobotDataset)` / `NavDatasetMetadata(LeRobotDatasetMetadata)` — **상속·확장** 재사용. (`create(...features=...)`, `add_frame(frame, task, timestamp)`, `save_episode(files)` 가 임의 feature dict를 지원)
- **기하/projection 공용 모듈**: `scripts/dataset_converters/gs_vlnpe/geometry_utils.py`(00에서 추출·검증됨) — depth scale, unprojection(K_inv), pose 해석(`action_to_c2w`), `compute_relative_transform`, FPV backward-warp, BEV(top-down) world-frame rasterize, mesh 앵커 검증(`check_against_scene_mesh`)을 모두 포함.
  - **pose는 반드시 `action_to_c2w(action, 'cam2world_gl')`를 거칠 것** (= `action @ diag(1,-1,-1,1)`). `action`을 직접 행렬 연산에 쓰면 축 컨벤션이 틀려 절대 좌표가 0.27~0.82m 어긋난다(아래 M1.0 절 참고). 상대 변환은 `compute_relative_transform`(= `inv(t1) @ t0`).
  - **핵심 진입점 2개, 입출력 형식 통일**: `check_alignment_fpv(rgb_t0, depth_t0, rgb_t1, depth_t1, pose_t0, pose_t1, k, convention, visualize, out_dir, label_prefix)`(카메라 프레임 재투영) / `check_alignment_bev(...동일 8개 인자..., cell_m, visualize, out_dir, label_prefix)`(월드 프레임 top-down 교차검증) — 앞 8개 위치 인자와 반환 dict 키(`metric_name`/`metric_value`/`coverage_frac`/`viz_paths`)가 두 함수 동일.
  - **FPV와 BEV는 서로 대체 불가다.** FPV는 상대 pose(`inv(A₁F)(A₀F)` = `F·(·)·F` 켤레)만 쓰므로 축 컨벤션 오류가 상쇄돼 **둔감**하고, BEV/world는 절대 pose(`A·F·X`, `F`가 한 번만 등장)라 그대로 드러나 **민감**하다. 둘 다 돌릴 것.
  - **여러 프레임을 하나의 world로 합쳐도 된다** — `unproject_to_world_frame`을 프레임마다 독립적으로 호출하면 그만이다(anchor 경유 불필요). 이 world 좌표계는 씬 mesh와 정확히 일치한다(표면거리 median 0.00003 m). *(과거 이 문서에 "독립 호출하면 정렬 안 되니 anchor를 경유하라"고 적혀 있었으나 오진이었다 — 진짜 원인은 위 축 flip 누락.)*
  - **회귀 게이트**: `python geometry_utils.py`가 identity 변환 + `CAM_CV_TO_GL` involution + **mesh 표면거리 < 1mm**를 assert한다. pose 수식을 건드리면 여기서 먼저 걸린다.
  - 04/06 등 이후 스크립트는 이 함수들을 그대로 호출할 것 — 직접 재구현하면 축 컨벤션/순서 버그를 반복할 위험이 있다.
- **소스 씬 (mesh)**: `data/scene_data/mp3d_n1/<scene>/matterport_mesh/` (Matterport3D). 씬 ID가 `vln_n1` traj 씬과 **직접 일치**(예: `17DRP5sb8fy`) → 정답지-mesh 1:1 대조 가능.
- **환경 래퍼**(Stage 2): `internnav/env/base.py`(`Env` 레지스트리) + `internnav/env/internutopia_env.py`(`InternutopiaEnv`, `@Env.register('internutopia')`), 데이터 수집은 `internnav/evaluator/utils/data_collector.py`(`DataCollector`).
- **로봇/자산**(Stage 2): H1 휴머노이드, `data/Embodiments/vln-pe/h1`.
- **GS→USD collision 보조툴**(Stage 3): `github.com/Galery23/SAGE-3D_Official`.
- **알고리즘 근거**: NavDP C1·C4, InternVLA-N1 C1·C5, VLN-PE(arXiv 2507.13019) §3.

---

## 실측한 `vln_n1` 스키마 (정답지 = 단일 진실원)

`data/.../vln_n1/traj_data/matterport3d_d435i/17DRP5sb8fy/meta/info.json` 및 `data/chunk-000/episode_000000.parquet` 실측:

**`meta/info.json`**
- `codebase_version: "v2.1"`, `robot_type: "unknown"`, `fps: 30`, `chunks_size: 1000`
- `data_path: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"`
- `video_path: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"`

**`data/chunk-000/episode_*.parquet` 컬럼**

| 컬럼 | dtype | shape | 의미 |
|---|---|---|---|
| `index` | int64 | scalar | 전역 프레임 인덱스 |
| `observation.camera_intrinsic` | float32 (list) | 3×3 | 카메라 내부 파라미터 (fx,fy,cx,cy — 480×270 기준) |
| `observation.camera_extrinsic` | float32 (list) | 4×4 | 에피소드 **내**에서는 상수지만 **에피소드마다 다르다** — translation z = 로봇 키 `h_b`, 회전 = 카메라 하향 pitch (world pose 아님) |
| `action` | float32 (list) | 4×4 | **매 프레임 변하는 실제 camera-to-world pose** |

> **M1.0 실측 확정 (2026-08-03 최종)**
> - `action[t]`이 per-frame pose이고, 정확한 규약은 **`action[t] @ diag(1,-1,-1,1)` = camera-to-world**
>   (`geometry_utils.action_to_c2w(action, 'cam2world_gl')`). mesh 표면거리 0.000029 m로 확정.
>   `action`을 직접 행렬로 쓰지 말고 반드시 이 함수를 거칠 것.
> - `camera_extrinsic`은 에피소드 **내**에서만 상수다 — **에피소드마다 달라지고**
>   translation z = 로봇 키 `h_b`, 회전 = 카메라 하향 pitch를 담는다
>   (`decompose_camera_extrinsic` / `compose_camera_extrinsic`). 바닥은 `floor_z = cam_z - h_b`.
> - depth 단위는 0.1mm (raw uint16 / 10000 = m). 1차 소스는 mp4가 아니라 프레임별 jpg/png.
> - **판별 근거는 photometric error가 아니라 씬 mesh 표면거리다** — 상대 지표는 축 컨벤션 오류를
>   상쇄해 좁은 baseline에서 정답/오답을 구분하지 못한다.
>
> 좌표 규약 상세·검출능력 비교·**과거 오진 목록**은 `.claude/memory/understanding_gs_vlnpe_fpv_bev_geometry.md`.
> 스키마 단일 진실원은 `scripts/dataset_converters/gs_vlnpe/target_schema.json`.

**`meta/` 부가 파일** (그대로 재현/대조 대상)
- `episodes.jsonl` / `tasks.jsonl`: `sub_instruction`, `sub_indexes`, `revised_sub_instruction` (서브 인스트럭션 단위 언어 라벨)
- `episodes_stats.jsonl`: 에피소드별 통계
- `pointcloud.ply`, `pointcloud_obstacle.npy`, `pointcloud_obstacle.npz`: **장애물 포인트클라우드** → ESDF/collision 검증의 실제 기준선

---

## 작업 디렉토리 (신규)

저장소 관례상 파이프라인 스크립트는 `scripts/` 아래, 변환 로직은 `scripts/dataset_converters/` 아래에 둔다.

```
scripts/dataset_converters/gs_vlnpe/
├── config.yaml                  # 씬 경로, 로봇/카메라(D435i) 스펙, voxel/clip, split, RGB 소스(mesh|gs)
├── viz_utils.py                 # (신규, 공용) logs/gs-vlnpe/에 원본 이미지 + self-contained report.html(base64 임베드) 생성
├── geometry_utils.py            # ✅ (신규, 공용) action_to_c2w(축 flip) + depth→3D→재투영 + check_alignment_fpv/bev + check_against_scene_mesh(회귀 게이트) — 04/06도 반드시 import(직접 재구현 금지)
├── 00_inspect_vln_n1.py         # ✅ 구현·검증 완료 — 로컬 vln_n1 샘플 로드 → target_schema.json + pose-convention 판별(mesh 앵커)
├── 01_prepare_scene.py          # (Stage1) mesh/USD/GT가 같은 프레임인지 검증 → scene_meta.json (USD 생성 안 함) ; (Stage3) GS+mesh 정합·floor 보정
├── 02_build_freemap_esdf.py     # mesh → occupancy/ESDF(0.05m voxel, r_b=0.25m truncate)
├── 03_sample_gt_paths.py        # start-goal 샘플 → A* → ESDF waypoint 최적화 → spline smoothing → filter
├── 04_render_obs.py             # ✅ Open3D 렌더 — 2-A(GT 재렌더링 검증) → 2-B(03 경로 따라 rgb/depth/extrinsic 생성)
├── 05_to_lerobot.py             # NavDataset 확장 → data/chunk-000/episode_*.parquet + meta/ + videos/  (vln_n1 포맷)
└── 06_verify.py                 # 포맷 diff + 품질 검증(collision-free, depth-pose 재투영 일치)
```
> Stage 1과 Stage 3은 **같은 스크립트**를 쓰고 `config.yaml`의 `rgb_source`(mesh|gs)와 `scene`만 바꾼다. 이게 "확장"의 실체.
> 파이프라인은 `00 → 01 → 02 → 03 → 04 → 05 → 06` 순서로 실행하며, 각 스크립트의 출력이 다음 스크립트의 입력이 된다 (아래 명세 참고). 화살표(`A → B`)는 "A의 출력 파일이 B의 입력 파일"이라는 의미다.

### 시각화 출력 · 검증 보고 컨벤션 (모든 스크립트 공통, 00에서 확립됨)

- **시각화는 `logs/gs-vlnpe/<script_name>/<scene>/episode_XXXXXX/`에 저장**한다 (`--log_dir` 기본값 `logs/gs-vlnpe`). `target_schema.json` 같은 파이프라인 canonical 산출물(`--out_dir`, 기본 `scripts/dataset_converters/gs_vlnpe/`)과는 경로를 분리한다 — 전자는 "사람이 보는 자료", 후자는 "다음 스크립트가 읽는 입력".
- **정렬/정합 확인은 정적 이미지 비교보다 "blink comparator"(화살표로 두 이미지를 같은 자리에서 번갈아 보여주기)가 훨씬 잘 보인다** — 사람 눈은 나란히 놓인 두 이미지보다 같은 자리에서 깜빡이며 바뀌는 것에서 어긋남을 훨씬 잘 감지한다. `viz_utils.py`의 `blink_widget_html(widget_id, states, title)`로 만든다(`states`: `[(label, image_path), ...]`, ‹ › 버튼으로 순환).
- **blink로 비교하는 두 이미지는 반드시 같은 격자(그리드)여야 한다.** `cv2.remap(rgb_t1, map_x, map_y)`로 만드는 backward-warp 결과는 **소스(t0)의 픽셀 격자를 그대로 유지한 채 색만 target(t1)에서 빌려온 것**이다 — 그래서 비교 대상은 항상 `rgb(t0)`다(self-supervised depth 학습에서 쓰는 표준 photometric reprojection 방식과 동일; `00_inspect_vln_n1.py`의 `run_pose_convention_check`). **target(t1)의 실제 사진과 직접 비교하려면 target 자체의 depth가 있어야 한다**(t0의 depth만으로 target 격자에 흩뿌리려면 z-buffer/홀 처리가 필요한 forward-splat이 되는데, 이 파이프라인은 t0의 depth만 검증 대상으로 삼으므로 불필요하게 복잡해진다) — 한 번 이 구분 없이 backward-warp 결과를 `rgb(t1)`과 비교했다가 서로 다른 시점(t0 grid vs t1 grid)을 비교하는 바람에 "warp 이미지와 원본 이미지 크기/스케일이 다르다"는 혼란이 생겼던 적이 있다. **blink widget은 항상 "원본 rgb(t0)" ↔ "backward-warp 결과"를 토글한다** — 소스와 같은 격자이므로 화살표를 눌러도 화면 위치가 그대로 유지되어 정렬 여부가 바로 보인다.
- **한 blink 시퀀스 안에 서로 다른 격자를 섞지 않는다 (여러 프레임을 보여줄 때 특히 주의).** pair 구간의 모든 프레임(t0..t1)을 보여주려고 `[실제t0, warp(t0→t1), 실제t1, warp(t0→t2), 실제t2, ...]`처럼 "재구성"과 "실제 촬영 프레임"을 한 시퀀스에 인터리브했다가, warp는 t0 그리드·실제tk는 tk 고유 그리드라서 화살표를 한 칸 넘길 때마다 그리드 자체가 바뀌어 실제 카메라 이동량만큼 화면이 튀는 문제가 생겼다 — projection 계산 자체는 정확했는데(픽셀 이동량이 parallax 공식과 정확히 일치, ground-truth 장애물 포인트클라우드로 depth scale도 교차검증됨), 시퀀스 설계 때문에 "재구성이 실제보다 더 많이 움직인 것처럼" 보이는 착시가 생겨 사용자가 재차 "align 안 됨"으로 오인했다. **해결**: 여러 프레임을 다 보여주고 싶으면 **그리드별로 시퀀스를 분리**한다 — ① 재구성 시퀀스(`[실제rgb(t0), warp(t0→t0+1), warp(t0→t0+2), ..., warp(t0→t1)]`, 전부 t0 그리드, 이게 진짜 정렬 확인용) ② 실제 촬영 시퀀스(`[실제t0, 실제t0+1, ..., 실제t1]`, 참고용, 프레임마다 고유 시점이라 화면이 움직이는 게 정상). 절대 ①②를 하나의 blink에 섞지 않는다(`00_inspect_vln_n1.py`의 `build_pair_step_sequence`가 두 개의 독립된 states 리스트를 반환하는 패턴 참고).
- **"align이 이상해 보인다"는 피드백을 받으면 다음 순서로 검증한다** (실측으로 효과 확인됨): ① 임의 픽셀에서 계산된 이동량을 pinhole parallax 공식(`(lateral_translation/depth)*fx`)으로 손계산해 대략적인 크기가 맞는지 확인(단, 이건 "그럴듯한 크기인지"만 확인할 뿐 **정렬 여부의 증거가 아니다** — 아래 참고) ② depth scale은 photometric reprojection error로 판별하지 말 것 — 회전 성분이 오차를 지배해서 depth를 실제보다 훨씬 크게 잡아도 오차가 단조 감소하는 착시가 있다(실측: 1/300~1/20000까지 스윕해도 반전점을 못 찾음). 대신 **ground-truth 참고 데이터**(이 경우 `meta/pointcloud_obstacle.npy`, world-scale 장애물 포인트클라우드)와 복원한 포인트의 공간적 범위(bounding box)를 비교해 물리적으로 말이 되는지 확인하는 게 더 강한 근거다 ③ **정렬 여부는 육안 판단으로 끝내지 말고 `cv2.matchTemplate`로 정량 측정**한다 — 두 이미지에서 같은 특징(의자 모서리, 그림 등)을 패치로 떼어 실제 어긋난 픽셀 수를 재야 한다. 육안으로는 "거의 맞는 것 같다"고 오판하기 쉽다(실제로 이 방법으로 재검증해서야 20~40px 어긋남을 확정할 수 있었다) ④ 정렬이 실제로 안 맞으면, **transform 합성 순서/방향 버그**부터 의심한다 — `inv(A)@B`와 `inv(B)@A`처럼 두 pose의 순서를 바꾼 조합, 역행렬을 씌운 조합 등을 전부 브루트포스로 대입해 어느 조합이 어긋남을 0에 가깝게 만드는지 직접 측정한다(아래 "M1.0 핵심 버그" 참고 — 실제로 `compute_relative_transform`에 t0/t1이 뒤바뀐 순서 버그가 있었고, "손계산 parallax 공식과 일치"라는 확인만으로는 이 버그를 못 잡았다).
- **원본/참고 자료(예: t0의 원본 rgb·depth)는 메인 흐름을 가리지 않도록 별도 모달(다른 창)에서 본다** — `viz_utils.py`의 `reference_button_html(label, entries)`로 버튼을 만들면, 클릭 시 공용 `<dialog>` 모달에 참고 이미지들을 띄운다(`entries`: `[(label, image_path), ...]`).
- **blink 위젯의 flex/grid 자식 요소에는 반드시 `min-width: 0`을 준다.** ‹ ›로 넘길 때 캡션 텍스트 길이가 state마다 다르면(예: "원본 rgb(t0)" → "cam2world backward-warp — err=33.08 valid_frac=0.80"), flex item의 기본 `min-width:auto`가 긴 텍스트의 content 너비를 강제해 카드 전체가 화면보다 넓어지고, 그 결과 이미지 뷰포트가 화면 밖으로 밀려나 **모바일에서 "이미지 넘기면 크기/위치가 달라 보인다"는 버그**가 생긴다(실제로 겪음 — `viz_utils.py`의 `.blink-label`/`.pair-cols`에 `min-width:0`, `minmax(0,1fr)` 추가로 수정). 이미지 자체는 픽셀 단위로 동일했고 원인은 순전히 CSS였다 — "크기가 다르다"는 사용자 리포트를 받으면 이미지 데이터뿐 아니라 **긴 캡션이 카드 레이아웃을 넓히는지도 의심**할 것.
- **이미 판별이 끝난 "틀린" 후보는 메인 비교 화면에 계속 보여주지 않는다** — pose convention처럼 여러 후보 중 하나를 고르는 검증이 끝나면, 남은 후보의 수치는 요약 표에만 남기고 blink 비교 등 메인 시각화는 **선택된 것 하나만** 보여준다. 기각된 후보를 계속 나란히 보여주면 "왜 틀린 것까지 보여주나"는 혼란만 준다.
- **재투영/reprojection 비교에서 "통계적 판별용 baseline"과 "육안 확인용 baseline"은 분리한다.** 두 후보(또는 두 방법)를 통계적으로 가르려면 baseline(프레임 간격)이 넓어야 차이가 뚜렷해지지만, baseline이 넓을수록 카메라가 실제로 많이 움직여 반사면·가려짐 등으로 **정답 쪽의 오차도 함께 커진다**(00_inspect_vln_n1.py에서 실측: gap=1→err≈5, gap=15→err≈42로 gap에 비례해 매끄럽게 증가 — 버그 아니라 실제 시차/반사 효과). 판별용 넓은 gap을 그대로 리포트에 보여주면 정확한 결과도 "안 맞아 보이는" 착시를 준다 — 실제로 이 착시 때문에 사용자가 재확인을 요청한 적이 있다. **판별은 넓은 gap으로, 리포트의 blink 시각화는 좁은 gap으로 따로 만든다**(`00_inspect_vln_n1.py`의 `--frame_gap`(판별) vs `--visual_gap`(시각화) 분리, `run_pose_convention_check`를 두 번 호출).
- **재투영/warp 결과에서 "데이터 없음" 픽셀은 검은색(0)으로 채우지 않는다** — 검은 배경은 실제 콘텐츠 영역을 원본보다 작아 보이게 하는 착시를 만든다(실측: cam2world 결과의 오른쪽 열 100%, world2cam 결과의 위쪽 행 77%가 검게 채워져 "이미지 크기가 다르다"는 오해를 유발함 — 실제 픽셀 크기는 항상 원본과 동일). 대신 `00_inspect_vln_n1.py`의 `mark_invalid_as_checkerboard(rgb, valid_mask)`처럼 **회색 체크무늬**로 채워 "크기는 같지만 이 영역은 데이터가 없다"는 걸 명확히 한다.
- 각 스크립트는 `save_gallery(out_dir, filename, title, summary_html, body_html)`로 위 컴포넌트들을 조합한 **base64 인라인 임베드 self-contained `report.html`**을 생성한다(원본 JPG/PNG도 같은 폴더에 그대로 저장). 이미지는 PNG보다 **JPEG(quality~90)로 저장해 파일 크기를 줄인다**(정확한 depth 값이 필요한 게 아니라 시각 확인용이므로 손실 압축으로 충분). `summary_html`은 공용 CSS 클래스(`stat-row`/`stat`, `pill good|bad|warn`, `table`)로 핵심 수치를 배지·표로 보여준다.
- **Claude(작업자)는 스크립트 실행 후 그 `report.html`을 Artifact 도구로 publish해 사용자에게 링크를 전달**한다 — 클로드 모바일/웹에서 바로 열어볼 수 있다. 같은 씬/에피소드를 재실행하면 같은 파일 경로이므로 Artifact 재호출 시 같은 링크가 갱신된다.
- **스크립트 실행 후에는 반드시 무엇을 검증했고 결과가 어땠는지 채팅으로 요약 보고**한다(수치+통과/실패 여부). `report.html` 링크만 던지고 끝내지 않는다 — 링크는 근거 자료이지 보고를 대체하지 않는다.
- 참고 구현: `00_inspect_vln_n1.py`의 `run_pose_convention_check`(pair별 rgb_t0/rgb_t1/depth_t0/warped_* 저장)와 `main()`의 리포트 조립부, `viz_utils.py` 전체(`blink_widget_html`/`reference_button_html`/`save_gallery`).

```
00_inspect_vln_n1.py  →  target_schema.json  ─────────────────────────────┐
                                                                            │ (스키마 대조)
01_prepare_scene.py   →  scene_meta.json (기존 fixed.usd 참조, 생성 안 함) │
        │                                                                  │
        ▼                                                                  │
02_build_freemap_esdf.py  →  esdf.npz                                      │
        │                                                                  │
        ▼                                                                  │
03_sample_gt_paths.py  →  paths.json                                       │
        │                                                                  │
        ▼                                                                  │
04_render_obs.py  →  obs/<scene>_<mode>/episode_%06d/{rgb,depth,extrinsic}/ + intrinsic.npy │
        │                                                                  │
        ▼                                                                  │
05_to_lerobot.py  →  data/chunk-000/episode_*.parquet + meta/ + videos/    │
        │                                                                  │
        ▼                                                                  ▼
06_verify.py  ←──────────────────────────────────────────── target_schema.json, esdf.npz, paths.json
```

---

## Stage 1 (M1) 태스크 분해 — VLN-N1 생성 & 검증

| ID | 스크립트 | 산출물 | 수락 기준 |
|---|---|---|---|
| **M1.0** | `00_inspect_vln_n1.py` | `target_schema.json` | ✅ **완료**(2026-08-01) — `info.json` features + parquet 컬럼(dtype/shape) 빠짐없이 기록; depth를 `action`으로 재투영해 rgb와 정합 확인, pose convention(cam2world) 실증 판별 |
| **M1.1** | `01_prepare_scene.py` | `scene_meta/<scene>.json` (**USD 생성 없음** — 이미 존재) | ✅ **완료**(2026-08-02) — 씬 정합 게이트로 축소. ① obj↔`fixed.usd` bounds 일치 ② up-axis Z + extent 자릿수 ③ **GT depth→world가 mesh 표면거리 <1mm** ④ GT 장애물·궤적이 bounds 안 + 카메라가 국소 바닥 위. 실패 시 non-zero exit |
| **M1.2** | `02_build_freemap_esdf.py` | `esdf/<scene>.npz` | ✅ **완료**(2026-08-03) — **3D occupancy만 저장**(2D ESDF는 `h_b` 의존이라 03이 도출). 검증: **GT 궤적 clearance median ∈ (0.30,1.00)·min>0.15**(유일한 독립 검증) + 맵 sanity. `.ply` 겹침은 순환논증이라 판정에 안 씀. negative test 2종 포함 |
| **M1.3** | `03_sample_gt_paths.py` + `03b/03c/03d` | `paths/<scene>.json`, `verify/`, `compare/`, `gridsearch/` | ✅ **C1(reproduce) 완료**(2026-08-04) — GT의 `h_b`/pitch/start/goal을 복사해 재현: A\* 20/20, **같은 루트 40/40**, chamfer 0.103~0.134 m, Fréchet 0.266~0.333 m, 하드 충돌 0. 03b가 기준선 대비 1.45× · 1.27×와 GT가 cubic spline임을 확인. 마지막 루트 불일치는 **격자 원점 위상**이 원인이었다. 03d가 8축 그리드서치로 전역 파라미터 한계를 재확인(기본값 불변). ✅ **C2(random) 완료**(2026-08-04) — `h_b~U(0.25,1.5)`/`pitch~U(0,30°)` + navigable 최대 연결 성분에서 무작위 start/goal(≥2 m), 계획 단계는 reproduce와 완전히 동일 코드 경로. 하드 게이트(r_b 위반/충돌) 위반 후보는 재추출(reject-and-resample). 두 씬 20/20 성공, 하드 충돌 0, 분포 참고범위 내. **버그 수정**(2026-08-04): `navigable`이 "장애물 없음"과 "mesh 데이터가 아예 없는 스캔 밖 허공"을 구분 못 해 무작위 start/goal이 건물 밖으로 샜다(씬2 navigable의 ~40%가 실제로는 무데이터 칸) — `compute_scan_coverage_mask`(전체 높이에서 점유 0인 열 제외)를 추가해 해결, reproduce 회귀(chamfer 0.103/0.134) 불변 확인 |
| **M1.4** | `04_render_obs.py` | `obs/<scene>_<mode>/episode_%06d/{rgb,depth,extrinsic}/`, `intrinsic.npy` | ✅ **완료**(2026-08-04) — Open3D `OffscreenRenderer`. **2-A**(`gt_replay`, GT `action` 그대로 재렌더링 vs 실제 캡처): 두 씬 PASS, depth median 0.0000 m/p95 0.0001 m, edge_corr 0.73~0.76. **2-B**(`reproduce`/`random`, 03 경로 따라 새 pose 합성 `synthesize_action_poses`): 두 씬×두 모드 PASS, mesh-anchor median 0.00000 m, 이동거리 위반 0건. RGB **(270,480,3)**, depth **raw uint16/10000=m**(문서 초안의 480×640·mm-clip·pitch 15°고정은 오기 — 실측대로 수정). **버그 수정**(2026-08-04): ① Open3D `set_projection`이 픽셀 모서리 원점 컨벤션이라 cx/cy에 +0.5px을 더해야 함(depth 1px 오프셋, MAE 0.0074→0.00005) ② 03의 `navigable`이 mesh 데이터가 전혀 없는 "건물 밖 허공"(씬2 navigable의 ~40%)을 걸러내지 못해 C2 랜덤 샘플링이 밖으로 샘 → `compute_scan_coverage_mask` 추가 ③ **rgb 명암 대비 과다**(채널 std가 실측 대비 1.3~1.5배) — vln_n1 GT rgb 자체가 실사진이 아니라 같은 mesh를 다른 렌더러로 렌더링한 **합성 데이터**임을 확인(README가 VLN-N1을 "Synthetic Data"로 명시, GT depth의 mesh-anchor 오차 2.9e-5m은 실제 센서로는 불가능한 정밀도) → "우리 렌더러 vs 그들 렌더러"의 톤매핑 차이로 보고 `ColorGrading` 톤매핑 6종을 실측 스윕, **REINHARD**만 std 비율 0.977(다른 후보는 1.3~1.4x)로 확정 + `INDIRECT_LIGHT_INTENSITY` 재보정(270000→310000, 평균밝기 113.4 vs 실측 113.75) |
| **M1.5** | `05_to_lerobot.py` | `data/chunk-000/episode_*.parquet` + `meta/` + `videos/` | `target_schema.json`과 **필드·dtype·shape 1:1 일치** (`camera_intrinsic` 3×3, `camera_extrinsic` 4×4, `action` 4×4) |
| **M1.6** | `06_verify.py` | `verify_report.json` | ① 포맷 diff 0 + InternNav LeRobot 로더 로드 성공 ② **depth+action 재투영 → rgb geometric 일치**(reprojection error 임계 이하) ③ gt-path collision-free ④ 통계 분포(경로 길이·depth 범위)가 로컬 `vln_n1`과 동류 |

**Stage 1 Definition of Done**: mp3d_n1 씬 1개에서 **약 20 에피소드**의 `rgb·depth·gt-path·action(pose)`이 생성되고, **로컬 `vln_n1`과 포맷 동일 + 품질 검증(재투영 일치·collision-free) 통과**. → 생성 알고리즘이 검증됨.

---

## 스크립트별 입출력 명세 (Stage 1)

각 스크립트는 "무엇을 읽어서 → 무엇을 계산해서 → 무엇을 어떤 이름/shape으로 저장하는지"를 아래처럼 고정한다. 배열 shape 표기는 `(축1, 축2, ...)`, dtype은 numpy 표기를 따른다.

### `00_inspect_vln_n1.py` — GT 데이터 구조 확인 (✅ 구현·검증 완료)

실제 파일: `scripts/dataset_converters/gs_vlnpe/00_inspect_vln_n1.py`. 실행 인터프리터는 `/workspace/isaaclab/_isaac_sim/python.sh`(pandas/pyarrow/cv2/numpy 확인됨; `lerobot`/`av`/`open3d`는 미설치라 의존하지 않음 — parquet은 `pyarrow`, 이미지/영상은 `cv2`만 사용).

- **하는 일**: 로컬 `vln_n1` 데이터셋에서 씬 1개·에피소드 1개를 읽어, parquet 컬럼/메타 파일의 dtype·shape을 빠짐없이 기록하고, **depth를 `action`(pose)으로 재투영해 rgb와 겹쳐보는 photometric sanity check**로 두 가지를 실증 확인한다: ① `action[t]`이 camera-to-world인지 world-to-camera인지, ② depth의 실제 단위 스케일. 이후 모든 스크립트가 따라야 할 GT 스키마를 고정하는 단계.
- **입력**
  - `<scene>/meta/info.json` — features 정의, fps, codebase_version
  - `<scene>/data/chunk-000/episode_{ep:06d}.parquet` — 컬럼: `index`(int64, scalar), `observation.camera_intrinsic`(float32, (3,3)), `observation.camera_extrinsic`(float32, (4,4), 에피소드 내 상수·**에피소드마다 다름** = `h_b`+pitch), `action`(float32, (4,4), **매 프레임 pose**)
  - `<scene>/videos/chunk-000/observation.images.rgb/episode_{ep:06d}_{frame:03d}.jpg` — uint8, (270,480,3) — **1차 rgb 소스**
  - `<scene>/videos/chunk-000/observation.images.depth/episode_{ep:06d}_{frame:03d}.png` — uint16, (270,480) — **1차 depth 소스** (raw / 10000 = meters; 0.1mm 단위. mm 단위로 가정하면 모든 프레임에서 최소 depth가 8.5~13m가 나와 실내 카메라로는 비현실적임을 실측으로 확인)
  - (참고용, 메트릭 계산엔 미사용) `observation.video.{rgb,depth}/episode_{ep:06d}.mp4` — h264 손실 재인코딩, depth mp4는 uint8 그레이스케일이라 원본 depth 값을 복원할 수 없음
- **출력**
  - `scripts/dataset_converters/gs_vlnpe/target_schema.json`(파이프라인 canonical, `--out_dir`) — 실제 구조: `source`(scene/episode/codebase_version), `fps`(선언값 30 vs mp4 실측 10fps), `frame_counts`(parquet/rgb/depth 파일 수 정합), `parquet_schema`(각 컬럼 dtype/shape/상수여부/row0 값), `image_streams`(rgb/depth/rgb_video/depth_video 각각의 path pattern·dtype·shape·주의사항), `pose_convention`(cam2world/world2cam 두 후보의 평균 재투영 오차·valid pixel 비율·판별 결과·margin)
  - `logs/gs-vlnpe/00_inspect_vln_n1/<scene>/episode_{ep:06d}/diagnostics/detection_gapNN/pair_{t0:04d}_{t1:04d}/{rgb_t0,depth_t0,warped_cam2world,warped_world2cam}.jpg`(시각화, `--log_dir`) — `--frame_gap`(기본 15) 기준 판별용 재투영 이미지.
  - `logs/gs-vlnpe/00_inspect_vln_n1/<scene>/episode_{ep:06d}/diagnostics/visual_gapNN/pair_{t0:04d}_{t1:04d}/{rgb_t{idx},warped_t{t0}_to_t{idx}_{convention},depth_t0,topdown_t0,bev_t0,bev_t1}.jpg`(시각화, `--log_dir`) — **pair 구간(t0..t1)의 모든 프레임**에 대해 개별 저장(중간 프레임을 건너뛰지 않음, 아래 "전체 프레임 표시" 참고) + top-down 참고 이미지(단일 프레임, 높이로 색칠) + **BEV(t0 vs t1) 교차검증 이미지**(같은 world grid, 아래 "BEV 교차검증" 참고).
  - `logs/gs-vlnpe/00_inspect_vln_n1/<scene>/episode_{ep:06d}/report.html` — pose_convention 요약(stat pill/표, cam2world·world2cam **둘 다** 수치로 표시, `--frame_gap` 기준, BEV overlap error 요약 포함) + 프레임 쌍마다 **blink comparator 3개**: ① 재구성 vs 원본(`--visual_gap` 기준, t0부터 t1까지 **모든 프레임을 하나도 건너뛰지 않고** `[실제 rgb(t0), warp(t0→t0+1), warp(t0→t0+2), ..., warp(t0→t1)]` — 전부 t0 그리드, `detected_convention` 하나만) ② 실제 촬영 프레임(참고용, 프레임마다 고유 시점) ③ **BEV(t0 vs t1, 같은 world grid)** — 세 위젯 모두 `geometry_utils.check_alignment_fpv`/`check_alignment_bev`가 반환한 `viz_paths`를 그대로 사용. + **reference 버튼**(클릭 시 모달에 colorized depth(t0) + top-down 포인트클라우드 표시). **이미 기각된 후보는 blink에 안 보여준다**. self-contained HTML (`viz_utils.save_gallery`) — Claude가 Artifact로 publish해 링크 제공
- **판별 gap vs 시각 확인 gap을 분리해야 한다 (중요, 실측으로 발견)**: pose convention을 통계적으로 가르려면 baseline이 넓어야(`--frame_gap`, 기본 15) 두 후보 오차 차이가 뚜렷해지지만, **baseline이 넓을수록 카메라가 실제로 많이 움직여 반사면(예: 이 씬의 sunburst 거울)·가려짐 등으로 정답 convention의 오차도 함께 커진다** — gap=1→err≈5, gap=3→err≈16~19(씬에 따라), gap=15→err≈42로 **gap에 비례해 매끄럽게 증가**함을 실측 확인(버그 아님, `t56->57`부터 `t56->71`까지 gap을 늘려가며 직접 확인). 그래서 gap=15 그대로 리포트에 보여주면 사람이 보기엔 "정확한 convention도 안 맞는 것처럼" 보인다(실제로 사용자가 이 착시로 재확인을 요청한 사례). **해결**: `--frame_gap`(넓은 값)으로 판별용 통계만 계산하고, `--visual_gap`(좁은 값, 기본 3)으로 리포트의 blink 이미지를 별도 생성한다. 좁은 gap에서는 같은 씬(t59→t62)에서 거울·캐비닛 위치가 거의 정확히 일치함을 확인.
- **전체 프레임 표시 (실측 피드백으로 추가)**: 처음엔 각 pair당 t0/t1 양끝 2개 state만 보여줬는데, 그 사이 중간 프레임(t0+1, t0+2, ...)이 시각화에 전혀 등장하지 않아 "화살표 넘기면 뭘 보고 있는지 불명확하다"는 피드백을 받았다. `build_pair_step_sequence()`(`00_inspect_vln_n1.py`)가 t0부터 t1까지 **매 프레임을 빠짐없이** `[실제, warp, 실제, warp, ..., 실제]`로 인터리브해 blink states를 만든다 — 화살표를 한 칸씩 넘길 때마다 "재구성 vs 실제"가 바로 이어져 비교되고, 프레임 전체를 훑을 수 있다.
- **카메라 파라미터·depth scale 참고용 시각화 (단일 프레임)**: photometric reprojection과는 독립적인 검증 수단으로, `internnav/model/utils/depth_rgb_to_bev_torch.py`(FPV depth→BEV 변환 유틸)의 방식을 참고해 top-down 포인트클라우드 시각화를 추가했다.
  - `unproject_to_camera_frame`을 그 파일의 `_build_ray_grid`와 동일한 `K_inv @ [u,v,1]` 명시적 구성으로 리팩터(수치는 `np.allclose`로 리팩터 전후 동일함을 확인 — 순수 정리, 회귀 없음).
  - `unproject_to_world_frame`(`geometry_utils.py`) + `render_topdown`(`00_inspect_vln_n1.py`, 참고용 단일 프레임)을 새로 추가: 그 파일의 `_world_to_bev_idx`+`scatter_add_` 아이디어를 numpy로 이식하되, 그 파일은 pitch각+높이만 아는 단순 카메라 모델이라 회전을 따로 재구성하는 반면 우리는 **`action[t]`가 이미 완전한 4×4 camera-to-world**(M1.0에서 실증 확정)라 `transform_points`를 그대로 재사용한다.
  - depth scale·camera parameter가 맞으면 한 프레임의 포인트클라우드는 카메라 위치(십자 마커로 표시)를 꼭짓점으로 하는 부채꼴 모양이어야 한다 — 실제로 그렇게 나타남을 확인(`reference` 모달에서 확인 가능).
  - 시각화 팁: 배경을 검정으로 하면 TURBO 컬러맵의 저채도(어두운 남색) 점이 배경과 구분이 안 된다 — 중간 회색 배경을 쓰고, 작은 캔버스는 `cv2.resize`(nearest-neighbor)로 최소 크기까지 확대해야 리포트에서 형태가 보인다(처음 버전은 109×170px라 거의 안 보였음).
- **BEV 교차검증 (t0 vs t1, `geometry_utils.check_alignment_bev`) — FPV와 독립적인 두 번째 정합 확인 (2026-08-02 추가)**: 위 `render_topdown`은 프레임 1개만 보는 참고용 시각화인 반면, 이건 **t0와 t1을 각자 world 프레임으로 unproject해서 같은 grid 위에 올려놓고 정합을 보는 진짜 검증**이다. FPV(`check_alignment_fpv`)가 "카메라(t1) 프레임 안에서 t0의 depth를 재투영하면 rgb(t0)와 맞는가"를 보는 반면, BEV는 "**월드 프레임에서 t0와 t1이 같은 물리적 위치에 합의하는가**"를 본다 — 서로 다른 각도의 독립적 교차검증.
  - `compute_shared_grid`(`geometry_utils.py`): t0/t1 두 포인트셋을 모두 포괄하는 공통 `(x_min, y_min, w, h, cell_m)` grid를 계산.
  - `rasterize_rgb_to_grid`: `depth_rgb_to_bev_torch.py`의 `depth_rgb_to_bev`(scatter_add로 색 합/카운트 누적 후 나누기 = cell당 평균색)를 numpy(`np.add.at`)로 이식. z-buffer 없이 단순 평균이라 겹치는 층이 섞이지만, "같은 world 위치에서 t0/t1이 비슷한 색을 내는가"만 보면 되므로 충분하다.
  - 겹치는 영역(둘 다 데이터가 있는 cell, `occ_t0 & occ_t1`)에서만 photometric error 계산(`compute_photometric_error` 재사용) — `metric_name='bev_overlap_error'`, `coverage_frac`은 전체 grid 대비 겹치는 cell 비율(두 시점이 다른 카메라 위치에서 봤으므로 FPV의 valid_frac보다 낮은 게 정상 — 실측 예: coverage≈0.17).
  - 시각화: BEV(t0)와 BEV(t1)를 같은 grid 위에서 blink 비교 — 정렬이 맞으면 화살표를 넘겨도 벽/바닥의 윤곽·위치가 같은 자리에 있고 음영(조명 방향에 따른 색상 차이)만 달라 보인다(`cv2.matchTemplate`으로 0px 어긋남까지 정량 확인, 셀 크기 `--bev_cell_m` 기본 0.05m).
    - **주의**: `check_alignment_bev`는 한동안 "t0 anchor 경유" 우회를 했는데 그건 축 flip 누락의 증상을
    가린 것이었다. 지금은 FPV와 동일한 상대 변환을 쓴다. 04/06에서 여러 프레임을 하나의 map으로
    합칠 때 anchor 우회는 필요 없다.

- **실행 결과 (2026-08-01)**: 씬 `17DRP5sb8fy`(239 프레임)·`s8pcmisQ38h`(145 프레임) 둘 다에서 프레임 수 정합(parquet=rgb=depth), `camera_intrinsic` 상수 확인, `camera_extrinsic`은 에피소드 내 상수(에피소드 간에는 다름 — 위 정정 참고). 인접 프레임(gap=1)은 baseline이 짧아 두 pose 후보 오차 차이가 거의 없었음(margin<3%) → `--frame_gap 15`로 넓혀 재실행하니 `cam2world`가 뚜렷이 우세. → **`action[t]`은 camera-to-world pose로 확정.**
- **버그 이력 요약**: `compute_relative_transform`은 2단계로 정정됐다(순서 → 축 flip). 최종형은
  `inv(action_to_c2w(t1)) @ action_to_c2w(t0)`이고, `python geometry_utils.py`의 5개 게이트가 회귀를 막는다.
  왜 두 번이나 오진했는지는 `understanding_gs_vlnpe_fpv_bev_geometry.md`의 "과거 오진 목록" 참고.

- 이후 리포트의 육안 확인용 blink는 `--visual_gap 3`로 별도 생성했고, 버그 수정 후 실제로 완벽히 정렬됨을 `cv2.matchTemplate`으로 재확인.
- **검증**
  - `target_schema.json`의 `frame_counts.consistent`가 true인지, `parquet_schema`의 각 필드가 실측 dtype/shape과 일치하는지 확인.
  - `pose_convention.confidence_margin`이 충분히 크고(≥5% 권장), `report.html`에서 (좁은 gap으로 만들어진) blink comparator를 ‹ ›로 토글해 화면이 안 튀고 잘 겹치는지 육안 확인.
  - 인접 프레임만으로 판별이 애매하면 `--frame_gap`을 늘려 재실행하되, 리포트의 `--visual_gap`은 그대로 좁게 유지한다(판별 근거와 육안 확인 목적이 다르므로 같이 늘리지 않는다).

### `01_prepare_scene.py` — mesh 씬 로드 (Stage1) / GS+mesh 정합 (Stage3)

- **하는 일**: **"확인하고 기록"만 한다 — 좌표 정규화도 USD 생성도 하지 않는다.** 원래 계획은 mesh의 up-axis/scale을 정규화하고 `scene.usd`를 만드는 것이었으나, 실측해 보니 (a) mesh가 **이미 Z-up·1.0=1m·GT와 동일 원점**이고 (b) **GT USD가 이미 존재**해서 둘 다 불필요했다(아래 실측 근거). 그래서 이 스크립트는 mesh·USD·GT 세 자산이 같은 프레임인지 정량 검증하고 그 결과를 `scene_meta.json`으로 고정하는 역할만 한다. Stage3에서는 이 스크립트만 GS splat과 mesh를 registration(정합)하는 모드로 바뀐다(아래 Stage 3 절 참고).

- **실측 근거 (2026-08-02)**

  | | X | Y | Z |
  |---|---|---|---|
  | obj mesh (trimesh) | -11.593 ~ 4.757 | -2.887 ~ 5.392 | -0.128 ~ 2.679 |
  | `fixed.usd` (pxr BBoxCache) | -11.593 ~ 4.757 | -2.887 ~ 5.392 | -0.128 ~ 2.679 |
  | GT `pointcloud_obstacle.npy` | -11.366 ~ 4.628 | -2.800 ~ 5.358 | -0.098 ~ 0.086 (바닥 슬라이스) |
  | GT 카메라 궤적 `action[:, :3, 3]` | -2.628 ~ 2.238 | -0.118 ~ 3.751 | 0.686 (상수) |

  Z extent 2.81m = 실내 층고 → 스케일 1.0=1m 확인. 카메라 높이 0.686m도 바닥(z≈0) 기준으로 타당.
  결정적으로 **GT depth를 world로 올리면 mesh 표면에 median 0.00003 m로 얹힌다** — mesh와 GT가 같은 프레임임을 mm 이하로 확증(M1.0 절 참고).

- **기존 USD (생성 불필요)**: `data/scene_data/mp3d_pe/<scene>/matterport_mesh/<hash>/`

  | 파일 | upAxis | metersPerUnit | 비고 |
  |---|---|---|---|
  | `fixed.usd` | Z | 0.01 | **InternNav가 실제 로드하는 것** |
  | `fixed_docker.usd` | Z | 0.01 | 컨테이너용 |
  | `isaacsim_<hash>.usd` | Z | 1.0 | metric 버전 |
  | `isaacsim_<hash>_non_metric.usd` | Y | 0.01 | Y-up |

  선택 로직은 `internnav/env/utils/episode_loader/generate_episode.py:15`
  (`'fixed_docker.usd' if is_in_container() else 'fixed.usd'`), 로드 시 `scene_scale=(1,1,1)` (:76,79).
  세 파일 모두 bounds 수치는 동일. `fixed.usd`의 `metersPerUnit=0.01` 선언은 좌표값과 불일치하지만
  InternNav가 이미 `scene_scale=(1,1,1)`로 쓰고 있으므로 **그 관례를 그대로 따른다**(01은 기록만).
  04에서 Isaac 렌더러를 쓸 경우 이 스케일 선언을 재확인할 것.

- **입력**
  - `data/scene_data/mp3d_n1/<scene>/matterport_mesh/*/*.obj` (Matterport3D raw mesh, `geometry_utils.find_scene_mesh`)
  - `data/scene_data/mp3d_pe/<scene>/matterport_mesh/*/fixed.usd` (존재 확인·메타 읽기만, `pxr`)
  - GT: `<data_root>/<scene>/meta/pointcloud_obstacle.npy`, `data/chunk-000/episode_{ep:06d}.parquet`, `videos/chunk-000/observation.images.depth/*.png`
- **출력**
  - `scene_meta/<scene>.json` (`--out_dir`, 파이프라인 canonical) — **씬별 파일**이다(단일 파일로 두면
    씬을 바꿔 돌릴 때 덮어쓴다). `scene_id`, `mesh_path`, `usd_path`/`usd_docker_path`,
    `bounds_min`/`bounds_max`, `up_axis`, `scale`, `usd_stage`, `floor_z_under_camera`, `gt_camera_z`,
    `gt_camera_height_m`, `pose_convention`, `frame_alignment`, `gt_extent_check`, `bounds_check`, `passed`
    → 02가 `bounds_min/max`·`mesh_path`를, 03이 `floor_z_under_camera`를, 04가 `usd_path`·카메라 높이를 읽는다.
  - `logs/gs-vlnpe/01_prepare_scene/<scene>/report.html` + 원본 이미지 (`--log_dir`, `viz_utils.save_gallery`)
- **검증**
  - obj bounds ↔ USD bounds(`pxr.UsdGeom.BBoxCache`) 일치 (tol 1e-3).
  - up-axis Z + 씬 extent가 상식적 자릿수(1~100m). **층고 같은 좁은 범위로 잡으면 안 된다** —
    Matterport에는 다층 건물이 섞여 있다(실측: `s8pcmisQ38h`는 Z extent 12.24m). 여기서 잡는 건
    100배/0.01배 스케일 오류뿐이고, 정밀한 스케일 근거는 아래 mesh 표면거리다.
  - **`geometry_utils.check_against_scene_mesh`로 GT depth→world→mesh 표면거리 median < 1mm** (M1.0의 회귀 게이트를 씬 단위로 재확인). 이게 M1.1의 핵심 수락 기준이다.
  - GT `pointcloud_obstacle.npy`·카메라 궤적이 mesh bounds 안 + 카메라가 **국소 바닥** 위 0.2~2.5m.
    바닥은 `mesh.bounds[0][2]`(전역 최소 z)가 아니라 **카메라 xy 주변 반경 0.4m 안에서 카메라보다
    낮은 정점의 최대 z**로 잡는다 — 다층 건물에서 전역 최소를 쓰면 2층 에피소드의 카메라 높이가
    4m대로 나온다(실측으로 겪음).
  - `scene_meta.json.scene_id`가 `vln_n1` traj 데이터의 씬 폴더명과 문자열 그대로 일치하는지 assert.
  - **Negative test**: `--pose_convention cam2world`(틀린 규약)로 돌려 `passed=false` + non-zero exit이
    실제로 나오는지 확인 — 게이트가 항상 통과만 내는 무의미한 체크가 아님을 보장한다.
    실측: `17DRP5sb8fy` median 0.42/0.20/0.23m, `s8pcmisQ38h`(world2cam) 4.84/8.44/5.03m로 전부 FAIL.
  - 시각화: mesh floorplan(높이 0.15~2.0m 단면 실루엣) 위에 blink 4-state — ① mesh만 ② +GT 장애물 점군
    ③ +GT 카메라 궤적 ④ +depth→world 점. 전부 같은 grid라 화살표를 넘겨도 화면이 안 튄다.
  - **실행 결과(2026-08-02)**: `17DRP5sb8fy` 표면거리 median 0.000029~0.000032m,
    `s8pcmisQ38h` 0.000028~0.000030m — 둘 다 PASS(exit 0). 궤적이 벽을 통과하지 않고 복도를 따라가는 것,
    depth 복원점이 의자·벽 윤곽에 얹히는 것을 floorplan에서 육안 확인.

### 논문 원문 — NavDP §3 Trajectory Generation (M1.2~M1.3의 유일한 근거)

> **Trajectory Generation.**
> To generate collision-free robot navigation trajectories, we first convert the scene meshes into
> a voxel map with a voxel size of 0.05m to estimate the Euclidean Signed Distance Field (ESDF) of
> the navigable areas. Navigable areas are defined as voxel elements with z-axis coordinates below
> the threshold h_nav, while obstacle areas are defined as voxel elements with z-axis coordinates
> exceeding the threshold h_obs. The thresholds h_nav and h_obs vary across scenes and depend on
> the robot height h_b. Voxels with distance values lower than the robot radius r_b are truncated
> to prevent collisions. The ESDF map of the navigable area is downsampled to 0.2m resolution to
> facilitate efficient A\* path planning. Navigation start and target points are selected randomly
> on the navigable area, and the A\* algorithm generates a planned path
> τ\* = [(x₀, y₀), (x₁, y₁), (x₂, y₂), . . . , (x_k, y_k)]. For each waypoint (x_n, y_n), a greedy
> search is performed in a local area of the original ESDF map to refine the position by maximizing
> the distance to nearby obstacles. This refinement process shifts waypoints further from obstacles.
> Finally, the refined waypoints are smoothed into a continuous navigation trajectory using cubic
> spline interpolation. Examples of the generated trajectories and global ESDF are shown in Appendix.

**7단계 → 우리 구현 대응** (계산은 전부 `esdf_utils.py`, 스크립트는 입출력·검증·리포트만)

| # | 논문 문장 | 우리 구현 | 검증 |
|---|---|---|---|
| 1 | voxel size 0.05 m 로 mesh → voxel map | `VOXEL_SIZE_M`, `voxelize_surface`(표면 3M점 샘플) | 02 맵 sanity |
| 2 | navigable = z < `h_nav`, obstacle = z > `h_obs`, 둘 다 **`h_b` 의존** | `derive_obstacle_2d` — 밴드 `(floor+h_nav, floor+h_b]` | 02 GT 궤적 clearance |
| 3 | ESDF 추정 | `compute_esdf_2d`(`distance_transform_edt`) | 03 게이트 ② |
| 4 | distance < `r_b` 인 voxel **truncate** | `truncate_navigable` | 03 게이트 ①④ |
| 5 | 0.2 m 다운샘플 → A\* → τ\* | `downsample_navigable`(`any()`) → `astar`(8-이웃) | 03 ④ `cost_ratio` |
| 6 | 각 waypoint를 **local area**에서 greedy search로 장애물 거리 최대화 | `greedy_refine(radius=0.15)` | 03b ⑤ 반경 역추정 |
| 7 | refined waypoint를 **cubic spline**으로 스무딩 | `smooth_cubic_spline` (+ 논문에 없는 `thin_waypoints`) | 03b ⑥ knot 복원 |

**원문만 봐서는 갈리는 해석 3건** — 전부 실측으로 결론냈다:

- **`truncate`는 navigable 집합에서 빼는 것**이지 값을 자르는 것이 아니다. 값 clipping으로 읽으면
  `r_b` 미만 셀이 navigable에 남아 A\*가 벽에 붙어 지나간다.
- **`h_obs` = `h_b`**(밴드 상한). 글자대로 "`h_obs` 초과가 장애물"로 읽으면 실내 천장 때문에 씬 전체가
  장애물이 되어 **경로 생성 0/20**이었다. 로봇을 높이 `h_b` 원통으로 보면 충돌 구간이
  `(h_nav, h_b]` 밴드라는 해석만 성립한다 — 이게 `h_obs`가 `h_b`에 의존하는 이유이기도 하다.
- **"start/target을 navigable에서 무작위 선택"** — 그래서 릴리스된 GT와 점 단위로 비교할 수 있는 것은
  `--mode reproduce`(start/goal/`h_b`/pitch를 GT에서 복사)뿐이다. 논문 그대로인 `--mode random`(C2)의
  검증 기준은 일치가 아니라 **분포 일치**다.

논문이 **값을 주지 않은 파라미터**는 **voxel 격자 원점**, `h_nav`/`h_obs`(씬·`h_b` 의존이라고만 함),
`r_b`, 6단계의 local area 크기·해석, 7단계 spline의 knot 간격·배치, A\* 연결성·tie-break이다.
이들은 GT에서 역으로 읽어낸다 (M1.3 아래 03b 참고).

> **원칙: 논문이 숫자로 준 값(voxel 0.05 m, A\* 0.2 m)은 고정하고, 안 준 값만 조정한다.**
> 이 원칙이 "저자가 하지 않았을 최적화"를 자동으로 배제한다. 실측으로 A\* 격자를 0.1 m로 줄이면
> 모든 지표가 개선되고 실행시간도 같지만(이 규모에서 "efficiency" 근거가 무의미), **채택하지 않는다.**
> `--astar_cell_m` 옵션으로만 남긴다.

### `02_build_freemap_esdf.py` — occupancy/ESDF 생성 (✅ 완료)

- **하는 일**: 씬 mesh를 0.05 m voxel occupancy로 만들어 저장하고, 그 맵이 정상인지 검증한다.
  계산은 `esdf_utils.py`(신규 공용 모듈)가 하고 02는 파일 입출력·검증·리포트만 담당한다.
- **입력**: `scene_meta/<scene>.json`(01) → `mesh_path` / `usd_path` / `floor_z`.
  `--geometry {obj,usd,both}` 기본 **obj** — 논문이 raw scene mesh를 썼고 GT도 그 mesh에서 렌더됐다.
  USD는 표면 동일 + collision Plane 2장인데 최종 navigable 차이가 0.4%(103셀)뿐이다.
  Stage 2(Isaac)에서 `usd`로 바꾼다.
- **격자 원점 = 월드 정렬** (`--grid_align`, 기본 `ASTAR_CELL_M`) — `origin = floor(mesh.bounds[0]/0.2)*0.2`.
  논문은 원점을 언급하지 않는데 `mesh.bounds[0]`을 그대로 쓰면 격자 위상이 **씬마다 제멋대로**가 되고
  (실측: 월드 0.2 m 격자선까지 y로 0.087 / 0.082 m), A\*가 내는 셀중심 좌표가 그만큼 옮겨진다.
  월드 정렬은 **자유 파라미터 0개의 씬 독립 규칙**이며 `s8pcmisQ38h` ep17의 루트 불일치를 해결했다
  (같은 루트 39/40 → **40/40**, chamfer 1.036 → 0.140, 위상 면적 1.745 → 0.000 m²).
  `--grid_align 0`으로 이전 동작 복원. 에피소드별 개선은 10/20·12/20으로 동전던지기이므로
  **채택 근거는 원칙과 40/40이지 chamfer 개선폭이 아니다.**
- **출력**: `esdf/<scene>.npz` — `occupancy`(bool 3D), `origin`, `voxel_size`/`h_nav`/`r_b`/`floor_z`,
  `ref_h_b`, `nav_mask_ref`/`esdf_ref`(참조 2D). 같은 이름 `.json`에 메타·검증 결과.
- **핵심 설계 — 2D ESDF는 저장하지 않는다**: `h_obs ≈ h_b`(로봇 키)라 에피소드마다 달라진다.
  03이 occupancy에서 매번 도출한다(328×166에서 수 ms).
- **`h_nav`/`h_obs`**: 로봇을 높이 `h_b` 원통으로 보면 충돌 구간은 바닥 위 `(h_nav, h_obs]` 밴드.
  `h_nav` 아래는 밟고 넘는 지면, `h_obs` 위는 밑으로 통과. **`truncate`는 값 clipping이 아니라
  navigable 집합에서 제외**다(논문 원문 오독 주의).
- **검증**
  - **GT 궤적 clearance**(유일한 독립 검증) — 에피소드마다 그 `h_b`로 맵을 만들어 궤적 clearance를 잰다.
    **범위 판정**: median ∈ (0.30, 1.00), min > 0.15. 6개 씬 실측이 median 0.461~0.680으로 씬마다
    다르므로 단일 기준선을 쓰면 안 된다(실제로 그렇게 잡았다가 2번째 씬을 오판했다).
  - 맵 sanity — 장애물 셀 `esdf==0`, 자유공간 `esdf>0`, occupancy가 bounds 안
  - `.ply` obstacle 겹침 — **판정에 쓰지 않는다**. 그 점들은 mesh 표면 위 0.0000 m라 순환논증이고
    바닥 슬라이스라 우리 밴드와 높이가 달라 겹침률이 낮은 게 정상이다.
  - **Negative test**: `--h_nav -0.1`(바닥을 장애물에 포함 → clearance 0으로 붕괴, exit 1),
    `--h_nav 1.0`(로봇 키보다 큼 → 설정 불가로 사전 차단)
- **실행 결과(2026-08-03)**: `17DRP5sb8fy` 328×166×57 점유 7.4% clearance 0.495/0.224 PASS,
  `s8pcmisQ38h` 477×200×245 점유 2.9% clearance 0.680/0.269 PASS.

### `03_sample_gt_paths.py` — GT 경로 샘플링 (✅ C1 reproduce / C2 random 둘 다 완료)

- **하는 일**: 논문 §3의 4~7단계 — navigable 다운샘플(0.2 m) → A\* → waypoint greedy refine
  (0.05 m 원본 ESDF) → cubic spline. 계산은 `esdf_utils.py`가 하고 03은 입출력·검증·리포트만.
- **두 모드**
  - `reproduce`(기본) — 에피소드별 `h_b`·pitch·start·goal을 **전부 GT에서 복사**해 정답지와 같은
    경로가 나오는지 본다. 분포 비교보다 날카로운 검증이다.
  - `random`(C2, 2026-08-04 완료) — 논문처럼 무작위 (`h_b ~ U(0.25,1.5)`, `pitch ~ U(0,30°)`).
    `nav_coarse`의 최대 연결 성분(`scipy.ndimage.label`)에서 서로 2 m 이상 떨어진 두 셀을 start/goal로
    뽑고, 이후 계획 단계(`plan_episode`)는 reproduce와 **완전히 동일한 코드 경로**를 탄다(분기는
    start/goal/h_b/pitch를 정하는 지점에만 guard clause로 들어간다). GT가 없어 chamfer로 못 재므로
    reproduce에서 실측한 GT 통계(길이 median 6.03 m, clearance median 0.46~0.68 m, 길이/직선 비율
    median 1.08)를 넓은 범위의 참고값으로만 쓴다(스모크 테스트, 하드 게이트 아님). **하드 게이트
    (r_b 위반/충돌)는 reproduce와 동일하게 유지** — 무작위 start/goal은 GT의 특정 경로보다 훨씬 자주
    낙관적 다운샘플의 병목(좁은 통로)을 지나가서 위반 후보가 나오는데(씬1 20개 중 10건, 씬2 2건),
    이를 실패로 두지 않고 **재추출(reject-and-resample, 최대 30회/슬롯)**해 최종 배치는 항상 두 하드
    게이트를 만족하게 했다. 결과: 두 씬 20/20 성공, 하드 충돌 0, 분포 참고범위 내. 산출물은
    `paths/<scene>_random.json`(reproduce의 `<scene>.json`과 별도 파일).
- **결정한 파라미터 (실측 근거)**
  - **`h_nav = 0.10`** (0.15 → 변경) — 논문의 "`h_nav`는 로봇 키에 의존" 서술을 안 따르던 고정값이
    **루트 불일치 3/20의 원인**이었다. 0.15면 낮은 가구·문턱이 장애물에서 빠져 GT에 없는 지름길이
    열린다. 0.05~0.12에서 20/20, 0.15에서 17/20(전환점 0.12~0.15). `--h_nav_ratio`로 비례형도 지원.
  - **`refine_radius = 0.10`** (0.15 → 변경) — 논문은 "a local area"라고만 한다. 처음엔 chamfer
    스윕으로 0.15를 골랐는데 그건 **지표를 보고 지표에 맞춘 순환논증**이었다. 03b가 GT에서 역추정한
    R\*는 **0.05**다(두 씬 일치). 0.05를 그대로 쓰면 낙관적 다운샘플이 통과시킨 셀을 되끌어올리지
    못해 `r_b`를 6/20 위반하므로 — **논문의 `r_b` 제약이 반경의 하한을 만든다** — 0.10을 쓴다.
    0.15 → 0.10에서 Fréchet가 두 씬 모두 개선(0.252→0.233, 0.383→0.366)되고 씬2 chamfer가 13%
    줄며, 무엇보다 ②(GT와의 clearance 차)가 두 씬 모두 0에 가까워진다(+0.020/+0.077 → −0.014/+0.035).
    키우면 waypoint가 방 중앙으로 끌려가 GT보다 벽에서 멀어진다(1.00→0.270, 3.00→1.061).
  - `r_b`=0.25 / 다운샘플 `any` / A\* 8-이웃 — 18조합 탐색에서 **현재 설정이 Pareto 최적**임을 확인.
    4-이웃은 Fréchet 0.295→0.414(계단 경로), center/mean 다운샘플은 계획 실패가 늘어난다.
  - **ESDF 맵 자체를 5건 실측했고 전부 "고칠 것 없음"**(2026-08-03): ① `floor_z`가 씬 전체에 하나라는
    가정 → 20 에피소드 **완전 동일**(표준편차 0.0000) ② `floor_z = cam_z − h_b`의 정확도 → 서 있을 수
    있는 면과의 차 +0.039 / **−0.024 m**(반셀 안) ③ 표면 3M 샘플의 구멍 → 3×3 closing이 navigable을
    **0셀** 바꿈 ④ `r_b` 스윕 → 0.25가 최적(0.15는 씬1 하드충돌 88, 0.30은 GT가 맵 밖 4~6/20)
    ⑤ 다운샘플 `majority` → 씬1만 개선(혼합, 기각). 상세는
    `.claude/memory/260803_gs_vlnpe_03_reproduce_result.md`.
    **씬1과 씬2가 서로 반대로 움직인다** — 전역 파라미터로는 더 줄일 수 없고, 2씬 40에피소드는
    판별 표본이 못 된다. 다음은 파라미터가 아니라 **씬 수를 늘리는 것**이다.
  - `--downsample_mode {any,majority,all}` 기본 **`any`**(낙관적) — 보수적 `all()`은 문틈을 지워 GT start/goal이
    0.2 m 격자에서 연결되지 않았다(같은 성분 2/12 → `any()` 12/12). 논문도 이 다운샘플을
    "to facilitate efficient A\*"라고만 한다. 안전은 0.05 m `r_b` truncate와 refinement가 담당하고,
    낙관적 A\*의 위험은 `check_path_navigable`(점 + **점 사이 선분**)로 잡는다.
  - `--smooth {cubic,bezier}` 기본 **cubic**(논문이 *"cubic spline interpolation"* 명시).
    `esdf_utils.SMOOTHERS` registry로 고르며 **기준선·03b·03c가 모두 같은 registry를 쓴다** —
    한 곳이 하드코딩하면 기준선이 다른 스무딩으로 계산돼 배수가 무의미해지고 F1이 깨진다(실제로 겪음).
    예외는 `recover_spline_knots`(⑥) — "GT가 cubic spline인가" 검정이라 **cubic 고정**이다.
    실측 (씬1 / 씬2):

    | 스무딩 | chamfer | Fréchet | ①회전 | ③헤딩 | traj min clr | 길이비 |
    |---|---|---|---|---|---|---|
    | **cubic** | **0.103 / 0.134** | **0.266** / 0.333 | 1.01 / 1.30 | 9.0 / 11.5 | 0.100 / **0.200** | **0.993 / 1.010** |
    | bezier | 0.112 / **0.133** | 0.274 / **0.322** | **0.67 / 0.86** | **8.5 / 8.8** | **0.141** / 0.180 | 0.955 / 0.985 |

    bezier는 convex hull 안에 머물며 코너를 자른다 → **오버슈트가 없어 clearance가 좋고 헤딩이 낫지만**,
    GT보다 **짧아지고**(길이비 0.955) **GT의 실제 회전까지 씻어낸다**(①이 1 미만 = 과도 스무딩).
    좁은 씬·큰 `r_b`·04의 방향 정합이 중요할 때 쓸 옵션이다.
  - **`--waypoint_spacing_m = 0.8`** — 스무딩 **전에** waypoint를 호길이 등간격으로 솎는다
    (`thin_waypoints`). A\*의 45° 격자 꺾임이 spline에 남아 GT보다 **2.8배 지그재그**였던 것을
    GT 수준(1.03배)으로 내린다. `r_b` 제약은 솎기 전 `wp_refined`에 걸리므로 안전 판정은 무관.
    Douglas-Peucker는 장애물을 안 보고 코너를 관통해 잘라내 **기각**(min clearance 0.000).
- **출력**: `paths/<scene>.json` — 에피소드별 `h_b`, `pitch_deg`, `start`, `goal`,
  `waypoints_astar`, `waypoints_refined`, `waypoints_thinned`, `trajectory`, `check_refined`,
  `check_trajectory`, `gt_compare{chamfer_m, frechet_m, len_ratio, same_route, turn_per_m_deg,
  clearance{...}, heading{...}}` + `stats`
- **검증**
  - **`r_b` 하드 제약은 refined waypoint에만** 걸고 스무딩 궤적은 측정만 한다 — GT 궤적도 13%(7/52)가
    min clearance < `r_b`이므로(cubic spline 오버슈트) 궤적에 걸면 GT조차 탈락한다.
  - refine이 clearance를 실제로 올렸는지 확인(논문 6단계의 목적).
  - **충돌**은 두 종류를 구분한다. **하드 충돌**(clearance = 0, 질점도 못 지남)은 맵/플래너 버그이므로
    **0을 게이트**로 걸고, **몸통 침범**(< `r_b`)은 GT도 위반하므로 측정만 한다. 점 + 점 사이
    선분을 0.01 m(= cell/5) 간격으로 찍는다 — 셀 크기로 찍으면 좁은 틈을 스치는 구간을 놓친다.
    실측 하드 충돌 **0/28,034 샘플**(두 씬 40 에피소드), 몸통 침범 3/20·0/20 에피소드
    (GT는 5/20·4/20). 원본 mesh를 3M점 샘플해 KD-tree로 **격자 없이** 다시 재도 하드 충돌 0이고
    ESDF와의 차이가 ±0.04 m(= voxel 양자화) 안이며 부호가 뒤집힌 에피소드가 없다 —
    자기참조(같은 맵으로 만들고 검사)를 피한 독립 검증이다.
  - 재현 정확도는 **chamfer**(순서 무시)와 **discrete Fréchet**(순서 고려)로. "같은 루트" 판정은
    Fréchet < 문 폭 0.8 m.
  - **보조 지표 ①②③** — 주 지표는 위치만 보므로 모양·거리·방향을 따로 잰다:
    ① **스무드니스**(미터당 회전각, GT 대비 배율 **0.5~2.0 게이트 — 양쪽이다**. 상한은 A\* 격자
    꺾임이 spline에 새는 것을, 하한은 스무딩이 GT의 실제 회전까지 씻어내는 것을 잡는다. bezier가
    0.67/0.86을 내면서 하한이 필요하다는 게 드러났다) ② **clearance 프로파일**
    (같은 ESDF에서 median 차, 양수면 refine 과다) ③ **방향 정합**(호길이 정렬 후 헤딩 차 median —
    04가 이 경로로 카메라를 놓으므로 위치보다 영향이 크다). ②③은 목표값을 못 박으므로 측정만 한다.
  - **모든 지표는 기준선 ⓒ와 함께 읽는다** — 아래 "기준선" 절 참고. 절대값만 보면 오독한다.
  - **Negative test**: `--r_b 1.0` → start/goal이 통행 불가로 20/20 실패, exit 1.
    `--waypoint_spacing_m 0.0` → chamfer 0.094·같은루트 20/20으로 **주 지표는 전부 통과**하는데
    ①이 2.80배를 잡아 FAIL. ①이 없으면 이 회귀가 개선으로 보인다.
    `--smooth bezier --waypoint_spacing_m 3.0` → ① **0.18**(과도 스무딩)로 하한 게이트가 잡아 FAIL.
- **실행 결과(2026-08-03)**

  | 씬 | A\* 성공 | 같은 루트 | chamfer | Fréchet | ① 회전배율 | ② clr 차 | ③ 헤딩 | ④ cost비 | ⑦ 위상 0인 ep |
  |---|---|---|---|---|---|---|---|---|---|
  | 17DRP5sb8fy | 20/20 | **20/20** | 0.103 m | 0.266 m | 1.01× | +0.000 m | 9.0° | 1.02 | 20/20 |
  | s8pcmisQ38h | 20/20 | **20/20** | 0.134 m | 0.333 m | 1.30× | +0.034 m | 11.5° | 1.02 | 20/20 |

  **같은 루트 40/40**이 됐다. 이전에 남았던 `s8pcmisQ38h` ep17은 ③이 42.6°, ⑦이 1.745 m²로 지목하고
  ④가 cost_ratio **1.014 = 동점**으로 진단했는데, 원인은 **격자 원점 위상**이었다(위 02 절 참고) —
  동점 상황에서 위상이 어느 루트가 이길지를 정하고 있었다. 월드 정렬 후 ⑦이 40/40에서 정확히 0.000이다.

#### 기준선 — 지표 절대값은 의미가 없다

**"경로가 같다"에는 세 가지 의미가 있고 값이 다르다.** 두 씬 20 에피소드 median:

| 지표 | ⓐ 완전 동일 | ⓑ 격자+스무딩 | **ⓒ 전 파이프라인** | 실측 | 실측/ⓒ |
|---|---|---|---|---|---|
| chamfer | 0.0000 | 0.052 / 0.062 | **0.080 / 0.109** | 0.107 / 0.154 | **1.19× / 1.54×** |
| Fréchet | 0.0000 | 0.122 / 0.130 | **0.179 / 0.242** | 0.233 / 0.366 | 1.18× / 1.70× |
| ① 회전배율 | 1.0000 | 1.139 / 1.417 | 1.081 / **1.574** | 1.118 / 1.232 | — |
| ② clr 차 | 0.0000 | 0.003 / 0.000 | +0.054 / +0.077 | −0.014 / +0.035 | — |
| ③ 헤딩 | 0.0000 | 6.19° / 6.29° | 6.13° / 8.49° | 8.03° / 10.35° | — |
| ④ cost비 | **1.016 / 1.028** | — | — | 1.024 / 1.023 | — |
| ⑦ 위상 | 0.0000 | 0.0000 | 0.0000 | 0.0000 (39/40) | — |

- **ⓐ** 같은 배열끼리 = 지표 정의상의 값. **ⓑ** GT 루트를 0.2 m 격자에 얹고 → 솎기 → spline.
  **ⓒ** ⓑ + `greedy_refine` = **A\*가 GT 루트를 정확히 골랐을 때 우리 파이프라인이 내는 값**.
- **기준선이 GT에서 벌어지는 이유** — 기준선은 GT가 *아니라* GT의 **루트**를 우리 파이프라인 어휘로
  다시 쓴 것이다. 단계별 chamfer 누적(씬1 / 씬2):

  | 단계 | chamfer | 증가분 | 원인 |
  |---|---|---|---|
  | s0 GT (0.05 m 리샘플만) | 0.011 / 0.011 | — | 점 분포 |
  | s1 **0.2 m 셀중심 스냅** | 0.045 / 0.045 | **+0.035 / +0.034** | A\*가 낼 수 있는 좌표가 셀중심뿐 |
  | s2 + 양끝 GT 고정 | 0.044 / 0.043 | −0.001 / −0.002 | 03의 보정 |
  | s3 + **`greedy_refine`** | 0.072 / 0.101 | **+0.028 / +0.058** | 벽에서 밀어낸다(GT와 무관) |
  | s4 + `thin_waypoints` | 0.069 / 0.093 | −0.003 / −0.008 | 격자 꺾임 제거 → 오히려 개선 |
  | s5 + cubic spline = **ⓒ** | **0.079 / 0.108** | +0.011 / +0.015 | 점 사이를 곡선으로 이음 |

  격자 스냅(+0.034, 두 씬 동일 = 순수 격자 성질)과 refine(공간이 넓은 씬2에서 2배)이 지배적이다.
  이론값과도 맞는다 — 0.2 m 셀 안 균일분포 점에서 셀중심까지 **평균 0.077 m, 최대 0.141 m**(반대각선).
- **ⓒ를 기준선으로 쓴다** (`esdf_utils.pipeline_floor_path`). 03은 `path_vs_gt()`라는 하나의 함수로
  우리 궤적과 기준선을 모두 재서 지표 정의가 갈리지 않게 한다.
- **chamfer 0을 목표로 삼을 수 없다.** 루트가 완벽해도 0.080/0.109가 나온다. 실측은 그 1.19/1.54배 =
  남은 차이의 84%/65%가 이산화이고 루트 선택 차이는 0.027/0.045 m뿐이다.
- **③ 헤딩의 하한이 6°다** — 격자에 얹는 순간 45° 꺾임이 생긴다. 실측 8.0°를 "8도 틀렸다"로 읽으면
  안 되고 "하한 대비 2도 초과"로 읽어야 한다. **④의 항등값은 0이 아니라 1.016/1.028**이다
  (경유점 제약 자체의 비용) — 실측이 이와 사실상 같으므로 GT 루트는 우리 맵에서 최적이다.
- **① 씬2는 실측(1.23)이 하한(1.57)보다 낮다** — `thin_waypoints`가 격자 지그재그를 실제로 지워낸다.
- **배수는 게이트가 아니다**(씬 난이도에 따라 달라진다). `FLOOR_RATIO_AT_LIMIT = 1.5` 이내를
  "이산화 한계"로 표시만 한다. 판정 게이트는 기존 5개 그대로.
- **F3(격자 스냅만)를 기준선으로 쓰면 안 된다** — 씬마다 크게 흔들려(0.105 vs 0.063) 같은 상태의
  두 씬을 "1.01× 한계 도달 / 1.94× 여지 있음"으로 갈라 보이게 했다. F2/F3는 **원인 분해 용도**로만
  남긴다(F2 0.011 m로 작아서 GT의 시간축 샘플링은 주원인이 아니다).

### `03b_verify_reproduction.py` — 재현도 역검증 (✅ 완료)

- **하는 일**: 03은 "얼마나 다른가"를 재고, **03b는 그 값이 좋은 값인지에 답한다.** 경로가 정말
  같아도 지표는 0이 아니다(격자 양자화·시간축 샘플링). 그 하한을 모르면 남은 오차가 개선 여지인지
  원리적 한계인지 알 수 없다. 또 논문이 값을 안 준 파라미터를 GT에서 **역으로 읽어낸다**.
- **입력**: `esdf/<scene>.npz`(02) + `paths/<scene>.json`(03) + GT parquet.
  `--refine_radius`/`--r_b`는 기본 None이고 **03이 실제로 쓴 값**을 JSON에서 읽는다 —
  03b가 자기 기본값을 쓰면 F1이 통째로 실패한다(실제로 겪음).
- **출력**: `verify/<scene>.json`, `logs/gs-vlnpe/03b_verify_reproduction/<scene>/report.html`
- **4가지 검사와 실측 결과**

  | 검사 | 17DRP5sb8fy | s8pcmisQ38h | 결론 |
  |---|---|---|---|
  | F1 결정성 (두 번 계획 → 비트 동일) | 20/20 | 20/20 | 비결정성 없음. **유일한 실패 조건** |
  | **기준선 ⓒ 대비** | 0.072 vs 0.103 = 1.45× | 0.100 vs 0.134 = 1.27× | 두 씬 모두 이산화 한계권 |
  | 원인 분해 (F3 격자 / F2 점분포) | 0.105 / 0.011 | 0.063 / 0.011 | F3는 씬마다 흔들려 기준선 부적합 |
  | ⑤ refine 반경 R\* (내장 대조군) | 0.05 (18/20) | 0.05 (15/20) | 우리 0.15가 과했다 → 0.10 |
  | ⑥ knot 복원 | 무릎 20/20, 0.79 m | 무릎 20/20, 0.90 m | **GT는 cubic spline이다** |

- **이전 결론 2건을 정정한다**
  - "GT는 컨트롤러 실행 궤적이라 점 분포를 못 맞춘다" → **틀렸다.** ⑥이 40/40에서 무릎을 찾았다
    (k\* median 7~9). GT는 논문 7단계 그대로 소수 knot의 cubic spline이고, 복원한 간격 0.79~0.90 m가
    우리 `--waypoint_spacing_m 0.8`과 일치한다(스무드니스로 고른 값의 **독립 확증**).
    남은 오차의 주원인은 이산화(격자 + refine + 스무딩)이며, 기준선 ⓒ가 그 크기를 준다.
  - ep17 "원인 불명" → ④가 답했다. cost_ratio 1.014 = 동점.
- **설계상 틀렸다가 고친 것** (둘 다 *대조군 없이 GT 수치를 해석하려 한* 것이 원인)
  - ④를 GT 궤적의 **셀 시퀀스 비용**으로 재면 45° 구간이 계단(1+1)으로 세어져 A\*의 대각(√2)보다
    비싸진다 → 같은 루트에서도 1.04~1.32가 나왔다. **양쪽 다 A\* 비용**이어야 성립(고친 뒤 1.00~1.08).
  - ⑤를 "refine 산출물 = ESDF 국소최대"로 검사하면 안 된다. greedy refine은 한 번만 적용되므로
    우리 `wp_refined`조차 국소최대 비율이 **14%**였다(양성 대조 실패). 대신 "다시 refine하면 얼마나
    움직이나" 프로파일을 후보 반경별로 만들어 L1 매칭하고, 알려진 반경 0.10/0.15/0.25/0.40을 전부
    복원하는 것을 **내장 대조군**으로 확인한다.
- **Negative test**: `--refine_radius 0.30` 또는 `--refine_mode min_move`로 03과 강제로 어긋내면
  F1 0/3, exit 1. (03b는 기본적으로 `params`에서 03이 쓴 값을 읽으므로 이 실패는 강제할 때만 난다.)
- **시각화** (`viz_utils.floorplan_canvas` / `line_chart` 공용, 03과 같은 grid·색 규약)
  - **기준선 분해** blink — 노랑 GT → 보라 ⓑ(격자+스무딩) → 하늘 ⓒ(+refine) → 초록 우리.
    보라가 노랑에서 벌어진 폭이 격자 양자화, 보라→하늘이 refine이 밀어낸 몫, 하늘→초록이 루트 차이다.
    **에피소드를 겹치면 마지막에 그린 초록이 다 덮어 판독이 안 된다** — 배수 중앙값/최댓값 2개만 낸다.
  - **knot 복원** blink — GT 위에 복원한 k\* knot(주황)과 그 knot으로 만든 spline(하늘).
    주황 점 4~13개가 만든 곡선이 GT에 정확히 포개진다 = **GT가 cubic spline이라는 시각적 증거**.
  - **refine 반경 프로파일** 차트 — 후보 반경별 곡선 중 GT(노랑)에 붙는 것이 R\*. 넓게 refine한
    경로는 ESDF 평탄부에 있어 아래에 깔린다. matplotlib을 안 쓰고 cv2로 그린다(self-contained 유지,
    축 라벨은 ASCII만 — `cv2.putText`는 한글을 못 그린다).

### `03c_compare_refine.py` — refine 해석 비교 (✅ 완료, argmax 유지)

- **하는 일**: 논문 6단계의 두 해석을 끝까지 돌려 비교한다. `argmax`(창 안 ESDF 최대점 = 논문 문장
  그대로, 기본) vs `min_move`(`r_b`를 만족하는 만큼만 이동). `esdf_utils.REFINERS` registry로 고르며
  03/03b도 `--refine_mode`로 같은 registry를 쓴다.
- **출력**: `compare/<scene>.json`, `logs/gs-vlnpe/03c_compare_refine/<scene>/report.html`
- **결과 — 기준선과 실제 경로가 반대 방향으로 움직인다** (median, 씬1 / 씬2)

  | 지표 | argmax | min_move | 우세 |
  |---|---|---|---|
  | **실제 경로 chamfer** | **0.107 / 0.154** | 0.142 / 0.165 | argmax |
  | **실제 경로 Fréchet** | **0.233 / 0.366** | 0.319 / 0.380 | argmax |
  | 기준선 ⓒ chamfer | 0.080 / 0.109 | **0.058 / 0.062** | min_move |
  | refine 이동거리 | 0.132 / 0.146 | 0.035 / 0.035 | — |
  | refined wp가 `r_b` 이상 | 20/20 | 20/20 | 둘 다 안전 |
  | 게이트 ②(refine이 clearance↑) | 통과 | **실패** | — |

- **왜 반대인가**: `argmax`는 어떤 루트든 **복도 중심선(ESDF 능선)으로 정규화**한다. GT도 refine을
  거쳐 능선 근처에 있으므로 우리 A\*의 계단 모양이 씻겨나가며 GT와 수렴한다. `min_move`는 계단을
  그대로 남겨 격자에서 온 요철이 차이로 남는다. 기준선은 **GT 자신의 셀 시퀀스**에서 출발하므로
  안 움직이는 쪽이 유리한데, 그 이점은 GT 루트를 모르는 실제 생성 상황에서 쓸 수 없다.
- **그래서 "기준선이 낮다"가 좋은 게 아니다.** 기준선은 표현 변위를 재는 진단값이다.
  변위가 작은 파이프라인이 *다른 루트*를 받았을 때 GT에 더 가까워지는 것은 아니다.
- **결론**: 기본값은 `argmax` 유지. **논문 문장과 측정이 같은 방향을 가리킨다.**
  `min_move`는 옵션으로만 남긴다. `min_move`는 반경에 둔감하다(0.15/0.30/0.60 수치 동일).
- **시각화**: 에피소드 bbox로 잘라 확대한다(씬 전체를 그리면 경로가 몇 픽셀이라 판독 불가).
  ① 실제 경로 blink — GT(노랑) / argmax(빨강) / min_move(초록) / A\* waypoint(흰 점).
  흰 점 위에 초록이 얹혀 있고 빨강이 노랑 쪽으로 옮겨간 것이 보인다.
  ② 같은 에피소드의 **기준선** blink — 여기서는 순서가 반대다.

  `s8pcmisQ38h`의 남은 1개(ep17)는 `h_nav`·`r_b`·다운샘플·연결성 어디에도 반응하지 않는다.
  GT가 시간축 샘플(회전 시 감속)로 보여 컨트롤러가 개입한 흔적이 있고, 논문에 없는 세부에서
  오는 것으로 본다 — 이 정보로 도달 가능한 상한으로 기록한다.
  상세 분석과 기각된 가설 6개는 `.claude/memory/260803_gs_vlnpe_03_reproduce_result.md` 참고.

### `03d_grid_search_params.py` — 8축 파라미터 그리드서치 (✅ 완료, 탐색 전용·기본값 불변)

- **하는 일**: `astar_cell_m`/`refine_radius`/`connectivity`/`clearance_weight`(A\* tie-break)/
  `spacing_m`/`smooth_step`/`h_nav`/`r_b`/`downsample_mode` 8개 축을 각각 독립적으로(다른 7개는
  프로덕션 기본값에 고정) 그리드서치하고, `d(GT,ⓒ)`/`d(GT,path)`/**`d(ⓒ,path)`**(신규 — 기준선과
  생성 경로 사이 거리)를 함께 비교한다. `esdf_utils.astar()`에 `connectivity`/`clearance`/
  `clearance_weight` optional 인자를 추가했다(기본값 8/None/0.0 = 완전 no-op, 03/03b/03c 무수정,
  self-check 게이트 10→11개). **순수 탐색이며 00~03c의 기본값은 바꾸지 않는다.**
- **출력**: `gridsearch/results.json`, `logs/gs-vlnpe/03d_grid_search_params/report.html`
- **내장 회귀 검증**: 9개 축 각각의 "현재 기본값" 지점이 이미 검증된 03 수치(0.103/0.134)와
  일치하는지 스크립트 안에서 자동 확인 — **9중 회귀 가드 전부 통과**.
- **새로 알게 된 것**
  - **`connectivity=4`가 17DRP5sb8fy에서 실제 하드 충돌 1건을 낸다** — 이전엔 스무드니스
    문제로만 알았는데, 안전 문제라는 게 새 정보다. 8-이웃 유지 근거가 하나 더 늘었다.
  - **`d(ⓒ,path)`가 "refine = 정규화" 메커니즘을 수치로 보여준다**: `refine_radius`를 키우면
    `d(GT,ⓒ)`는 커지는데(0.054→0.190) `d(ⓒ,path)`는 작아진다(0.136→0.043) — 기준선과 생성
    경로가 서로 수렴하면서 둘 다 GT에서 멀어진다(03c의 결론을 숫자로 재확인).
  - **`clearance_weight`(A\* tie-break 가중치)는 신뢰할 수 있는 개선이 없다** — 씬1은 모든
    양의 가중치에서 악화, 씬2는 노이즈성 혼재. 도입 안 함.
  - `smooth_step`은 완전한 no-op이 아니다(작을수록 두 씬 모두 근소하게 낫다, ~1%) — 처음으로
    두 씬이 같은 방향으로 움직인 축이지만 효과가 작아 기본값을 바꿀 근거는 못 된다.
  - `h_nav`/`refine_radius`는 현재값이 최적임을 재확인. `r_b` 축소는 하드충돌 폭증(0.15에서
    씬1 8건), `downsample_mode='majority'`는 씬1만 이기고 씬2는 진다 — **또 씬1/씬2가 반대로
    움직이는 패턴**(이 세션 내내 반복). 전역 파라미터로는 더 못 줄인다는 신호가 재확인됐다.
  상세 표는 `.claude/memory/260804_gs_vlnpe_03d_grid_search_result.md` 참고.

**"논문과 완전히 동일하게 했으면 GT와 똑같이 나오나?"** — 아니다. (1) 격자 원점·A\* tie-break·
refine 반경·솎기 규칙 4개가 논문에 미기재이고 실제로 결과를 바꾼다(원점만 바꿔도 ep17이
갈렸다), (2) refine의 목적함수가 "GT 재현"이 아니라 "장애물에서 멀어지기"라 충실한 구현일수록
GT에서 멀어지는 역설이 있다(`d(ⓒ,path)`가 이를 수치로 보여준다, 03d). "GT는 컨트롤러 실행
궤적이라 못 맞춘다"던 이전 가설은 반증됐다(GT는 40/40에서 cubic spline으로 잘 재현됨) — 남은
간극은 실행 노이즈가 아니라 **미기재 자유도** 때문이다. 상세는
`.claude/memory/260804_gs_vlnpe_reproducibility_limits.md` 참고.

### `04_render_obs.py` — RGB/Depth/Pose 렌더링 (✅ 완료, 2026-08-04)

**초안(위 표·트리)은 00 이전에 쓰여 지금 확정된 실측과 어긋난다** — RGB shape은 `(480,640,3)`이
아니라 **`(270,480,3)`**, depth는 mm-clip이 아니라 **raw uint16/10000=meters**, pitch는 "15도
고정"이 아니라 **에피소드마다 다르다**(00~03 전체가 이미 이렇게 다룬다). "정답지 원칙"에 따라
`target_schema.json` 실측값을 따랐다. 렌더러는 `pyrender`/`vtk`가 없어 IsaacLab python에 이미
있는 **Open3D 0.19 `OffscreenRenderer`**로 새 의존성 없이 구현했다.

- **두 단계** — 렌더러(mesh+intrinsic+pose→rgb/depth)를 검증 없이 새 경로에 바로 쓰면 결과가
  이상해도 "경로가 이상한지 렌더러가 이상한지"를 못 가른다.
  - **2-A `--mode gt_replay`(기본)**: vln_n1 parquet의 실제 `action` 시퀀스(재계획 아님, 원본
    그대로)로 mesh를 렌더링해 실제 캡처(`observation.images.rgb/depth`)와 프레임별로 비교한다.
    depth median/p95 절대오차[m](mesh-anchor보다 강한 검증 — pose·intrinsic·mesh·렌더러 체인
    전체가 실제 센서와 일치하는지 직접 잰다), rgb는 엣지맵 상관계수(질감/노출이 달라도 "같은
    구조를 같은 자리에서 보는가"). **두 씬 모두 PASS**: depth median 0.0000 m / p95 0.0001 m
    (기준 0.01/0.05 m), edge_corr median 0.73~0.76(기준 0.5). 초기 스모크 테스트는 depth median
    0.003 m·edge_corr 0.43~0.48 정도였는데, 사용자가 "GT보다 렌더 rgb가 어둡고 depth가 픽셀만큼
    밀린다"고 지적해 재조사 — `cv2.matchTemplate`/MAE 스윕으로 두 문제 다 실측 확정했다: ①
    Open3D `Camera.set_projection`은 픽셀 **모서리**가 정수좌표인 컨벤션이라 intrinsics의 cx/cy에
    **+0.5px**를 더해야 OpenCV 컨벤션(intrinsic.npy 실측값)과 맞다(안 하면 depth 1px 오프셋,
    4프레임 MAE 0.0074→0.00005) ② `indirect_light_intensity`를 90000→270000으로 재보정(밝기
    실측 65→116 vs 실제 116, 씬2는 같은 값에서 124 — 실카메라 auto-exposure를 전역 상수 하나로는
    두 씬 다 못 맞춘다는 03d와 같은 패턴). 두 수정 후 depth median이 0.003→0.0000 m,
    edge_corr median이 0.43~0.48→0.71~0.74로 뛰었다 — 픽셀 오프셋이 rgb 정합도 같이 갉아먹고
    있었다는 뜻.

    이후 사용자가 다시 "그래도 GT rgb와 다르다"고 지적해 재조사 — **먼저 vln_n1의 GT rgb 자체가
    실사진이 아니라 합성 데이터임을 확인했다**(`data/InternData-N1-v0.5-mini/README.md`가
    VLN-N1을 "Synthetic Data"로 명시, GT depth의 mesh-anchor 오차 2.9e-5m은 실제 depth 센서로는
    불가능한 정밀도 — mesh를 직접 렌더링해야 나오는 값이다). 즉 이 비교는 "실물 vs 렌더"가 아니라
    **"그들의 렌더러 vs 우리 렌더러"**다. 채널별 mean/std를 다시 실측하니 밝기 차보다 **명암
    대비(std)가 렌더 쪽에서 1.3~1.5배 높다**는 게 훨씬 뚜렷한 패턴이었다(단일 ambient light만
    쓰고 보조광/AO가 없어 어두운 구석은 더 어둡고 밝은 벽은 더 밝게 나옴). `ColorGrading` 톤매핑
    6종(LINEAR/ACES/ACES_LEGACY/FILMIC/REINHARD/UCHIMURA)을 3프레임 쌍(두 씬)에서 실측
    스윕했더니 **REINHARD만 std 비율 0.977**(하이라이트 압축·그림자 리프트 표준 곡선이라 대비
    과다에 정확히 대응, 다른 후보는 전부 1.3~1.4x로 개선 없음). 톤매핑을 바꾸면 평균 밝기도
    같이 변해 `INDIRECT_LIGHT_INTENSITY`를 270000→310000으로 재보정(밝기 113.4 vs 실측 113.75,
    std 45.0 vs 실측 44~45 — 둘 다 사실상 일치). depth(0.0000 m)와 mesh-anchor(0.00000 m)는
    이 변경으로 전혀 흔들리지 않았다(색상 전용 변경) — edge_corr median은 0.71~0.74→0.73~0.76로
    소폭 개선.
  - **2-B `--mode reproduce`/`random`**: 2-A를 통과한 같은 `render_along`으로 `paths/<scene>
    [_random].json`(03의 출력)을 따라 새 에피소드의 rgb/depth/extrinsic을 만든다. 신규 로직은
    **카메라 pose 합성**(`geometry_utils.synthesize_action_poses`) 하나뿐이다 — 위치는
    `resample_by_arclength`로 프레임 간격(0.035 m, GT 실측 median과 동일) 리샘플, yaw는
    프레임간 tangent(중앙차분 `atan2`), pitch는 그 에피소드의 `h_b`/`pitch_deg`를
    `compose_camera_extrinsic`으로 합성한다. **회전 합성 공식은 analytic 유도만으로 확정하지
    않고 실측으로 검증했다** — 4개 씬/에피소드에서 진짜 `action[t]`를 역산해 정확한 yaw로
    재구성하면 오차 3e-8(부동소수점 수준), tangent 추정 yaw로도 median 0.1°/max 0.9°에 불과함을
    확인(`geometry_utils.py` self-check 게이트 ⑥). 두 씬 × reproduce/random 모두 **PASS**
    (위 두 렌더 수정 반영 후 mesh-anchor median 0.00000 m, 인접 프레임 이동거리 위반 0건).
- **좌표 컨벤션**: 렌더러가 받는 `c2w`는 `action_to_c2w`가 반환하는 것과 완전히 같은 컨벤션
  (OpenCV, x=right/y=down/z=forward). Open3D `Camera.look_at(center,eye,up)`은 world 벡터만
  받으므로 `eye=c2w[:3,3]`, `forward=c2w[:3,2]`, `up=-c2w[:3,1]`을 그대로 넘기면 Open3D 내부
  컨벤션(OpenGL, -Z를 봄)과 무관하게 정확히 재현된다 — 새 축 변환을 발명하지 않았다. mesh는
  `open3d.io.read_triangle_model`(멀티 머티리얼 텍스처 유지)로 로드한다 — `read_triangle_mesh` +
  단일 `MaterialRecord`는 텍스처가 날아가 회색 평면이 되는 것을 실제로 겪었다.
- **입력**: `02`의 mesh(`geometry_utils.find_scene_mesh`), `03`의 `paths/<scene>[_random].json`,
  intrinsic은 그 씬의 실제 parquet에서 읽는다(전체 데이터셋 공통 상수, 발명하지 않음).
- **출력** (`obs/<scene>_<mode>/episode_%06d/` 아래, 실측 스키마)
  - `rgb/frame_%04d.jpg` — uint8, shape **`(270, 480, 3)`**
  - `depth/frame_%04d.png` — uint16, shape **`(270, 480)`**, **raw/10000 = meters**, invalid=0,
    저장 전 clip `[0.1, 3.0]` m(D435i 스펙)
  - `extrinsic/frame_%04d.npy` — float32 `(4,4)`, vln_n1의 `action`과 **정확히 같은 포맷**
    (`action_to_c2w(..., 'cam2world_gl')`로 읽으면 cam2world가 나온다)
  - `intrinsic.npy` — float32 `(3,3)`, 그 씬의 실측 intrinsic
- **검증**: depth clip 범위 assert(저장 시점에 이미 강제), 인접 프레임 `extrinsic` 이동거리가
  프레임 간격(0.035 m)의 3배를 넘는지 assert(0건), 저장된 intrinsic/extrinsic으로 depth를 다시
  unproject해 씬 mesh 표면까지의 거리(mesh-anchor)를 재는 자기 검증(기준 0.01 m — mesh를 직접
  렌더링한 것이라 GT의 센서 노이즈가 없어 실측 GT보다도 작아야 정상). rgb/depth 나란히 blink 리포트.
- 상세: `.claude/memory/260804_gs_vlnpe_04_render_obs_result.md`.

### `05_to_lerobot.py` — LeRobot v2.1 변환

- **하는 일**: `04`가 만든 프레임 단위 파일들을 하나씩 순회하면서 `NavDataset`(`scripts/dataset_converters/vlnce2lerobot.py`의 `NavDataset(LeRobotDataset)`을 상속한 클래스) 객체에 `add_frame`으로 프레임을 채우고, `save_episode`로 parquet과 비디오 파일을 저장한다. **`action`은 04가 프레임마다 기록한 `extrinsic/frame_%04d.npy`(camera-to-world pose) 값을 그대로 옮겨 담는다**(00에서 확정된 대로 `action`은 프레임 간 delta가 아니라 그 자체가 camera-to-world pose). `camera_extrinsic`은 매 프레임 다시 계산할 필요 없이 **04의 카메라 마운트 config(고정 pitch·높이)로부터 한 번 계산한 상수를 모든 행에 반복 기록**한다(vln_n1과 동일하게 에피소드 내내 동일한 값).
- **입력**
  - `episode_%06d/{rgb,depth,extrinsic}/`, `intrinsic.npy` (04의 출력, 에피소드 수만큼)
  - `target_schema.json` (00의 출력, feature dict 정의에 사용)
- **출력** (vln_n1과 동일한 디렉토리 레이아웃)
  - `data/chunk-000/episode_%06d.parquet` — 컬럼: `index`(int64), `observation.camera_intrinsic`(float32 (3,3)), `observation.camera_extrinsic`(float32 (4,4)), `action`(float32 (4,4))
  - `videos/chunk-000/observation.video.rgb/episode_%06d.mp4`, `videos/chunk-000/observation.video.depth/episode_%06d.mp4` (정답지와 동일하게 `video_key` 이름을 맞춤)
  - `meta/info.json` — `codebase_version: v2.1`, `fps: 30`, `total_episodes`, `total_frames`, features 정의
  - `meta/episodes.jsonl`, `meta/tasks.jsonl` — Stage 1에서는 `sub_instruction` 필드에 placeholder 문자열(또는 정답지 문구 재사용) 기록
  - `meta/pointcloud_obstacle.npy` — 02의 `esdf.npz`에서 `sdf<0` voxel 좌표를 추출해 저장, shape `(M, 3)` float32
- **검증**
  - 생성된 parquet+meta+videos를 InternNav LeRobot 로더로 실제 인스턴스화하고 첫 배치를 iterate — 예외 없이 로드되는지 확인.
  - `meta/info.json`의 `total_episodes`/`total_frames`가 실제 저장된 episode 수·parquet 행 수와 일치하는지 assert.
  - 저장된 `action`이 04가 기록한 `extrinsic/frame_%04d.npy` 값과 프레임별로 완전히 동일한지(diff==0) 확인. 저장된 `camera_extrinsic`이 모든 행에서 동일한 상수이고 04의 카메라 마운트 config 값과 일치하는지 확인.
  - mp4의 프레임 수가 parquet 행 수와 같은지 확인.

### `06_verify.py` — 포맷·품질 검증 (Stage 1의 목적)

- **하는 일**: 생성한 데이터셋을 GT 스키마·ESDF·GT 데이터(`vln_n1`)의 분포, 이렇게 4가지 관점에서 비교하고 그 결과를 리포트로 남긴다.
- **입력**
  - `data/chunk-000/episode_*.parquet` + `meta/` + `videos/` (05의 출력)
  - `target_schema.json` (00), `esdf.npz`(02), `paths.json`(03)
  - 대조 기준선: 로컬 `data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i/<scene>/` 원본
- **처리**
  - **포맷 diff**: 생성 parquet의 컬럼명/dtype/shape을 `target_schema.json`과 1:1 비교, 불일치 0건 확인 + InternNav LeRobot 로더로 실제 로드.
  - **기하 정합**: 각 프레임의 `depth + action`(pose)으로 point cloud를 복원 → 다른 프레임에 재투영(backward-warp) → source 프레임(depth를 가진 쪽)의 rgb와 픽셀 단위로 얼마나 잘 맞아떨어지는지(reprojection error, 단위 px) 계산. (`camera_extrinsic`은 상수 마운트 오프셋이라 이 계산에 사용하지 않음 — `geometry_utils.check_alignment_fpv`를 그대로 호출, 직접 재구현 금지. 추가로 `geometry_utils.check_alignment_bev`로 월드 프레임 교차검증도 함께 수행 가능. **주의**: backward-warp 결과는 source 프레임과 같은 격자이므로 반드시 source의 rgb와 비교해야 한다 — target 프레임의 rgb와 비교하면 서로 다른 시점을 비교하는 셈이라 오차가 의미 없어진다.)
  - **경로 안전성**: `paths.json`의 모든 waypoint를 `esdf.npz`에 투영해 `d ≥ d_safe` assert.
  - **분포 비교**: 경로 길이·depth 히스토그램을 정답지 `vln_n1`과 비교.
- **출력**
  - `verify_report.json` — `{"schema_diff": [], "loader_ok": true, "mean_reprojection_error_px": 1.3, "collision_free_ratio": 1.0, "path_length_hist": {...}, "depth_hist": {...}}`
  - `reproj_overlay/episode_%06d_frame_%04d.jpg` — 재투영 vs 원본 rgb 오버레이 시각화(육안 점검용)
- **검증** (검증 스크립트 자체를 검증)
  - `verify_report.json`의 임계값(`mean_reprojection_error_px` 상한, `d_safe` 등)이 하드코딩이 아니라 `config.yaml`에서 오는지 확인.
  - **Negative test**: pose나 depth를 의도적으로 어긋나게 만든 입력을 넣어 `06_verify.py`가 실제로 `loader_ok: false`나 높은 `mean_reprojection_error_px`로 실패를 잡아내는지 확인 — 검증 스크립트가 항상 통과만 내보내는 무의미한 체크가 아님을 보장.
  - `reproj_overlay/` 이미지 몇 장을 무작위로 열어 리포트 수치(예: reprojection error 낮음)와 육안 인상이 일치하는지 교차 확인.

---

## Stage 2 (M2) — VLN-PE 전환 (개요)
- Stage 1 파이프라인 위에 **물리**를 얹는다: `01`을 InternUtopia USD 물리 씬(`data/scene_data/mp3d_pe`)으로, `04`를 **H1 controller(Move-by-Speed/Move-along-Path) rollout**(`04b`)로 확장해 실제 dynamics 관측 수집. 환경은 `InternutopiaEnv`(`@Env.register('internutopia')`), 수집은 `DataCollector`(`internnav/evaluator/utils/data_collector.py`) 재사용. VLN-PE LeRobot 포맷으로 저장.
- **라벨 보강**: critic value(전역 ESDF, NavDP C4), pixel-goal(farthest-visible 투영, InternVLA-N1 C1).
- **지표 도입**: FR(roll>15°/pitch>35°/COM-to-foot)·StR(50step<0.2m·15°) (arXiv 2507.13019 §3).

## Stage 3 (M3) — 실제 GS-map 확장 (개요)
- `config.yaml`의 `scene`=취득 GS-map, `rgb_source`=**gs**로 전환. `01`에서 **GS↔mesh 정합 + floor gap 보정**(필요 시 SAGE-3D) 후 USD.
- `04`의 RGB만 **GS splat 렌더**로 교체(depth는 mesh 유지). 나머지 `02·03·05·06` 그대로.
- Stage 1/2에서 검증된 파이프라인이므로 여기서는 **씬 정합·GS 렌더 품질**만 새 리스크.

---

## Stage 1 검증 방법 (end-to-end)
1. `00_inspect_vln_n1.py` → `target_schema.json` + 레퍼런스 재투영 이미지.
2. `01`~`05` 순차 실행 → mp3d_n1 씬 20 에피소드.
3. `06_verify.py`:
   - 스키마 diff 0 + 로더 20/20 로드.
   - **depth+action 재투영 rgb 일치**: 평균 reprojection error가 임계(예: 수 px) 이하.
   - gt-path 전부 collision-free assert.
   - 경로 길이·depth 분포가 로컬 `vln_n1`과 동류.
4. 무작위 에피소드 rgb/depth/gt-path 오버레이 육안 점검.

## 리스크 (Stage 1 국한)
- **포맷 미세 불일치**(dtype/shape) → 로더 실패: `target_schema.json`을 단일 진실원으로 M1.6 강제 diff. 특히 `camera_extrinsic`/`action`이 **4×4 float32 list**임에 유의.
- **pose 규약 불일치**(좌표 프레임·회전 표기): ✅ M1.0에서 실증 확정됨 — `action[t]`이 camera-to-world pose(00_inspect_vln_n1.py의 backward-warp 재투영으로 확인). 남은 리스크는 04에서 새로 렌더링하는 pose도 동일 규약(camera-to-world)으로 기록하는 것.
- **depth-rgb 비정합**(렌더러 intrinsic 오설정): M1.6 재투영 검증이 게이트. D435i intrinsic을 정답지 `camera_intrinsic`과 대조.

## 참고 소스
- 정답지·레퍼런스: `data/InternData-N1-v0.5-mini/vln_n1/`, mesh: `data/scene_data/mp3d_n1/`
- 변환 베이스: `scripts/dataset_converters/vlnce2lerobot.py` (`NavDataset`, `NavDatasetMetadata`)
- 환경/수집: `internnav/env/base.py`, `internnav/env/internutopia_env.py`, `internnav/evaluator/utils/data_collector.py`
- SAGE-3D(`Galery23/SAGE-3D_Official`), VLN-PE(arXiv 2507.13019 §3)
- 논문: NavDP, InternVLA-N1, DualVLN
