# gs_vlnpe → 학습 가능한 데이터셋 (L1 라벨링 · L2a vln_pe 조립)

> 노은역 3DGS 씬에서 렌더한 8,378 프레임을 **로더가 실제로 읽는 데이터셋**으로 만드는 실행 스펙.
> 작성 2026-09-08. 근거는 전부 이 저장소 코드를 직접 읽어 확인한 것이고, 확인하지 못한 것은 §7에 따로 적었다.

## 지금 상태

| | |
|---|---|
| 있는 것 | `apply_real/paths/noeun_station_mid_random.json` (20 ep, 충돌 0, median 14.41 m)<br>`apply_real/obs/noeun_station_mid_random_isaac_d455_nominal/` (**8,378 프레임**, 842 MB, mesh-anchor median 4.6 mm)<br>`apply_real/esdf/noeun_station_mid.npz` (voxel 0.05, navigable 0.8796) |
| 없는 것 | `apply_real/format_validation/validate_noeun_vln_formats.py`의 `missing_vln_fields`가 그대로 열거한다 — parquet, **language instruction**, discrete action, robot state, progress/step, finish status. 같은 파일에 `direct_loader_compatible: False`가 하드코딩돼 있다 |

**포맷 정정.** `docs/execution-staged.md`는 `05_to_lerobot.py`가 `vln_n1` 레이아웃을 목표로 잡았지만, 실측 결과 **InternVLA-N1 트레이너는 `vln_ce`만 읽고** `vln_pe`는 CMA/RDP/Seq2Seq가 `LerobotAsLmdb`로 읽는다. **재렌더 없이 갈 수 있는 건 `vln_pe`뿐**이므로 L2a의 목표를 `vln_pe`로 바꾼다. `execution-staged.md`의 05/06 슬롯은 이 문서가 대체한다.

---

## 1. 저장소 경계 — 라벨링은 여기서 하지 않는다

라벨링은 별도 저장소 **`/home/universe/git/vln-annotator`** 에서 한다. 그쪽은 이미 완성돼 있다 (GT-match 0.951, turns/ep 0.59가 R2R GT와 일치하도록 캘리브레이션됨). 이 저장소에 합치지 않는 이유:

- vln-annotator는 InternNav 의존성이 0이다 — 원격 vLLM만 쓴다. torch/Isaac 환경으로 끌고 오면 async openai 클라이언트가 Isaac Sim 컨테이너에 얹힌다.
- 캘리브레이션된 하이퍼파라미터(`p_stop=0.494`, `p_wait=0.305`, `p_anchor=0.231`, `p_turn_unanchored=0.42`)를 깨뜨릴 위험이 있다.

**이 저장소가 맡는 것은 어댑터 2개**다 — `geometry_utils`·`esdf_utils`·`dataset_utils`가 필요한 쪽만.

```
gs_vlnpe 00–04  →  [06_export_for_annotator.py]  →  vln-annotator CLI  →  annotated.json.gz
                                                                                  │
                            8,378 프레임 ─────────────────────┐                   │
                                                              ▼                   ▼
                                                      [07_to_vlnpe.py]  ←─────────┘
                                                              │
                                                              ▼
                                    data/gs_vlnpe/.../vln_pe/traj_data/r2r/noeunstationmid/
```

---

## 2. 신규 파일

```
scripts/dataset_converters/gs_vlnpe/apply_real/
├── 06_export_for_annotator.py   # gs_vlnpe 출력 → annotator 입력 3종
├── r2r_vocab.py                 # vln_pe 2504-어휘 복원 (공용)
├── 07_to_vlnpe.py               # annotator 출력 + 프레임 → vln_pe
└── 07b_verify_vlnpe.py          # 검증 게이트 + negative test
```

`.gitignore`에 추가: `apply_real/annotator_export/`, `apply_real/vlnpe/`

> `apply_real/` 디렉토리는 현재 `root:root`다(컨테이너가 root로 파일을 만들어왔다). 호스트에서 쓰려면 chown이 필요하거나, 컨테이너 안에서 만든다.

---

## 3. `06_export_for_annotator.py` — annotator 입력 만들기

### 3.1 목표 계약 (vln-annotator 코드에서 직접 확인)

**① `--frames-dir`** — `vision.py:103,209`
```
<export>/frames/episode_%06d/
├── start.jpg          # 시작 시점
├── turn_1.jpg  turn_2.jpg  turn_3.jpg           # 각 회전 지점의 정면 (최대 3개)
└── turn_side_turn_1.jpg  turn_side_turn_2.jpg  turn_side_turn_3.jpg   # 회전 방향 쪽
```
확장자는 `jpg|jpeg|png` 순으로 탐색한다. **최대 3개 회전만** 쓴다.

**② `--midpoints-dir`** — `vision.py:153-166`
```
<export>/midpoints/episode_%06d/
├── midpoints.json     # {"frames": [{"label": "mid_1", "path": "mid_1.jpg"}, ...]}
└── mid_1.jpg  mid_2.jpg  ...
```
`path`는 `episode_%06d/` 기준 상대경로다.

**③ `--gt-path`** — `dataset_assembler.py:load_source`, `assemble_episode`
`.json.gz`, 최상위 `{"episodes": [...], "instruction_vocab": {...}}`. 에피소드당 필요한 키:
`episode_id`, `trajectory_id`, `scene_id`, `start_position`, `start_rotation`, `info.geodesic_distance`, `goals`, `instruction.{instruction_text,instruction_tokens}`, `reference_path`.
`assemble_episode`가 `instruction`만 갈아끼우고 나머지는 **그대로 복사**하므로, 구조 필드는 우리가 채워야 한다.

### 3.2 재사용할 것 (직접 재구현 금지)

| 필요 | 재사용 | 위치 |
|---|---|---|
| OpenGL `action` → OpenCV c2w | `action_to_c2w(M, 'cam2world_gl')` | `apply_real/geometry_utils.py:187` |
| JPEG 저장 (RGB→BGR 처리 포함) | `save_jpg(rgb, path, quality)` | `apply_real/geometry_utils.py:84` |
| ESDF 클리어런스 조회 | `sample_esdf_at(esdf, xy, origin, 0.05)` | `apply_real/esdf_utils.py:1121` |
| world↔cell | `world_to_cell` / `cell_to_world` | `apply_real/esdf_utils.py:1111,1116` |
| arclength 리샘플 | `resample_by_arclength` | `apply_real/esdf_utils.py:680` |
| 리포트 HTML | `save_gallery`, `blink_widget_html`, `floorplan_canvas` | `apply_real/viz_utils.py:259,224,282` |

`geometry_utils.load_rgb_frame`(:62)은 **쓰지 말 것** — 경로 템플릿이 `episode_%06d_%03d.jpg`(vln_n1 레이아웃)이고, 04의 출력은 `episode_%06d/rgb/frame_%04d.jpg`다.

### 3.3 회전·midpoint 지점 고르기

`extrinsic/frame_%04d.npy`를 전부 읽어 pose 시퀀스를 만든 뒤:

```python
P    = [action_to_c2w(np.load(f), 'cam2world_gl') for f in sorted(...)]
xy   = P[:, :2, 3]
yaw  = np.arctan2(P[:,1,2], P[:,0,2])        # R_cv[:,2] = world forward
s    = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
dyaw = wrap(np.diff(yaw))
```

- **회전 지점**: 누적 |Δyaw|가 30°를 넘는 구간(vln-annotator `config.turn_threshold_deg=30.0`과 같은 값)을 찾아 크기 순 상위 3개. NMS는 arclength 1.5 m 배제창.
- **`turn_side_*`**: 회전 방향 쪽을 본 프레임이 필요하다. 우리 렌더는 카메라가 항상 경로 접선을 보므로 **측면 프레임이 없다.** → 회전 **직후** 프레임(이미 그쪽을 향한 상태)을 대용으로 쓴다. 이건 근사이므로 `report.html`과 provenance에 명시한다.
- **midpoint**: 회전 사이 직진 구간의 중간점. `min_segment_dist_m=0.3` 미만 구간은 버린다.
- **`start`**: 프레임 0.

### 3.4 GT `.json.gz` 만들기 — 어휘가 갈리는 지점

**⚠️ 이게 이 스펙에서 가장 틀리기 쉬운 부분이다.** `vln_ce`와 `vln_pe`는 **서로 다른 어휘 id 공간**이다. 실측:

| | `vln_ce/raw_data/r2r/train/train.json.gz` | `vln_pe/raw_data/r2r/train/train.json.gz` |
|---|---|---|
| `instruction_vocab` | **있다** — `word2idx_dict` 2,711개, max id 2710 | **없다** (`episodes` 키만) |
| `go` / `the` / `right` | 1067 / 2389 / 1968 | 982 / 2202 / 1819 |
| `,` | 없음 (→ UNK) | 103 |

`vln_ce` 어휘로 인코딩한 결과를 `vln_pe` 에피소드의 토큰과 비교하면 **3,000건 중 0건 일치**, 그리고 **2,717건이 id ≥ 2504**가 되어 `internnav/configs/model/cma.py:13`의 `vocab_size=2504` 임베딩 범위를 벗어난다.

따라서:

- **annotator에 넘기는 `--gt-path`의 `instruction_vocab`**: `vln_ce` 것을 그대로 복사해 넣는다. annotator의 `VLNTokenizer`(`dataset_assembler.py`)가 이걸 읽어 `instruction_tokens`를 만드는데, 그 토큰은 **`vln_ce` 공간**이다. `vln_ce` 타깃(L2b)에는 그대로 쓸 수 있다.
- **`vln_pe`/CMA 타깃(L2a)**: annotator가 만든 `instruction_tokens`를 **버리고**, `instruction_text`만 가져와 `r2r_vocab.py`로 다시 인코딩한다.

### 3.5 구조 필드 채우기

| 필드 | 값 |
|---|---|
| `episode_id` | `paths/*.json`의 `episode_id` |
| `trajectory_id` | `episode_id`와 동일 |
| `scene_id` | `"noeun_station_mid"` |
| `start_position` / `goals[0].position` | 궤적 첫/마지막 waypoint |
| `start_rotation` | 첫 프레임 yaw → 쿼터니언 |
| `reference_path` | `paths/*.json`의 `waypoints_thinned` (없으면 `trajectory`) |
| `info.geodesic_distance` | `path_length_m` (`assemble_episode`가 없으면 `_geodesic`으로 계산해준다) |
| `instruction.instruction_text` | **placeholder** — annotator가 갈아끼운다 |
| `instruction.instruction_tokens` | 200개 0 배열 (placeholder) |

### 3.6 CLI · 출력 · 게이트

```
python .../apply_real/06_export_for_annotator.py \
  --scene noeun_station_mid --mode random --obs_tag isaac_d455_nominal \
  --vocab_source data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/train/train.json.gz \
  --max_turns 3 --turn_threshold_deg 30 --min_segment_dist_m 0.3 \
  --out_dir scripts/dataset_converters/gs_vlnpe/apply_real \
  --log_dir logs/gs-vlnpe/apply_real
```

출력: `apply_real/annotator_export/{frames,midpoints}/episode_%06d/…`, `annotator_export/noeun_gt.json.gz`, `logs/.../06_export_for_annotator/noeun_station_mid/report.html`

| 게이트 | 임계 |
|---|---|
| E1 에피소드 수 | 20/20이 `start.jpg`를 갖는다 |
| E2 회전 검출 | 에피소드당 1–3개. 0개인 에피소드는 목록에 남긴다(직진만인 짧은 궤적은 정상) |
| E3 GT 로드 | `dataset_assembler.load_source()`로 실제 읽히고, `VLNTokenizer(vocab_source)`가 `num_vocab=2711`로 뜬다 |
| E4 좌표 일관성 | `reference_path` 첫/끝점이 `start_position`/`goals[0]`과 1 cm 이내 |
| **negative** | 한 에피소드의 `frames/`를 비워 **E1이 실패하는지** 확인 |

---

## 4. `r2r_vocab.py` — `vln_pe` 2504-어휘 복원

`vln_pe` 쪽 `word2idx_dict`는 디스크에 없다. `embeddings.json.gz`는 2504×50 임베딩 **행렬**일 뿐이다. 그런데 `train.json.gz`의 에피소드 10,819개가 `instruction_text`와 `instruction_tokens`를 **둘 다** 갖고 있어 역산할 수 있다.

```python
SPLIT_RE = re.compile(r'(\W+)')
def r2r_tokenize(s):
    s = s.lower().replace(',', '').replace('?', '')
    return [t.strip() for t in SPLIT_RE.split(s) if t.strip()]
```

이 토크나이저로 정렬하면 **모호한 단어 0개**, 단어 약 2,347개, **라운드트립 정확 일치 99.4%** (10,752/10,819). 상수: `PAD=0`, `UNK=1`, `MAX_INSTR_LEN=200`, `VOCAB_SIZE=2504`.

API: `build_vocab(raw_root, cache=...) -> dict`, `encode(text, w2i) -> list[int]` (200으로 절단·제로패딩), `oov_report(text, w2i) -> dict`.

**200 제로패딩이 필수인 이유**: `bert_tokenizer is None`(기본 `CMA_Policy`)이면 `extract_instruction_tokens`가 **호출되지 않는다**(`cma_lerobot_dataset.py:154` 가드). 따라서 downstream 패딩이 없다. `:162`가 리스트를 `(T,L)`로 타일하고 `cma_collate_fn`이 `torch.stack(dim=1)`을 하므로 배치 전체에서 `L`이 같아야 한다. `instruction_encoder.py:73`이 길이를 `(instruction != 0).sum(dim=1)`로 구하므로 trailing 0이 패딩 규약이다. 릴리스 데이터도 전부 정확히 200이다.

`__main__` 자기검사 2개: ① 모호 0개 + 라운드트립 ≥ 0.99, ② `len(encode(...)) == 200` and `max(id) < 2504`.

**OOV는 정보 손실이다** — UNK=1로 뭉개지면 CMA에 신호가 0이 된다. 그래서 L1 결과에 OOV 게이트를 걸고, 초과하면 `needs_review`로 표시하고 조용히 넘기지 않는다.

---

## 5. `07_to_vlnpe.py` — `vln_pe` 조립

### 5.1 재사용: 라이터는 이미 있다

`scripts/visualization/eval_gt_collect.py`의 `save_episode()`(:122–241)가 parquet 15컬럼 · `(T,256,256,3) uint8`/`(T,256,256) float32` npy · `meta/{episodes,tasks,episodes_stats}.jsonl`을 **전부 쓴다.** `_stats()`(:107), `finalize_info_json()`(:245)도 그대로 쓴다.
**새로 쓸 것은 "렌더 프레임 → `step_buffer`" 어댑터뿐이다.**

### 5.2 action 이산화 — 이게 핵심이고, 최적화가 아니라 정확성 요건이다

렌더된 궤적은 **연속**이다. 실측: 프레임당 이동이 `FRAME_STEP_M = 0.035`로 균일(`04_render_obs_isaac.py:112`), 프레임당 |Δyaw| median 0.31°, **회전 전용 프레임 0개**.

VLN-PE는 action 1개 = step 1개다. 그리고 `internnav/configs/model/cma.py:12`가 `max_step=200`, `cma_lerobot_dataset.py:148`이 모든 배열을 `[:200]`으로 **하드 절단**한다. 프레임마다 라벨만 붙이면 285–912 step 에피소드가 되어 **절반 이상이 조용히 버려지고 `progress`가 0.5에서 멈춘다.**

`eval_gt_collect.py:51–100`의 결정 규칙을 그대로 가져오되(`FORWARD_DIST=0.25`, `TURN_ANGLE_RAD=15°`, `|delta| > 7.5°`이면 회전), env를 스텝하는 대신 **렌더된 프레임 위를 걷는다**:

```python
i, theta, out = 0, yaw[0], []
while True:
    j = np.searchsorted(s, s[i] + 0.25)
    if j >= len(s): break
    bearing = atan2(*(xy[j] - xy[i])[::-1])
    for _ in range(24):                                   # 루프 가드
        delta = wrap(bearing - theta)
        if abs(delta) <= radians(15) * 0.5: break
        sgn = 1 if delta > 0 else -1
        theta = wrap(theta + sgn * radians(15))
        k = argmin_{m in [i..j]} |wrap(yaw[m] - theta)|    # 회전을 실제 프레임에 묶는다
        out.append((k, 2 if sgn > 0 else 3))
    out.append((i, 1))
    i = j
out.append((len(s) - 1, 0))                               # 종료 STOP
```

**action 알파벳은 `{0,1,2,3}`이고 5는 없다.** 릴리스 `vln_pe` parquet **2,813개 전부**를 세어 확인했다: `{0: 2571, 1: 166395, 2: 48416, 3: 53870}`. 의미는 `h1_vln_move_by_flash_controller.py:65-83`에서 직접 확인 — `1`=전진(`forward_distance`만큼 yaw 방향), `2`=좌회전(`yaw + rotation_angle`), `3`=우회전(`yaw − rotation_angle`), 그 외=제자리. (`eval_gt_collect.py`는 종료 STOP을 `step_buffer` 루프 **밖에서** 스텝하므로 0을 안 쓴다 — 우리는 쓴다.)

같이 측정된 것: 릴리스 에피소드 길이는 min 1 / **median 88** / max 501이다. 즉 `max_step=200` 절단은 릴리스 데이터에도 일어난다. 우리 리샘플 결과(최대 T=145)는 이 분포 안에 들어간다.

**한계를 명시할 것.** 04는 카메라 yaw를 경로 접선으로 렌더했으므로 이 데이터셋에는 **진짜 제자리 회전 관측이 없다.** 위 방식은 "제자리 15° 회전"을 "곡선상 15° 더 간 지점의 가장 가까운 렌더 프레임"으로 근사한다. 실측 오차: yaw median 1.2° / max 7.4°, 위치 median 0.245 m / max 0.284 m. 둘 다 구조적으로 상한이 있다(각각 ±7.5° 판정 임계, 0.25 m 한 스텝). 이 분포를 `report.html`과 provenance에 넣는다.
`--action_mode forward_only`(전부 1 + 종료 0, pose 허구 0)를 대조군으로 함께 제공한다 — action 분포가 ~98% 전진으로 퇴화하는데, 그게 대조군의 요점이다.

**측정된 결과: 8,378 프레임 → 약 1,342 step, 최대 T=145.** RGB 스택 264 MB, depth 352 MB.

### 5.3 pose → `vln_pe` 쿼터니언

`dataset_utils.py:107-110`의 `vlnpe_orientation_to_c2w_rotation`(`R_cv = [-r[:,1], -r[:,2], r[:,0]]`)의 역:
```python
r = np.column_stack([R_cv[:,2], -R_cv[:,0], -R_cv[:,1]])   # Isaac [forward, left, up]
q_xyzw = Rotation.from_matrix(r).as_quat()
q_wxyz = q_xyzw[[3,0,1,2]]
camera_yaw = atan2(r[1,0], r[0,0])
```
**pose 수학은 `geometry_utils.action_to_c2w`를 반드시 경유한다.** 직접 재구현하면 축 컨벤션 버그를 반복한다 (`docs/execution-staged.md`의 M1.0 절 참고 — `action`을 그대로 쓰면 절대좌표가 0.27~0.82 m 어긋난다).

### 5.4 이미지 변환 (실측 확정치)

| | 우리 출력 | `vln_pe` 목표 | 변환 |
|---|---|---|---|
| RGB | 480×270 JPG. `save_jpg`가 RGB→BGR 후 저장하므로 **디스크 JPG는 정상 RGB** | `(T,256,256,3) uint8` | PIL로 열면 RGB 그대로. `cv2.INTER_AREA`로 256² |
| Depth | uint16 PNG, `depth_scale_m=0.001` → m | `(T,256,256) float32`, 실측 범위 **0.001–1.0** (m = 값 × 10) | `val = raw * 1e-4`; `val[raw==0] = 1.0`; `clip(0,1)`; `INTER_NEAREST` |

Depth를 `INTER_NEAREST`로 하는 것은 `eval_gt_collect.py:179`(기본 bilinear)와 **의도적으로 다르다.** 거기서는 Isaac이 이미 256²로 렌더해 리사이즈가 no-op이었지만, 여기는 실제 1.875× 축소이므로 bilinear는 경계마다 두 표면 사이의 없는 깊이값을 만들어낸다.

**⚠️ 경로에 `3dgs`를 넣지 말 것.** `cma_lerobot_dataset.py:57`(및 `rdp_lerobot_dataset.py:124`, `cma_lmdb_dataset.py:68`)이 `if '3dgs' in lerobot_features_dir`면 RGB 채널을 뒤집는다. `gs_vlnpe`는 `3dgs`가 아니므로 안전하지만, 지뢰이므로 스크립트에 주석으로 남긴다.

**intrinsic 불일치는 기하 문제가 아니라 도메인 시프트다.** CMA/Seq2Seq/RDP는 intrinsic을 읽지 않는다(`cma_policy.py:267-323`은 `instruction`/`rgb`/`depth`/`progress`/`prev_actions`만 쓴다). 우리 렌더는 hfov가 정확히 90°로 VLN-PE와 **같고**(`camera_profiles.py:52`), vfov만 58.7° vs 90°로 다르다. 비등방 리사이즈 후 유효 `K = [[128,0,128],[0,227.6,128],[0,0,1]]` — 수평은 정확히 일치. 크롭(수평 FOV 44% 손실)이나 레터박스(44%가 합성 패딩)보다 낫다. 유효 `K`를 provenance에 기록한다.

### 5.5 나머지 컬럼

| 컬럼 | 값 |
|---|---|
| `observation.camera_position` / `_orientation` / `_yaw` | 묶인 프레임의 실제 pose (§5.3) |
| `observation.robot_position` | `[x_cam, y_cam, floor_z]` — 아래 참고 |
| `observation.robot_orientation` / `_yaw` | `Rz(camera_yaw)` wxyz / `= camera_yaw` |
| `observation.progress` | `s[k] / s[-1]` — 첫 행 0.0, 종료 STOP 1.0 |
| `observation.step` | 렌더 프레임 인덱스 (추적용) |
| `observation.action` | 0/1/2/3 |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | `eval_gt_collect.py:191-197`과 동일. `task_index = 3*ep` |

**`robot_position`은 판단이 필요하다.** VLN-PE 관계 `cam = rob + R(rob_ori) @ [0.2, 0, 0.72]`(`dataset_utils.py:261-267`, 릴리스 데이터에서 8.2 mm로 검증됨)를 역산할 수 없다 — stage 03이 에피소드마다 `h_b`를 0.27–1.42 m로 **랜덤화**해서 `rob_z = floor_z + h_b − 0.72`가 20개 중 7개에서 **바닥 아래**가 된다. 이 데이터셋에 H1은 없다(`scene_meta`의 `gt_robot_params: null`). → 기본값은 카메라의 바닥 투영이고, **0.72 m 몸통 오프셋을 꾸며내지 않는다.** provenance에 명시.

**`progress`는 학습 타깃이다** — `cma_trainer.py:98-107`이 `tanh(progress_monitor(x))`와 MSE를 취하고 `configs/model/cma.py:38-41`이 `progress_monitor.use=True`다. GT replay는 항상 목표에 도달하므로 정규화 누적 arclength가 맞다(단조 비감소, 0→1).

**`finish_status`는 장식이 아니다.** `cma_lerobot_dataset.py`에서 append 블록 **전체**(:89-152)가 `if self.config.il.filter_failure.use:` 안에 있다. `finish_status='success'`면 `min_rgb_nums=15` 필터를 건너뛰는데, 이게 중요하다 — 에피소드 04는 T=10밖에 안 나온다. `'stuck'`은 25프레임 tail drop(:104-116)을 유발하므로 절대 쓰지 않는다. → `finish_status='success'`, `fail_reason='success'`.

### 5.6 `meta/` — 에피소드당 지시문 3개

`cma_lerobot_dataset.py:201`이 `return self.length * 3  # each trajectory corresponds to 3 instructions`이고, `LerobotAsLmdb.get_data_by_key()`(`internnav/utils/loader.py:184-216`)가 `episodes_stats.jsonl`의 `task_index` **min~max 범위**로 `tasks.jsonl` 행들을 모아온다.

- `meta/episodes.jsonl`: `{"episode_index": i, "tasks": [txt0, txt1, txt2]}`
- `meta/tasks.jsonl`: **3행** (`task_index` = `3i`, `3i+1`, `3i+2`)
- `meta/episodes_stats.jsonl`: `"task_index": {"min": 3i, "max": 3i+2, "count": T}`

`eval_gt_collect.py`는 에피소드당 1개만 쓰므로 **여기가 유일하게 고쳐야 하는 부분**이다. 릴리스 씬 `s8pcmisQ38h`가 16 ep / 48 task로 이 규약을 쓴다.

**`tasks.jsonl` 필드는 네 개를 다 쓴다.** 소비자가 셋으로 갈린다:

| 소비자 | 읽는 필드 | 위치 |
|---|---|---|
| `CMA_Policy` (기본, `bert_tokenizer=None`) | `instruction_tokens` | `cma_lerobot_dataset.py:138` |
| `CMA_CLIP_Policy` / `model.text_encoder` 있는 설정 | `instruction_text` | `cma_lerobot_dataset.py:134` |
| RDP | `task` | `rdp_lerobot_dataset.py:195-198` |
| `LerobotAsLmdb.save_video` | `task` | `internnav/utils/loader.py:284` |

릴리스 `tasks.jsonl`에는 `instruction_text`가 **없다**(`task_index`, `task`, `instruction_tokens`, `finish_status`, `fail_reason`). → `task` · `instruction_text` · `instruction_tokens` + `finish_status`/`fail_reason` 전부 쓴다. 몇 KB 비용으로 네 소비자가 다 돈다.

`meta/info.json`: `finalize_info_json`을 쓰되 `robot_type`은 `"camera_only"`(H1이 없다). 릴리스 `info.json`은 **stale**하다 — `vln_n1` 스키마를 선언하고 있고 실제 parquet에 없는 컬럼을 적어놨다(`dataset_utils.py:17-18`이 이미 경고한다). `LerobotAsLmdb`는 `info.json`을 열지 않으므로 우리 실제 parquet에 맞게 쓴다.

`meta/noeun_provenance.json` — **신규·추가 사이드카.** 이 데이터셋이 실제 H1 데이터로 오해될 수 없게 한다: 소스 씬/프레임 수, 렌더 `K`와 리사이즈 후 유효 `K`, hfov/vfov, depth 변환식, `action_mode`와 회전 잔차 분포, `robot_frame`, `no_inplace_turn_observations: true`, 라벨러 모델·어휘·OOV 통계.

### 5.7 출력 위치 — 릴리스 데이터를 건드리지 않는다

```
data/gs_vlnpe/noeun_station_mid/vln_pe/traj_data/r2r/noeunstationmid/
├── data/chunk-000/episode_{000000..000019}.parquet
├── videos/chunk-000/observation.images.{rgb,depth}/episode_%06d.npy
└── meta/{info,episodes,episodes_stats,tasks}.json*  +  noeun_provenance.json
```

**⚠️ 씬 디렉토리 이름에 `_`를 쓸 수 없다.** `internnav/utils/loader.py:147`이 키를 `f"{scan}_{scene_index}_{chunk:03d}_{episode:06d}"`로 만들고 `:158-175`가 `key.split('_')`로 되돌려 `scan=parts[0]`, `scene_index=parts[1]`을 취한다. `noeun_station_mid`는 `noeun`으로 읽힌다. → **`noeunstationmid`**.
그리고 `allow_scan_list=['r2r']`가 하드코딩(`cma_lerobot_dataset.py:41`)이므로 스캔 디렉토리 이름은 **`r2r`**여야 한다. 이러면 코드 변경이 없다.

릴리스 트리(`data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/`)에 씬을 끼워넣는 방식은 **쓰지 않는다.** 새 루트라서 61개 씬이 그대로 남고 노은역만으로 학습할 때 필터링이 필요 없다. 디스크 616 MB / 153 GB.

템플릿 지시문 버전도 `data/gs_vlnpe/noeun_station_mid_template/...`로 하나 더 만든다 — 같은 프레임·같은 action, 지시문만 다르다. 지시문 ablation의 대조군이고 비용은 명령 하나와 616 MB다.

---

## 6. `07b_verify_vlnpe.py` — 게이트

| # | 게이트 | 임계 |
|---|---|---|
| V1 | 스키마: parquet 컬럼명·Arrow 타입이 릴리스와 동일, npy dtype/shape | diff 0 |
| V2 | 개수: `rgb.shape[0] == depth.shape[0] == parquet 행수`, `info.total_tasks == 3 × episodes` | 정확히 |
| V3 | `LerobotAsLmdb`: `get_all_keys(['r2r'])`가 20키, `get_data_by_key`가 20/20에서 `finish_status=='success'` + `len(episodes_in_json)==3` | 20/20 |
| V4 | **CMA 데이터로더 첫 배치**: `cma_collate_fn`으로 1배치. `rgb[-3:]==(256,256,3)`, `depth[-3:]==(256,256,1)`, `instruction[-1]==200`, `action ⊂ {0,1,2,3}`, `len(dataset)==60` | 예외 없음 |
| V5 | pose 왕복: `vlnpe_orientation_to_c2w_rotation(q) ≈ R_cv` | < 1e-6 |
| V6 | depth 왕복: `stored*10` vs `raw*0.001`; `stored ∈ (0,1]` | < 1e-6 |
| V7 | 이산화: `progress` 단조·0→1, 전진 간격 0.25 m ± tol, 회전 잔차 분포 | yaw p99 ≤ 7.5°, xy p99 ≤ 0.30 m |
| V8 | 지시문: 60개 전부 길이 200, `max(id) < 2504`, `task == instruction_text` | 정확히 |
| V9 | 분포 대조: action/T/depth 히스토그램 vs 릴리스 — **참고용, FAIL 아님** | 보고만 |
| **NEG** | 5개 변형을 **복사본**에 넣어 각각 해당 게이트만 FAIL시킨다 | 5/5 검출 |

negative test 5개: ① parquet 행 3개 제거 → V2 ② `camera_yaw`에 +0.15 rad → V5 ③ depth를 m 단위로 저장 → V6 ④ `instruction_tokens`를 137개로 절단 → V8(+V4가 `torch.stack`에서 예외) ⑤ `episodes_stats.task_index`를 `{min:3i,max:3i}`로 → V3

---

## 7. 확인하지 못한 것

1. **Cosmos-Reason1을 아직 로드해본 적 없다.** 원격 `config.json`이 `qwen2_5_vl`/gated 아님인 것과 클래스 import는 확인했지만, 가중치를 내려받아 이 GPU에서 돌린 적은 없다. 로컬 컨테이너에 `vllm`이 없어 OpenAI 호환 서빙에는 vLLM 설치나 shim이 필요하다.
2. **지시문 품질은 미측정.** 게이트는 전부 형식적이다(길이·어휘·회전 부호 일치). NuRec Gaussian 렌더에서 지하철 랜드마크를 제대로 지목할지는 스모크 전까지 모른다. 퇴화하면 템플릿 베이스라인 + 키프레임당 랜드마크 지목 1회 호출로 대체한다.
3. **릴리스 `r2r` npy의 채널 순서를 눈으로 확인하지 않았다.** `BRG_to_RGB`가 non-`3dgs`에서 off인 것과 `eval_gt_collect.py:168-172`가 Isaac `obs['rgb']`를 그대로 저장하는 것으로부터 RGB라고 추론했다. 07b 리포트에 소스 JPG와 저장된 프레임을 나란히 놓고 육안 확인하는 항목을 넣는다.
4. **`turn_side_*`를 회전 직후 프레임으로 대용하는 것의 타당성.** 진짜 측면 관측은 재렌더 없이는 못 만든다. annotator의 Phase 1c 결과가 쓸 만한지는 스모크로 판단한다.
5. **`observation.step` 의미.** 릴리스는 stride 50의 sim-step 카운터(`0,50,100,…`)이고 우리는 렌더 프레임 인덱스를 쓴다. `'stuck'` tail-drop 경로만 소비하고 우리는 거기 안 걸리지만, 나중에 이 씬으로 평가하면 `internnav/evaluator/`가 읽는지 확인할 것.
6. **어휘 복원에서 제외한 67건(0.6%)** — 길이가 안 맞는 에피소드. 모호 매핑 0개라 무해하지만, 완전히 닫으려면 그 67건의 원문에서 어떤 문자를 원 전처리가 다르게 다뤘는지 봐야 한다(아포스트로피/하이픈 추정).
7. **`--turn_frame nearest_yaw` vs `current`** 중 무엇이 학습에 나은지는 미검증. 전자는 yaw 오차(median 1.2°)를 줄이고 위치 오차(0.245 m)를 늘리며, 후자는 반대다. 둘 다 구현하고 분포를 리포트에 넣되, 미관으로 고르지 않는다.
