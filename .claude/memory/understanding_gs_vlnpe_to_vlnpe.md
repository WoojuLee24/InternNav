# gs_vlnpe → vln_pe(Isaac) 포맷 생성·비교 방법

"reproduce"는 이 파이프라인에서 **두 가지 다른 뜻**이고 스크립트도 다르다. 먼저 어느 쪽인지 정해야 한다.

| | A. 렌더 재현 | B. 경로 재현 |
|---|---|---|
| 무엇을 재현 | **이미지만** 우리 렌더로 교체 | **궤적을 GT 안 보고** 다시 계획 |
| GT에서 받는 것 | camera pose 전부 | start/goal · `h_b` · pitch만 |
| parquet·meta | GT에서 **바이트 복사** | 직접 써야 함 |
| 스크립트 | `05_export_vlnpe.py` | `03f_reproduce_batch.py` |
| 상태 | **61씬 전수 완료** (116 GB) | 측정만 완료, **저장 미구현(Phase C)** |

---

## A. 렌더 재현 — `05_export_vlnpe.py`

GT parquet의 camera pose를 그대로 재생하고 rgb/depth만 우리 Isaac 렌더로 바꾼다.
궤적·action·지시문이 GT와 같아서 **action 이산화·라벨링·통계 재계산이 전부 불필요**하고,
구조 동일성이 복사로 보장돼 프레임 단위 SSIM 대조가 정확해진다.

한 씬 스모크 (전수 전에 반드시 먼저):
```
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05_export_vlnpe.py --dataset vln_pe --scenes 17DRP5sb8fy --max_episodes 3 --out_root data/InternData-N1-v0.5-mini/vln_pe_render --log_dir logs/gs-vlnpe
```
전수:
```
timeout --signal=KILL 86400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05_export_vlnpe.py --dataset vln_pe --n_scenes 0 --out_root data/InternData-N1-v0.5-mini/vln_pe_render --log_dir logs/gs-vlnpe
```

| 인자 | 뜻 |
|---|---|
| `--n_scenes 0` | 전수(기본). 표본만 보려면 숫자, 특정 씬은 `--scenes A,B` |
| `--max_episodes` | 씬당 상한. 0이면 전부 |
| `--light`/`--rtx_ambient`/`--film_iso`/`--tonemap_op` | 생략하면 **확정 프리셋**(`render_profiles.vln_pe_measured`) 사용 — 생략이 정답 |
| `--headroom_gib 20` | 남길 디스크 여유. npy 헤더로 프레임 수를 먼저 세어 용량을 가드한다 |
| `--no_skip_existing` | 이미 된 씬도 다시 |

실측: **61씬 · 2,813 ep · 271,252 frame · 116 GB · 5.2 h.**

⚠️ **씬당 프로세스 1개**다. 한 프로세스에서 `build_renderer`를 두 번 부르면 `/World/RenderCamera`
충돌로 죽고, 씬을 3번 올리면 `AnnotatorRegistryError`가 난다(둘 다 실측).
⚠️ 성공 판정은 **종료코드가 아니라 sentinel 파일**로 한다 — Isaac은 정상 종료에도 세그폴트를 낸다.

## B. 경로 재현 — `03f_reproduce_batch.py`

`01`(scene_meta) → `02`(ESDF) → `03`(경로+GT대조)를 필요한 것만 돌린다. **전부 CPU**, Isaac 안 띄운다.
판정 기준 `same_route` = Fréchet < 0.8 m(문 너비). 같은 루트면 GT 지시문이 그대로 유효하다.

```
timeout --signal=KILL 21600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03f_reproduce_batch.py --dataset vln_pe --n_scenes 0 --path_profile reproduce_v1 --work_dir scripts/dataset_converters/gs_vlnpe/logs --log_dir logs/gs-vlnpe/03f_v1
```

실측(45씬 2,274 ep — 61씬 중 15씬은 vln_n1 GT가 없어 제외, 1씬은 GT가 1프레임):

| | legacy | reproduce_v1 |
|---|---|---|
| chamfer / Fréchet 중앙값 | 0.191 / 0.553 | **0.167 / 0.480** |
| 같은 루트 | 73.1 % | **78.8 %** |
| 하드 충돌 에피소드 | 18 | **0** |
| 쓸 수 있는 ep | 1,649 | **1,700** |

⚠️ `--r_b`·`--h_nav_ratio`는 이 방식으로 스윕하면 안 된다 — 그 값이 `02`가 만든 ESDF에 박혀 있어
`esdf/`를 하드링크로 재사용하면 **옛 지도로 계획**한다(조용히 틀린다). 그 두 축은 ESDF부터 다시.

**파라미터 천장 78~79%** — 시험한 30여 조합이 전부 73~79%였다. 더 올리려면 A* 비용함수를 바꿔야 한다.

**이 경로 1,700개를 렌더해 저장하는 `07_render_paths_to_vlnpe.py`가 아직 없다(Phase C).**
필요한 조각 둘: ① action 이산화(알고리즘은 `docs/gs-vlnpe-l1-l2a.md` §5.2에 확정) ② 조립.
라이터는 이미 있다 — `vlnpe_writer.py`(자기검사 `--self_test`, GPU 불필요).

---

## 비교·검증 — 두 층이다

### 1) 기계 검증 — `05b_verify_vlnpe.py` (Isaac 안 띄움, 디스크만 읽음)

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05b_verify_vlnpe.py --render_root data/InternData-N1-v0.5-mini/vln_pe_render/traj_data/r2r --gt_root data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --scenes all --require_complete --negative --deep --log_dir logs/gs-vlnpe
```

| 게이트 | 무엇 |
|---|---|
| V1 | 트리: 디렉토리·파일명·개수가 GT와 동일 |
| V2 | npy dtype/shape 동일, rgb·depth·parquet **행수 일치** |
| V3 | parquet 컬럼·Arrow 타입·`huggingface` 메타 동일 (복사이므로 sha256도 같아야) |
| V4 | depth 값역 `0 < v <= 1`, NaN 0, rgb uint8 |
| V5 | `LerobotAsLmdb` 키 수 == GT, 에피소드당 지시문 3개 |
| V6 | `cma_collate_fn` 첫 배치가 예외 없이 생성 |
| V7 | **GT 대조 SSIM — 보고용, FAIL 아님** |

| 인자 | 뜻 |
|---|---|
| `--require_complete` | GT에 있는데 빠진 에피소드도 잡는다. **전수 검사에 필수** |
| `--deep` | parquet·meta sha256까지 GT와 비교 |
| `--negative` | 산출물 **복사본**을 5가지로 망가뜨려 해당 게이트만 FAIL 하는지 확인 |
| `--self_test` | GT를 산출물 자리에 넣어 검증기 자체를 검사(전 게이트 PASS가 기준선) |

**전수 실측 결과** (`logs/gs-vlnpe/06b_verify_vlnpe_render/verify.json`, 61씬):

| | 결과 |
|---|---|
| V1~V5 | **61/61 PASS** (전부) |
| V7 SSIM | 씬 중앙값 **0.8295** · 최악 씬 0.6669 · **0.75 이상 56/61** |
| negative | **5/5 검출** (행 제거→V2 · depth m단위→V4 · rgb float32→V2 · …) |

SSIM 0.8232(톤맵 전수 측정값)와 일치한다. 에피소드 단위로 보면 편차가 크다 —
17DRP5sb8fy ep3은 median 0.43/min 0.18로 낮고, `verify.json`의 `scenes.<씬>.V7.per_episode`에
에피소드별 `ssim_median`/`ssim_min`/`frames`가 다 들어 있다.

### 2) 눈으로 비교 — `04e_sample_compare.py`

씬 여러 개를 한 페이지에서 GT ↔ 우리 렌더로 나란히 본다. 전수 저장 **전에** 설정이 맞는지 보는 단계다.

```
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04e_sample_compare.py --dataset vln_pe --n_scenes 6 --both_settings --n_frames 4 --log_dir logs/gs-vlnpe
```

`--both_settings` = 기존·확정 설정 둘 다 렌더해 같은 프레임에서 비교(씬당 렌더 2회).
출력 `logs/gs-vlnpe/04e_sample_compare/<dataset>_<n>scenes/report.html`.

⚠️ 프레임마다 **정적 이미지로** 나란히 놓는다 — blink 위젯만 쓰면 스크립트 없는 뷰어에서 빈칸으로 보인다(실측).

---

## 한 씬을 처음부터 만들 때 순서

```
02_build_freemap_esdf.py   # occupancy + ESDF (지도)
03_sample_gt_paths.py      # A* 경로 (--mode reproduce | random)
04_render_obs_isaac.py     # 렌더 (--mode gt_replay 로 먼저 렌더러 검증)
05_export_vlnpe.py         # vln_pe 레이아웃으로 저장
05b_verify_vlnpe.py        # 게이트 + GT 대조
```

`04`를 `--mode gt_replay`로 먼저 돌려 **렌더러가 GT를 재현하는지**부터 확인하는 게 원칙이다 —
이게 통과해야 그 뒤 경로 생성 결과를 믿을 수 있다.

관련: `scripts/dataset_converters/gs_vlnpe/reports.md`(리포트 링크 단일 소스),
`.claude/memory/260910_vlnpe_render_and_reproduce_result.md`(상세 경위),
`.claude/memory/260807_gs_vlnpe_vlnpe_dataset_result.md`(vln_pe 지원 근거).
