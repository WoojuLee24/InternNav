# 다른 세션용 프롬프트 — `01_prepare_scene.py` 에 `--dataset` 추가 (작업 B)

같은 머신에서 **작업 A**(46씬 경로 재생성)가 동시에 돌고 있다. 충돌하지 않도록 아래
"동시 작업 주의"를 반드시 지킬 것.

---

저장소 `/ws/src/InternNav`, 브랜치 `feature/input_v0.1`. 먼저 `git pull`.
Python 은 `/workspace/isaaclab/_isaac_sim/python.sh` (Isaac 컨테이너).

## 문제

`vln_pe` 61씬 중 **15씬이 지도(ESDF)를 못 만든다.** 파이프라인 첫 단계 `01_prepare_scene.py` 가
**`vln_n1` 전용**이라서다. 이 15씬은 `vln_pe` 에만 있고 `vln_n1` 에는 없다(확인함). mesh 는 15/15 다 있다.

```
5q7pvUzZiYa  GdvgFV5R1Z5  HxpKQynjfin  PX4nDJXEHrG  Pm6F8kyY3z2
Uxmj2M2itWa  V2XKFyX4ASd  VVfe2KiqLaN  aayBHfsNo7d  b8cTxDM8gDG
e9zR4mvMWw7  gZ6f7yhEvPG  kEZ7cmS4wCh  qoiz87JEwZ2  rPc6DW4iMge
```

막히는 지점은 `01` 의 `load_gt_reference()`(L236) 다 — `observation.camera_extrinsic` 컬럼을 읽는데
그 컬럼은 **`vln_n1` 에만 있다**. 거기서 `floor_z = cam_z − h_b` 를 구해 `scene_meta` 에 넣고,
`02` 가 그 `floor_z` 로 높이대를 자른다. 그래서 `scene_meta` 가 없으면 `02`→`03` 이 전부 막힌다.

## 할 일

`01` 에 `--dataset {vln_n1,vln_pe,vln_ce}` 를 추가한다. **`02`/`03` 에 이미 같은 패턴이 들어가 있으니
그걸 그대로 따른다**(`02_build_freemap_esdf.py` 의 `--dataset` 과 로컬 `load_gt_episode` 래퍼 참고).

핵심: `dataset_utils.load_gt_episode(data_root, scene, episode, dataset, rig=...)` 가 **세 데이터셋을
모두 같은 계약으로** 돌려준다 — `poses_c2w` / `cam_xyz` / `body_xyz` / `h_b` / `pitch_deg` /
`floor_z` / `k`. `01` 이 직접 parquet 을 읽는 대신 이걸 쓰면 된다.

지켜야 할 것:
- **`--dataset` 기본값은 `vln_n1`** 로 두어 생략 시 종전 동작이 완전히 같게 한다
- `--data_root` 를 생략하면 `dataset_utils.default_data_root(args.dataset)` 로 채운다
- `vln_ce` 는 rig 가 있으므로 `--rig`(기본 `125cm_30deg`)도 같이 받는다 — `02`/`03` 과 같은 이름
- 기존 분기를 건드리지 말고, 가능하면 **guard clause + 위임**으로 (`dataset_utils.load_gt_episode` 가
  이미 `vln_ce` 를 `vlnce_gt.py` 로 위임하는 방식이 선례다)
- `01` 의 검증 ④ 중 **`pointcloud.ply` 를 쓰는 부분은 `vln_n1` 에만 있다**(실측: vln_pe·vln_ce 의
  `meta/` 에는 info.json/episodes*.jsonl/tasks.jsonl 뿐). `02` 가 이미 같은 문제를 겪고
  exhaustive `if/elif/else` + `assert False` 로 건너뛰게 고쳐 뒀으니 **같은 방식**으로 처리할 것

## 검증

1. **회귀가 최우선** — `--dataset` 생략 시 기존 씬 산출물이 **바이트 동일**해야 한다:
```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/01_prepare_scene.py --scene 17DRP5sb8fy --out_dir /tmp/b_check --log_dir /tmp/b_check_logs
```
   → `/tmp/b_check/scene_meta/17DRP5sb8fy.json` 을 `scripts/dataset_converters/gs_vlnpe/logs/scene_meta/17DRP5sb8fy.json`
   과 비교. `cmp` 로 동일해야 한다(경로 문자열만 다를 수 있음 — 그 경우 그 차이만 있는지 확인).

2. **15씬 `01`** — `--dataset vln_pe` 로 돌려 `scene_meta` 가 나오는지. `floor_z` 가
   0.2~2.5 m 카메라 높이 가정과 모순되지 않는지(검증 ④) 확인.

3. **15씬 `02`** — `--dataset vln_pe --r_b 0.20 --h_nav_ratio 0.12` 로 ESDF 생성.
   씬당 수 분 걸린다. clearance 게이트가 PASS 하는지 본다
   (vln_pe 기준은 `median>0.10 & max zero_frac<30%` — vln_n1 기준과 다르다).

4. **15씬 `03`** — 아래 설정으로. 작업 A 가 46씬에 쓰고 있는 것과 **같은 설정**이어야 합친 수치가 의미 있다:
```
--dataset vln_pe --r_b 0.20 --plan_r_b 0.25 --h_nav_ratio 0.12 --refine_radius 0.20 --post_refine --num_episodes 999
```

## ⚠️ 동시 작업 주의 — 같은 머신에서 작업 A 가 돌고 있다

작업 A 는 `03_sample_gt_paths.py` 로 **esdf 가 있는 46씬**의 경로를 다시 만드는 중이다.

- **고쳐도 되는 파일**: `01_prepare_scene.py` 뿐. `02`/`03`/`esdf_utils.py`/`dataset_utils.py` 는
  **건드리지 말 것** (A 가 실행 중이고, 이미 `--dataset` 지원이 들어가 있다)
- **출력**: `01`/`02` 는 `--out_dir scripts/dataset_converters/gs_vlnpe/logs` 로 써도 되지만,
  **없는 15씬에만** 쓴다. 기존 46씬의 `scene_meta`/`esdf` 를 덮어쓰면 A 가 읽는 중인 파일이 깨진다.
  불안하면 `--out_dir /tmp/b_out` 으로 받아 나중에 **없는 것만** 복사할 것
- `03` 결과는 `--out_dir /tmp/b_paths` 같은 임시 경로로 받는다
- **git**: push 전에 반드시 `git pull --rebase`. A 쪽도 같은 브랜치에 커밋한다

## 참고

- `02`/`03` 의 `--dataset` 추가 커밋: `2b0a6b2` (vln_ce 지원) — 같은 패턴을 보면 된다
- `dataset_utils.load_gt_episode` 의 guard clause 위임과 `vlnce_gt.py` 가 선례
- `vln_pe` 로봇: `r_b 0.20` · 카메라 높이 1.66 m · H1 휴머노이드
- 결과는 `.claude/memory/<날짜>_01_dataset_result.md` 에 정리하고 커밋
- 수치가 예상과 **크게 다르면 먼저 보고**할 것. 조용히 넘어가지 말 것

---

## ⚠️ 2026-10-12 추가 — `03` 설정이 바뀌었다

위 4번의 `03` 설정은 **더 이상 쓰지 말 것.** 5씬 파라미터 탐색 결과 아래가 더 좋다
(r_b 미달 8 → 1, 관통 0 유지, 에피소드 손실 217→211):

```
--dataset vln_pe --r_b 0.20 --plan_r_b 0.28 --h_nav_ratio 0.12 --refine_radius 0.20 --post_refine --post_refine_margin 0.05 --num_episodes 999
```

바뀐 점은 둘:
- `--plan_r_b` 0.25 → **0.28** (spline 이 여유를 median 0.05 · p90 0.15 깎아먹어서 0.05 로는 부족)
- `--post_refine_margin` 0(기본) → **0.05** (수리 목표를 정확히 `r_b` 로 잡으면 재스무딩이
  같은 양만큼 다시 파고든다. 실측: 수리 성공 4/12 → 32/33)

`--downsample_mode majority` 는 **쓰지 말 것** — vln_ce 와 달리 vln_pe 에서는 에피소드가
217→151 로 급감한다(r_b 0.20 이 커서 coarse 격자가 과하게 막힌다).

이미 0.25 로 돌렸다면 **다시 돌려야** 작업 A 의 45씬과 합쳐서 의미 있는 수치가 된다.
