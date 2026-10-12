# habitat 세션용 프롬프트 — vln_ce navmesh 기준 검증

이 컨테이너(isaaclab)에는 habitat이 없어 `.navmesh`를 읽을 수 없다(`habitat_sim.nav.PathFinder` 필요).
아래 내용을 habitat 컨테이너 세션에 그대로 붙여넣는다.

---

저장소 `/ws/src/InternNav`, 브랜치 `feature/input_v0.1` (먼저 `git pull`).
이 컨테이너는 habitat 환경이고 모든 스크립트는 `/usr/bin/python` 으로 실행한다.

## 배경 — 왜 하는가

`gs_vlnpe` 가 2026-10-11 에 vln_ce 경로 재현을 지원하게 됐다(`--dataset vln_ce`).
지금은 **우리 occupancy(밴드 투영)** 로 계획하고 검사하는데, 이 지도에는 알려진 허점이 있다:
**"이 높이에 바닥이 있는지"를 묻지 않아** 다층 집에서 다른 층 공간이 새어 들어온다
(`3dloader_vlnce` 실측: s8pcmisQ38h 거짓 승인의 66.6 %).

그런데 **vln_ce 의 GT 경로는 habitat navmesh 위에서 만들어졌다.** 그래서 navmesh 가
"더 정확한 지도"가 아니라 **GT 의 자**다. 우리 지도로 쟀더니 **GT 자신이 7.2 % 위반**으로 나왔는데,
이게 GT 의 결함인지 우리 지도의 오차인지 가려야 한다.

`3dloader_vlnce/03_verify_navmesh_gt.py` 가 이미 "GT 는 navmesh 에서 벽을 뚫지 않는다"를
2 씬에서 확인했다(waypoint 474/474 · 531/531). 이번엔 **20 씬**으로 넓히고, 거기에 더해
**우리가 생성한 경로**도 같은 자로 재는 것이 목표다.

## 할 일

### 0. preflight (수 초)

```
/usr/bin/python -c "import habitat_sim; print(habitat_sim.__version__)"
/usr/bin/python -c "import skimage" || pip install scikit-image
ls data/scene_data/mp3d_ce/mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh
ls data/scene_data/mp3d_pe/17DRP5sb8fy/matterport_mesh
ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | wc -l     # 61 이어야 한다
```
없는 자산이 있으면 **거기서 멈추고 무엇이 없는지 보고**해라. 추측으로 진행하지 말 것.

### 1. 우리 경로를 재생성한다 (CPU 만, habitat 불필요)

`logs/` 는 gitignore 라 `scene_meta`·`esdf` 가 저장소에 없다. 20 씬에 대해 `01`→`02`→`03` 을 돌린다.
`01`/`02`/`03` 은 Isaac 을 쓰지 않는다(obj 경로).

씬 20개:
```
17DRP5sb8fy 1LXtFkjw3qL 1pXnuDYAj8r 29hnd4uzFmX 2n8kARJN3HM 5LpN3gDmAk7 759xd9YjKW5
7y3sRwLe3Va 82sE5b5pLXE 8WUmhLawc2A B6ByNegPMKs D7G3Y4RVNrH D7N2EKCX4Sj E9uDoFAP3SH
EDJbREhghzL JF19kD82Mey JeFG25nYj2p JmbYfDe2QKZ PuKPg4mmafe S9hNv5qa7GM
```

씬마다 (한 줄씩, `$GS=scripts/dataset_converters/gs_vlnpe`):
```
/usr/bin/python $GS/01_prepare_scene.py --scene <S> --mesh_root data/scene_data/mp3d_pe --out_dir $GS/logs --log_dir logs/gs-vlnpe
```
```
/usr/bin/python $GS/02_build_freemap_esdf.py --dataset vln_ce --rig 125cm_30deg --scene <S> --mesh_root data/scene_data/mp3d_pe --scene_meta_dir $GS/logs --out_dir $GS/logs --log_dir logs/gs-vlnpe
```
```
/usr/bin/python $GS/03_sample_gt_paths.py --dataset vln_ce --rig 125cm_30deg --scene <S> --r_b 0.10 --downsample_mode majority --num_episodes 20 --esdf_dir $GS/logs/esdf --out_dir $GS/logs --log_dir logs/gs-vlnpe
```

⚠️ **`--downsample_mode majority` 를 빠뜨리지 말 것.** `r_b=0.10` 이 작아서 기본 `any` 다운샘플이
막힌 칸을 통과시킨다(실측: 충돌 에피소드 8/20 → 0/20).

### 2. navmesh 캐시를 만든다 (habitat 필요)

```
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/03_verify_navmesh_gt.py --scenes <20씬 콤마구분> --r_bs 0.10 --episodes 20 --out_dir logs/embodiment_augment/s1_navmesh_vlnce
```
`data/embodiment_aug/navmesh/<scene>_rb0.10.navmesh` 가 씬마다 생긴다.

### 3. navmesh 기준으로 충돌을 다시 잰다 — **이게 핵심**

`3dloader_vlnce/navmesh_grid.py` 의 `load_mask` + `esdf_from_mask` 로 navmesh 를 우리 격자의
esdf 규약으로 바꾼 뒤, `gs_vlnpe/esdf_utils.check_path_navigable` 로 두 가지를 잰다:

- **(a) GT 실행 궤적** — `dataset_utils.load_gt_episode(..., 'vln_ce', rig='125cm_30deg')['cam_xyz']`
- **(b) 우리 생성 경로** — `$GS/logs/paths/<scene>_vlnce.json` 의 `episodes[].trajectory`

⚠️ **navmesh 는 이미 `r_b=0.10` 만큼 깎인 configuration space 다.** `esdf_from_mask` 가
`esdf = (마스크 안) ? 경계까지거리 + r_b : 0` 로 규약을 맞춰 주므로, 거기에 `r_b` 를 **또** 적용하지 말 것.
`navmesh_grid.py` 머리 주석에 이 관계가 적혀 있다.

### 4. GT ↔ 생성 경로 비교

```
/usr/bin/python $GS/checks/03g_compare_to_reference.py --dataset vln_ce --rig 125cm_30deg --paths_dir $GS/logs/paths --label vln_ce_navmesh --log_dir logs/gs-vlnpe
```
세 쌍이 나온다 — `우리↔계획GT`, `우리↔GT실행`, `GT실행↔계획GT`(참고선).

### 5. 매끄러움

`03` 의 `stats` 에 이미 있다. GT 대비로 읽는다:

| 지표 | 뜻 |
|---|---|
| `turn_per_m_deg` / `gt_turn_per_m_deg` | m 당 회전량. **비율이 1 근처여야** GT 와 비슷하게 완만 |
| `kappa_p95` / `gt_kappa_p95` | 곡률 상위 5 %. 우리가 GT 보다 작으면 더 완만 |
| `turn_ratio` | 위 둘의 비 |

참고(17DRP5sb8fy, occupancy 기준): `turn_per_m_deg 33.6` vs `gt 25.7` (비 1.24),
`kappa_p95 1.63` vs `gt 3.77`. **회전량은 GT 보다 많고 곡률 피크는 더 낮다**는 뜻이다.

### 6. 래스터화 격자를 npz 로 내보낸다

isaaclab 컨테이너에서도 쓸 수 있도록 **habitat 없이 읽히는 형태**로 저장한다.
씬마다 `mask`(bool) · `esdf`(float32) · `origin` · `cell` · `floor_y` · `r_b` 를 담아:
```
data/embodiment_aug/navmesh_grid/<scene>_rb0.10.npz
```
이게 있으면 다음부터는 habitat 없이 navmesh 기준 검사를 할 수 있다.

### 7. 리포트와 기록

- `3dloader_vlnce/publish_artifact_report.py` 로 Artifact 발행
- `.claude/memory/<날짜>_vlnce_navmesh_result.md` 에 정리하고 커밋·push

## 비교 기준값 (우리 occupancy 로 잰 실측 — 이번 결과와 대조할 것)

| | occupancy 기준 |
|---|---|
| **GT 실행 궤적 충돌** | **29 / 400 (7.2 %)** · 완전관통 27 |
| 우리 경로 충돌 (`r_b 0.10`) | 21 / 367 |
| 우리 경로 충돌 (`r_b 0.15` 로 계획) | 1 / 354 |
| 우리 ↔ 계획GT | chamfer 0.217 · Fréchet 0.584 · 같은루트 73.7 % |
| 우리 ↔ GT실행 | chamfer 0.192 · Fréchet 0.487 · 같은루트 73.1 % |
| GT실행 ↔ 계획GT | chamfer 0.135 · Fréchet 0.433 · 같은루트 80.8 % |

**가장 알고 싶은 것**: navmesh 기준으로 재면 GT 충돌이 **0 에 가까워지는가**.
그렇다면 occupancy 의 7.2 % 는 지도 오차이고, 우리 경로의 21 건 중 몇 건이 같은 이유인지도 갈린다.
`3dloader` 의 2 씬 실측(`waypoint 474/474 · 531/531`, 뚫린 점은 **전부 깊이 1 셀** = 격자 반올림)이
20 씬에서도 유지되는지 보는 것이다.

## 주의

- 2026-10-11 에 `vlnce_align.py` 와 `pixel_goal_utils.py` 가 옮겨졌다
  (`gs_vlnpe/vlnce_align`, `gs_vlnpe/vlnce_rig`, `3dloader_vlnce/goal_adjust`).
  import 가 깨지면 그것부터 확인.
- 검증 실행은 **`--out_dir` 을 임시 경로로** 줄 것. 생산 산출물(`$GS/logs/paths` 등)을 덮어쓴 적이 있다.
- `03` 의 `=> FAIL` 은 종합 게이트 판정이라 경로가 안 나온 것과 다르다. `stats.n_ok` 를 볼 것.
- 수치가 위 기준값과 **크게 다르면 먼저 보고**하고 원인을 찾을 것. 조용히 넘어가지 말 것.
