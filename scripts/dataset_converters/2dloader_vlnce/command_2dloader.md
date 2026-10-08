# 2dloader_vlnce 명령어

이 컨테이너는 habitat 환경 → **전부 `/usr/bin/python`**. 모든 명령은 repo 루트(`/ws/src/InternNav`)에서 실행.

## self-check (프레임워크 없는 회귀 테스트)

```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/episode_io.py
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/local_map.py
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/obstacle_synth.py
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/augment2d.py
```

## 검증 게이트

```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w0_frame.py --scene 17DRP5sb8fy --episode 0 --preset 125cm_0_30 --n_frames 6
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w1_oracle.py --scene 17DRP5sb8fy --episode 0 --preset 125cm_0_30 --n_frames 6 --esdf_dir scripts/dataset_converters/gs_vlnpe/logs/esdf
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w2_obstacle.py --scene 17DRP5sb8fy --episode 0 --n_frames 6 --seed 0
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w3_embodiment.py --scene 17DRP5sb8fy --episode 0 --n_frames 6 --r_bs 0.10,0.20,0.35,0.50 --obstacle 0
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w4_pixelgoal.py --scene 17DRP5sb8fy --episode 0 --n_frames 6 --r_bs 0.10,0.20,0.35,0.50 --goal_adjust retreat
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w5_budget.py --scene 17DRP5sb8fy --episode 0 --n 50
```
```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w6_paired.py --scene 17DRP5sb8fy --episode 0 --n_frames 4 --r_bs 0.10,0.20,0.35,0.50 --obstacle 1
```

두 번째 씬으로 일반화 확인 (`--out_dir`를 바꿔 덮어쓰기 방지):

```
/usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w3_embodiment.py --scene s8pcmisQ38h --episode 0 --n_frames 6 --obstacle 0 --out_dir logs/embodiment_augment2d/s2_w3
```

## 리포트 발행 (Artifact)

```
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/publish_artifact_report.py --stage_dir logs/embodiment_augment2d/w2 --title "W2 — 장애물 합성 RGB/depth/BEV 정합" --eyebrow "2dloader_vlnce · embodiment augmentation"
```

## argument 의미

| argument | 의미 |
|---|---|
| `--scene` | MP3D scan id (`data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/<scene>`) |
| `--episode` | 에피소드 인덱스 |
| `--preset` | `<H>cm_<pitch1>_<pitch2>`. 학습 preset과 같은 형식 (`125cm_0_30`, `60cm_15_15` 등). **pose·depth·pixel goal은 전부 pitch_2(룩다운) rig 기준**이고 pitch_1은 S2가 보는 FPV RGB |
| `--n_frames` | goal이 있는 프레임 중 균등 추출 개수 |
| `--r_bs` | 로봇 반경 목록 (m). ESDF 임계 `truncate_navigable(esdf, r_b)`로 들어간다 |
| `--h_nav` / `--h_b` | BEV slab `(h_nav, h_b]`. 밟고 넘는 높이 / 로봇 높이. `depth_to_bev_occ_ros2`의 `z_min`/`z_max` |
| `--goal_adjust` | `retreat`(2D 기본) / `shift` / `nearest` — goal이 이 `r_b`에서 통행 불가일 때 옮기는 방식 |
| `--unknown` | `nontraversable`(기본) / `free` / `block` — 미관측 셀 정책. §결과 메모 참고 |
| `--obstacle` | 1이면 합성 장애물도 함께 (W3/W4/W6) |
| `--seed` | 장애물 샘플링 시드 |
| `--esdf_dir` | W1 전용. 3D 오라클 occupancy npz (`gs_vlnpe/logs/esdf/<scene>.npz`) |
| `--bev_sizes` | W5 전용. planning 격자 해상도 sweep |
| `--out_dir` | 리포트 출력 (`logs/embodiment_augment2d/<stage>`) |

## 코드 정책

- 신규 코드는 전부 이 폴더. `gs_vlnpe/`·`3dloader_vlnce/`·`internnav/`는 **import만** 하고 수정하지 않는다.
- 재사용: `esdf_utils`(ESDF/A*/refine/thin/spline/feasibility), `pixel_goal_utils`(rig intrinsics·pixel goal
  왕복·goal 조정), `viz_utils`(HTML 갤러리), `geometry_utils`(depth colormap/jpg),
  `depth_rgb_to_bev_torch.depth_to_bev_occ_ros2`(학습·평가와 **같은** BEV 함수),
  `vlnce_align`+`verify_pose_mesh`(W1의 3D 오라클 정합), `publish_artifact_report`(Artifact 발행).
- 장애물 종류를 늘릴 때 손댈 곳은 `obstacle_synth.render_obstacle`의 `kind` 분기 + `footprint_xy` 뿐이다
  (BEV는 합성된 depth에서 다시 계산되므로 하류는 kind를 모른다).

## 발행된 리포트 목록
[`reports.md`](reports.md) — 7개 게이트 + 두 번째 씬 6개 + 3dloader R2 정정의 Artifact 주소와 결과 요약.
