# `00_inspect_vln_n1.py` (M1.0) — 완료

정답지 `vln_n1`에서 씬 1개를 읽어 ① 이후 파이프라인이 지킬 스키마를 `target_schema.json`으로 고정
② pose 규약을 실증 판별 ③ 에피소드별 로봇 파라미터 분포를 기록한다.

> 기하·좌표 규약의 상세와 **과거 오진 이력**은 `understanding_gs_vlnpe_fpv_bev_geometry.md` 참고.
> 이 문서는 스크립트의 입출력·수치만 다룬다.

## 실행

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/00_inspect_vln_n1.py --scene 17DRP5sb8fy --episode 0
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--data_root` | `data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i` | GT 루트 |
| `--mesh_root` | `data/scene_data/mp3d_n1` | 씬 mesh — pose 규약 판별의 **절대 앵커** |
| `--scene` / `--episode` | `17DRP5sb8fy` / `0` | |
| `--frame_gap` | `15` | 규약 **판별용** 간격 (baseline이 넓어야 후보가 갈림) |
| `--visual_gap` | `3` | 리포트 blink **육안용** (판별용 gap을 쓰면 정답도 안 맞아 보임) |
| `--num_robot_param_episodes` | `0`(전체) | `camera_extrinsic` 스캔 에피소드 수 |
| `--num_pairs` / `--bev_cell_m` / `--max_depth_m` | `5` / `0.05` / `10.0` | |
| `--out_dir` / `--log_dir` | `scripts/dataset_converters/gs_vlnpe` / `logs/gs-vlnpe` | canonical / 시각화 |

## 입력

| 파일 | 내용 |
|---|---|
| `<scene>/meta/info.json` | fps, codebase_version |
| `<scene>/data/chunk-000/episode_*.parquet` | `camera_intrinsic`(3,3 상수), `camera_extrinsic`(4,4 **에피소드별** `h_b`+pitch), `action`(4,4 매 프레임 pose) |
| `videos/chunk-000/observation.images.rgb/*.jpg` | uint8 (270,480,3) — **1차 rgb 소스** |
| `videos/chunk-000/observation.images.depth/*.png` | uint16 (270,480), raw/10000 = m — **1차 depth 소스** |
| `<mesh_root>/<scene>/matterport_mesh/*/*.obj` | pose 규약 판별용 절대 앵커 |

`observation.video.{rgb,depth}.mp4`는 8bit 손실 재인코딩본 — **메트릭 용도 금지**.

## 출력

| 경로 | 내용 |
|---|---|
| `target_schema.json` (canonical) | `source`, `fps`, `frame_counts`, `parquet_schema`, `image_streams`, **`robot_params`**, `pose_convention`, `notes` |
| `logs/gs-vlnpe/00_inspect_vln_n1/<scene>/episode_*/diagnostics/` | 판별용(gap 15) / 육안용(gap 3) 진단 이미지 |
| `logs/.../report.html` | pose 규약 요약 + blink 3종(재구성·실제촬영·BEV) + reference 모달 |

`robot_params`는 `camera_extrinsic`을 전 에피소드 스캔해 `h_b`·`pitch_down`·`floor_z` 분포를 남긴 것.
03/04가 여기서 샘플링한다. 분해식은 `understanding_...` §2.

## 실행 결과

```
17DRP5sb8fy (52 ep)   cam2world_gl  err= 2.076  mesh_dist=0.000029 m   <- 채택 (margin 1.000)
                      cam2world     err=41.606  mesh_dist=0.418680 m
                      world2cam     err=46.754  mesh_dist=0.957407 m
                      robot_params  h_b 0.259~1.474, pitch 0~30 deg, floor_z -0.014 (상수)
                      visual        FPV err=2.347 / BEV err=2.646 (coverage 0.178)

s8pcmisQ38h (99 ep)   cam2world_gl  err= 2.662  mesh_dist=0.000028 m   <- 동일 결론
                      robot_params  h_b 0.254~1.457, floor_z -0.001 (상수)
```

**판별 근거는 photometric error가 아니라 mesh 표면거리다** — 상대 지표는 축 컨벤션 오류를 상쇄해
좁은 baseline에서 정답/오답을 구분하지 못한다.

## 참고

- Artifact: https://claude.ai/code/artifact/de729add-5777-444e-8491-d4ae1dc966f4
- 설계: `docs/execution-staged.md` M1.0 절
