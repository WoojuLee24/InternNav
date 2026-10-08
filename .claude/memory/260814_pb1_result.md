# P-B1 결과 — EmbodimentAugmenter 독립 구현 + 검증

작성 2026-08-14. 신규 `3dloader_vlnce/embodiment_augment.py`, `3dloader_vlnce/06_validate_augment.py`.
리포트 `logs/embodiment_augment/pb/` + Artifact. 기존 학습/데이터 코드 **무수정**(3dloader_vlnce 정책).

## 판정: PASS — augmenter end-to-end 동작 확인

파이프라인(전부 기존 코드 재사용): vln_ce 에피소드 → `vlnce_align`(T_sf2mesh) →
`plan_episode`로 r_b별 GT path 재계획 → 새 path에서 **depth-only 렌더(타일 상주, G3 최적화)** →
`depth_to_bev_occ_ros2`로 BEV.

17DRP5sb8fy ep0(정합 resid 0.0035):
- r_b=0.15 → path 6.93m, r_b=0.30 → path 6.04m (**같은 start/goal, 다른 경로** — 오버레이에서 파랑=작은
  로봇이 벽에 붙고 노랑=큰 로봇이 clearance 확보하며 우회), r_b=0.50 → 복도 좁아 infeasible.
- 각 r_b에서 새 path의 depth 6프레임 + BEV 렌더 성공(문/방 구조 depth, robot-centric BEV occupancy).

## 핵심 구현 포인트
- **depth-only + 타일 씬 상주**(G3): `render_along`(clear+add+rgb+depth 매번) 대신 타일 1회 add 후
  프레임마다 `set_camera_pose + render_to_depth_image`만. per-frame ~10ms.
- **다층 씬 floor_z**: occ npz의 global floor_z가 아니라 **에피소드 층**(`median(cam_z)-cam_height`)을
  plan_episode에 넘겨야 함(안 그러면 다른 층이라 start/goal not navigable).
- **start/goal 스냅**: 벽 근처 GT 끝점이 r_b dilation으로 지워지므로 e-navigable 최근접 셀로 스냅
  (`_snap_navigable`). embodiment 투영으로 정당. 스냅 후에도 좁은 복도는 infeasible(정상 — 큰 로봇 못 지나감).

## 한계/관측
- v1은 r_b/h_nav만 → 관측(depth/BEV)은 **GT path가 바뀌면서 간접적으로** 바뀜(카메라가 새 경로를 따라감).
  cam_height/pitch 직접 변화(관측 직접 변경)와 r_b-dilated BEV(논문 C1 모델 변경)는 v2.
- 작은 씬(17DRP)은 r_b 큰 값에서 feasibility 제한 큼. 큰/복잡 씬일수록 경로 분기 다양.

## 다음: P-B2 (이식) — **사용자 시각화 검증 후**
- `internnav/dataset/internvla_n1_lerobot_dataset.py` `__init__`+`__getitem__:~1238` guard 1줄로 연결.
- 플래그 threading(Params/train_argv/DataArguments, 기본 off), 신규 config `s1.bev.occ_aug_s2.fpv.py`.
- traj_poses/traj_depths를 dataloader 포맷(상대 pose)으로 repackage.
- 검증: embodiment_aug=False로 loss 불변.
