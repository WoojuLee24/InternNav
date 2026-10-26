# nav3d 경로(sg·wp) 에 b1 · '후' 라벨 -> relabel_half 2x2 (진행 기록) — 2026-10-26~27 KST

## 라벨 이름 대응 (셀·체크포인트 파일명 255자 한도 때문에 짧은 이름)
| 짧은 이름 | 경로 | 라벨 (instruction_labels_nav3d/<이름>) | gen_hash |
|---|---|---|---|
| n3sg-b1 | nav3d_sg | b1_model-gemma31b-bf16 | a69461a93ffb (= R2R b1) |
| n3sg-post | nav3d_sg | b1_facts-memo-v1_knobs-v2_delivery-joint3-v1_writer-gepa-v3-joint-gemma_model-gemma31b-bf16 | f9fcc41eb6ab (= R2R '후') |
| n3wp-b1 | nav3d_wp | b1_model-gemma31b-bf16 | a69461a93ffb |
| n3wp-post | nav3d_wp | (n3sg-post 와 같음) | f9fcc41eb6ab |
- 라벨: train 10,684 (형제 라벨로 채운 1개: jh4fc5c5qoQ ep 82/85 — 사람 문장이 같아 R2R episode_id 가 겹침) / val_unseen 1,839, 생성 오류 0

## 데이터 만든 순서와 검증
1. **경로 npz 재생성** (nav3d 21, 이 서버에 원본 없음): `nav3d/logs/common/refine_paths{,_val_unseen}/scenes/` train 61 · val 11.
   원본 렌더 기록과 대조: 같은 follower 로 걸으면 action 이 **8,342/8,342 경로 완전히 같음** → 원본 경로와 같다.
   nav3d 출발점은 R2R 출발점과 최대 0.7 m 다르다 (경로 npz 의 첫 점을 써야 함).
2. **카메라 5종 렌더** (mini 학습과 같은 카메라: 125cm 0/30/45°, 60cm 15/30°): `configs/render_rig5.yaml`, `habitat_rig_sim.py` 에 rig 별 높이·hfov (optional).
   릴리스 GT 재생 대조 (3 scene x 8 ep): 5종 모두 SSIM ≥ 0.972, depth 정확히 일치 ≥ 99.2 %, 위치 오차 < 0.1 mm.
   **60cm hfov = 69.0°** (vlnce_rig.py 의 68.7 은 근사: SSIM 0.89 · depth 11 %).
   51_render --config render_rig5.yaml -> `vln_ce_nav3d_rig5` (GPU 당 1 프로세스, 2026-10-26 20:52~10-27 00:07). 기존 nav3d 와 대조: 색인 같음, 표본 80 action·pose 같음, 125cm_0deg 이미지 SSIM 1.0000.
   게이트 일부 FAIL (P1 1 묶음, P4 4 묶음) 은 경로 자체 판정 — 원본 nav3d 와 같은 경로.
3. **라벨** `nav3d/common/label_paths.py` 에 `--split val_unseen`, joint('후') 분기 (경로 단위로 형제를 묶어 joint._generate_traj; R2R 형제 수·캐시 문제 회피) 추가. 기본 동작 불변.
4. **학습·평가 데이터** `relabel_vlnce/build_nav3d_relabel.py`: `vln_ce_<짧은 이름>/traj_data` = rig5 scene 링크 + 문장만 라벨, eval = R2R val_unseen 1,839 (출발·도착·경로 원본) + nav3d val 라벨 문장. yaml = ld30.

## 학습 (relabel_half 와 같은 학습법: mini 61 scene, s1 1 epoch, 카메라 4 데이터셋, val_unseen 1,839 ld30)
- 비교 기준 (R2R 경로, 같은 학습법): GT 42.5 · b1 37.4 · '후' 37.3 (label->GT)
- 결과: (학습 후 채움)
