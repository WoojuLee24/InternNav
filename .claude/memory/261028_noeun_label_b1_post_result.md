# 노은역 nav3d 데이터에 b1 · '후' instruction 라벨 — 2026-10-28 KST

## 데이터 (Drive `InternNav/data/InternNav/data/`, 2026-10-28 21:00 KST 업로드판, md5 일치)
- 원본 tar: `data-vol2/nav3d/noeun/drive_20261028_2100kst/` · 풀린 곳: `data-vol1/nav3d/noeun/<이름>/`
- 4 데이터셋 = {vln_ce, vln_pe} x {rbq_v6, rbq_hunav_v1}. vln_ce 와 vln_pe 는 경로가 같다 (reference_path·start 일치).
- split: `traj_data/r2r` (train) + `traj_data/r2r_val_seen` + `raw_data/r2r/{train,val_seen}/*.json.gz`. v6 200+40, hunav_v1 187+37 경로. 같은 노은역 scene 안 split (unseen 아님).
- 렌더 rig 는 60cm_15deg 하나 (나머지 4 rig 는 빈 값). 이미지 jpg.
- 이전에 받은 train 만 있던 판(05:20 업로드)은 이번 판 train 과 파일·크기·meta·parquet 동일 → `data-vol1/nav3d/noeun/_old_train_only_20261028/` 로 비켜 둠 (지워도 됨).

## 라벨 설정 (사용자 확인)
| 이름 | 설정 | gen_hash |
|---|---|---|
| b1 | method b1 + model=gemma31b-bf16 + frames.rig=60cm_15deg + route.rooms=noeun-v1 | 0d611d54de08 |
| 후 (post) | b1 + route.facts=memo-v1 · style.knobs=v2 · delivery=joint3-v1 · style.writer=gepa-v3-joint-gemma + 같은 3개 | 5bc6460a380b |
- nav3d n3sg-b1(a69461a93ffb) / n3sg-post(f9fcc41eb6ab) 와 다른 점은 rig 와 rooms 두 가지뿐 (노은역에 다른 rig 가 없고 .house 방 주석이 없어서).
- 경로당 문장 3 개 (형제). vln_ce tasks = 형제 0, vln_pe tasks = 형제 0·1·2, raw json instruction = 형제 0.
- route.rooms=noeun-v1: 구역 16 개 (층 B2/B1/G 높이 띠 + 계단 C1~C3 + 층 안 k-means). 이름은 확인 페이지 https://claude.ai/artifact/XcBq1oUwbaG671KHhKiLh7 에서 사용자 확인.
  예: "Area changes along the path (from station layout): leave the platform, enter the stairs, enter the paid area, enter the ticket gates, enter the concourse. You stop in the concourse."

## 코드 (전부 신규, route.yaml 은 rooms.noeun-v1 항목 추가만)
| 파일 | 하는 일 |
|---|---|
| `nav3d/noeun/61_build_zones.py` | 구역 점 파일 `nav3d/noeun/zones/noeun_zones_v1.json` 생성 |
| `nav3d/noeun/noeun_rooms.py` | `rooms.room_sequence` 노은역판 (가장 가까운 구역 점) |
| `nav3d/noeun/62_label.py` | 라벨 생성 (e2e.generate / label_paths.joint_results 그대로 호출, rooms.room_sequence 만 바꿔 끼움) |
| `nav3d/noeun/63_build_labeled.py` | 라벨을 붙인 새 데이터셋 (data/videos 는 원본 링크) |
- `/usr/bin/python scripts/dataset_converters/nav3d/noeun/62_label.py --methods b1,post --managed` (gemma-4-31B-it vLLM DP8, 전체 약 6 분)
- `/usr/bin/python scripts/dataset_converters/nav3d/noeun/63_build_labeled.py`

## 결과
- 라벨: `data-vol1/nav3d/noeun/instruction_labels/<라벨 이름>/<ds>_<split>/` (meta/ep_*.json + json.gz). 2,784 문장, 오류 0.
- 데이터셋: `data-vol1/nav3d/noeun/{vln_ce,vln_pe}_noeun_nav3d_rbq_{v6,hunav_v1}_{b1,post}/` 8 개.
- 단어 평균: b1 26.5~28.4 · post 23.5~24.9.
- ⚠️ 품질 경고 `no_landmark` b1 20.1 % · post 22.3 % (nav3d sg 는 0.3 % · 2.3 %): e2e.LANDMARK_NOUNS 가 집 물건(table, bed, kitchen …)만 있어 역 랜드마크(escalator, ticket gates, pillar)를 못 센 **거짓 경고**. 그런데 이 때문에 5 번 재시도해 마지막 시도(temperature 0.35 -> 0.75) 문장이 남았다.
- b1 형제 3 개 중 똑같은 문장이 있는 경로 39/464 (b1 은 형제가 seed 만 다름). post 는 0.
