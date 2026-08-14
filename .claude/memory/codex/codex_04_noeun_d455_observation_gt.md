# Codex 04 — 노은역 D455 RGB/depth/pose GT

## 목적

03의 20개 경로를 원본 `04_render_obs_isaac.py`로 따라가며 RGB, uint16 depth, camera extrinsic GT를 생성한다.

## 카메라 규약

- profile: `d455_nominal`
- 해상도: 480×270
- K: [[240,0,240],[0,240,135],[0,0,1]]
- depth unit: 0.001m/raw
- synthetic storage cutoff: 10m
- render near/far: 0.05/12m

0.001m/raw는 D400 계열 Z16의 기본 depth unit과 맞춘 저장 단위다. uint16 이론 표현 한계는 65.534m지만, 이 데이터의 10m는 합성 데이터 저장 cutoff이며 실제 D455의 정확도 보장 거리라는 뜻은 아니다. 실제 장치 사용 시에는 RealSense API의 실측 intrinsics와 `get_depth_scale()`을 기록해야 한다. 기존 N1/D435i 기본 profile은 0.0001m/raw, 3m cutoff로 유지해 InternNav N1 계약과 충돌하지 않는다.

## 실행

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py --scene noeun_station_mid --mode random --num_episodes 20 --camera d455_nominal --out_dir scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex --log_dir logs/gs-vlnpe/codex_noeun`

## 실제 결과

- 판정: PASS, 20/20
- 총 8,378 frames
- 모든 episode에서 RGB/depth/extrinsic 프레임 수 일치
- depth 전부 uint16, 270×480
- 저장 샘플 최대: 10,000 raw = 10.0m
- 모든 경로 step 위반: 0
- 전체 mesh-anchor median: 0.00462m
- episode별 frame 수: 285, 344, 282, 611, 72, 682, 487, 383, 708, 407, 159, 845, 65, 420, 181, 417, 72, 463, 583, 912
- 출력 전체 크기: 약 3.1GB
- report: `logs/gs-vlnpe/codex_noeun/04_render_obs_isaac/noeun_station_mid_random_d455_nominal/report.html`

모든 저장과 PASS 출력 후 Isaac 종료 훅이 오래 대기했다. 결과와 report가 완성된 것을 확인한 뒤 종료 신호로 세션만 정리했다. 데이터 생성 실패는 아니다.
