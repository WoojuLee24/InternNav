# Codex 04 기본 Open3D — 노은역 실행 결과

## 실행 대상

Isaac 버전이 아닌 원본 `04_render_obs.py`를 노은역 03 random 경로 20개에 실행했다. HTML은 이 스크립트의 기존 `render_generate_report`와 `save_gallery`가 생성했다.

## 최소 입력 연결

- 기본 04는 Open3D renderer이며 OBJ/PLY 입력만 지원한다.
- Open3D 0.19는 USDZ를 직접 읽지 못한다.
- 노은역 USDZ 안에 포함된 `mesh.ply`를 임시 디렉터리에 추출해 기존 `read_triangle_model`과 trimesh 검증 경로에 전달했다.
- D455 nominal K, 0.001m/raw, 10m cutoff를 공용 `camera_profiles.py`에서 읽는다.
- Matterport 기본값과 기존 dataset camera 경로는 유지했다.

## 실행

`EGL_PLATFORM=surfaceless /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs.py --scene noeun_station_mid --mode random --num_episodes 20 --usd_path data/noeun_station/noeun_station_collision.usdz --camera d455_nominal --out_dir scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex --log_dir logs/gs-vlnpe/codex_noeun`

Conda/mamba는 사용하지 않았다.

## 기하/depth 결과

- 20개 episode, 총 8,378 frames 생성
- RGB/depth/extrinsic 개수 전부 일치
- depth: uint16 270×480
- 저장 최대: 10,000 raw = 10m
- 인접 pose 간격 위반: 0
- mesh-anchor median: 0.00001m

## RGB 문제와 최종 판정

USDZ 내 `mesh.ply` 헤더에는 position과 triangle index만 있다. vertex color, normal, UV, texture가 모두 없다. 따라서 Open3D 기본 렌더러 결과는 거의 흰색이다.

- 첫 프레임 평균 밝기 범위: 249.03~254.75
- 첫 프레임 표준편차 범위: 1.02~11.24
- 여러 episode에서 245보다 어두운 픽셀이 1% 미만

기존 04는 step과 mesh-anchor만 검사해 처음에는 PASS로 출력했지만, 이 상태는 RGB-D 데이터로 쓸 수 없다. 노은역 분기에서 RGB mean/std 진단을 기존 report 요약에 추가했고 최종 판정은 FAIL로 바로잡았다.

- depth/pose 데이터: 정상
- RGB 데이터: 사용 불가
- 전체 RGB-D dataset: FAIL
- HTML: `logs/gs-vlnpe/codex_noeun/04_render_obs/noeun_station_mid_random/report.html`
- 출력: `noeun_pipeline_codex/obs/noeun_station_mid_random` 약 3.1GB

노은역의 색/재질은 USDZ의 `.nurec` 및 USD/Isaac 렌더 경로가 담당한다. 따라서 실제 RGB-D 생성에는 앞서 PASS한 `04_render_obs_isaac.py` 결과를 사용해야 한다.
