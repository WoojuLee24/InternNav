# 노은역 랜드마크 파이프라인 재현

순서대로 돌리면 `landmarks.json` 까지 나온다. 각 단계는 결과 파일이 있으면 건너뛴다.

```bash
R=scripts/dataset_converters/nav3d/noeun
A=scripts/apptainer/noeun_apptainer.sh

# 0) 준비물 — usdz 에서 원본 mesh 를 꺼낸다 (A5000 의 noeun_station_mid.ply 와 같다)
python3 -c "import zipfile,shutil; z=zipfile.ZipFile('data/GS_USDZ/noeun_station_collision.usdz'); \
  open('data/scene_data/noeun_station/noeun_station_mid.ply','wb').write(z.read('mesh.ply'))"

# 1) 바닥 구멍 조사 -> holes.csv  (75 초)
$A habitat python3 $R/11_check_assets.py --scene noeun_station \
  --usdz data/GS_USDZ/noeun_station_collision.usdz \
  --nav01 scripts/dataset_converters/nav3d/logs/noeun/12_build_navmesh/noeun_station

# 2) 씬 스캔 — 설 자리 계산(habitat) 후 렌더(isaac).  167 자리 x 4 방향 = 666 장, 35 분
$A habitat python3 scripts/eval/noeun_scene_scan.py --make_spots --grid_m 5 \
  --spots_json logs/scene_scan/spots.json --out logs/scene_scan
$A isaac "NOEUN_AGENT_SERVER=0 NOEUN_HEADLESS=1 NOEUN_DATA=<raw_data/r2r> \
  /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/noeun_scene_scan.py \
  --spots_json /ws/src/InternNav/logs/scene_scan/spots.json --out /ws/src/InternNav/logs/scene_scan"

# 3) Gemma 서버 (GPU 1) — 문장·이름 뽑기에 쓴다
$A habitat bash -c "PYTHONPATH=/ws/src/InternNav/.venv_label/site:/ws/src/InternNav \
  CUDA_VISIBLE_DEVICES=1 python3 scripts/dataset_converters/nav3d/common/vlm_server.py \
  --model_path <gemma-4-31B-it> --served_name gemma-4-31B-it --port 8200"

# 4) 랜드마크 — 이름 뽑기 -> 어휘 검증 -> 검출 -> 묶기  (약 50 분)
$A habitat python3 $R/69_scene_landmarks.py --snaps logs/scene_scan --out logs/scene_landmarks

# 5) 구멍 근처 랜드마크 제거 + 목적지 후보 찾기
$A habitat python3 $R/73_goal_candidates.py --landmarks logs/scene_landmarks \
  --holes scripts/dataset_converters/nav3d/logs/noeun/11_check_assets/noeun_station/holes.csv \
  --out logs/goal_candidates

# 6) .house 생성 (기존 MP3D 파이프라인이 그대로 읽는다)
$A habitat python3 $R/70_make_house.py --landmarks logs/scene_landmarks \
  --out data/scene_data/noeun_station/noeun_station.house

# 그림 · 편집 도구
$A habitat python3 $R/71_landmark_viz.py --snaps logs/scene_scan --landmarks logs/scene_landmarks --out logs/scene_landmarks/viz
$A habitat python3 $R/72_landmark_tool_data.py --landmarks logs/scene_landmarks --snaps logs/scene_scan --out logs/landmark_tool/data.json
```

검증 스크립트는 `scripts/apptainer/checks/` 참고.
