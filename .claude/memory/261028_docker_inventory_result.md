# docker 이미지 정리 (2026-10-28)

용어
- **commit 이미지**: `docker commit` 으로 컨테이너 상태를 저장한 이미지. 무엇이 들어갔는지 기록이 없어 다시 만들 수 없다.
- **부모**: 그 이미지가 위에 얹혀 있는 이미지 (층 공유). 부모를 지워도 자식이 있으면 공간이 안 줄어든다.
- 크기는 이미지 단독 크기(공유 층 포함)라 더해도 실제 디스크 사용량이 아니다. 실제: 이미지 436 GB 중 396 GB 가 실행 중 컨테이너와 무관, 빌드 캐시 190 GB.

## 새 구조 (`InternNav/docker/`, 옛 파일은 `docker_old/`)

| 새 이미지 | 대체하는 것 | 내용 |
|---|---|---|
| `dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090` / `-h200` | habitat-2.9.0(commit), habitat-2.9.1, gdsim, h200-v2 | torch 2.9.0 cu130, habitat-sim(빌드 포함), transformers 4.51, gdsim apt |
| `dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090` / `-h200` | issacsim5.1.0-5090(commit), issaclab-5090, torch2.7.0-issacsim | Isaac Sim 5.1 + Lab + ROS2, CUDA 12.8 툴킷, InternUtopia 의존성 |

- 패키지 버전은 지금 쓰는 컨테이너(`internnav-habitat-navmesh`, `internnav-isaac`)의 `pip freeze` 에서 베이스 이미지 것을 뺀 목록으로 고정 (`requirements-*.txt`).
- commit 층에서 버린 것: `/root/.cache/pip` 11 GB, vscode/antigravity 서버 3.6 GB, `/root/.cache/ov` 2.5 GB, 깨진 torchcodec dev 빌드, 아무도 안 쓰는 cu12 nvidia 휠.
- 코드(InternNav, habitat-lab, InternUtopia)는 이미지에 넣지 않고 마운트 + `.pth` (기존 `pip install -e` 와 같은 효과).

## 전체 목록

| 이미지 | 크기 | 쓰는 곳 | 제안 |
|---|---|---|---|
| **gd_vln — 지금 사용 중** | | | |
| internnav:torch2.9.0-cuda13.0-issacsim5.1.0-ros2-jazzy-5090 (commit) | 58.3 | `internnav-isaac` 실행 중, run_vln_stack.sh, command.md, nav3d 문서 | 새 isaac-5090 검증 후 삭제 |
| internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090-old (commit, 원래 태그는 새 이미지가 가져감) | 53.5 | `internnav-habitat-navmesh` 실행 중, nav3d 문서 | 새 habitat-5090 검증 후 삭제 (같은 태그를 새 이미지가 가져가므로 먼저 `-old` 태그로 옮김) |
| internnav:torch2.9.1-cuda13.0-habitat-jazzy-5090-gdsim | 47.4 | command.md, build_and_verify.sh, `navverse-dl`(정지) | 〃 |
| `<none>` 29b183 (옛 gdsim) | 47.4 | `internnav-gdsim` 실행 중 | 〃 |
| internnav:torch2.9.1-cuda13.0-habitat-jazzy-5090 | 47.4 | gdsim 의 부모 | 〃 |
| **gd_vln — 안 씀 (참조 없음 또는 docker_old 만)** | | | |
| internnav:torch2.7.0-cuda12.8-issacsim5.1.0-ros2-jazzy-5090 | 51.9 | docker_old run 파일만 | 삭제 |
| internnav:torch2.9.0-cuda13.0-issaclab-ros2-jazzy-5090 | 39.9 | isaacsim-5090 · vln-challenge 의 부모 | 자식 지우면 삭제 |
| internnav:torch2.9.0-cuda13.0-issac-ros2-jazzy-h200-v2 | 41.5 | 없음 (H200 옛 이미지) | 삭제 |
| internnav:`<none>` 781e | 52.0 | 없음 | 삭제 |
| internnav:torch2.9.0-cuda13.0-issac-ros2-jazzy-5090 | 48.5 | **ultralytics(gd_detector) 의 부모** | gd_detector 정리 때 |
| vln-challenge:torch2.7.0-cuda12.8-issacsim5.1.0-ros2-jazzy-5090 | 43.7 | VLN-Challenge/docker run 파일 | 확인 필요 |
| isaac-lab-ros2-sim5:latest | 18.1 | 없음 | 삭제 |
| isaac_sim:5.1.0-jazzy | 19.1 | 없음 | 삭제 |
| **베이스 — 유지** | | | |
| isaac-lab-ros2:latest (= dnwn24/isaac-lab-ros2, 같은 ID) · isaac-lab-base | 18.1 | 새 isaac 이미지의 부모 (docker-isaac/ 로 빌드) | 유지 |
| nvcr.io/nvidia/isaac/ros:noble-ros2_jazzy_… | 38.8 | 새 habitat 이미지 · gs-sdf 의 부모 | 유지 |
| isaac_sim:5.1.0-lab-jazzy-torch2.8.0-cuda12.9 | 31.8 | agentic_control 의 부모 | 유지 |
| **sim / gdm** | | | |
| sim_docker-sim:latest · sim_docker-sim:3dgrut | 29.8 · 50.4 | Isaac Sim (`isaac-sim` 컨테이너), compose.3dgrut | 유지 |
| `<none>` 04ae07 (sim_docker-sim 자식, 10 일 전) | 48.7 | 없음 (3dgrut 옛 빌드로 보임) | 삭제 |
| gdm_docker-gdm:latest | 20.9 | gdm | 유지 |
| **다른 프로젝트 — 유지** | | | |
| cvl:* (8 개, `<none>` 3 개 포함), hcnet, ccvpe | | 다른 이미지 (사용자) | 유지 |
| agentic_control:* | 62.6 | 다른 이미지 (사용자) | 유지 |
| 3dgrut:cuda128 · dnwn24/3dgrut:… | 24.2 · 29.0 | ply → usdz 변환 (사용자) | 유지 |
| gs-sdf, ultralytics:*, yolov8, gr00t-5090, vln-annotator | | gd_mapping · gd_detector · gd_vla · vln-annotator | 유지 |
| conf-watch:dev + `<none>` 142 MB 3 개 | 0.1 | conf-watch 옛 빌드 | `<none>` 삭제 |
| **삭제함** | | | |
| navverse-gd:local · cached · customized-isaaclab | 53.2 | (사용자 지시) | 2026-10-28 삭제 |

## 삭제 후보와 확보 용량 (2026-10-28 측정, 아직 안 지움)

새 이미지 검증이 끝나면 지운다. 컨테이너가 쓰는 이미지는 컨테이너를 먼저 내려야 지워진다.
확보 = 그 이미지(또는 묶음)만 가진 층의 크기 (`docker system df -v` 의 UNIQUE + 같이 지우는 부모 층).

| 묶음 | 이미지 (`dnwn24/internnav:` 뒤) | 쓰는 컨테이너 | 확보 |
|---|---|---|---|
| 옛 isaac | `torch2.9.0-cuda13.0-issacsim5.1.0-ros2-jazzy-5090`, `torch2.7.0-cuda12.8-issacsim5.1.0-ros2-jazzy-5090`, 부모 `torch2.9.0-cuda13.0-issaclab-ros2-jazzy-5090` | internnav-isaac | 약 50 GB (vln-challenge 를 남기면 약 28 GB) |
| 옛 habitat | `torch2.9.0-cuda13.0-habitat-jazzy-5090-old` | internnav-habitat-navmesh | 약 15 GB |
| 옛 gdsim | `torch2.9.1-cuda13.0-habitat-jazzy-5090(-gdsim)`, `<none>` 29b183 | internnav-gdsim, navverse-dl | 약 9 GB |
| 옛 H200 | `torch2.9.0-cuda13.0-issac-ros2-jazzy-h200-v2` | — | 약 3 GB |
| 참조 없음 | `<none>` 781e (부모 issac-ros2-5090 은 ultralytics 가 써서 남김) | — | 약 3.5 GB |
| 참조 없음 | `isaac-lab-ros2-sim5:latest`, `dnwn24/isaac_sim:5.1.0-jazzy` | — | 약 6.6 GB |
| 확인 필요 | `dnwn24/vln-challenge:torch2.7.0-…-5090` (VLN-Challenge run 파일) | — | 1.3 GB (+ 위 22 GB) |
| 거의 0 | `<none>` 04ae07 (sim_docker-sim:3dgrut 와 층 공유), conf-watch `<none>` 3 개 | — | ~0 |
| **이미지 합계** | | | **약 88 GB** |
| **빌드 캐시** | `docker builder prune` (870 개, 전부 미사용) | — | **234 GB** |
