# gd_vln docker

이미지 2 종(habitat, isaac) × GPU 2 종(5090, H200). 파일 이름 = 이미지 태그 (`Dockerfile-<태그>`, `run-<태그>.sh`).
옛 Dockerfile · run 파일은 `../docker_old/` (참고용, 쓰지 않음).

이름 규칙: `dnwn24/internnav:torch{}-cuda{}-{환경}[-{기능}]-{ros2}-{gpu}`
- isaac 의 torch 는 Isaac python 의 것 (2.7.0+cu128).
- 기능(3dgrut, tensorrt 등)은 이미지에 넣지 않고 마운트된 작업 폴더의 venv 로 둔다 (예: 경량화 TRT = `vln-deploy/quantization_kit/convenv_x86`, `tools/setup_venv.sh`). 그래서 이름에 기능이 없다. 기능을 이미지에 꼭 넣어야 할 때만 붙인다.

| 이미지 (`dnwn24/internnav:` 뒤) | 하는 일 | 실행 | 컨테이너 |
|---|---|---|---|
| `torch2.9.0-cuda13.0-habitat-jazzy-5090` | 학습 · habitat 평가(VLN-CE) · HF 원본 기준값 · sim 연동(gdsim) | `run-torch2.9.0-cuda13.0-habitat-jazzy-5090.sh` | `internnav-habitat` |
| `torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090` | Isaac Sim 5.1 (VLN-PE / InternUtopia · nav3d 렌더) · 경량화(`vln-deploy/docs/command.md`) · sim 스택 S1/S2 | `run-torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090.sh` | `internnav-isaacsim` |
| `torch2.9.0-cuda13.0-habitat-jazzy-h200` | 위 habitat 의 H200 판 (Kubeflow Notebook) | Kubeflow 에 이미지 지정 | — |
| `torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-h200` | 위 isaac 의 H200 판 (Kubeflow Notebook) | Kubeflow 에 이미지 지정 | — |

```bash
docker/run-torch2.9.0-cuda13.0-habitat-jazzy-5090.sh     # 이미지가 없으면 빌드, 이미 떠 있으면 셸만 하나 더 연다
```

- 5090 run 파일: root 셸, 작업 폴더를 `/ws` 에 마운트, 이미지가 없으면 같은 이름의 Dockerfile 로 빌드한다 (label `gdvln.git` = 빌드한 git 커밋).
- H200 Dockerfile 은 5090 이미지 위에 얹는다 (Triton arch sm90, 코드 위치 `/home/irteam/git`, uid 1000 = irteam, Jupyter Lab). 빌드 명령은 각 Dockerfile 첫 줄. H200 run 파일은 그 이미지를 로컬에서 띄워 보는 용도다.
- H200 노트북을 처음 만든 뒤 한 번 `ln -s /home/irteam/data-vol1 /home/irteam/git/InternNav/data` (옛 `install_habitat_h200.sh` 의 [0] 단계. 나머지 단계는 이미지 빌드에 들어갔다).

규칙
- **`docker commit` 금지.** 바꿀 것이 생기면 Dockerfile 이나 `requirements-*.txt` 를 고치고 다시 빌드한다.
- 코드(InternNav, habitat-lab, InternUtopia)는 이미지에 넣지 않는다. 5090 `/ws/src`, H200 `/home/irteam/git` 아래에 두면 이미지의 `gdvln-src.pth` 가 import 경로에 넣는다 (옛 `pip install -e` 와 같은 효과).
- 파이썬 패키지 버전은 `requirements-*.txt` 에 고정돼 있다 (2026-10-28 5090 에서 쓰던 컨테이너의 `pip freeze` 기준). `--no-deps` 로 깔아 resolver 가 버전을 바꾸지 못하게 한다 — 패키지를 추가할 때는 의존성까지 직접 적는다.
- isaac 베이스 `dnwn24/isaac-lab-ros2:latest` 는 `../docker-isaac/` (Isaac Lab 공식 docker) 로 만든다.
