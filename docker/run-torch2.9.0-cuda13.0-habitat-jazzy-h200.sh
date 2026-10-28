#!/bin/bash
# H200 이미지 로컬 확인용 -> http://localhost:8888 (Jupyter Lab). 이 PC 에서는 빌드하지 않는다 (Dockerfile-torch2.9.0-cuda13.0-habitat-jazzy-h200 첫 줄 참고).
# 클러스터에서는 Kubeflow Notebook 생성 화면에 이미지 dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-h200 만 넣는다. 거기서는 이 스크립트를 쓰지 않는다.
# 코드는 /home/irteam/git/{InternNav,habitat-lab,InternUtopia} 에 두면 import 된다 (이미지의 gdvln-src.pth). 데이터는 /home/irteam/data-vol1
set -e
docker run --rm -it --gpus all --runtime=nvidia -p 8888:8888 --shm-size 32g dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-h200
