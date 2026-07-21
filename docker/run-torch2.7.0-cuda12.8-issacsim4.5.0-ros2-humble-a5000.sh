#!/bin/bash

xhost +local:root
docker run --name internnav --entrypoint bash -it --gpus '"device=0"' --runtime=nvidia --cpus="12" --memory="60g" --shm-size="16gb" -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v /home/dnwn24/gd_project/modules/gd_vln/workspace/:/ws \
   -v /media/TrainDataset:/ws/src/InternNav/data \
   -w /ws \
   dnwn24/internnav:torch2.7.0-cuda12.8-issasim4.5.0-ros2-jazzy-a5000
