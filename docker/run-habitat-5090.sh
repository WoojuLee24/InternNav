#!/bin/bash

xhost +local:root

docker run --name internnav-habitat \
    --shm-size 32g \
    --entrypoint bash \
    -it --rm \
    --network=host \
    --gpus all \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v /home/universe/gd_project/modules/gd_vln/workspace:/ws:rw \
    -v /home/universe/gd_project/modules/gd_vln/workspace:/gd_vln:rw \
    -v /home/universe/data:/datasets:ro \
    -v /media/TrainDataset:/ws/src/InternNav/data:rw \
    -w /ws \
    dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090
