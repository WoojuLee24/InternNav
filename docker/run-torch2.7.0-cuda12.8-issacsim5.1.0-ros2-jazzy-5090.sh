#!/bin/bash

   # -v /home/universe/gd_project/modules/gd_vln/workspace/src/InternNav:/workspace/isaaclab/InternNav \
   # -v /home/universe/gd_project/modules/gd_vln/workspace/src/InternUtopia:/workspace/InternUtopia \

xhost +local:root
docker run --name internnav-torch2.7.0-cuda12.8-issacsim5.1.0-ros2-jazzy-5090 --shm-size 32g --entrypoint bash -it --gpus all --runtime=nvidia -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v $HOME/.Xauthority:/home/irteam/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/home/irteam/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/home/irteam/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/home/irteam/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/home/irteam/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/home/irteam/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/home/irteam/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/home/irteam/Documents:rw \
   -v /home/universe/gd_project/modules/gd_vln/workspace:/ws:rw \
   -v /home/universe/data:/datasets:ro \
   -v /media/TrainDataset:/ws/src/InternNav/data:rw \
   -v /media/TrainDataset/InternData-N1-v0.5-mini/scene_data/mp3d_pe:/isaac-sim/Matterport3D/data/v1/scans:ro \
   -w /ws \
   dnwn24/internnav:torch2.7.0-cuda12.8-issacsim5.1.0-ros2-jazzy-5090
