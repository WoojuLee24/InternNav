# Project Guidelines

## Python / pip 실행 경로
- `python` / `python3` → `/workspace/isaaclab/_isaac_sim/python.sh`
- `pip` / `pip3` → `/workspace/isaaclab/_isaac_sim/python.sh -m pip`
- `.bashrc`의 alias는 non-interactive shell에서 적용되지 않으므로 항상 full path 사용

## 제한사항
- 코드 변경은 최소한으로 (conservative changes)
  - 코드 변경할 부분이 많으면 새로운 파일을 생성해서 기존 코드에 영향을 적게
  - 기존 코드의 argument를 줘서 해결할 수 있으면 코드 변경 x
- 한 번에 하나의 파일만 수정

## scripts/train/base_train/
- start_1gpu.sh: CPU core 24, RAM 128G, RTX 5090 1GPU, shm 32g
- start_1node.sh: CPU core 128, RAM 2TB, H200 8GPU, shm 1.8TB

## 명령어 출력 규칙
- 멀티라인 `\` 이음 명령어는 copy-paste 시 `\` 뒤 공백으로 실패하므로 **항상 한 줄**로 제공

## r2r_h1_replay 데이터 수집 (eval_r2r_h1_replay.py)
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/eval/eval_r2r_h1_replay.py --config scripts/eval/configs/h1_internvla_n1_async_cfg_orig.py --r2r_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --save_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_h1_replay --scenes <SCENE> --max_eps <N>
```
- `--scenes` 생략 시 전체 scene, `--max_eps` 생략 시 scene당 전체 episode
- `--skip_existing` : 이미 저장된 scene/episode 건너뜀

## debug 이미지 저장 (save_debug_images.py)
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/eval/save_debug_images.py --src_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --scan <SCENE> --ep <N>
```
- `--src_dir` 에 `r2r` 또는 `r2r_h1_replay` 경로를 지정
- `--scan` / `--ep` 생략 시 전체 처리
- 출력: `traj_data/r2r_debug/<dataset_name>/<scan>/episode_XXXXXX/rgb/frame_XXXX.jpg` 및 `depth/frame_XXXX.png`
