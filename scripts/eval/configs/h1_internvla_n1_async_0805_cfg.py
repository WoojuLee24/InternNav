import os
import sys

# runner.py --config로 돌릴 때 PARAMS가 필요해서 추가 (default_config.py는
# scripts/train_eval/qwenvl_train/에 있음). scripts/eval/eval.py로 직접 돌릴
# 때도 동작하도록 경로를 직접 계산해서 넣는다 (runner.py의 sys.path 설정에
# 의존하지 않음).
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "train_eval", "qwenvl_train"
))
import default_config as base  # noqa: E402

# from scripts.eval.configs.agent import *
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8024,
        model_name='internvla_n1',
        ckpt_path='',
        model_settings={
            'env_num': 1,
            'sim_num': 1,
            'model_path': "/ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN", # "/home/irteam/git/InternNav/checkpoints/InternVLA-N1-w-NavDP", # 
'mode': 'dual_system',
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            'resize_w': 384,
            'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
            'infer_mode': 'partial_async',  # You can choose "sync" or "partial_async", but for this model, "partial_async" is better.
            # debug
            'vis_debug': True,  # If vis_debug=True, you can get visualization results
            'vis_debug_path': '/media/T7/InternNav_eval/test_n1_0805/vis_debug',
        },
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'max_episodes': 100,
            'use_fabric': False,  # Please set use_fabric=False due to the render delay;
            'headless': True, # Isaac-sim
        },
    ),
    task=TaskCfg(
        task_name='test_n1_0805',
        task_settings={
            'env_num': 1,
            'use_distributed': False,  # If the others setting in task_settings, please set use_distributed = False.
            'proc_num': 1,
            'max_step': 1000,  # If use flash mode，default 1000; descrete mode, set 50000
        },
        scene=SceneCfg(
            scene_type='mp3d',
            scene_data_dir='/ws/src/InternNav/data/InternData-N1-v0.5-mini/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_flash=True,  # If robot_flash is True, the mode is flash (set world_pose directly); else you choose physical mode.
        robot_platform_size=0.3, # 0.12 (Default), None (robot prim) # The robot platform size (diameter) used for collision detection in flash mode. If None, it will be set to 0.12m by default.
        flash_collision='stop',  # None: no detection, 'stop': stop on collision, 'reset': episode failure on collision
        robot_usd_path='/ws/src/InternNav/data/InternData-N1-v0.5-mini/Embodiments/vln-pe/h1/h1_internvla.usd',
        camera_resolution=[640, 480],  # (W,H)
        camera_prim_path='torso_link/h1_1_25_down_30',
        one_step_stand_still=True,  # For dual-system, please keep this param True.
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            'base_data_dir': '/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r',
            'split_data_types': ['val_unseen'],  # 'val_seen'
            'filter_stairs': True,  # For iros challenge, this is False; For results in the paper, this is True.
            # 'selected_scans': ['zsNo4HB9uLZ'],
            # 'selected_scans': ['8194nk5LbLH', 'pLe4wQe7qrG'],
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': True,
        'vis_output': True, # True,
        'show_rgb': False, # True,
        'use_agent_server': False, # True,  # If use_agent_server=True, please start the agent server first.
    },
)

# eval-only config: 위 eval_cfg는 그대로 유지하고, runner.py 자체의 북키핑
# (nproc/debug_dir/wandb 등)에만 쓰이는 기본 Params를 그대로 가져온다.
# runner.py로 돌릴 때는 반드시 --no-train과 함께 써야 한다 (이 config는
# 학습 하이퍼파라미터를 나타내지 않음).
PARAMS = base.PARAMS
