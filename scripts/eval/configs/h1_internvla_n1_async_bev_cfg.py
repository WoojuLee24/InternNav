"""H1 (Isaac Sim) async dual-system eval with BEV visual input.

Same as h1_internvla_n1_async_cfg.py except:
  - model_name='internvla_n1_bev'  → InternVLAN1AgentBEV + InternVLAN1NetBEV
    (new files, parent agent/policy untouched)
  - model_settings gains the visual_provider / bev_* keys

Set visual_provider='fpv' to reproduce the original behaviour exactly.
H1 camera: torso_link/h1_1_25_down_30 → height 1.25 m, pitched 30° down,
640×480 @ fx=fy=585. Obs depth is normalised [0, 1] over 10 m, hence
bev_depth_scale=10.0 for the S2 path (S1 depth is already metric).
"""

# Registers Agent 'internvla_n1_bev' (import side effect — do not remove)
import internnav.agent.internvla_n1_agent_bev  # noqa: F401  # isort: skip

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
        server_port=8023,
        model_name='internvla_n1_bev',
        ckpt_path='',
        model_settings={
            'env_num': 1,
            'sim_num': 1,
            'model_path': "/ws/src/InternNav/checkpoints/InternVLA-N1-w-NavDP",
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
            'infer_mode': 'partial_async',
            # debug
            'vis_debug': False,
            'vis_debug_path': './logs/test_n1_bev/vis_debug',
            # ---------------- BEV visual input ----------------
            'visual_provider': 'bev_image',  # 'fpv' | 'bev_image' | 'bev_feature'
            'bev_s1': True,   # replace NavDP images_dp with BEV frames
            'bev_s2': True,   # append BEV image to the LLM look-down turn
            'bev_cam_height': 1.25,
            'bev_cam_pitch_deg': 30.0,  # h1_1_25_down_30
            'bev_fx': 585.0,
            'bev_fy': 585.0,
            'bev_cx': 320.0,
            'bev_cy': 240.0,
            'bev_ref_width': 640,
            'bev_ref_height': 480,
            'bev_depth_scale': 10.0,  # S2 obs depth [0, 1] × 10 = metres
        },
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': False,
        },
    ),
    task=TaskCfg(
        task_name='test_n1_bev',
        task_settings={
            'env_num': 1,
            'use_distributed': False,
            'proc_num': 1,
            'max_step': 1000,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            scene_data_dir='/ws/src/InternNav/data/InternData-N1-v0.5-mini/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_flash=True,
        robot_platform_size=0.3,
        flash_collision='stop',
        robot_usd_path='/ws/src/InternNav/data/InternData-N1-v0.5-mini/Embodiments/vln-pe/h1/h1_internvla.usd',
        camera_resolution=[640, 480],  # (W,H)
        camera_prim_path='torso_link/h1_1_25_down_30',
        one_step_stand_still=True,
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            'base_data_dir': '/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r',
            'split_data_types': ['val_unseen'],
            'filter_stairs': True,
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': True,
        'vis_output': False,
        'show_rgb': False,
        'use_agent_server': False,
    },
)
