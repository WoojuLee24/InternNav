# nav3d 06r 용 H1 env 설정 — scripts/eval/configs/h1_cma_cfg.py 를 복사해 경로만 이 PC 기준 절대 경로로 고쳤다.
# vln_pe 릴리스 카메라 (torso_link/h1_pano_camera_0, 256x256, hfov 90) · 물리 모드 (robot_flash 없음 = False).
# 원본의 'data/Embodiments/...' · 'data/scene_data/...' 는 이 PC 에 없다 (InternData-N1-v0.5-mini 아래에 있음).
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
        server_port=8087,
        model_name='cma',
        ckpt_path='checkpoints/r2r/fine_tuned/cma_plus',
        model_settings={},
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='cma_plus_eval',
        task_settings={
            'env_num': 1,
            'use_distributed': False,
            'proc_num': 8,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            scene_data_dir='/ws/src/InternNav/data/InternData-N1-v0.5-mini/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_usd_path='/ws/src/InternNav/data/InternData-N1-v0.5-mini/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
        camera_resolution=[256, 256],  # (W,H)
        camera_prim_path='torso_link/h1_pano_camera_0',
        vlnce=False,  # vlnpe by default
        obstacle_detection=False,  # whether allow flash across obstacle
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            'base_data_dir': '/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r',
            'split_data_types': ['val_unseen', 'val_seen'],
            'filter_stairs': True,
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': True,
        'vis_output': True,
        'use_agent_server': True,
    },
)
