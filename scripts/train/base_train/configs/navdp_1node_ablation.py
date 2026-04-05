from internnav.configs.model.navdp import navdp_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss

navdp_1node_exp_cfg = ExpCfg(
    name='navdp_1node',
    model_name='navdp',
    
    torch_gpu_id=0,
    torch_gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    output_dir='checkpoints/%s/ckpts',
    tensorboard_dir='checkpoints/%s/tensorboard',
    checkpoint_folder='checkpoints/%s/ckpts',
    log_dir='checkpoints/%s/logs',
    local_rank=0,
    # device = None,
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_seen'],
        ckpt_to_load='',
        max_steps=195,
        sample=False,
        success_distance=3.0,
        start_eval_epoch=-1,
        step_interval=50,
    ),
    il=IlCfg(
        epochs=10,
        batch_size=64, # 32, 64 (optimal speed), 256
        lr=1e-4,
        num_workers=4, # 4 (optimal speed), 8, 16 
        weight_decay=1e-4,  # TODO
        warmup_ratio=0.05,  # TODO
        use_iw=True,
        inflection_weight_coef=3.2,
        save_interval_epochs=5,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='',
        lmdb_map_size=1e12,
        dataset_r2r_root_dir='data/vln_pe/raw_data/r2r',
        dataset_3dgs_root_dir='',
        dataset_grutopia10_root_dir='',
        lmdb_features_dir='r2r',
        lerobot_features_dir='data/vln_pe/traj_data/r2r',
        camera_name='pano_camera_0',
        report_to='none',  # wandb, tensorboard, none
        dataset_navdp='data/datasets/navdp_dataset_lerobot.json',
        root_dir='/home/irteam/git/InternNav/data/InternData-N1-v0.5-mini/vln_n1/traj_data',
        image_size=224,
        scene_scale=1.0,
        preload=False,
        random_digit=False,
        prior_sample=False,
        use_scipy_kdtree=True,  # ablation: True=scipy cKDTree O(N logM), False=numpy broadcast O(N*M)
        memory_size=8,
        predict_size=24,
        pixel_channel=4,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        channels=3,
        dropout=0.1,
        scratch=False,
        finetune=False,
        ddp_find_unused_parameters=False,
        val_ratio=0.1,
        # val_interval_steps=500,
        filter_failure=FilterFailure(
            use=True,
            min_rgb_nums=15,
        ),
        loss=Loss(
            alpha=0.0001,
            dist_scale=1,
        ),
    ),
    model=navdp_cfg,
)
