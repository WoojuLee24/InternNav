"""
habitat_vln_ce_chrononav_full_cfg.py — ChronoNav full val_unseen eval (1839 eps).

DualVLN S2-only 4-bit + temporal cache τ=0.92 (inference-time ChronoNav).
This is the Nov-1-critical SOTA validation run — no episode cap.

Model:   InternVLA-N1-DualVLN (4-bit BnB) + vision_cache_tau=0.92
Split:   R2R val_unseen, all 1839 episodes
Target:  Confirm SR ≥ 72.82% (n=504 best run) on full split
Runtime: ~16-30 hours (hardstop disabled for paper eval)

Launch:
    docker run -d --name vln_ce_chrononav_full --gpus all \\
      -v /home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev:/workspace/InternNav-dev:rw \\
      -v /home/kemal/VLNav/VLNav/workspaces/model/InternNav:/workspace/InternNav:ro \\
      -v /home/kemal/VLNav/VLNav/workspaces/model/ChronoNav:/workspace/ChronoNav:ro \\
      -v /home/kemal/VLNav/VLNav/checkpoints:/checkpoints:ro \\
      -v /mnt/nvme0/vln_habitat/habitat_data:/workspace/InternNav/data:rw \\
      -v /mnt/nvme0/vln_habitat/logs:/workspace/InternNav/logs:rw \\
      vlnav/habitat-eval:rebuilt bash -c "cd /workspace/InternNav && \\
        PYTHONPATH=/workspace/InternNav-dev:/workspace/InternNav/src/diffusion-policy \\
        python scripts/eval/eval.py \\
        --config /workspace/InternNav-dev/scripts/eval/configs/habitat_vln_ce_chrononav_full_cfg.py"
"""
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",
            "model_path": "/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 3,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 64,
            "vis_debug": False,
            "attn_implementation": "eager",
            "load_in_4bit": True,
            "use_kv_cache": False,  # disable to prevent OOM on RTX 3080 Ti (12GB) at step 32
            "vision_cache_tau": 0.92,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # Full 1839-episode val_unseen (no num_episode_sample limit)
            'config_path': '/workspace/ChronoNav/scripts/vln_eval/vln_r2r_debug.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/vln_ce_full/chrononav_full_val_unseen",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2337",
        "dist_url": "env://",
        "use_wandb": False,
        # No episode cap — run all 1839 for paper validation (hardstop is session-level, not code)
    },
)
