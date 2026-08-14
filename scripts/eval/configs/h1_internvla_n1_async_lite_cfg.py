"""H1 (Isaac Sim) async dual-system eval on the QUANTIZED stack.

Same env/task/dataset as h1_internvla_n1_async_cfg.py so the two are directly
comparable; only the agent changes:

  model_name='internvla_n1_llamacpp'
    System-2 : Qwen2.5-VL as a Q4_K_M-latent GGUF, run by the `llama-s2-runner`
               C++ subprocess over a stdin/stdout pipe (no 7B in this process).
    System-1 : the 180 MB S1-only safetensors, denoise loop on TensorRT FP16.

Everything both systems need lives in ONE directory, so --model-path selects both:

    --model-path /ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN-lite

built by vln-deploy/deploy/tools/make_eval_bundle.sh (WORKFLOW.md Step 10).
eval.py overrides model_settings['model_path'] with --model-path, and the agent
resolves <model_path>/gguf/ and <model_path>/bin/ from it.

TensorRT is NOT in Isaac Sim's bundled python; `use_trt_s1=True` needs
quantization_kit/trt_py311 on PYTHONPATH (WORKFLOW.md Step 9). runner.py injects it
automatically. Without it, set use_trt_s1=False (S1 ~110 ms/infer instead of ~20 ms).
"""
import os
import sys

VLN_DEPLOY = "/ws/src/InternNav/vln-deploy"
sys.path.insert(0, VLN_DEPLOY)  # provides `quantization_kit` (S1LiteModel / trt_s1)

# Env relays, same convention runner.py uses for BEV_DEBUG_DIR / EVAL_HEADLESS: eval.py
# runs as a subprocess that re-imports this file, so CLI flags can't reach it directly.
# EVAL_MAX_EPISODES caps the episode list (internutopia_env.py:40) -- set it identically
# for both variants so they evaluate the same slice. VLN_TIMING turns on per-call timing.
_MAX_EP = os.environ.get("EVAL_MAX_EPISODES")
_MAX_TOK = os.environ.get("EVAL_MAX_NEW_TOKENS")
if os.environ.get("VLN_TIMING"):
    sys.path.insert(0, os.path.join(VLN_DEPLOY, "verify"))
    import eval_timing  # noqa: E402

    eval_timing.install()

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "train_eval", "qwenvl_train"
))
import default_config as base  # noqa: E402

# Registers Agent 'internvla_n1_llamacpp' (import side effect — do not remove)
import internnav.agent.internvla_n1_agent_llamacpp  # noqa: F401  # isort: skip

from internnav.configs.agent import AgentCfg  # noqa: E402
from internnav.configs.evaluator import (  # noqa: E402
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8023,
        model_name='internvla_n1_llamacpp',
        ckpt_path='',
        model_settings={
            'env_num': 1,
            'sim_num': 1,
            # eval.py --model_path overrides this. Must be an eval bundle: S1 weights at the
            # top level, S2 GGUF in gguf/, llama-s2-runner + libs in bin/.
            'model_path': "/ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN-lite",
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            # Match orig's S2 input format: planning 384, look_down at camera resolution.
            # (448 "112-aligned parity" was obsolete — mtmd smart-resizes internally.)
            'resize_w': 384,
            'resize_h': 384,
            'lookdown_full_res': True,
            # 128, not 1024: matches the production evaluator's hardcoded value.
            # EVAL_MAX_NEW_TOKENS overrides; unset -> this config's own value.
            'max_new_tokens': int(_MAX_TOK) if _MAX_TOK else 128,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
            'infer_mode': 'partial_async',
            # ---- quantized-stack knobs (see internvla_n1_policy_llamacpp.py) ----
            'use_trt_s1': True,   # needs Step 9 on PYTHONPATH; False = PyTorch fp32 fallback
            's1_steps': 5,        # deployment default (5 -> ~30 Hz S1; 10 = slower, higher quality)
            'kv_reuse': False,    # conservative: fresh S2 conversation each step
            # runner_bin / model_gguf / mmproj_gguf / lib_dir are resolved from model_path.
            # Set them here to override (e.g. to point at a different quantization set).
            # debug
            'vis_debug': False,
            'vis_debug_path': './logs/lite/llamacpp',
        },
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            # EVAL_MAX_EPISODES=N caps the run (unset -> full split)
            **({'max_episodes': int(_MAX_EP)} if _MAX_EP else {}),
            'use_fabric': False,  # Please set use_fabric=False due to the render delay;
            'headless': True,  # Isaac-sim
        },
    ),
    task=TaskCfg(
        task_name='lite',
        task_settings={
            'env_num': 1,
            'use_distributed': False,
            'proc_num': 1,
            'max_step': 1000,  # If use flash mode，default 1000; descrete mode, set 50000
        },
        scene=SceneCfg(
            scene_type='mp3d',
            scene_data_dir='/ws/src/InternNav/data/InternData-N1-v0.5-mini/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_flash=True,
        robot_platform_size=None,
        flash_collision=None,
        robot_usd_path='/ws/src/InternNav/data/InternData-N1-v0.5-mini/Embodiments/vln-pe/h1/h1_internvla.usd',
        camera_resolution=[640, 480],  # (W,H)
        camera_prim_path='torso_link/h1_1_25_down_30',
        one_step_stand_still=True,  # For dual-system, please keep this param True.
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            'base_data_dir': '/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r',
            'split_data_types': ['val_unseen'],
            'filter_stairs': True,
            # 'selected_scans': ['zsNo4HB9uLZ'],  # smoke test: one scene
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

PARAMS = base.PARAMS
