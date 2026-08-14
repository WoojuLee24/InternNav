"""Dual-system agent whose System-2 runs on llama.cpp (GGUF) and System-1 on the
standalone lite diffusion model. Reuses InternVLAN1Agent's threading / step loop
verbatim — only the policy construction changes (no 7B Qwen is loaded in-process).

Select via the eval config: AgentCfg(model_name='internvla_n1_llamacpp', model_settings={...}).
Extra model_settings keys (defaults shown):
    chat_url : "http://127.0.0.1:8080"   # llama-server  (-m text.gguf  --mmproj mmproj.gguf)
    embd_url : "http://127.0.0.1:8081"   # llama-server  (-m text-latent.gguf --mmproj ... --embeddings --pooling none)
    resize_w : 448                       # 112-aligned (REQUIRED for parity)
Run the two servers first (see internvla_n1_policy_llamacpp.py header), then eval.py as usual.
"""
import atexit
import os
import threading
from pathlib import Path

import imageio
import torch

from internnav.agent.base import Agent
from internnav.agent.internvla_n1_agent import InternVLAN1Agent
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp import (
    InternVLAN1LlamaCppPolicy,
)
from internnav.model.utils.misc import set_random_seed
from internnav.model.utils.vln_utils import S1Input, S1Output, S2Input, S2Output


_BUNDLE_KEYS = ('runner_bin', 'model_gguf', 'mmproj_gguf', 'lib_dir')


def _resolve_bundle_assets(model_path):
    """Find the System-2 assets sitting next to the System-1 weights in an eval bundle.

    Layout produced by ``vln-deploy/deploy/tools/make_eval_bundle.sh`` (WORKFLOW.md Step 10):
        <model_path>/gguf/*-latent.gguf , *mmproj*.gguf     S2 quantized
        <model_path>/bin/llama-s2-runner , lib*.so*         S2 runtime
    Lets a single --model-path select both systems. Only consulted for keys the config
    omits, so an explicit model_settings entry always wins.
    """
    mp = Path(model_path)
    hint = (f"point --model-path at a bundle built by "
            f"vln-deploy/deploy/tools/make_eval_bundle.sh, or set it in model_settings")

    def _one(sub, pattern, key):
        hits = sorted((mp / sub).glob(pattern))
        if not hits:
            raise FileNotFoundError(f"{key}: no {sub}/{pattern} under {mp} -- {hint}")
        return str(hits[0])

    runner = mp / 'bin' / 'llama-s2-runner'
    if not runner.exists():
        raise FileNotFoundError(f"runner_bin: {runner} not found -- {hint}")
    return {
        'runner_bin': str(runner),
        'lib_dir': str(mp / 'bin'),
        'model_gguf': _one('gguf', '*-latent.gguf', 'model_gguf'),
        'mmproj_gguf': _one('gguf', '*mmproj*.gguf', 'mmproj_gguf'),
    }


@Agent.register('internvla_n1_llamacpp')
class InternVLAN1LlamaCppAgent(InternVLAN1Agent):
    def __init__(self, config: AgentCfg):
        Agent.__init__(self, config)                       # skip InternVLAN1Agent.__init__ (HF load)
        set_random_seed(0)
        vln = self.config.model_settings
        ms = ModelCfg(**vln)
        self.device = torch.device(ms.device)
        self.mode = getattr(ms, 'infer_mode', 'sync')
        self.sys2_max_forward_step = getattr(ms, 'sys2_max_forward_step', 8)

        # S2 asset paths: explicit model_settings keys win; anything omitted is resolved
        # from the eval bundle at model_path, so --model-path alone can select S1 + S2.
        assets = {k: vln[k] for k in _BUNDLE_KEYS if vln.get(k)}
        if len(assets) < len(_BUNDLE_KEYS):
            assets = {**_resolve_bundle_assets(ms.model_path), **assets}

        # ---- POLICY: llama.cpp S2 via persistent llama-s2-runner CLI + lite S1 (no 7B in-process) ----
        self.policy = InternVLAN1LlamaCppPolicy(
            runner_bin=assets['runner_bin'],
            model_gguf=assets['model_gguf'],
            mmproj_gguf=assets['mmproj_gguf'],
            lib_dir=assets['lib_dir'],
            s1_model_path=ms.model_path,
            device=ms.device,
            num_history=getattr(ms, 'num_history', 8),
            resize=vln.get('resize_w', 448),
            max_new_tokens=getattr(ms, 'max_new_tokens', 128),
            use_trt_s1=getattr(ms, 'use_trt_s1', False),
            s1_steps=getattr(ms, 's1_steps', 5),
            kv_reuse=getattr(ms, 'kv_reuse', False),
            lookdown_full_res=vln.get('lookdown_full_res', False),
        )
        self.policy.eval()

        self.camera_intrinsic = self.get_intrinsic_matrix(ms.width, ms.height, ms.hfov)

        # ---- identical to InternVLAN1Agent.__init__ below ----
        self.episode_step = 0
        self.episode_idx = 0
        self.look_down = False

        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.dual_forward_step = 0
        self.sys1_infer_times = 0
        self.sys1_depth_threshold = 5.0
        self.sys1_forward_step = 4

        self.s1_input = S1Input()
        self.s2_input = S2Input()
        self.s2_output = S2Output()
        self.s1_output = S1Output()

        self.s2_thread = None
        self.s2_input_lock = threading.Lock()
        self.s2_output_lock = threading.Lock()
        self.s2_agent_lock = threading.Lock()
        self._start_s2_thread()

        self.vis_debug = vln['vis_debug']
        if self.vis_debug:
            self.debug_path = vln['vis_debug_path']
            os.makedirs(self.debug_path, exist_ok=True)
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_dp.mp4", fps=5)
            self.output_pixel = None
            atexit.register(lambda: self.fps_writer.close() if self.fps_writer else None)
            atexit.register(lambda: self.fps_writer2.close() if self.fps_writer2 else None)
