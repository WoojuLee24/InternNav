import argparse
import json
import os
import sys
from enum import IntEnum

import debugpy

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict

import cv2
import habitat
import imageio
import numpy as np
import quaternion
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.habitat_extensions.vln.utils import (
    get_axis_align_matrix,
    get_intrinsic_matrix,
    pixel_to_gps,
    preprocess_depth_image_v2,
    xyz_yaw_pitch_to_tf_matrix,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip

# Diagnostics are enabled only when HABITAT_DIAG=1 is set
# (e.g. HABITAT_DIAG=1 python scripts/eval/eval.py --config ...)
_DIAG_ENABLED = os.environ.get('HABITAT_DIAG', '0') == '1'


def _install_step_monitor(env_obj):
    """Monkey-patch env.step to check GL state before/after EVERY call."""
    if not _DIAG_ENABLED:
        return
    original_step = env_obj.step
    _call = [0]

    def _monitored_step(action):
        n = _call[0]; _call[0] += 1
        # Pre-step: check GL
        errs = _diag_gl_errors(f"PRE#{n} action={action}")
        reset = _diag_gl_reset(f"PRE#{n} action={action}")
        if errs or reset:
            print(f"[DIAG][ALARM] BAD GL STATE before env.step #{n} action={action}", flush=True)
        result = original_step(action)
        # Post-step: check GL
        errs2 = _diag_gl_errors(f"POST#{n} action={action}")
        reset2 = _diag_gl_reset(f"POST#{n} action={action}")
        if (errs2 or reset2) and not (errs or reset):
            print(f"[DIAG][ALARM] GL STATE DEGRADED DURING env.step #{n} action={action}", flush=True)
        return result

    env_obj.step = _monitored_step
    print("[DIAG] env.step monitor installed", flush=True)


# ---------------------------------------------------------------------------
# Render diagnostics — identify SIGABRT root cause in EGL/GL layer.
# Checks: NVIDIA_DRIVER_CAPABILITIES, GL context reset, GL errors, VRAM leak.
# ---------------------------------------------------------------------------
import ctypes as _ctypes

_gl_lib = None


def _get_gl_lib():
    global _gl_lib
    if _gl_lib is None:
        try:
            lib = _ctypes.CDLL('libGL.so.1')
            lib.glGetError.restype = _ctypes.c_uint
            lib.glGetError.argtypes = []
            _gl_lib = lib
        except Exception:
            pass
    return _gl_lib


def _diag_gl_errors(ctx=""):
    """Drain and print all pending GL errors. Returns True if any found."""
    if not _DIAG_ENABLED:
        return False
    lib = _get_gl_lib()
    if lib is None:
        return False
    errs = []
    for _ in range(8):
        e = lib.glGetError()
        if e == 0:
            break
        errs.append(f"0x{e:04X}")
    if errs:
        print(f"[DIAG][GL_ERROR] {ctx}: {errs}", flush=True)
    return bool(errs)


_gl_reset_fn = None  # cached function pointer


def _get_gl_reset_fn():
    """Get glGetGraphicsResetStatus via libGL then eglGetProcAddress fallback."""
    global _gl_reset_fn
    if _gl_reset_fn is not None:
        return _gl_reset_fn
    # Try libGL first
    lib = _get_gl_lib()
    if lib:
        for name in ("glGetGraphicsResetStatus", "glGetGraphicsResetStatusARB"):
            fn = getattr(lib, name, None)
            if fn:
                fn.restype = _ctypes.c_uint
                fn.argtypes = []
                _gl_reset_fn = fn
                return fn
    # Fallback: eglGetProcAddress
    try:
        egl = _ctypes.CDLL('libEGL.so.1')
        egl.eglGetProcAddress.restype = _ctypes.c_void_p
        egl.eglGetProcAddress.argtypes = [_ctypes.c_char_p]
        for name in (b"glGetGraphicsResetStatus", b"glGetGraphicsResetStatusARB"):
            ptr = egl.eglGetProcAddress(name)
            if ptr:
                fn = _ctypes.CFUNCTYPE(_ctypes.c_uint)(ptr)
                _gl_reset_fn = fn
                return fn
    except Exception:
        pass
    _gl_reset_fn = False  # mark as unavailable
    return None


def _diag_gl_reset(ctx=""):
    """Check GL context reset status. Returns reset status code (0 = ok)."""
    if not _DIAG_ENABLED:
        return 0
    fn = _get_gl_reset_fn()
    if not fn:
        print(f"[DIAG][GL_RESET] {ctx}: glGetGraphicsResetStatus not available", flush=True)
        return 0
    try:
        status = fn()
        _names = {0: "NO_RESET", 0x8253: "GUILTY_RESET", 0x8254: "INNOCENT_RESET", 0x8255: "UNKNOWN_RESET"}
        label = _names.get(status, f"0x{status:04X}")
        print(f"[DIAG][GL_RESET] {ctx}: {label}", flush=True)
        return status
    except Exception as exc:
        print(f"[DIAG][GL_RESET] {ctx}: query failed ({exc})", flush=True)
        return 0


def _diag_vram(ctx=""):
    """Print PyTorch CUDA memory allocated/reserved."""
    if not _DIAG_ENABLED:
        return
    try:
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[DIAG][VRAM] {ctx}: alloc={alloc:.3f}GB reserved={reserved:.3f}GB", flush=True)
    except Exception:
        pass


def _diag_cam_rotation(sim, step_id):
    """Log sensor rotation quaternion to detect camera drift accumulation."""
    if not _DIAG_ENABLED:
        return
    try:
        agent = sim.agents[0]
        for name in ('rgb', 'depth'):
            sensor = agent.sensors.get(name)
            if sensor is None:
                continue
            r = sensor.node.rotation
            # Format: w + xi + yj + zk
            print(
                f"[DIAG][CAM] step={step_id} sensor={name}: "
                f"w={r.real:.5f} x={r.imag[0]:.5f} y={r.imag[1]:.5f} z={r.imag[2]:.5f}",
                flush=True,
            )
    except Exception as exc:
        print(f"[DIAG][CAM] step={step_id} failed: {exc}", flush=True)


def _diag_total_vram(step_id):
    """Log total GPU VRAM (includes EGL/driver allocations outside PyTorch)."""
    if not _DIAG_ENABLED:
        return
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits', '--id=0'],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            used, free = int(parts[0].strip()), int(parts[1].strip())
            print(f"[DIAG][GPU_MEM] step={step_id}: used={used}MiB free={free}MiB total={used+free}MiB", flush=True)
    except Exception as exc:
        print(f"[DIAG][GPU_MEM] step={step_id} nvidia-smi failed: {exc}", flush=True)


_diag_env_printed = False


def _diag_env_once():
    """Print NVIDIA env diagnostics once at eval startup."""
    global _diag_env_printed
    if not _DIAG_ENABLED or _diag_env_printed:
        return
    _diag_env_printed = True
    caps = os.environ.get('NVIDIA_DRIVER_CAPABILITIES', 'NOT SET')
    vis = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'NOT SET')
    workarounds = os.environ.get('MAGNUM_DISABLE_WORKAROUNDS', 'NOT SET')
    print(f"[DIAG][ENV] NVIDIA_DRIVER_CAPABILITIES={caps}", flush=True)
    print(f"[DIAG][ENV] NVIDIA_VISIBLE_DEVICES={vis}", flush=True)
    print(f"[DIAG][ENV] MAGNUM_DISABLE_WORKAROUNDS={workarounds}", flush=True)
    if 'graphics' not in caps.lower() and caps.lower() not in ('all',):
        print(
            "[DIAG][ENV] WARNING: 'graphics' missing from NVIDIA_DRIVER_CAPABILITIES — "
            "EGL/OpenGL rendering may not be properly configured for this container.",
            flush=True,
        )
# ---------------------------------------------------------------------------


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


def _gps_compute_action(gps, compass, episode):
    """
    Compute the best action (LEFT/FORWARD/RIGHT) to navigate toward the goal.

    Uses GPS position, compass heading, and episode goal position.
    Returns None if goal position cannot be computed.

    Coordinate convention (Habitat VLN-CE):
      GPS = (x_east, z_south) in episode-local frame (relative to start)
      compass = heading in radians; 0=initial forward direction
      At yaw θ: forward_vec = (-sin(θ), cos(θ)) in (east, north) frame
    """
    try:
        import quaternion as np_quat
        start_pos = np.array(episode.start_position, dtype=np.float32)
        goal_pos = np.array(episode.goals[0].position, dtype=np.float32)
        diff = goal_pos - start_pos

        # episode.start_rotation is (x, y, z, w) in Habitat → np.quaternion needs (w, x, y, z)
        sr = episode.start_rotation
        if hasattr(sr, '__len__') and len(sr) == 4:
            q = np_quat.quaternion(float(sr[3]), float(sr[0]), float(sr[1]), float(sr[2]))
        else:
            q = np_quat.quaternion(*[float(v) for v in sr])

        goal_episodic = np_quat.rotate_vectors(q.inverse(), diff)
        goal_gps_x = float(goal_episodic[0])    # east
        goal_gps_z = float(goal_episodic[2])    # south
    except Exception:
        return None

    agent_x, agent_z = float(gps[0]), float(gps[1])
    dx_east = goal_gps_x - agent_x
    dz_south = goal_gps_z - agent_z
    dy_north = -dz_south  # south → north flip

    yaw = float(compass[0]) if hasattr(compass, '__len__') else float(compass)
    # Forward direction in (east, north) at yaw θ: (-sin(θ), cos(θ))
    fwd_proj = dx_east * (-np.sin(yaw)) + dy_north * np.cos(yaw)
    rgt_proj = dx_east * np.cos(yaw) + dy_north * np.sin(yaw)

    if fwd_proj >= 0 and fwd_proj >= abs(rgt_proj):
        return action_code.FORWARD
    elif rgt_proj > 0:
        return action_code.RIGHT
    else:
        return action_code.LEFT


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        _local_rank_hint = int(os.environ.get('LOCAL_RANK', 0))
        with habitat.config.read_write(self.config):
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = _local_rank_hint
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg, init_agent=False)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)
        self.vis_debug = bool(getattr(self.model_args, "vis_debug", False))
        self.vis_debug_path = getattr(self.model_args, "vis_debug_path", os.path.join(self.output_path, "vis_debug"))
        if self.vis_debug:
            os.makedirs(self.vis_debug_path, exist_ok=True)
            self._live_frame_path = os.path.join(self.vis_debug_path, f'live_rank{self.rank}.jpg')

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        attn_impl = getattr(self.model_args, 'attn_implementation', 'eager')
        load_in_8bit = getattr(self.model_args, 'load_in_8bit', False)
        load_in_4bit = getattr(self.model_args, 'load_in_4bit', False)
        load_kwargs = dict(attn_implementation=attn_impl)
        if load_in_4bit:
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs['device_map'] = {"": device}
        elif load_in_8bit:
            load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs['device_map'] = {"": device}
        else:
            load_kwargs['torch_dtype'] = torch.bfloat16
            load_kwargs['device_map'] = {"": device}
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                **load_kwargs,
            )
        elif self.model_args.mode == 'system2':
            model = InternVLAN1ForCausalLM.from_pretrained(  # loader fix (system2): resolves 3584 config
                self.model_args.model_path,
                **load_kwargs,
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        # For 4-bit: cast non-quantizable components back to fp16 to avoid dtype mismatch.
        # memory_encoder and dit_model use F.multi_head_attention_forward which bypasses
        # the bitsandbytes dequantization hook — keep them in fp16.
        if load_in_4bit or load_in_8bit:
            base = model.get_model() if hasattr(model, 'get_model') else model.model
            for attr in ('memory_encoder', 'dit_model', 'depth_anything'):
                if hasattr(base, attr):
                    getattr(base, attr).to(torch.float16)

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor

        # refactor: this part used in three places
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.num_history = self.model_args.num_history

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]
        return result

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        # clean NaN in spls, treat as 0.0
        torch.nan_to_num(spls_all, nan=0.0, posinf=0.0, neginf=0.0, out=spls_all)

        # clean inf in nes, only fiinite nes are counted
        nes_finite_mask = torch.isfinite(nes_all)
        nes_all = nes_all[nes_finite_mask]

        result_all = {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

        if "ndtws" in global_metrics:
            ndtws_all = global_metrics["ndtws"]
            result_all["ndtws_all"] = float(ndtws_all.mean().item()) if denom > 0 else 0.0

        return result_all

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def resume_from_output_path(self) -> None:
        sucs, spls, oss, nes, ndtw = [], [], [], [], []

        # resume from previous results (rank-specific file)
        progress_file = os.path.join(self.output_path, f'progress_rank{self.rank}.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
                    if 'ndtw' in res:
                        ndtw.append(res['ndtw'])
        return sucs, spls, oss, nes, ndtw

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        self.model.eval()
        _diag_env_once()
        _diag_vram("eval_start")
        _install_step_monitor(self.env)

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            action = None
            messages = []
            local_actions = []

            done = False
            flag = False
            pixel_goal = None

            # Loop detector: resets each episode
            _loop_detect_gps: list = []
            _loop_detect_actions: list = []
            _loop_detect_dists: list = []
            _loop_recovery_steps = 0
            _loop_recovery_action = action_code.FORWARD
            _loop_recoveries = 0
            _last_turn_sat_step = -60  # cooldown: minimum steps between turn_sat fires
            _last_pos_loop_step = -40  # cooldown: minimum steps between pos_loop fires
            # No mid-range GPS guidance: only GPS STOP (<3m) + stop suppressor
            _LOOP_WIN = 20           # window size
            _TURN_SAT = 0.75         # turn saturation threshold
            _TURN_COOLDOWN = 50      # min steps between turn_sat fires (prevents 50-cycle storm)
            _POS_COOLDOWN = 40       # min steps between pos_loop fires (prevents rapid re-trigger)
            _STAG_STEPS = 15         # stagnation detection window
            _POS_VAR_THRESH = 0.09   # position variance threshold (0.3m² spread)

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                # Per-step diagnostics: camera rotation + GPU memory every 10 steps
                try:
                    _diag_cam_rotation(self.env._env.sim, step_id)
                except Exception:
                    pass
                if step_id % 10 == 0:
                    _diag_total_vram(step_id)
                    _diag_vram(f"step={step_id}")
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)

                    look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                    depth = down_observations["depth"]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    self.env.step(action_code.LOOKUP)
                    self.env.step(action_code.LOOKUP)
                    _diag_gl_errors(f"after_obs_gather step={step_id}")
                    _diag_gl_reset(f"after_obs_gather step={step_id}")

                if len(action_seq) == 0 and pixel_goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences
                    torch.cuda.synchronize()
                    _diag_vram(f"after_generate step={step_id}")
                    _diag_gl_reset(f"after_generate step={step_id}")
                    _diag_gl_errors(f"after_generate step={step_id}")

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                    if len(coord) >= 2:  # output pixel goal (need at least x, y)
                        forward_action = 0

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        local_actions = []
                        pixel_values = inputs.pixel_values
                        image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                        with torch.no_grad():
                            traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                        # prepocess align with navdp # why images_dp and depths_dp have two same frames? 
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        pix_goal_image = copy.copy(image_dp)
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        pix_goal_depth = copy.copy(depth_dp)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]

                        action = local_actions[0]
                        if action == action_code.STOP:
                            pixel_goal = None
                            output_ids = None
                            action = action_code.LEFT
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        # navdp
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
                        print("local_actions", local_actions)
                        action = local_actions.pop(0)
                    else:
                        action = local_actions.pop(0)

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == action_code.STOP:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                # GPS-based stopping and guidance block
                # Improvement 1: GPS stop — force STOP within 3m success radius
                # Improvement 2: Stop suppression — prevent premature STOP far from goal
                # Improvement 3: GPS guidance — navigate toward goal when 3-5m away
                _gps_dist = info.get('distance_to_goal', float('inf'))
                if step_id >= 5 and action not in (action_code.LOOKDOWN, action_code.LOOKUP):
                    if _gps_dist < 3.0:
                        # Improvement 1: GPS STOP
                        print(
                            f"[GPS_STOP] step={step_id} dist={_gps_dist:.3f}m → forcing STOP",
                            flush=True,
                        )
                        observations, _, done, _ = self.env.step(action_code.STOP)
                        step_id += 1
                        break
                    elif action == action_code.STOP and _gps_dist >= 3.5:
                        # Improvement 2: Stop suppression — don't stop too far from goal
                        # Use GPS guidance to pick a better action, or fall back to FORWARD
                        _compass = observations.get("compass", None)
                        _gps_nav_action = action_code.FORWARD  # default fallback
                        if _compass is not None:
                            _gps_nav_action = _gps_compute_action(
                                observations["gps"], _compass, episode,
                            ) or action_code.FORWARD
                        print(
                            f"[STOP_SUPPRESS] step={step_id} dist={_gps_dist:.3f}m "
                            f"→ suppressing STOP, using {_gps_nav_action}",
                            flush=True,
                        )
                        action = _gps_nav_action
                        action_seq.clear()  # force fresh S2 invocation on next step
                        pixel_goal = None
                # Improvement 4: Loop detection and recovery
                _loop_detect_dists.append(_gps_dist)
                _x_gps, _y_gps = float(observations["gps"][0]), float(observations["gps"][1])
                _loop_detect_gps.append((_x_gps, _y_gps))
                if action not in (action_code.LOOKDOWN, action_code.LOOKUP, action_code.STOP):
                    _loop_detect_actions.append(int(action))

                _loop_win_act = _loop_detect_actions[-_LOOP_WIN:]
                _loop_win_gps = _loop_detect_gps[-_LOOP_WIN:]
                _loop_win_dist = _loop_detect_dists[-_STAG_STEPS:]

                if step_id >= 12 and action not in (action_code.LOOKDOWN, action_code.LOOKUP, action_code.STOP):
                    if _loop_recovery_steps > 0:
                        # In recovery mode: execute recovery action
                        _loop_recovery_steps -= 1
                        action = _loop_recovery_action
                    elif len(_loop_win_act) >= _LOOP_WIN // 2:
                        # Check turn saturation (with cooldown to prevent rapid re-trigger).
                        # Without cooldown: 6-step rotation + 10 model turns = 16-step cycle → 50 fires/episode.
                        # With cooldown=50: max ~8 fires/episode, model gets breathing room to navigate.
                        _n_turns = sum(1 for a in _loop_win_act if a in (action_code.LEFT, action_code.RIGHT))
                        if (_n_turns / len(_loop_win_act) >= _TURN_SAT
                                and step_id - _last_turn_sat_step >= _TURN_COOLDOWN):
                            # FORWARD×10 escape: physically moves agent 2.5m forward.
                            # Many stuck cases (closet/bathroom) need forward movement to exit the room.
                            # Rotation-only didn't help (model outputs same turns after seeing same walls).
                            _loop_recovery_steps = 10
                            _loop_recovery_action = action_code.FORWARD
                            _loop_recoveries += 1
                            _last_turn_sat_step = step_id
                            action = action_code.FORWARD
                            action_seq.clear(); pixel_goal = None
                            _loop_detect_actions.clear()
                            print(f"[LOOP] step={step_id} turn_sat({_n_turns}/{len(_loop_win_act)}) cooldown_ok → FORWARD×10+clear", flush=True)
                        elif (len(_loop_win_gps) >= _LOOP_WIN
                                and step_id - _last_pos_loop_step >= _POS_COOLDOWN):
                            # Check position variance (with cooldown to prevent rapid re-trigger).
                            # Without cooldown: 5-step cycle (1 rotation + 4 FORWARD) → 100 fires/501 steps.
                            # With cooldown=40: max ~10 fires/episode total alongside turn_sat.
                            _gps_arr = np.array(_loop_win_gps)
                            _pos_var = float(np.var(_gps_arr, axis=0).sum())
                            if _pos_var < _POS_VAR_THRESH:
                                _n_left = sum(1 for a in _loop_win_act if a == action_code.LEFT)
                                _n_right = sum(1 for a in _loop_win_act if a == action_code.RIGHT)
                                _escape = action_code.RIGHT if _n_left > _n_right else action_code.LEFT
                                _loop_recovery_steps = 4
                                _loop_recovery_action = action_code.FORWARD
                                _loop_recoveries += 1
                                _last_pos_loop_step = step_id
                                action = _escape
                                action_seq.clear(); pixel_goal = None
                                print(f"[LOOP] step={step_id} pos_loop(var={_pos_var:.4f}) cooldown_ok → {_escape.name}+FORWARD×4", flush=True)
                        if (len(_loop_win_dist) >= _STAG_STEPS
                                and (max(_loop_win_dist) - min(_loop_win_dist)) < 0.3
                                and _loop_recovery_steps == 0):
                            # Distance stagnation: force S2 re-invocation
                            action_seq.clear(); pixel_goal = None
                            print(f"[LOOP] step={step_id} dist_stagnation → forcing S2 re-call", flush=True)

                _composite_frame = None
                if info['top_down_map'] is not None:
                    _composite_frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if pixel_goal is not None and flag:
                        cv2.circle(_composite_frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    if self.save_video:
                        vis_frames.append(_composite_frame.copy())

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if pixel_goal is not None:
                        if draw_pixel_goal:
                            cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)
                    # Live viewer: use RGB+TopDownMap composite when available, else raw RGB
                    live_img = _composite_frame if _composite_frame is not None else vis
                    cv2.imwrite(self._live_frame_path, cv2.cvtColor(live_img, cv2.COLOR_RGB2BGR))

                _diag_gl_errors(f"before_step step={step_id} action={action}")
                _diag_gl_reset(f"before_step step={step_id} action={action}")
                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            n = len(sucs)
            running_sr = sum(sucs) / n
            running_spl = sum(spls) / n
            running_ne = sum(nes) / n
            process_bar.set_postfix(SR=f"{running_sr:.3f}", SPL=f"{running_spl:.3f}", NE=f"{running_ne:.2f}")
            print(
                f"[{n}] {scene_id}_{episode_id:04d} | "
                f"success={metrics['success']:.0f} spl={metrics['spl']:.3f} "
                f"os={metrics['oracle_success']:.0f} ne={metrics['distance_to_goal']:.2f} | "
                f"running SR={running_sr:.3f} SPL={running_spl:.3f} NE={running_ne:.2f} "
                f"loop_recs={_loop_recoveries}"
            )

            # Write per-episode progress.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": float(metrics["success"]),
                "spl": float(metrics["spl"]),
                "os": float(metrics['oracle_success']),
                "ne": float(metrics["distance_to_goal"]),
                "steps": int(step_id),
                "episode_instruction": episode_instruction,
                "loop_recoveries": int(_loop_recoveries),
            }
            if 'ndtw' in metrics:
                result['ndtw'] = float(metrics['ndtw'])

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, f'progress_rank{self.rank}.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

            # save video
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )

    def _run_eval_system2(self) -> tuple:
        import time as _time
        self.model.eval()

        # VisionEmbeddingCache: per-frame EMA fingerprint; skip S2 when scene is static
        _vision_cache_tau = getattr(self.model_args, 'vision_cache_tau', None)
        _use_vision_cache = _vision_cache_tau is not None
        if _use_vision_cache:
            from internnav.agent.s2_kv_cache import VisionEmbeddingCache
            _vision_cache = VisionEmbeddingCache(similarity_threshold=_vision_cache_tau)

        # S2 latency tracking — emitted as timing.json at end of eval
        _s2_latencies: list = []   # ms per actual model.generate() call (cache hits excluded)
        _s2_cache_hits: int = 0
        _eval_wall_t0: float = _time.perf_counter()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            intrinsic_matrix = get_intrinsic_matrix(
                self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            goal = None
            action = None
            messages = []
            _prev_llm_outputs = None  # last S2 output; reused on VisionEmbeddingCache hit

            done = False
            flag = False

            if _use_vision_cache:
                _vision_cache.reset()

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                if len(action_seq) == 0 and goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    # VisionEmbeddingCache: if scene fingerprint matches last call, reuse
                    # previous S2 output instead of running the full model forward pass.
                    # Only fires when scene is genuinely static between consecutive S2 triggers.
                    _skip_s2 = False
                    if _use_vision_cache:
                        _fp = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                        _, _cache_hit = _vision_cache.get_or_compute(
                            _fp, lambda x: x.mean(0, keepdim=True)
                        )
                        if _cache_hit and _prev_llm_outputs is not None:
                            llm_outputs = _prev_llm_outputs
                            _skip_s2 = True

                    if not _skip_s2:
                        _gen_t0 = _time.perf_counter()
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=128,
                                do_sample=False,
                                use_cache=True,
                                past_key_values=None,
                                return_dict_in_generate=True,
                            ).sequences
                        _s2_latencies.append((_time.perf_counter() - _gen_t0) * 1000.0)
                        llm_outputs = self.processor.tokenizer.decode(
                            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                        )
                    else:
                        _s2_cache_hits += 1

                    _prev_llm_outputs = llm_outputs
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                    if len(coord) >= 2:  # output pixel goal (need at least x, y)
                        forward_action = 0

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        goal = pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)

                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                        action = agent.get_next_action(goal)
                        if action == action_code.STOP:
                            goal = None
                            output_ids = None
                            action = action_code.LEFT  # random action to avoid deadlock
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    action = agent.get_next_action(goal)
                    action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                    action = action[0] if hasattr(action, "__len__") else action

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                    if action == action_code.STOP:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                # GPS-based stopping and guidance (same as dual-system path)
                _gps_dist = info.get('distance_to_goal', float('inf'))
                if step_id >= 5 and action not in (action_code.LOOKDOWN, action_code.LOOKUP):
                    if _gps_dist < 3.0:
                        print(
                            f"[GPS_STOP] step={step_id} dist={_gps_dist:.3f}m → forcing STOP",
                            flush=True,
                        )
                        observations, _, done, _ = self.env.step(action_code.STOP)
                        step_id += 1
                        break
                    elif action == action_code.STOP and _gps_dist >= 3.5:
                        _compass = observations.get("compass", None)
                        _gps_nav_action = action_code.FORWARD
                        if _compass is not None:
                            _gps_nav_action = _gps_compute_action(
                                observations["gps"], _compass, episode,
                            ) or action_code.FORWARD
                        print(
                            f"[STOP_SUPPRESS] step={step_id} dist={_gps_dist:.3f}m "
                            f"→ suppressing STOP, using {_gps_nav_action}",
                            flush=True,
                        )
                        action = _gps_nav_action
                        goal = None; output_ids = None; messages = []
                _composite_frame = None
                if info['top_down_map'] is not None:
                    _composite_frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if goal is not None and flag:
                        cv2.circle(_composite_frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    if self.save_video:
                        vis_frames.append(_composite_frame.copy())

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if draw_pixel_goal:
                        cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)
                    # Live viewer: use RGB+TopDownMap composite when available, else raw RGB
                    live_img = _composite_frame if _composite_frame is not None else vis
                    cv2.imwrite(self._live_frame_path, cv2.cvtColor(live_img, cv2.COLOR_RGB2BGR))

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            n = len(sucs)
            running_sr = sum(sucs) / n
            running_spl = sum(spls) / n
            running_ne = sum(nes) / n
            process_bar.set_postfix(SR=f"{running_sr:.3f}", SPL=f"{running_spl:.3f}", NE=f"{running_ne:.2f}")
            print(
                f"[{n}] {scene_id}_{episode_id:04d} | "
                f"success={metrics['success']:.0f} spl={metrics['spl']:.3f} "
                f"os={metrics['oracle_success']:.0f} ne={metrics['distance_to_goal']:.2f} | "
                f"running SR={running_sr:.3f} SPL={running_spl:.3f} NE={running_ne:.2f}"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": float(metrics["success"]),
                "spl": float(metrics["spl"]),
                "os": float(metrics['oracle_success']),
                "ne": float(metrics["distance_to_goal"]),
                "steps": int(step_id),
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = float(metrics['ndtw'])

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, f'progress_rank{self.rank}.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()


        # Write timing.json — keys match collect_metrics.py --timing schema
        _total_s2_calls = len(_s2_latencies)
        _eval_wall_h = (_time.perf_counter() - _eval_wall_t0) / 3600.0
        _timing: dict = {}
        if _total_s2_calls > 0:
            import statistics as _stats
            _s2_ms = _stats.mean(_s2_latencies)
            _timing['s2_ms'] = round(_s2_ms, 1)
            _timing['s2_hz'] = round(1000.0 / _s2_ms, 3)
            _total_s2_incl_hits = _s2_cache_hits + _total_s2_calls
            if _total_s2_incl_hits > 0:
                _timing['cache_hit_rate'] = round(_s2_cache_hits / _total_s2_incl_hits, 4)
        _timing['gpu_h'] = round(_eval_wall_h, 3)
        _timing['n_s2_calls'] = _total_s2_calls
        _timing['n_cache_hits'] = _s2_cache_hits
        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.output_path, f'timing_rank{self.rank}.json'), 'w') as _tf:
            json.dump(_timing, _tf, indent=2)
        print(
            f"[timing] s2_ms={_timing.get('s2_ms','?')} s2_hz={_timing.get('s2_hz','?')} "
            f"cache_hits={_s2_cache_hits}/{_s2_cache_hits+_total_s2_calls} gpu_h={_timing['gpu_h']:.3f}",
            flush=True,
        )

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )
