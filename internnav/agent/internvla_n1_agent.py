import copy
import os
import threading
import time
import atexit

import cv2
import imageio
import numpy as np
import torch
from gym import spaces
from PIL import Image

from internnav.agent.base import Agent
from internnav.agent.navcot import NavCoTWrapper
from internnav.model.basemodel.nexusnav.bev_fusion import BEVFusion
from internnav.agent.latent_reasoning import LatentReasoner, S2ThoughtCache
from internnav.agent.memory.body_memory import BodyMemory
from internnav.agent.memory.task_memory import TaskMemory
from internnav.agent.memory.world_memory import WorldMemory
from internnav.agent.world_predictor import WorldPredictor
try:
    from internnav.agent.lacp import LACP, LACPConfig
    _HAS_LACP = True
except Exception:
    _HAS_LACP = False
try:
    from internnav.agent.bev_fusion import BEVFusion as _AgentBEVFusion, BEVConfig
    _HAS_BEV = True
except Exception:
    _HAS_BEV = False
try:
    from internnav.agent.nav_cot import NavCoTBuilder, NavCoTParser, NavCoTConfig
    _HAS_NAVCOT = True
except Exception:
    _HAS_NAVCOT = False
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model import get_config, get_policy
from internnav.model.utils.misc import set_random_seed
from internnav.model.utils.vln_utils import S1Input, S1Output, S2Input, S2Output


@Agent.register('internvla_n1')
class InternVLAN1Agent(Agent):
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,
    )

    def __init__(self, config: AgentCfg):
        super().__init__(config)
        set_random_seed(0)
        vln_sensor_config = self.config.model_settings
        self._model_settings = ModelCfg(**vln_sensor_config)
        self.device = torch.device(self._model_settings.device)
        self.mode = getattr(self._model_settings, 'infer_mode', 'sync')
        self.sys2_max_forward_step = getattr(self._model_settings, 'sys2_max_forward_step', 8)

        policy = get_policy(self._model_settings.policy_name)
        policy_config = get_config(self._model_settings.policy_name)
        model_config = {'model': self._model_settings.model_dump()}
        self.policy = policy(config=policy_config(model_cfg=model_config))
        self.policy.eval()

        self.camera_intrinsic = self.get_intrinsic_matrix(
            self._model_settings.width, self._model_settings.height, self._model_settings.hfov
        )

        self.episode_step = 0
        self.episode_idx = 0
        self.look_down = False

        # for async dual sys
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

        # Thread management
        self.s2_thread = None

        # Thread locks
        self.s2_input_lock = threading.Lock()
        self.s2_output_lock = threading.Lock()
        self.s2_agent_lock = threading.Lock()

        # LatentReasoner: N=4 GRU ThoughtCells, silent chain-of-thought in embedding space
        _thought_dim = 256
        self.latent_reasoner = LatentReasoner(
            thought_dim=_thought_dim, context_dim=_thought_dim, n_steps=4, vocab_size=256
        ).to(self.device)
        self.latent_reasoner.eval()
        self.s2_thought_cache = S2ThoughtCache(thought_dim=_thought_dim)

        # Memory Triad: passive state trackers updated per S2 step (no LLM input change)
        self.task_memory = TaskMemory()
        self.world_memory = WorldMemory(max_nodes=200)
        self.body_memory = BodyMemory()

        # WorldPredictor: online BCE reachability (Prophet-style), generates training signal
        self.world_predictor = WorldPredictor(feature_dim=256, device=str(self.device))

        # C11 LACP: anticipatory KERV gate — predicts latent K steps ahead
        self.lacp = LACP(LACPConfig()).to(self.device) if _HAS_LACP else None
        if self.lacp is not None:
            self.lacp.eval()
        self._lacp_h_pred = None  # pending prediction to compare K steps later

        # C10 NavCoT: structured chain-of-thought elicitation before each S2 step
        self.navcot = NavCoTWrapper(enabled=True, max_think_tokens=128)
        self._action_history: list = []

        # C12 BEVFusion: depth → BEV occupancy → 4 CLIP tokens for S2 context
        # Enabled via config.use_bev_fusion (default False — set True when training with C12)
        self.bev_fusion = (
            BEVFusion(token_dim=3584, n_tokens=4).to(self.device)
            if getattr(self._model_settings, 'use_bev_fusion', False)
            else None
        )

        # Recovery counter: non-zero forces backtrack actions in S1 loop
        self._recovery_steps_remaining = 0

        # Start S2 thread
        self._start_s2_thread()

        # vis debug
        self.vis_debug = vln_sensor_config['vis_debug']
        if self.vis_debug:
            self.debug_path = vln_sensor_config['vis_debug_path']
            os.makedirs(self.debug_path, exist_ok=True)
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_dp.mp4", fps=5)
            self.output_pixel = None
            # Patch: 
            atexit.register(lambda: self.fps_writer.close() if self.fps_writer else None)
            atexit.register(lambda: self.fps_writer2.close() if self.fps_writer2 else None)

    def reset(self, reset_index=None):
        '''reset_index: [0]'''
        if reset_index is not None:
            self.episode_idx += 1
            if self.vis_debug:
                self.fps_writer.close()
                self.fps_writer2.close()
        else:
            self.episode_idx = -1

        self.episode_step = 0
        self.s1_input = S1Input()
        with self.s2_input_lock:
            self.s2_input = S2Input()
        with self.s2_output_lock:
            self.s2_output = S2Output()
        self.s1_output = S1Output()

        # for async dual sys
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.dual_forward_step = 0
        self.sys1_infer_times = 0

        # Reset s2 agent
        with self.s2_agent_lock:
            self.policy.reset()

        # Reset latent thought state and memory triad for new episode
        self.s2_thought_cache.reset()
        self.task_memory.reset([])
        self.world_memory.reset()
        self.body_memory.reset()
        self._recovery_steps_remaining = 0
        self.navcot.reset()

        # Reset C10/C11/C12 per-episode state
        if self.lacp is not None:
            self.lacp.reset()
        self._lacp_h_pred = None
        if self.bev_fusion is not None:
            self.bev_fusion.reset()
        self._action_history = []

        if self.vis_debug:
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_dp.mp4", fps=5)

    def get_intrinsic_matrix(self, width, height, hfov) -> np.ndarray:
        width = width
        height = height
        fov = hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        return intrinsic_matrix

    def get_last_thought(self) -> "torch.Tensor | None":
        """Return the current latent thought vector from the last S2 call, or None if no call yet."""
        return self.s2_thought_cache.current_thought

    def _start_s2_thread(self):
        def s2_thread_func():
            while True:
                # Check if inference is needed
                should_infer = self.s2_input.should_infer
                if should_infer:
                    with self.s2_input_lock:
                        self.s2_input.should_infer = False
                        s2_infer_idx = self.s2_input.idx
                else:
                    time.sleep(0.5)  # Sleep briefly if inference is not needed
                    continue

                # # Check if currently inferring
                # if self.mode == "sync":
                #     if not self.s2_output.is_infering:
                #         with self.s2_output_lock:
                #             self.s2_output.is_infering = True
                #     else:
                #         time.sleep(0.5)  # Sleep briefly if already inferring
                #         continue

                # Build memory context to inject into S2 prompt
                _mem_ctx_parts = []
                try:
                    if self.body_memory.is_oscillating():
                        _mem_ctx_parts.append("You appear to be revisiting a location. Change direction.")
                        self._recovery_steps_remaining = 3
                    if hasattr(self.task_memory, 'current_segment'):
                        _seg = self.task_memory.current_segment()
                        if _seg:
                            _mem_ctx_parts.append(f"Current instruction step: {_seg}")
                except Exception:
                    pass

                # C12 BEVFusion: prepend spatial context from depth → BEV
                _bev_context = ""
                try:
                    if self.bev_fusion is not None and self.s2_input.depth is not None:
                        _depth_t = torch.from_numpy(
                            np.array(self.s2_input.depth, dtype=np.float32)
                        ).to(self.device)
                        if _depth_t.dim() == 3:
                            _depth_t = _depth_t[:, :, 0]
                        _bev_context = self.bev_fusion.get_context_string(_depth_t)
                        if _bev_context:
                            _mem_ctx_parts.insert(0, _bev_context)
                except Exception:
                    pass

                # C10 NavCoT: rebuild memory context via structured CoT prompt
                _memory_context = " ".join(_mem_ctx_parts)
                # Augment instruction with CoT reasoning prompt for S2
                _instruction = self.s2_input.instruction
                try:
                    if self.navcot_builder is not None and _instruction:
                        _cot_prompt = self.navcot_builder.build(
                            instruction=_instruction,
                            step_idx=self.episode_step,
                            memory_context=" ".join(_mem_ctx_parts),
                            bev_context=_bev_context,
                            action_history=self._action_history,
                        )
                        _memory_context = _cot_prompt.user
                    elif self._navcot_simple is not None and _instruction:
                        _instruction = self._navcot_simple.augment_instruction(
                            _instruction, step=self.episode_step
                        )
                except Exception:
                    pass

                # NavCoT (C10): prepend structured CoT elicitation to memory context
                try:
                    _pose_for_cot = self.s2_input.pose
                    _heading = float(_pose_for_cot[2]) * 57.3 if _pose_for_cot is not None and len(_pose_for_cot) > 2 else 0.0
                    _landmarks = NavCoTWrapper.extract_landmarks_from_memory(_memory_context)
                    _memory_context = self.navcot.build_cot_context_prefix(
                        instruction=self.s2_input.instruction or "",
                        steps_done=self.episode_step,
                        heading=_heading,
                        memory_context=_memory_context,
                        landmarks=_landmarks,
                    )
                    self.navcot.increment_step()
                except Exception:
                    pass  # CoT prefix is best-effort

                # BEV tokens (C12): compute 4 spatial tokens from depth before S2 call
                _bev_tokens = None
                if self.bev_fusion is not None and self.s2_input.depth is not None:
                    try:
                        _depth_t = torch.from_numpy(self.s2_input.depth).float().to(self.device)
                        with torch.no_grad():
                            _bev_tokens = self.bev_fusion(_depth_t)  # (1, 4, 3584)
                    except Exception:
                        _bev_tokens = None  # BEV tokens are best-effort

                # Execute inference
                success = True
                try:
                    with self.s2_agent_lock:
                        current_s2_output = self.policy.s2_step(
                            self.s2_input.rgb,
                            self.s2_input.depth,
                            self.s2_input.pose,
                            _instruction,
                            self.camera_intrinsic,
                            self.s2_input.look_down,
                            memory_context=_memory_context,
                            bev_tokens=_bev_tokens,  # C12: top-down spatial awareness
                        )
                except Exception as e:
                    print(f"s2 infer error: {e}")
                    self.s2_output.is_infering = False
                    self.policy.reset()
                    success = False
                if not success:
                    try:
                        current_s2_output = self.policy.s2_step(
                            self.s2_input.rgb,
                            self.s2_input.depth,
                            self.s2_input.pose,
                            _instruction,
                            self.camera_intrinsic,
                            False,
                            memory_context=_memory_context,
                            bev_tokens=_bev_tokens,  # C12
                        )
                    except Exception as e:
                        print(f"s2 infer error: {e}")
                        self.s2_output.is_infering = False
                        self.policy.reset()
                        self.s2_output.output_pixel = None
                        self.s2_output.output_action = [0]  # finish the inference
                        self.s2_output.output_latent = None
                        continue

                print("s2 infer finish!!")

                # Update latent thought state: use output_latent from S2 when available,
                # else zero context (graceful degradation — still tracks episode dynamics)
                try:
                    _out_latent = getattr(current_s2_output, 'output_latent', None)
                    if _out_latent is not None and _out_latent.numel() >= 256:
                        _ctx = _out_latent.detach().reshape(1, -1)[:, :256].to(self.device)
                    else:
                        _ctx = torch.zeros(1, 256, device=self.device)
                    with torch.no_grad():
                        _, _thought, _conf = self.latent_reasoner(_ctx, self.s2_thought_cache.current_thought)
                    self.s2_thought_cache.update(_thought)
                    _ctx_src = 'latent' if _out_latent is not None else 'zero'
                    print(f"[LatentReasoner] ctx={_ctx_src} {self.s2_thought_cache.stats_str()} conf={_conf.item():.3f}")
                except Exception as _le:
                    print(f"[LatentReasoner] update skipped: {_le}")

                # Memory Triad: passive per-step updates (no LLM input change; state for training)
                try:
                    _pose = self.s2_input.pose  # [x, y, theta] or None
                    if _pose is not None:
                        _odom = (_pose[0], _pose[1], _pose[2] if len(_pose) > 2 else 0.0)
                        self.body_memory.update(_odom)
                        _pose_arr = np.array(_odom, dtype=np.float32)
                        _emb_vec = np.zeros(256, dtype=np.float32)
                        self.world_memory.add_node(_pose_arr, _emb_vec)
                    _llm_text = getattr(current_s2_output, 'output_text', '') or ''
                    if _llm_text and self.s2_input.instruction:
                        self.task_memory.update(_llm_text, [self.s2_input.instruction])
                except Exception as _me:
                    pass  # memory updates are best-effort

                # WorldPredictor: online BCE reachability update
                try:
                    if self.s2_input.rgb is not None and _pose is not None:
                        _rgb_t = torch.from_numpy(
                            np.array(self.s2_input.rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
                        ).unsqueeze(0)
                        _dp = torch.tensor([_odom[0], _odom[1], _odom[2]], dtype=torch.float32)
                        _reward = 1.0 if getattr(current_s2_output, 'output_action', [0])[0] != 0 else 0.0
                        self.world_predictor.update(_rgb_t, _dp, _reward)
                except Exception:
                    pass  # world predictor updates are best-effort

                # C11 LACP: update K-step ahead prediction from current S2 latent
                try:
                    if self.lacp is not None:
                        _out_lat = getattr(current_s2_output, 'output_latent', None)
                        _out_act = getattr(current_s2_output, 'output_action', None)
                        if _out_lat is not None and _out_lat.numel() >= 256:
                            _h_now = _out_lat.detach().reshape(1, -1)[:, :256].float().to(self.device)
                            if _out_act is not None and len(_out_act) > 0:
                                _traj_k = torch.tensor(
                                    _out_act[:self.lacp.cfg.K], dtype=torch.float32
                                ).unsqueeze(0).to(self.device)
                            else:
                                _traj_k = torch.zeros(1, self.lacp.cfg.K, 2, device=self.device)
                            with torch.no_grad():
                                self._lacp_h_pred = self.lacp(_h_now, _traj_k)
                except Exception:
                    pass

                # Update output state
                with self.s2_output_lock:
                    print("get s2 output lock")
                    # S2 output

                    self.s2_output.output_pixel = current_s2_output.output_pixel
                    self.s2_output.output_action = current_s2_output.output_action
                    self.s2_output.output_latent = current_s2_output.output_latent
                    self.s2_output.idx = s2_infer_idx
                    self.s2_output.rgb_memory = self.s2_input.rgb
                    self.s2_output.depth_memory = self.s2_input.depth
                    self.s2_output.is_infering = False
                time.sleep(0.01)  # Sleep briefly after completing inference

        self.s2_thread = threading.Thread(target=s2_thread_func)
        self.s2_thread.daemon = True
        self.s2_thread.start()

    def should_infer_s2(self, mode="partial_async"):
        """Function: Enables the sys2 inference thread depending on the mode.
        mode: just support 2 modes: "sync" and "partial_async".
        "sync": Synchronous mode (navdp_version >= 0.0), Sys1 and Sys2 execute in a sequential inference chain.
        "partial_async": Asynchronous mode (navdp_version > 0.0, e.g., 0.1),
                         Sys2 performs a single inference, while Sys1 performs multiple inference cycles.
        """
        if self.episode_step == 0:
            return True

        if self.s2_output.is_infering:
            return False

        # 1. Synchronous mode: infer S2 every frame to provide to S1 for execution
        if mode == "sync":
            if self.s2_output.output_action is None:
                return True
            else:
                return False
        # 2. Partial async mode: S2 infers 1 frame while S1 executes multi frames
        if mode == "partial_async":
            if self.dual_forward_step >= self.sys2_max_forward_step:
                return True
            if (
                self.s2_output.output_action is None
                and self.s2_output.output_pixel is None
                and self.s2_output.output_latent is None
            ):
                # This normally only occurs when output is discrete action and discrete action has been fully executed
                return True

            # C11 LACP anticipatory gate: trigger S2 early when predicted latent diverges
            try:
                if (self.lacp is not None and self._lacp_h_pred is not None
                        and self.s2_output.output_latent is not None):
                    _lat = self.s2_output.output_latent.detach()
                    if _lat.numel() >= 256:
                        _h_actual = _lat.reshape(1, -1)[:, :256].float()
                        if self.lacp.should_trigger_s2(_h_actual, self._lacp_h_pred):
                            self._lacp_h_pred = None
                            return True
            except Exception:
                pass

            return False
        raise ValueError("Invalid mode: {}".format(mode))

    def step(self, obs):
        mode = self.mode  # 'sync', 'partial_async'

        obs = obs[0]  # do not support batch_env currently?
        rgb = obs['rgb']
        depth = obs['depth']
        instruction = obs['instruction']
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # S2 inference is done in a separate thread
        if self.should_infer_s2(mode) or self.look_down:  # The look down frame must be inferred
            print(f"======== Infer S2 at step {self.episode_step}========")
            with self.s2_input_lock:
                self.s2_input.idx = self.episode_step
                self.s2_input.rgb = rgb
                self.s2_input.depth = depth
                self.s2_input.pose = pose
                self.s2_input.instruction = instruction
                self.s2_input.should_infer = True
                self.s2_input.look_down = self.look_down
                self.s2_output.is_infering = True  # for async

            self.dual_forward_step = 0
        else:
            # Even if this frame doesn't do s2 inference, rgb needs to be provided to ensure history is correct
            self.policy.step_no_infer(rgb, depth, pose)
        # S1 inference is done in the main thread
        while self.s2_output.is_infering:
            time.sleep(0.5)

        while not self.s2_output.validate():
            time.sleep(0.2)

        output = {}
        # Simple branch:
        # 1. If S2 output is full discrete actions, don't execute S1 and return directly
        # Recovery override: force LEFT turn (action=2) when oscillation was detected
        if self._recovery_steps_remaining > 0:
            output['action'] = [2]  # LEFT — escape the loop
            self._recovery_steps_remaining -= 1
            self.episode_step += 1
            self.dual_forward_step += 1
            return output

        print('===============', self.s2_output.output_action, '=================')
        if self.s2_output.output_action is not None:
            output['action'] = [self.s2_output.output_action[0]]

            with self.s2_output_lock:
                self.s2_output.output_action = self.s2_output.output_action[1:]
                if self.s2_output.output_action == []:
                    self.s2_output.output_action = None
            if output['action'][0] == 5:
                self.look_down = True
                # Clear action list when looking down
                with self.s2_output_lock:
                    self.s2_output.output_action = None
                    self.s2_output.output_pixel = None
                    self.s2_output.output_latent = None
                output['action'] = [-1]
                self.sys1_infer_times = 0
            else:
                self.look_down = False
                if self.sys1_infer_times > 0:
                    self.dual_forward_step += 1

        else:
            self.look_down = False
            # 2. If output is in latent form, execute latent S1
            if self.s2_output.output_latent is not None:
                self.output_pixel = copy.deepcopy(self.s2_output.output_pixel)
                print(self.output_pixel)

                #  Patch: Ensure 3D array (C, H, W) for NumPy inputs
                if self.s2_output.depth_memory.ndim == 2:
                    self.s2_output.depth_memory = self.s2_output.depth_memory[..., np.newaxis]
                if depth.ndim == 2:
                    depth = depth[..., np.newaxis]

                if mode != 'sync':
                    processed_pixel_rgb = (
                        np.array(Image.fromarray(self.s2_output.rgb_memory).resize((224, 224))) / 255.0
                    )
                    processed_pixel_depth = (
                        np.array(Image.fromarray(self.s2_output.depth_memory[:, :, 0]).resize((224, 224))) * 10.0
                    )
                    processed_pixel_depth[processed_pixel_depth > self.sys1_depth_threshold] = self.sys1_depth_threshold

                    processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255.0
                    processed_depth = (
                        np.array(Image.fromarray(depth[:, :, 0]).resize((224, 224))) * 10.0
                    )  # should be 0-10m (TODO: why 10.0?)
                    processed_depth[processed_depth > self.sys1_depth_threshold] = self.sys1_depth_threshold

                    rgbs = (
                        torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                        .unsqueeze(0)
                        .to(self.device)
                    )  # [1, 2, 224, 224, 3]
                    depths = (
                        torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .to(self.device)
                    )  # [1, 2, 224, 224, 1]
                    self.s1_output = self.policy.s1_step_latent(rgbs, depths, self.s2_output.output_latent)
                else:
                    self.s1_output = self.policy.s1_step_latent(rgb, depth * 10000.0, self.s2_output.output_latent)

            else:
                assert False, f"S2 output should be either action or latent, but got neither!  {self.s2_output}"

            if self.s1_output.idx == []:
                output['action'] = [-1]
            else:
                output['action'] = [self.s1_output.idx[0]]
            with self.s2_output_lock:
                if len(self.s1_output.idx) > 1:
                    self.s2_output.output_action = self.s1_output.idx[1:]
                    if self.s2_output.output_action == []:
                        self.s2_output.output_action = None
                else:
                    self.s2_output.output_action = None

                self.s2_output.output_pixel = None  # TODO: now just for visulization
                if mode == 'sync':
                    self.s2_output.output_latent = None
                else:
                    # already reach the pixel-goal
                    if len(self.s1_output.idx) < self.sys1_forward_step:
                        all_step_ = len(self.s1_output.idx) + self.dual_forward_step
                        if all_step_ < self.sys2_max_forward_step:
                            self.dual_forward_step = self.sys2_max_forward_step - len(self.s1_output.idx)

                    self.sys1_infer_times += 1
                    self.dual_forward_step += 1

                    if self.dual_forward_step > self.sys2_max_forward_step:
                        print("!!!!!!!!!!!!")
                        print("ERR: self.dual_forward_step ", self.dual_forward_step, " > ", self.sys2_max_forward_step)
                        print("Potential reason: sys1 infers empty trajectory list []")
                        print("!!!!!!!!!!!!")

        print('Output discretized traj:', output['action'], self.dual_forward_step)

        # Visualization
        if self.vis_debug:
            vis = rgb.copy()
            if 'action' in output:
                vis = cv2.putText(vis, str(output['action'][0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.output_pixel is not None:
                pixel = self.output_pixel
                vis = cv2.putText(
                    vis,
                    f"{pixel[1]}, {pixel[0]} ({self.s2_output.idx})",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.circle(vis, (pixel[1], pixel[0]), 5, (0, 255, 0), -1)
                self.output_pixel = None
            self.fps_writer.append_data(vis)

            if self.s1_output.vis_image is not None:
                Image.fromarray(self.s1_output.vis_image).save(
                    os.path.join("./vis_debug_pix/", f"ttttt_{self.episode_step}.png")
                )
                self.fps_writer2.append_data(self.s1_output.vis_image)

        # C10 NavCoT: track action history for future CoT context
        try:
            if 'action' in output and output['action']:
                _act_id = output['action'][0]
                _act_names = {0:"STOP",1:"MOVE_FORWARD",2:"TURN_LEFT",3:"TURN_RIGHT",4:"LOOK_UP",5:"LOOK_DOWN"}
                self._action_history.append(_act_names.get(_act_id, str(_act_id)))
                if len(self._action_history) > 12:
                    self._action_history = self._action_history[-12:]
        except Exception:
            pass

        self.episode_step += 1
        if 'action' in output:
            return [{'action': output['action'], 'ideal_flag': True}]
        elif 'velocity' in output:
            return [{'action': output['velocity'], 'ideal_flag': False}]
        else:
            assert False
