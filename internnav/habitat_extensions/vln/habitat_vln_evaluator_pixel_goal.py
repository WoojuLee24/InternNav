"""Habitat VLN evaluator with the pixel_goal S1 conditioning (train parity).

Subclass of ``HabitatVLNEvaluatorBEV`` — neither ``habitat_vln_evaluator.py``
nor ``habitat_vln_evaluator_bev.py`` is modified.

Selected via config ``eval_type='habitat_vln_pixel_goal'``. Reuses
HabitatVLNEvaluatorBEV's ``bev_s1_mode='fpv'|'bev'`` switch and provider
intrinsics wiring unchanged — both pixel_goal training configs
(``base_s1.fpv_s2.fpv_rgb_gt.pixel_goal.py`` / ``s1.bev_s2.fpv_rgb_gt.pixel_goal.py``)
set ``bev=True`` (-> ``visual_provider='bev_image'``), so ``self.provider.processor``
(real Habitat sensor intrinsics) always exists regardless of ``bev_s1_mode``.

Two injection points, marked ``# === PIXEL_GOAL ===``:
  - ``__init__``: the parent always loads ``InternVLAN1ForCausalLM`` (it has no
    reason to load anything else). ``generate_traj_pixel()`` only exists on
    ``InternVLAN1BEVProviderPixelGoalForCausalLM``, so this subclass reloads
    the checkpoint with that class right after ``super().__init__()`` instead
    of touching the parent's inline model-loading code.
  - ``_run_eval_dual_system`` (a copy of the BEV parent's method, per its own
    established convention of copying + marking injection points): the old
    branch called ``generate_latents()`` + ``generate_traj()``, which never
    actually conditions the diffusion policy on the pointed-at pixel — see
    ``.claude/tasks/260706_pixelgoal_result.md``. It's replaced with
    ``self.model.generate_traj_pixel()``, the exact inference path
    ``internvla_n1_pixel_goal_agent.py`` uses for InternUtopia/Isaac, so eval
    here matches training's ``pixel_goal_mode`` (``'prepend'`` by default, or
    ``'mlp_cond'``) end-to-end — same pixel-decoding convention
    (``fpv_pixel_to_normalized`` against the native 640x480 frame), same
    metric conversion (``generate_traj_pixel``'s ``_pixel_norm_to_metric``).

Camera intrinsics for that pixel -> metric conversion are passed in from
``self.provider.processor`` (real Habitat sensor fx/fy/cx/cy/ref_width/
ref_height, already populated by ``HabitatVLNEvaluatorBEV.__init__``) rather
than left at ``generate_traj_pixel``'s Isaac-camera defaults, since Habitat's
FOV need not match the robot camera pixel_goal was trained on.
"""

import copy
import json
import os
import random
import re
from collections import OrderedDict  # noqa: F401  (kept for parity with parent imports)

import cv2
import imageio
import numpy as np
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from PIL import Image

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import Evaluator
from internnav.habitat_extensions.vln.habitat_vln_evaluator import (
    DEFAULT_IMAGE_TOKEN,
    MAX_LOCAL_STEPS,
    MAX_STEPS,
    _diag_cam_rotation,
    _diag_env_once,
    _diag_gl_errors,
    _diag_gl_reset,
    _diag_total_vram,
    _diag_vram,
    _install_step_monitor,
    action_code,
)
from internnav.habitat_extensions.vln.habitat_vln_evaluator_bev import HabitatVLNEvaluatorBEV
from internnav.habitat_extensions.vln.utils import preprocess_depth_image_v2
from internnav.model.basemodel.internvla_n1.internvla_n1_bev_provider import _BASE_H, _BASE_W
from internnav.model.basemodel.internvla_n1.internvla_n1_pixel_goal import (
    InternVLAN1BEVProviderPixelGoalForCausalLM,
)
from internnav.model.utils.pixel_goal_utils import fpv_pixel_to_normalized
from internnav.model.utils.vln_utils import split_and_clean


class HabitatVLNEvaluatorPixelGoal(HabitatVLNEvaluatorBEV):
    def __init__(self, cfg: EvalCfg):
        super().__init__(cfg)

        # === PIXEL_GOAL: swap in the model class that has generate_traj_pixel().
        # Guard clause — 'system2' mode never uses S1/pixel_goal at all. ===
        if self.model_args.mode != 'dual_system':
            return
        del self.model
        torch.cuda.empty_cache()
        self.model = InternVLAN1BEVProviderPixelGoalForCausalLM.from_pretrained(
            self.model_args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
        self.model.eval()

    def _pixel_coord_to_norm_tensor(self, coord):
        """[col, row] digits parsed from S2's text output (native 640x480 camera
        frame, matching training's ``_extract_pixel_coords`` convention) ->
        (1, 2) normalized [x, y] tensor for ``generate_traj_pixel``. Mirrors
        ``internvla_n1_pixel_goal.py``'s ``s1_step_pixel``."""
        col_row = np.array([coord[0], coord[1]])
        pixel_norm = fpv_pixel_to_normalized(col_row, (_BASE_W, _BASE_H))
        return torch.tensor(pixel_norm, dtype=torch.bfloat16).unsqueeze(0).to(self.device)

    def _generate_traj_pixel(self, pixel_tensor, images_dp, depths_dp):
        """generate_traj_pixel with real Habitat sensor intrinsics/pitch (from
        self.provider), instead of the Isaac-camera constants it defaults to."""
        proc = self.provider.processor
        return self.model.generate_traj_pixel(
            pixel_coord_norm=pixel_tensor,
            images_dp=images_dp,
            depths_dp=depths_dp,
            predict_step_nums=self.predict_step_num,
            cam_pitch_deg=self.provider.s1_pitch_deg or 0.0,
            fx=proc.fx, fy=proc.fy, cx=proc.cx, cy=proc.cy,
            ref_w=proc.ref_width, ref_h=proc.ref_height,
        )

    # ------------------------------------------------------------------ loop

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        """Copy of HabitatVLNEvaluatorBEV._run_eval_dual_system with the
        pixel_goal branch (marked ``# === PIXEL_GOAL ===``) swapped to
        generate_traj_pixel; all BEV injection points (``# === BEV ===``)
        untouched."""
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

            # === BEV: clear per-episode provider state ===
            self.provider.reset()

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

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
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
                    # === BEV: keep the look-down frame pair for the provider (depth in metres) ===
                    look_down_rgb_np = rgb
                    look_down_depth_m = depth / 1000.0
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
                    # === BEV: keep the look-down frame pair for the provider (depth in metres) ===
                    look_down_rgb_np = down_observations["rgb"]
                    look_down_depth_m = depth / 1000.0
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
                    s2_extra = []  # === BEV ===
                    s2_mode = getattr(self.provider, 's2_mode', 'fpv_bev')  # === BEV ===
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        # === BEV: BEV image(s) of the look-down frame either
                        # replace ('bev') or follow ('fpv_bev') the FPV frame ===
                        s2_extra = self.provider.get_s2_extra(
                            look_down_rgb_np, look_down_depth_m, is_lookdown=True
                        )
                        if s2_extra and s2_mode == 'bev':
                            input_images += s2_extra
                            input_img_id = -len(s2_extra)
                        else:
                            input_images += [look_down_image] + s2_extra
                            input_img_id = -(1 + len(s2_extra))
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
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

                    # === BEV: one <image> token per image of this turn ===
                    if s2_extra and s2_mode == 'bev':
                        prompt = random.choice(self.conjunctions) + "the bird's-eye view of your surroundings:"
                        prompt += ('\n' + DEFAULT_IMAGE_TOKEN) * len(s2_extra)
                    else:
                        prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                        if s2_extra:
                            prompt += " This is the bird's-eye view of your surroundings: "
                            prompt += ('\n' + DEFAULT_IMAGE_TOKEN) * len(s2_extra)
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

                    # FPV image [384, 384], depth [384, 384]; BEV ??
                    # look_down_images [640, 480], look_down_depths_m [640, 480]
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

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        local_actions = []

                        # === PIXEL_GOAL: [col, row] text -> normalized [x, y] in the
                        # native camera frame (same convention as training's
                        # _extract_pixel_coords / Isaac's s1_step_pixel) ===
                        pixel_tensor = self._pixel_coord_to_norm_tensor(coord)

                        # === FPV: images_dp: [B=1, T=2, H=224, W=224, 3], depths_dp: [B=1, T=2, H=224, W=224]
                        self.provider.set_goal(look_down_rgb_np, look_down_depth_m)

                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        pix_goal_image = copy.copy(image_dp)
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        pix_goal_depth = copy.copy(depth_dp)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        # === BEV: provider may replace the NavDP visual input ===
                        # images_dp: [B=1, T=2, H=224, W=224, 3], depths_dp: [B=1, T=2, H=224, W=224]
                        images_dp, depths_dp = self._apply_s1_provider(images_dp, depths_dp)

                        # === PIXEL_GOAL: generate_traj_pixel (not generate_latents +
                        # generate_traj) — the same S1 inference path
                        # internvla_n1_pixel_goal_agent.py uses, so eval matches
                        # config.pixel_goal_mode ('prepend'/'mlp_cond') from training ===
                        with torch.no_grad():
                            dp_actions = self._generate_traj_pixel(pixel_tensor, images_dp, depths_dp)

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

                        # === BEV: provider may replace the NavDP visual input ===
                        images_dp, depths_dp = self._apply_s1_provider(images_dp, depths_dp)

                        # === PIXEL_GOAL: same pixel_tensor as when the goal was first
                        # decoded this dual-forward segment ===
                        with torch.no_grad():
                            dp_actions = self._generate_traj_pixel(pixel_tensor, images_dp, depths_dp)

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

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

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
            process_bar.update(1)

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

            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            if self.rank == 0:
                os.makedirs(self.output_path, exist_ok=True)
                with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
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


# Evaluator.register's decorator returns None, so register without decorating
# to keep the class name importable.
Evaluator.register('habitat_vln_pixel_goal')(HabitatVLNEvaluatorPixelGoal)
