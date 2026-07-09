"""Habitat VLN evaluator for UnifiedImageProvider (fpv + bev-or-depth S2).

Subclass of ``HabitatVLNEvaluator`` — habitat_vln_evaluator.py is NOT modified.
Selected via config ``eval_type='habitat_vln_unified'`` with
``visual_provider='unified_image'``.

``_run_eval_dual_system`` is derived from HabitatVLNEvaluatorBEV with the key
change that S2 BEV is provided every step (not only on LOOKDOWN turns), matching
the training setup where dual cameras (pitch_1 FPV + pitch_2 lookdown) are
available for every decision frame.

NOTE: this module must be imported for the registry entry to exist — config files
do this with ``import internnav.habitat_extensions.vln.habitat_vln_evaluator_unified``.
"""

import copy
import json
import math
import os
import random
import re

import cv2
import imageio
import numpy as np
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat.utils.visualizations.maps import calculate_meters_per_pixel, colorize_topdown_map
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from PIL import Image

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import Evaluator
from internnav.habitat_extensions.vln.habitat_vln_evaluator import (  # noqa: F401
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
from internnav.habitat_extensions.vln.utils import preprocess_depth_image_v2
from internnav.model.utils.unified_image_provider import _OCC_MODE_MAP
from internnav.model.utils.visual_input_provider import create_visual_provider
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions

_HabitatVLNEvaluator = Evaluator.evaluators['habitat_vln']
_LOOKDOWN_PITCH_OFFSET_DEG = 60.0


class HabitatVLNEvaluatorUnified(_HabitatVLNEvaluator):
    def __init__(self, cfg: EvalCfg):
        debug_dir = (cfg.agent.model_settings or {}).get('debug_dir')
        bev_range  = (cfg.agent.model_settings or {}).get('bev_range', 5.0)
        self._has_topdown_cam = False
        if debug_dir:
            import internnav.habitat_extensions.vln.habitat_vln_evaluator as _hve_mod
            import habitat as _habitat
            from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig
            from omegaconf import OmegaConf
            _orig_get_cfg = _hve_mod.get_habitat_config
            bev_cam_h   = bev_range * 2.0
            bev_hfov    = math.degrees(2.0 * math.atan(bev_range / bev_cam_h))
            def _patched(path):
                config = _orig_get_cfg(path)
                with _habitat.config.read_write(config):
                    cam_cfg = HabitatSimRGBSensorConfig(
                        height=500, width=500, hfov=int(round(bev_hfov)),
                        position=[0.0, bev_cam_h, 0.0],
                        orientation=[-math.pi / 2, 0.0, 0.0],
                    )
                    config.habitat.simulator.agents.main_agent.sim_sensors.update(
                        {'topdown_rgb': cam_cfg}
                    )
                    node = config.habitat.simulator.agents.main_agent.sim_sensors.topdown_rgb
                    OmegaConf.set_struct(node, False)
                    node.uuid = 'topdown_rgb'
                return config
            _hve_mod.get_habitat_config = _patched
            try:
                super().__init__(cfg)
            finally:
                _hve_mod.get_habitat_config = _orig_get_cfg
            self._has_topdown_cam = True
        else:
            super().__init__(cfg)

        settings = dict(cfg.agent.model_settings)
        sensor = self.sim_sensors_config.depth_sensor
        base_pitch = settings.get('bev_cam_pitch_deg', 0.0)
        settings.setdefault('bev_fx', float(self._fx))
        settings.setdefault('bev_fy', float(self._fy))
        settings.setdefault('bev_cx', (sensor.width - 1) / 2.0)
        settings.setdefault('bev_cy', (sensor.height - 1) / 2.0)
        settings.setdefault('bev_ref_width', sensor.width)
        settings.setdefault('bev_ref_height', sensor.height)
        settings.setdefault('bev_cam_height', float(self._camera_height))
        settings.setdefault('bev_s1_pitch_deg', base_pitch + _LOOKDOWN_PITCH_OFFSET_DEG)
        # S2 source frame depends on s2_image_view (:394-397 below — 'bev_ld'=lookdown,
        # 'bev'/other=FPV) — pitch must match the ACTUAL camera angle of that source or
        # the depth->world projection (depth_rgb_to_bev_torch._build_R_c2w) is wrong.
        s2_image_view = settings.get('s2_image_view', 'fpv')
        if s2_image_view == 'bev_ld':
            settings.setdefault('bev_s2_pitch_deg', base_pitch + _LOOKDOWN_PITCH_OFFSET_DEG)
        else:
            settings.setdefault('bev_s2_pitch_deg', base_pitch)
        self.provider = create_visual_provider(settings, device=str(self.device))
        self.provider.s2_depth_in_meters = True
        self._tdm_mpp = None
        self.s2_source_view = settings.get('s2_image_view', 'fpv')
        print(
            f"[unified] s1_view={settings.get('s1_image_view', 'fpv')} "
            f"s2_view={settings.get('s2_image_view', 'fpv')} "
            f"s2_combine={settings.get('s2_combine_mode', 'none')} "
            f"s2_source={self.s2_source_view}",
            flush=True,
        )

    # ------------------------------------------------------------------ S1

    def _apply_s1_provider(self, images_dp: torch.Tensor, depths_dp: torch.Tensor):
        s1v = self.provider.get_s1_input(images_dp, depths_dp)
        images_dp = s1v.images if s1v.images is not None else images_dp
        depths_dp = s1v.depths if s1v.depths is not None else depths_dp
        # extra_images (s1_combine_mode='concat'): BEV computed for both [goal, cur]
        # slots — keep only "cur" (index -1); the goal-frame BEV is discarded, same
        # acceptance already implicit in 'replace' mode substituting both slots.
        bev_images = s1v.extra_images[:, -1:] if s1v.extra_images is not None else None
        # 'replace' substitutes images_dp with BEV; 'concat' returns it via extra_images
        # instead (see above) — both compute a BEV through _save_s1_debug (bumping
        # provider._debug_step), so both need the matching world-frame tdmap overlay.
        if self.provider.s1_view == 'bev' and self.provider.s1_combine in ('replace', 'concat') and self.provider.debug_dir:
            self._save_eval_tdmap_debug()
        return images_dp, depths_dp, bev_images

    # ------------------------------------------------------------------ S2

    def _apply_s2_provider(self, rgb_np, depth_m, file_stem='s2_step'):
        s2_extra = self.provider.get_s2_extra(rgb_np, depth_m, is_lookdown=True)
        if s2_extra and self.provider.s2_view in ('bev', 'bev_ld') and self.provider.debug_dir:
            self._save_eval_tdmap_debug(step=self.provider._s2_debug_step - 1, file_stem=file_stem)
        return s2_extra

    def _save_eval_tdmap_debug(self, step=None, file_stem='step'):
        try:
            from habitat.tasks.nav.nav import TopDownMap as _TopDownMap

            s         = (self.provider._debug_step - 1) if step is None else step
            debug_dir = self.provider.debug_dir
            os.makedirs(debug_dir, exist_ok=True)

            agent_angle = float(
                _TopDownMap.get_polar_angle(self.env._env.sim.get_agent_state())
            )
            S         = self.provider.processor.bev_size
            bev_range = self.provider.processor.bev_range
            c_px      = S // 2
            px_m      = S / (2.0 * bev_range)
            fa        = int(2.5 * px_m)
            rot_deg   = math.degrees(math.pi + agent_angle)

            if self._has_topdown_cam:
                sim_obs = self.env._env.sim.get_sensor_observations()
                cam_rgb = sim_obs.get('topdown_rgb')
                if cam_rgb is None:
                    return
                img_bgr = cv2.cvtColor(
                    cv2.resize(cam_rgb, (S, S), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_RGB2BGR,
                )
            else:
                info = self.env.get_metrics()
                if info is None:
                    return
                tdm = info.get('top_down_map')
                if tdm is None or tdm.get('map') is None:
                    return
                raw_map  = tdm['map']
                fog_mask = tdm.get('fog_of_war_mask')
                tdm_rgb  = colorize_topdown_map(raw_map, fog_mask)
                if self._tdm_mpp is None:
                    try:
                        self._tdm_mpp = calculate_meters_per_pixel(
                            1024, pathfinder=self.env._env.sim.pathfinder
                        )
                    except Exception:
                        self._tdm_mpp = 1.0
                coord = tdm['agent_map_coord']
                agent_row, agent_col = coord[0]
                half_px = int(math.ceil(bev_range / self._tdm_mpp))
                H_map, W_map = raw_map.shape[:2]
                r0, r1 = agent_row - half_px, agent_row + half_px
                c0, c1 = agent_col - half_px, agent_col + half_px
                pad_t = max(0, -r0);  pad_b = max(0, r1 - H_map)
                pad_l = max(0, -c0);  pad_r = max(0, c1 - W_map)
                crop_src = tdm_rgb[max(0, r0):min(H_map, r1), max(0, c0):min(W_map, c1)]
                crop = cv2.copyMakeBorder(crop_src, pad_t, pad_b, pad_l, pad_r,
                                          cv2.BORDER_CONSTANT, value=(30, 30, 30))
                img_bgr = cv2.cvtColor(
                    cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_RGB2BGR,
                )

            def _rotate_world(im):
                if abs(rot_deg % 360) > 0.5:
                    M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), rot_deg, 1.0)
                    im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
                return im

            def _draw_agent(im):
                h, w = im.shape[:2]
                cx, cy_px = w // 2, h // 2
                fa_l = int(2.5 * (w / (2.0 * bev_range)))
                ac = int(cx + fa_l * math.sin(agent_angle))
                ar = int(cy_px + fa_l * math.cos(agent_angle))
                cv2.arrowedLine(im, (cx, cy_px), (ac, ar), (0, 220, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)
                cv2.circle(im, (cx, cy_px), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(im, (cx, cy_px), 8, (255, 255, 255), 1, cv2.LINE_AA)
                return im

            # tdmap content only depends on current sim state, not on which loop branch
            # triggered the save — so S1 uses 'step', ALL S2 branches (LOOKDOWN-turn or
            # every-loop) unify under 's2_step' to match get_s2_extra's own file prefix.
            depth_src = self.provider.processor.depth_source
            if file_stem == 'step':
                # S1 BEV debug filenames (UnifiedImageProvider._save_s1_debug, unified_image_provider.py)
                prefix = 'step'
                bev_prefix = f"step_{s:06d}_b00"
                pairs = [(f'_3_bev_{depth_src}.jpg', f'_w3_bev_world_{depth_src}.jpg')]
            elif file_stem in ('s2_el', 's2_ld', 's2_step'):
                # S2 BEV debug filenames (UnifiedImageProvider.get_s2_extra, unified_image_provider.py)
                prefix = 's2_step'
                type_sfx = (f'_{_OCC_MODE_MAP.get(self.provider.s2_image_mode, "rgb")}'
                            if self.provider.s2_type == 'depth' else '')
                bev_prefix = f"s2_step_{s:06d}"
                pairs = [(f'_2_bev{type_sfx}_{depth_src}.jpg', f'_w2_bev{type_sfx}_world_{depth_src}.jpg')]
            else:
                assert False, f"unreachable file_stem={file_stem!r}"

            cv2.imwrite(f"{debug_dir}/{prefix}_{s:06d}_b00_5_gt_cam.jpg", img_bgr)
            cv2.imwrite(f"{debug_dir}/{prefix}_{s:06d}_w5_gt_topdown_world.jpg",
                        _draw_agent(_rotate_world(img_bgr)))

            for src_suffix, dst_suffix in pairs:
                bev_img = cv2.imread(f"{debug_dir}/{bev_prefix}{src_suffix}")
                if bev_img is not None:
                    cv2.imwrite(f"{debug_dir}/{bev_prefix}{dst_suffix}", _rotate_world(bev_img))
        except Exception:
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ loop

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        """HabitatVLNEvaluatorBEV._run_eval_dual_system with always-S2.

        Key difference from BEV evaluator: S2 BEV is provided every step
        (else branch), not only on LOOKDOWN turns, matching training where
        dual cameras give S2 at every decision frame.
        """
        self.model.eval()
        _diag_env_once()
        _diag_vram("eval_start")
        _install_step_monitor(self.env)

        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            self.provider.reset()

            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

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
                    fpv_rgb_np = rgb          # FPV source: save before depth gets overwritten
                    fpv_depth_m = depth / 1000.0

                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)

                    look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                    depth = down_observations["depth"]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
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
                    s2_extra = []
                    s2_combine = self.provider.s2_combine  # 'none' | 'replace' | 'concat'
                    if action == action_code.LOOKDOWN:
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        s2_extra = self._apply_s2_provider(look_down_rgb_np, look_down_depth_m, file_stem='s2_ld')
                        if not s2_extra:
                            input_images += [look_down_image]
                            input_img_id = -1
                        elif s2_combine == 'replace':
                            # BEV replaces the look-down FPV frame entirely
                            input_images += s2_extra
                            input_img_id = -len(s2_extra)
                        elif s2_combine == 'concat':
                            # BEV follows the look-down FPV frame (both kept)
                            input_images += [look_down_image] + s2_extra
                            input_img_id = -(1 + len(s2_extra))
                        else:
                            assert False, f"unreachable s2_combine={s2_combine!r} (s2_extra non-empty implies combine != 'none')"
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

                        # === S2: always — source selected by s2_image_view ('bev_ld'=lookdown, 'bev'=FPV) ===
                        if self.s2_source_view == 'bev_ld':
                            s2_src_rgb, s2_src_depth = look_down_rgb_np, look_down_depth_m
                        else:
                            s2_src_rgb, s2_src_depth = fpv_rgb_np, fpv_depth_m
                        s2_extra = self._apply_s2_provider(s2_src_rgb, s2_src_depth, file_stem='s2_el')
                        if s2_extra:
                            if s2_combine == 'replace':
                                # BEV replaces current FPV: rebuild without cur_images to match token count
                                input_images = [rgb_list[i] for i in history_id] + s2_extra
                            elif s2_combine == 'concat':
                                # BEV supplements current FPV: keep cur, append BEV
                                input_images = input_images + s2_extra
                            else:
                                assert False, f"unreachable s2_combine={s2_combine!r} (s2_extra non-empty implies combine != 'none')"

                    s2_image_label = getattr(self.provider, 's2_image_label', "the bird's-eye view of your surroundings")
                    if s2_extra and s2_combine == 'replace':
                        prompt = random.choice(self.conjunctions) + f"{s2_image_label}:"
                        prompt += ('\n' + DEFAULT_IMAGE_TOKEN) * len(s2_extra)
                    else:
                        prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                        if s2_extra:
                            prompt += f" This is {s2_image_label}: "
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

                    if bool(re.search(r'\d', llm_outputs)):
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        local_actions = []
                        pixel_values = inputs.pixel_values
                        image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                        with torch.no_grad():
                            traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                        self.provider.set_goal(look_down_rgb_np, look_down_depth_m)

                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        pix_goal_image = copy.copy(image_dp)
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        pix_goal_depth = copy.copy(depth_dp)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        images_dp, depths_dp, bev_images = self._apply_s1_provider(images_dp, depths_dp)

                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp, bev_images=bev_images)

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
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        images_dp, depths_dp, bev_images = self._apply_s1_provider(images_dp, depths_dp)

                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp, bev_images=bev_images)

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


Evaluator.register('habitat_vln_unified')(HabitatVLNEvaluatorUnified)
