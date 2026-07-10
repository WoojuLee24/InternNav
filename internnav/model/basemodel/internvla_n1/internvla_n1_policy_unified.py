"""InternVLA-N1 policy with pluggable unified (view/type/mode/combine) visual input
(Isaac Sim / H1 path).

Subclass of ``InternVLAN1Net`` — internvla_n1_policy.py is NOT modified. Isaac-Sim
analog of ``internvla_n1_policy_bev.py``'s ``InternVLAN1NetBEV``, but built on
``UnifiedImageProvider`` (``create_unified_provider``) instead of the legacy
``BEVImageProvider`` (``create_visual_provider``), so it supports the same
``s1_image_*``/``s2_image_*`` axes the ``image_base`` Habitat configs already use.

Selected via config: ``model_name='internvla_n1_unified'`` (see
``internnav/agent/internvla_n1_agent_unified.py``). With every axis at its default
(``fpv``/``rgb``/``raw``/``none``) every override degenerates to the parent behaviour.

Ground-truth topdown debug jpgs (``internnav/model/utils/isaac_topdown_debug.py``) are
sourced from Isaac's own ``topdown_camera_500`` sensor obs — see
``internnav/agent/internvla_n1_agent_unified.py``'s ``step()`` override, which snapshots
``obs['topdown_rgb']``/``obs['globalrotation']`` onto this policy via
``set_debug_topdown`` before ``s1_step_latent``/``s2_step`` run (those only ever
receive ``rgb``/``depth``, not the full obs dict).
"""

import copy
import re

import numpy as np
import torch
from PIL import Image

from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net
from internnav.model.utils.isaac_topdown_debug import save_isaac_topdown_debug
from internnav.model.utils.unified_image_provider import create_unified_provider
from internnav.model.utils.vln_utils import S1Output, S2Output, chunk_token, split_and_clean, traj_to_actions


class InternVLAN1NetUnified(InternVLAN1Net):
    """InternVLAN1Net + UnifiedImageProvider. Owns all unified-provider logic for the H1 path."""

    def __init__(self, config):
        super().__init__(config)
        settings = config.model_cfg['model']
        self.provider = create_unified_provider(settings, device=str(self.model_config.device))
        self._debug_topdown_rgb = None
        self._debug_global_rotation = None

    def reset(self):
        super().reset()
        self.provider.reset()

    # ------------------------------------------------------------- debug hooks

    def set_debug_topdown(self, topdown_rgb, global_rotation) -> None:
        """Called once per agent.step() (see internvla_n1_agent_unified.py) with the
        raw obs fields needed for the ground-truth topdown debug jpg."""
        self._debug_topdown_rgb = topdown_rgb
        self._debug_global_rotation = global_rotation

    def _save_isaac_topdown_debug(self, prefix: str, step: int) -> None:
        if self._debug_topdown_rgb is None or not self.provider.debug_dir:
            return
        try:
            from omni.isaac.core.utils.rotations import quat_to_euler_angles

            _, _, yaw = quat_to_euler_angles(np.array(self._debug_global_rotation))
            save_isaac_topdown_debug(
                self.provider.debug_dir, prefix, step, self._debug_topdown_rgb, float(yaw),
                # Crop+resize to the same physical area/pixel size as the BEV jpg
                # (bev_range/bev_size) so ..._b00_4_gt_topdown_world.jpg is directly
                # scale-comparable to ..._b00_3_bev_gt_cur.jpg — topdown_camera_500's raw
                # frame otherwise covers ~2x the area (see isaac_topdown_debug.py's
                # _TOPDOWN_CAMERA_WORLD_WIDTH_M).
                crop_range_m=self.provider.processor.bev_range,
                resize_to=self.provider.processor.bev_size,
            )
        except Exception:
            import traceback

            traceback.print_exc()

    # ------------------------------------------------------------------- S2

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        # 'bev' must be computed from the CURRENT frame on every S2 turn, not just
        # look-down turns — matches training's hardcoded is_lookdown=True
        # (internvla_n1_lerobot_dataset.py:1139) and habitat's "S2: always" branch
        # (habitat_vln_evaluator_unified.py:426-431). h1's own look_down flag (the
        # discrete look-down ACTION turn) is a different concept and must not gate this.
        extra_images = self.provider.get_s2_extra(rgb, depth, is_lookdown=True)
        if not extra_images:
            # fpv / combine='none': parent behaviour, unchanged.
            return super().s2_step(rgb, depth, pose, instruction, intrinsic, look_down)

        if self.provider.s2_view in ('bev', 'bev_ld') and self.provider.debug_dir:
            self._save_isaac_topdown_debug('s2_step', self.provider._s2_debug_step - 1)

        # 'replace' already raised inside get_s2_extra (no training-side implementation
        # — see unified_image_provider.py); 'none' never returns a non-empty list. So
        # the only combine mode reachable here is 'concat'.
        s2_combine = self.provider.s2_combine
        assert s2_combine == 'concat', (
            f"unreachable s2_combine={s2_combine!r} "
            "('replace' already raised inside get_s2_extra; 'none' never returns non-empty)"
        )
        image = Image.fromarray(rgb).convert('RGB')

        if not look_down:
            # --- regular turn: same image/history bookkeeping as parent's non-look-down
            # branch (internvla_n1_policy.py:114-137), plus provider image appended ---
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            self.conversation_history = []
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (self.DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'
            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images + list(extra_images)
            input_img_id = 0
            self.episode_idx += 1
        else:
            # --- look-down turn: continue previous conversation, same as parent's
            # look-down branch (internvla_n1_policy.py:138-146), plus provider image ---
            turn_images = [image] + list(extra_images)  # look-down FPV + extra image(s)
            self.input_images.extend(turn_images)
            input_img_id = -len(turn_images)  # this turn's images sit at the tail
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN
        prompt += f" This is {self.provider.s2_image_label}: "
        prompt += ('\n' + self.DEFAULT_IMAGE_TOKEN) * len(extra_images)
        sources[0]["value"] += f" {prompt}."

        # --- remainder mirrors the parent tail (prompt build -> generate) ---
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"============ output {self.episode_idx}  {self.llm_output}")
        output = S2Output()

        if bool(re.search(r'\d', self.llm_output)):  # Output pixel goal
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            output.output_pixel = np.array(pixel_goal)

            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)
            output.output_latent = traj_latents

            self.provider.set_goal(rgb, depth)
        else:  # Output action
            action_seq = self.parse_actions(self.llm_output)
            output.output_action = action_seq

        return output

    # ------------------------------------------------------------------- S1

    def s1_step_latent(self, rgb, depth, latent):
        s1v = self.provider.get_s1_input(rgb, depth)
        images_dp = s1v.images if s1v.images is not None else rgb
        depths_dp = s1v.depths if s1v.depths is not None else depth
        # extra_images (s1_combine_mode='concat'): keep only the "cur" slot (index -1),
        # same acceptance already implicit in 'replace' mode substituting both slots —
        # see habitat_vln_evaluator_unified.py's _apply_s1_provider.
        bev_images = s1v.extra_images[:, -1:] if s1v.extra_images is not None else None

        if (
            self.provider.debug_dir
            and self.provider.s1_view == 'bev'
            and self.provider.s1_combine in ('replace', 'concat')
        ):
            self._save_isaac_topdown_debug('step', self.provider._debug_step - 1)

        with torch.no_grad():
            dp_actions = self.model.generate_traj(
                traj_latents=latent, images_dp=images_dp, depths_dp=depths_dp, bev_images=bev_images
            )

        # --- remainder mirrors InternVLAN1Net.s1_step_latent ---
        if self.continuous_traj:
            action_list = traj_to_actions(dp_actions)
        else:
            random_choice = np.random.choice(dp_actions.shape[0])
            action_list = chunk_token(dp_actions[random_choice])

        action_list = [x for x in action_list if x != 0]

        output = S1Output(idx=action_list[:4])
        return output
