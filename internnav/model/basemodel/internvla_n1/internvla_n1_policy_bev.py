"""InternVLA-N1 policy with pluggable BEV visual input (Isaac Sim / H1 path).

Subclass of ``InternVLAN1Net`` — internvla_n1_policy.py is NOT modified.
A ``VisualInputProvider`` (created from model_settings) decides what S1/S2 see:

  - S1: ``s1_step_latent`` asks the provider for replacement images_dp/depths_dp.
        bev_s1_mode='fpv'     → original [goal, current] FPV stack (unchanged)
        bev_s1_mode='bev'     → [goal_bev, current_bev] (depths stay FPV — same
                                convention as the rgb_gt training mode)
        bev_s1_mode='fpv_bev' → [goal, current, goal_bev, current_bev] along T
                                (fpv_concat_gt training convention, T doubles)
  - S2: on a look-down turn the provider may return BEV images:
        bev_s2_mode='fpv'     → look-down FPV only (unchanged)
        bev_s2_mode='bev'     → BEV replaces the look-down FPV frame
        bev_s2_mode='fpv_bev' → BEV appended after the look-down FPV frame
        <image> tokens are kept aligned with the image list in every mode.

Selected via config: ``model_name='internvla_n1_bev'`` (see
``internnav/agent/internvla_n1_agent_bev.py``). With
``visual_provider='fpv'`` every override degenerates to the parent behaviour.
"""

import copy
import re

import numpy as np
import torch
from PIL import Image

from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net
from internnav.model.utils.visual_input_provider import create_visual_provider
from internnav.model.utils.vln_utils import S1Output, S2Output, chunk_token, split_and_clean, traj_to_actions


class InternVLAN1NetBEV(InternVLAN1Net):
    """InternVLAN1Net + VisualInputProvider. Owns all BEV logic for the H1 path."""

    def __init__(self, config):
        super().__init__(config)
        settings = config.model_cfg['model']
        self.provider = create_visual_provider(settings, device=str(self.model_config.device))

    def reset(self):
        super().reset()
        self.provider.reset()

    # ------------------------------------------------------------------- S2

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        extra_images = self.provider.get_s2_extra(rgb, depth, is_lookdown=look_down)
        if not (look_down and extra_images):
            # FPV / non-look-down turns: parent behaviour, unchanged.
            return super().s2_step(rgb, depth, pose, instruction, intrinsic, look_down)

        # --- look-down turn with BEV injection ----------------------------
        # Mirrors the look_down branch of InternVLAN1Net.s2_step, with the BEV
        # image(s) either appended after ('fpv_bev') or replacing ('bev') the
        # look-down FPV frame; <image> tokens stay aligned with the image list.
        s2_mode = getattr(self.provider, 's2_mode', 'fpv_bev')
        image = Image.fromarray(rgb).convert('RGB')
        if s2_mode == 'bev':
            turn_images = list(extra_images)                  # BEV only
        else:  # 'fpv_bev'
            turn_images = [image] + list(extra_images)        # look-down FPV + BEV
        self.input_images.extend(turn_images)
        input_img_id = -len(turn_images)          # this turn's images sit at the tail

        assert self.llm_output != "", "Last llm_output should not be empty when look down"
        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
        self.conversation_history.append(
            {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
        )

        # === BEV INJECTION: one <image> token per image of this turn ===
        if s2_mode == 'bev':
            prompt = self.conjunctions[0] + "the bird's-eye view of your surroundings:"
            prompt += ('\n' + self.DEFAULT_IMAGE_TOKEN) * len(extra_images)
        else:  # 'fpv_bev'
            prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN
            prompt += " This is the bird's-eye view of your surroundings: "
            prompt += ('\n' + self.DEFAULT_IMAGE_TOKEN) * len(extra_images)
        sources[0]["value"] += f" {prompt}."

        # --- remainder mirrors the parent tail (prompt build → generate) ---
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

            # === BEV INJECTION: goal-frame snapshot hook (no-op for the
            # stateless BEVImageProvider; kept for stateful providers) ===
            self.provider.set_goal(rgb, depth)
        else:  # Output action
            action_seq = self.parse_actions(self.llm_output)
            output.output_action = action_seq

        return output

    # ------------------------------------------------------------------- S1

    def s1_step_latent(self, rgb, depth, latent):
        # === BEV INJECTION: the provider may replace images_dp / depths_dp ===
        s1v = self.provider.get_s1_input(rgb, depth)
        images_dp = s1v.images if s1v.images is not None else rgb
        depths_dp = s1v.depths if s1v.depths is not None else depth

        with torch.no_grad():
            dp_actions = self.model.generate_traj(
                traj_latents=latent, images_dp=images_dp, depths_dp=depths_dp
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
