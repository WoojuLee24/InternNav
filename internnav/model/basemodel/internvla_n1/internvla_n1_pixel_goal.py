"""pixel_goal conditioning: BEV image processing (optional) + pixel-coord S1 conditioning.

InternVLAN1BEVProviderPixelGoalForCausalLM inherits BEVProvider.forward() which applies
BEV when bev_s1_mode != 'fpv', then calls super().forward(). The _build_cond_token()
hook in InternVLAN1ForCausalLM.forward() is overridden here to replace the VLM latent
with a projected pixel coord. bev_s1_mode='fpv' (default) skips BEV entirely.

Training:  internvla_n1_pixel_goal_trainer.py monkey-patches this in.
Inference: InternVLAN1PixelGoalAgent uses InternVLAN1PixelGoalNet.
"""

import copy
import re
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    TRAJ_TOKEN_INDEX,
    InternVLAN1ModelConfig,
)
from internnav.model.basemodel.internvla_n1.internvla_n1_bev_provider import (
    InternVLAN1BEVProviderForCausalLM,
)
from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net
from internnav.model.utils.vln_utils import S1Output, S2Output, chunk_token, split_and_clean, traj_to_actions

_LATENT_EMB_SIZE = 768


class InternVLAN1BEVProviderPixelGoalForCausalLM(InternVLAN1BEVProviderForCausalLM):
    """BEV image processing (optional) + pixel-coord S1 conditioning.

    Call chain at training:
        BEVProvider.forward() → applies BEV to traj_images → super().forward()
        InternVLAN1ForCausalLM.forward() → self._build_cond_token() [overridden here]
    """

    def __init__(self, config):
        super().__init__(config)
        self.get_model().pixel_cond_projector = nn.Sequential(
            nn.Linear(2, _LATENT_EMB_SIZE),
            nn.GELU(approximate="tanh"),
            nn.Linear(_LATENT_EMB_SIZE, _LATENT_EMB_SIZE),
        )

    def setup_pixel_goal_decoder(self, tokenizer, resize_w: int, resize_h: int):
        self._pixel_goal_tokenizer = tokenizer
        self._pixel_goal_resize_w = resize_w
        self._pixel_goal_resize_h = resize_h

    def _extract_pixel_coords(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """(B, 2) normalized [x, y] pixel coords decoded from logits at answer-token positions."""
        IGNORE_INDEX = -100
        result = []
        for b in range(labels.shape[0]):
            lab = labels[b]
            mask = (lab != IGNORE_INDEX) & (lab != TRAJ_TOKEN_INDEX)
            positions = mask.nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                result.append(torch.zeros(2, dtype=logits.dtype, device=logits.device))
                continue
            with torch.no_grad():
                pred_ids = logits[b, positions - 1].detach().argmax(dim=-1).tolist()
            text = self._pixel_goal_tokenizer.decode(pred_ids, skip_special_tokens=True)
            nums = re.findall(r'\d+', text)
            if len(nums) < 2:
                result.append(torch.zeros(2, dtype=logits.dtype, device=logits.device))
                continue
            col = min(int(nums[0]), self._pixel_goal_resize_w - 1)
            row = min(int(nums[1]), self._pixel_goal_resize_h - 1)
            result.append(torch.tensor(
                [col / self._pixel_goal_resize_w, row / self._pixel_goal_resize_h],
                dtype=logits.dtype, device=logits.device,
            ))
        return torch.stack(result, dim=0)

    def _build_cond_token(self, memory_tokens, traj_hidden_states, traj_images, logits=None, labels=None):
        if logits is None or labels is None or not hasattr(self, '_pixel_goal_tokenizer'):
            return self.get_model().cond_projector(traj_hidden_states)
        pixel_coord_norm = self._extract_pixel_coords(logits, labels)
        T = traj_images.size(1)
        repeated = pixel_coord_norm.unsqueeze(1).repeat(1, T, 1).flatten(0, 1)
        return self.get_model().pixel_cond_projector(repeated.to(memory_tokens.dtype)).unsqueeze(1)

    def generate_traj_pixel(
        self,
        pixel_coord_norm,
        images_dp,
        depths_dp=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
    ):
        """generate_traj() nextdit_async path with pixel_coord_norm instead of VLM latent.

        Args:
            pixel_coord_norm: (B, 2) normalized [x, y] in [0, 1]
            images_dp: (B, 2, 224, 224, 3) [goal_frame, current_frame]
        """
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.utils.torch_utils import randn_tensor

        assert 'nextdit' in self.get_system1_type() and 'async' in self.get_system1_type(), \
            "generate_traj_pixel only supports nextdit_async system1"

        scheduler = FlowMatchEulerDiscreteScheduler()
        device = pixel_coord_norm.device
        dtype = pixel_coord_norm.dtype
        batch_size = pixel_coord_norm.shape[0]

        pixel_cond = self.get_model().pixel_cond_projector(pixel_coord_norm).unsqueeze(1)

        with torch.no_grad():
            images_dp_norm = (images_dp.permute(0, 1, 4, 2, 3) - self._resnet_mean) / self._resnet_std
            self.get_model().rgb_model.to(dtype)
            images_dp_feat = (
                self.get_model()
                .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                .unflatten(dim=0, sizes=(batch_size, -1))
            )
            memory_feat = self.get_model().memory_encoder(images_dp_feat.flatten(1, 2))
            memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
            memory_tokens = self.get_model().rgb_resampler(memory_feat)

        hidden_states = torch.cat([memory_tokens, pixel_cond], dim=1)
        hidden_states_input = torch.cat([torch.zeros_like(hidden_states), hidden_states], 0)
        hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

        latents = randn_tensor(
            shape=(batch_size * num_sample_trajs, predict_step_nums, 3),
            generator=None, device=device, dtype=dtype,
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        for t in scheduler.timesteps:
            latent_features = self.get_model().action_encoder(latents)
            pos_ids = torch.arange(latent_features.shape[1]).reshape(1, -1).repeat(batch_size, 1).to(device)
            latent_features += self.get_model().pos_encoding(pos_ids)
            latent_model_input = latent_features.repeat(2, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.get_model().traj_dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device, torch.long),
                z_latents=hidden_states_input,
            )
            noise_pred = self.get_model().action_decoder(noise_pred)
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        return latents


class InternVLAN1PixelGoalNet(InternVLAN1Net):
    """Inference policy: S2 outputs pixel coords, S1 uses generate_traj_pixel.

    Inherits all of InternVLAN1Net except __init__ (loads the pixel_goal model class)
    and s2_step (skips the expensive generate_latents call).
    """

    def __init__(self, config: Union[InternVLAN1ModelConfig, ModelCfg]):
        # Skip InternVLAN1Net.__init__ to load the correct model class.
        PreTrainedModel.__init__(self, config)
        self.model_config = ModelCfg(**config.model_cfg['model'])

        self.model = InternVLAN1BEVProviderPixelGoalForCausalLM.from_pretrained(
            self.model_config.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.model_config.device},
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(self.model_config.model_path)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = 'left'

        self.init_prompts()

        self.num_frames = self.model_config.num_frames
        self.num_history = self.model_config.num_history
        self.num_future_steps = self.model_config.num_future_steps
        self.continuous_traj = self.model_config.continuous_traj
        self.resize_w = self.model_config.resize_w
        self.resize_h = self.model_config.resize_h

        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """Same as InternVLAN1Net.s2_step but skips generate_latents: output_pixel only."""
        # 1. Preprocess input
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)

        # 2. Prepare input
        if not look_down:
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
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        parts = split_and_clean(copy.deepcopy(sources[0]["value"]))

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

        # 3. Model inference
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
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"============ output {self.episode_idx}  {self.llm_output}")
        output = S2Output()

        # 4. Post-process: pixel goal only — skip generate_latents
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            output.output_pixel = np.array([int(coord[1]), int(coord[0])])
        else:
            output.output_action = self.parse_actions(self.llm_output)

        return output

    def s1_step_pixel(self, rgb, depth, pixel_coord_s2, debug_dir=None, step=None):
        """S1 inference using pixel coords from S2 as goal conditioning.

        Args:
            rgb: (1, 2, 224, 224, 3) [goal_frame, current_frame]
            depth: (1, 2, 224, 224, 1)
            pixel_coord_s2: np.array [col, row] in resize_w x resize_h space
            debug_dir: if set, saves visualization
            step: episode step for debug filename
        """
        from internnav.model.utils.pixel_goal_utils import fpv_pixel_to_normalized, visualize_pixel_goal

        pixel_norm = fpv_pixel_to_normalized(pixel_coord_s2, (self.resize_w, self.resize_h))

        if debug_dir is not None:
            cur_frame = (rgb[0, 1].cpu().numpy() * 255).astype(np.uint8)
            scale_x, scale_y = 224.0 / self.resize_w, 224.0 / self.resize_h
            pixel_224 = np.array([pixel_coord_s2[0] * scale_x, pixel_coord_s2[1] * scale_y])
            visualize_pixel_goal(cur_frame, pixel_224, pixel_norm, debug_dir, step=step, mode='fpv')

        pixel_tensor = torch.tensor(pixel_norm, dtype=torch.bfloat16).unsqueeze(0).to(self.model_config.device)

        with torch.no_grad():
            dp_actions = self.model.generate_traj_pixel(
                pixel_coord_norm=pixel_tensor,
                images_dp=rgb,
                depths_dp=depth,
                predict_step_nums=getattr(self.model_config, 'predict_step_nums', 32),
            )

        if self.continuous_traj:
            action_list = traj_to_actions(dp_actions)
        else:
            action_list = chunk_token(dp_actions[np.random.choice(dp_actions.shape[0])])

        action_list = [x for x in action_list if x != 0]
        return S1Output(idx=action_list[:4])
