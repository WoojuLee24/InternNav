"""pixel_goal agent: S2 outputs pixel coords, S1 uses generate_traj_pixel.

Identical to InternVLAN1Agent except the S1 dispatch branch in step():
  base: output_latent → s1_step_latent
  here: output_pixel  → s1_step_pixel (no generate_latents required)

Policy must be InternVLAN1PixelGoal_Policy (InternVLAN1PixelGoalNet).
Requires infer_mode != 'sync' (pixel_goal runs async only).
"""

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
from internnav.agent.internvla_n1_agent import InternVLAN1Agent
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model import get_config, get_policy
from internnav.model.utils.misc import set_random_seed
from internnav.model.utils.vln_utils import S1Input, S1Output, S2Input, S2Output


@Agent.register('internvla_n1_pixel_goal')
class InternVLAN1PixelGoalAgent(InternVLAN1Agent):
    """pixel_goal agent: identical to InternVLAN1Agent except S1 dispatch uses s1_step_pixel."""

    def __init__(self, config: AgentCfg):
        super().__init__(config)
        self.debug_dir = getattr(self._model_settings, 'debug_dir', None)

    def step(self, obs):
        mode = self.mode

        obs = obs[0]
        rgb = obs['rgb']
        depth = obs['depth']
        instruction = obs['instruction']
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        if self.should_infer_s2(mode) or self.look_down:
            print(f"======== Infer S2 at step {self.episode_step}========")
            with self.s2_input_lock:
                self.s2_input.idx = self.episode_step
                self.s2_input.rgb = rgb
                self.s2_input.depth = depth
                self.s2_input.pose = pose
                self.s2_input.instruction = instruction
                self.s2_input.should_infer = True
                self.s2_input.look_down = self.look_down
                self.s2_output.is_infering = True
            self.dual_forward_step = 0
        else:
            self.policy.step_no_infer(rgb, depth, pose)

        while self.s2_output.is_infering:
            time.sleep(0.5)
        while not self.s2_output.validate():
            time.sleep(0.2)

        output = {}
        print('===============', self.s2_output.output_action, '=================')
        if self.s2_output.output_action is not None:
            output['action'] = [self.s2_output.output_action[0]]
            with self.s2_output_lock:
                self.s2_output.output_action = self.s2_output.output_action[1:]
                if self.s2_output.output_action == []:
                    self.s2_output.output_action = None
            if output['action'][0] == 5:
                self.look_down = True
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
            # pixel_goal path: use output_pixel + s1_step_pixel (no output_latent needed)
            if self.s2_output.output_pixel is not None:
                self.output_pixel = copy.deepcopy(self.s2_output.output_pixel)
                print(self.output_pixel)

                if self.s2_output.depth_memory.ndim == 2:
                    self.s2_output.depth_memory = self.s2_output.depth_memory[..., np.newaxis]
                depth_ = depth if depth.ndim == 3 else depth[..., np.newaxis]

                processed_pixel_rgb = np.array(Image.fromarray(self.s2_output.rgb_memory).resize((224, 224))) / 255.0
                processed_pixel_depth = np.array(Image.fromarray(self.s2_output.depth_memory[:, :, 0]).resize((224, 224))) * 10.0
                processed_pixel_depth[processed_pixel_depth > self.sys1_depth_threshold] = self.sys1_depth_threshold
                processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255.0
                processed_depth = np.array(Image.fromarray(depth_[:, :, 0]).resize((224, 224))) * 10.0
                processed_depth[processed_depth > self.sys1_depth_threshold] = self.sys1_depth_threshold

                rgbs = torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)]).unsqueeze(0).to(self.device)
                depths = torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)]).unsqueeze(0).unsqueeze(-1).to(self.device)

                self.s1_output = self.policy.s1_step_pixel(
                    rgbs, depths, self.s2_output.output_pixel,
                    debug_dir=self.debug_dir, step=self.episode_step,
                )
            else:
                assert False, f"S2 output should be pixel or action, got neither: {self.s2_output}"

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

                self.s2_output.output_pixel = None
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

        self.episode_step += 1
        if 'action' in output:
            return [{'action': output['action'], 'ideal_flag': True}]
        elif 'velocity' in output:
            return [{'action': output['velocity'], 'ideal_flag': False}]
        else:
            assert False
