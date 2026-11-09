"""S2 단독 (Qwen2.5-VL System2) 에이전트 — VLN-PE(Isaac) 평가용.

habitat `HabitatVLNEvaluator._run_eval_system2` 의 한 에피소드 루프를 env step 하나씩 나눠 그대로 옮겼다
(같은 프롬프트 · conjunction · 지난 프레임 linspace(0, step_id-1, num_history) · '↓' 대화 이어가기 ·
pixel goal 추종 최대 MAX_STEPS 뒤 재질의 · 해석 불가 출력 = STOP). 다른 점:
  - pixel goal -> 3D: habitat 은 yaw·pitch 로 카메라 자세를 만든다. 여기서는 Isaac 카메라 prim 의 world 자세
    (obs camera_position / camera_orientation, USD 카메라 = -Z 전방 · +Y 위) 를 그대로 쓴다.
  - 추종: habitat 은 navmesh ShortestPathFollower(goal_radius 0.25). Isaac 에는 navmesh 가 없어 goal 방향으로
    15 deg 회전 / 0.25 m 전진 하는 직선 추종 (방향 오차 > rotation/2 이면 회전, xy 거리 < goal_radius 이면 도착).
    막히면 flash 컨트롤러가 전진을 무시하고, MAX_STEPS 뒤 재질의된다.
  - '↓': habitat 은 카메라를 내린다. 60cm_15deg rig 는 학습 데이터에서 현재 = look-down 같은 그림이라 제자리
    (stand_still) 한 step 뒤 그 그림을 look-down 이미지로 쓴다 (`lookdown_same_view`).
  - WAIT: 출력에 'WAIT' 가 있으면 제자리 한 step (stand_still). habitat 에는 없다 (action 4 = LOOKUP 이라 쓰지 않음).

반환: evaluator 규약 [{'action': [a], 'ideal_flag': True}] — a: 0 STOP · 1 전진 · 2 좌 · 3 우 · -1 제자리.
"""
import itertools
import json
import math
import os
import random
import re
import time
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg

DEFAULT_IMAGE_TOKEN = '<image>'
MAX_STEPS = 8  # habitat_vln_evaluator.MAX_STEPS
STOP, FORWARD, LEFT, RIGHT, LOOKDOWN, STAND = 0, 1, 2, 3, 5, -1
WAIT = 'WAIT'
PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on "
    "track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully "
    "completed the task."
)
CONJUNCTIONS = [
    'you can see ',
    'in front of you is ',
    'there is ',
    'you can spot ',
    'you are toward the ',
    'ahead of you is ',
    'in your sight is ',
]
ACTIONS2IDX = OrderedDict({'STOP': [STOP], "↑": [FORWARD], "←": [LEFT], "→": [RIGHT], "↓": [LOOKDOWN]})
DEFAULTS = {
    'device': 'cuda:0',
    'width': 640,
    'height': 480,
    'hfov': 68.7,  # 60cm_15deg rig (gs_vlnpe/apply_real/camera_profiles.py)
    'resize_w': 384,
    'resize_h': 384,
    'num_history': 8,
    'max_new_tokens': 128,
    'max_depth': 10.0,  # evaluator norm_depth 범위
    'goal_radius': 0.25,  # habitat ShortestPathFollower goal_radius
    'rotation_deg': 15.0,
    'lookdown_same_view': True,
    'wait_token': WAIT,
    'seed': 0,
    'log_dir': '',  # 비면 기록 안 함. 있으면 에피소드마다 steps jsonl + (save_images 면) 입력 이미지
    'save_images': False,
    # pixel goal 추종: 'straight' = 아래 _follow 직선 추종 | 'navmesh' = habitat 처럼 navmesh 최단 경로
    # (scripts/eval/noeun_navmesh_server.py 가 navmesh_url 에 떠 있어야 한다. base_height = base z - 바닥, H1 1.05 · RBQ10 0.55)
    'follower': 'straight',
    'navmesh_url': 'http://127.0.0.1:8040',
    'navmesh_base_height': 1.05,
    # pixel goal 하나를 따라가는 최대 step (넘으면 재질의). habitat 8. 줄이면 S2 를 더 자주 묻는다
    'max_follow_steps': MAX_STEPS,
    # STOP 이 이 횟수만큼 연달아 나와야 멈춘다 (1 = habitat 과 같음). 아니면 같은 관측으로 대화를 새로 열어 다시 묻는다
    'stop_confirm': 1,
}


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_wxyz_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def parse_pixel_goal(llm_outputs, height, width):
    """habitat_extensions/vln/utils.parse_pixel_goal 과 같다 ("x y" -> [v, u], 무효면 None). habitat 의존성 없이 쓰려고 옮김."""
    coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
    if len(coord) < 2:
        return None
    v, u = coord[1], coord[0]
    if v >= height or u >= width:
        return None
    return [v, u]


@Agent.register('internvla_n1_s2')
class InternVLAN1S2Agent(Agent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        ms = dict(config.model_settings)
        self.s = {k: ms.get(k, v) if ms.get(k, v) is not None else v for k, v in DEFAULTS.items()}
        self.model_path = ms['model_path']
        self.device = torch.device(self.s['device'])
        self._load_model()
        w, h = self.s['width'], self.s['height']
        fx = (w / 2.0) / math.tan(math.radians(self.s['hfov'] / 2.0))
        self.K = np.array([[fx, 0.0, (w - 1.0) / 2.0], [0.0, fx, (h - 1.0) / 2.0], [0.0, 0.0, 1.0]])
        self.episode_count = -1
        self._log = None
        if self.s['follower'] == 'straight':
            self._navmesh = None
        elif self.s['follower'] == 'navmesh':
            from internnav.agent.navmesh_follower import NavmeshFollower

            self._navmesh = NavmeshFollower(self.s['navmesh_url'], self.s['navmesh_base_height'])
        else:
            assert False, f"unreachable follower={self.s['follower']!r}"
        self.reset()

    # ---------------------------------------------------------------- model
    def _load_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            device_map={'': self.device},
        )
        self.model.eval()

    def _generate(self, messages, images):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.s['max_new_tokens'],
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences
        return self.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

    # ---------------------------------------------------------------- episode state
    def reset(self, reset_index=None):
        self.rng = random.Random(self.s['seed'])
        self.rgb_list = []
        self.action_seq = []
        self.input_images = []
        self.messages = []
        self.llm_outputs = ''
        self.goal = None
        self.pixel_goal = None
        self.forward_action = 0
        self.step_id = 0
        self.prev_action = None
        self.look_down_image = None
        self.n_wait = 0
        self.n_query = 0
        self.invalid_outputs = 0
        self.n_stop_votes = 0  # 이번 step 에서 연달아 나온 STOP 수
        self.n_stop_rejected = 0  # stop_confirm 으로 다시 물은 횟수 (ep 누적)
        self.episode_count += 1
        self._open_log()

    def _open_log(self):
        if self._log is not None:
            self._log.close()
            self._log = None
        if not self.s['log_dir']:
            return
        self.ep_dir = os.path.join(self.s['log_dir'], f'ep{self.episode_count:04d}')
        os.makedirs(self.ep_dir, exist_ok=True)
        self._log = open(os.path.join(self.ep_dir, 'steps.jsonl'), 'a')

    def _record(self, **kw):
        if self._log is None:
            return
        self._log.write(json.dumps(kw, ensure_ascii=False) + '\n')
        self._log.flush()

    # ---------------------------------------------------------------- geometry
    def _pixel_to_world(self, pixel, depth_m, cam_pos, cam_quat):
        v, u = pixel
        d = float(depth_m[v, u])
        p_cam = np.array([(u - self.K[0, 2]) * d / self.K[0, 0], -(v - self.K[1, 2]) * d / self.K[1, 1], -d])
        return quat_wxyz_to_matrix(cam_quat) @ p_cam + np.asarray(cam_pos, dtype=float), d

    def _follow(self, pos, quat):
        """goal 로 가는 다음 행동 (ShortestPathFollower 대신 직선 추종). 도착이면 STOP."""
        if self._navmesh is not None:
            return self._navmesh.next_action(pos, quat_wxyz_to_yaw(quat), self.goal)
        dx, dy = self.goal[0] - pos[0], self.goal[1] - pos[1]
        if math.hypot(dx, dy) < self.s['goal_radius']:
            return STOP
        err = math.atan2(dy, dx) - quat_wxyz_to_yaw(quat)
        err = (err + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > math.radians(self.s['rotation_deg']) / 2:
            return LEFT if err > 0 else RIGHT
        return FORWARD

    @staticmethod
    def _parse_actions(output):
        regex = re.compile('|'.join(re.escape(a) for a in ACTIONS2IDX))
        return list(itertools.chain.from_iterable(ACTIONS2IDX[m] for m in regex.findall(output)))

    # ---------------------------------------------------------------- S2 query (habitat 과 같은 대화)
    def _query(self, instruction):
        if self.prev_action == LOOKDOWN:
            self.input_images += [self.look_down_image]
            self.messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_outputs}]})
            input_img_id = -1
            value = ''
        else:
            value = PROMPT.replace('<instruction>.', instruction[:-1])  # habitat 과 같은 치환 (마침표 둘 다 빠짐)
            cur_images = self.rgb_list[-1:]
            if self.step_id == 0:
                history_id = []
            else:
                history_id = np.unique(
                    np.linspace(0, self.step_id - 1, self.s['num_history'], dtype=np.int32)
                ).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                value += f' These are your historical observations: {placeholder}.'
            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
        value += f' {self.rng.choice(CONJUNCTIONS) + DEFAULT_IMAGE_TOKEN}.'
        from internnav.model.utils.vln_utils import split_and_clean

        content = []
        for part in split_and_clean(value):
            if part == DEFAULT_IMAGE_TOKEN:
                content.append({'type': 'image', 'image': self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({'type': 'text', 'text': part})
        self.messages.append({'role': 'user', 'content': content})
        t0 = time.time()
        self.llm_outputs = self._generate(self.messages, self.input_images)
        self.n_query += 1
        return time.time() - t0

    # ---------------------------------------------------------------- one env step
    def step(self, obs):
        ob = obs[0]
        rgb, depth = ob['rgb'], ob['depth'][..., 0] * self.s['max_depth']
        pos, quat = np.asarray(ob['globalgps'], dtype=float), np.asarray(ob['globalrotation'], dtype=float)
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGB')
        instruction = ob['instruction'] if 'instruction' in ob else ''

        action, latency, event = None, 0.0, ''
        while action is None:
            if self.prev_action == LOOKDOWN:
                self.look_down_image = image
            else:
                self.rgb_list.append(image.resize((self.s['resize_w'], self.s['resize_h'])))
                if self.s['save_images'] and self._log is not None:
                    image.save(os.path.join(self.ep_dir, f'rgb_{len(self.rgb_list) - 1:04d}.jpg'))

            if len(self.action_seq) == 0 and self.goal is None:
                latency += self._query(instruction)
                out = self.llm_outputs
                has_digit = bool(re.search(r'\d', out))
                parsed = parse_pixel_goal(out, *depth.shape[:2]) if has_digit else None
                self.invalid_outputs += int(has_digit and parsed is None)
                if parsed is not None:
                    self.forward_action = 0
                    self.pixel_goal = parsed
                    self.goal, d = self._pixel_to_world(parsed, depth, ob['camera_position'], ob['camera_orientation'])
                    event = f'pixel_goal {parsed} depth {d:.2f} -> world {np.round(self.goal, 2).tolist()}'
                    a = self._follow(pos, quat)
                    if a == STOP:  # habitat: 이미 도착이면 왼쪽으로 돌고 재질의 (교착 방지)
                        self.goal, self.messages = None, []
                        action = LEFT
                        break
                elif self.s['wait_token'] in out:
                    self.n_wait += 1
                    event = 'wait'
                    action = STAND
                    break
                else:
                    self.action_seq = self._parse_actions(out)

            if len(self.action_seq) != 0:
                action = self.action_seq.pop(0)
            elif self.goal is not None:
                a = self._follow(pos, quat)
                self.forward_action += 1
                if self.forward_action > self.s['max_follow_steps'] or a == STOP:
                    self.goal, self.messages = None, []
                    self.step_id += 1
                    self.forward_action = 0
                    self.prev_action = a
                    continue  # habitat 과 같이 env step 없이 같은 관측으로 다시
                action = a
            else:
                action = STOP
            if action == STOP and self.n_stop_votes + 1 < self.s['stop_confirm']:
                self.n_stop_votes += 1
                self.n_stop_rejected += 1
                self.action_seq, self.goal, self.messages = [], None, []
                self.step_id += 1
                self.prev_action = STOP
                event = 'stop_vote'
                action = None
        if action != STOP:
            self.n_stop_votes = 0

        self._record(
            step_id=self.step_id,
            output=self.llm_outputs if latency > 0 else None,
            latency_s=round(latency, 3),
            action=int(action),
            event=event,
            pos=np.round(pos, 3).tolist(),
            goal=None if self.goal is None else np.round(self.goal, 3).tolist(),
            n_wait=self.n_wait,
            stop_rejected=self.n_stop_rejected,
            **({} if self._navmesh is None else {'nav': self._navmesh.last}),
        )
        self.prev_action = action
        if action == LOOKDOWN:
            env_action = STAND if self.s['lookdown_same_view'] else LOOKDOWN
        else:
            self.step_id += 1
            self.messages = []
            env_action = action
        return [{'action': [int(env_action)], 'ideal_flag': True}]
