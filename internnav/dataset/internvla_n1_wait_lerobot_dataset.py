"""NavPixelGoalDataset + WAIT 행동 (action 4: 사람 때문에 제자리 대기, 노은역 hunav 데이터).

기본 로더(`internvla_n1_lerobot_dataset.NavPixelGoalDataset`)는 고치지 않는다. 부모가 샘플 목록을 다 만든 뒤
`list_data_dict` 와 `scene_keys` 를 같이(1:1 정렬 유지) 고친다.

대기 프레임 = 부모와 같은 시프트(`actions = item['actions'][1:] + [0]`) 후 `actions[t] == 4` 인 프레임 t.
(노은역 hunav: sidecar `waiting[t]` 와 parquet `action[t+1]` 이 81/81 일치)

wait_mode
  wait    : 대기 프레임마다 정답 텍스트 `WAIT` 인 샘플을 wait_weight 번 넣는다 (STOP 의 stop_weight 와 같은 방식).
  exclude : WAIT 샘플을 넣지 않는다 (같은 데이터에서 WAIT 만 없는 대조군).
  visible : wait 와 같되 `wait_keep` 파일에 있는 대기 프레임(사진에 사람이 보이는 것)만 WAIT, 나머지 대기 프레임은 exclude 처럼 뺀다.
            wait_keep = {"keep": [[scene, episode_index, frame], ...]} (scene = 형제 번호를 뺀 이름, 예 "hunav"; nav3d/noeun/66_wait_keep.py)
  둘 다   : 시작 프레임이 대기 프레임인 pixel goal / turn 샘플은 뺀다 (사람 앞에서 "저기로 가" 를 배우지 않게),
            turn 목록은 4 에서 자른다 (부모 `idx2actions` 에 4 가 없어 KeyError).
wait_mode='none' 이면 이 클래스를 쓰지 않는다 (internvla_n1_wait_trainer.py 가 바꿔 끼우지 않음) -> 기본 경로 그대로.

⚠️ habitat 평가기에서 action 4 는 LOOKUP 이다. 평가 쪽에서 WAIT 를 번호 4 로 넘기지 말고 `WAIT` 텍스트로 따로 처리할 것.
"""

import json
import os
import re

from internnav.dataset.internvla_n1_lerobot_dataset import (
    NavPixelGoalDataset,
    data_list,
    get_annotations_from_lerobot_data,
    rank0_print,
)

WAIT_ACTION = 4
WAIT_TEXT = "WAIT"
WAIT_MODES = ("wait", "exclude", "visible")

_PENDING = {}  # internvla_n1_wait_trainer.py 가 argv 에서 뗀 설정 (HfArgumentParser 를 고치지 않으려고)


def set_pending_wait_settings(settings: dict):
    _PENDING.clear()
    _PENDING.update(settings)


def parse_wait_cli_args(argv: list) -> tuple:
    """--wait_mode <m> --wait_weight <n> 을 떼어 (settings, 나머지 argv) 로."""
    settings, rest, i = {"wait_mode": "wait", "wait_weight": 5, "wait_keep": ""}, [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("--wait_mode", "--wait_weight", "--wait_keep"):
            settings[a[2:]] = argv[i + 1]
            i += 2
            continue
        for k in ("--wait_mode=", "--wait_weight=", "--wait_keep="):
            if a.startswith(k):
                settings[k[2:-1]] = a[len(k):]
                break
        else:
            rest.append(a)
        i += 1
    settings["wait_weight"] = int(settings["wait_weight"])
    assert settings["wait_mode"] in WAIT_MODES, f"wait_mode={settings['wait_mode']!r} (wait | exclude)"
    assert settings["wait_weight"] >= 1, settings["wait_weight"]
    assert (settings["wait_mode"] == "visible") == bool(settings["wait_keep"]), "wait_keep 은 wait_mode=visible 에서만 (그리고 꼭) 준다"
    return settings, rest


class NavPixelGoalWaitDataset(NavPixelGoalDataset):
    def __init__(self, tokenizer, data_args):
        super().__init__(tokenizer, data_args)
        self.wait_mode = _PENDING.get("wait_mode", getattr(data_args, "wait_mode", "wait"))
        self.wait_weight = int(_PENDING.get("wait_weight", getattr(data_args, "wait_weight", 5)))
        assert self.wait_mode in WAIT_MODES, f"wait_mode={self.wait_mode!r}"
        self.wait_keep = None
        if self.wait_mode == "visible":
            keep_path = _PENDING.get("wait_keep", getattr(data_args, "wait_keep", ""))
            self.wait_keep = {tuple(k) for k in json.load(open(keep_path))["keep"]}
            rank0_print(f"[wait] keep {len(self.wait_keep)} frames from {keep_path}")
        self.idx2actions = dict(self.idx2actions)
        self.idx2actions[WAIT_ACTION] = WAIT_TEXT
        waits = self._wait_frames(data_args)
        self.wait_stats = self._apply_wait(waits)
        rank0_print(f"[wait] mode={self.wait_mode} weight={self.wait_weight} {self.wait_stats}")

    def _wait_frames(self, data_args) -> dict:
        """{(video, ep_id, instruction): (data_path, height, pitch_1, pitch_2, [대기 프레임 t, ...])} — 부모와 같은 데이터셋 목록·경로 규칙."""
        out = {}
        for data in data_list(data_args.vln_dataset_use.split(",")):
            height, pitch_1, pitch_2 = data.get("height"), data.get("pitch_1"), data.get("pitch_2")
            data_path = data["data_path"]
            if getattr(data_args, "data_root", None):
                data_path = os.path.join(data_args.data_root, data_path)
            ann = get_annotations_from_lerobot_data(data_path, f"{height}cm_{pitch_2}deg")
            for item in ann["episodes"]:
                actions = item["actions"][1:] + [0]   # 부모 __init__ 과 같은 시프트
                if len(actions) < 4:                  # 부모가 건너뛰는 에피소드
                    continue
                ts = [t for t, a in enumerate(actions) if a == WAIT_ACTION]
                if ts:
                    out[(item["video"], item["id"], item["instructions"])] = (data_path, height, pitch_1, pitch_2, ts)
        return out

    def _apply_wait(self, waits: dict) -> dict:
        wait_set = {(k[0], k[1], k[2], t) for k, v in waits.items() for t in v[4]}
        new_items, new_keys = [], []
        n_drop = n_trim = 0
        for item, key in zip(self.list_data_dict, self.scene_keys):
            ep_id, data_path, video, h, p1, p2, instr, (s, e), action, pose = item
            if (video, ep_id, instr, s) in wait_set and not (pose is None and action == 0):   # 대기 프레임에서 시작하는 pixel/turn 샘플
                n_drop += 1
                continue
            if pose is None and isinstance(action, list) and WAIT_ACTION in action:     # turn 목록은 4 에서 자름
                action = action[:action.index(WAIT_ACTION)]
                n_trim += 1
                if not action:
                    n_drop += 1
                    continue
                item = (ep_id, data_path, video, h, p1, p2, instr, (s, e), action, pose)
            new_items.append(item)
            new_keys.append(key)
        n_wait = n_skip = 0
        if self.wait_mode in ("wait", "visible"):
            for (video, ep_id, instr), (data_path, h, p1, p2, ts) in sorted(waits.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
                scene = video[len(data_path):].lstrip('/').split('/')[0]
                scene_key = f"{data_path.rstrip('/').split('/')[-1]}/{scene}"
                base = re.sub(r"_s\d+$", "", scene)                    # 형제 scene (hunav_s0 ...) -> hunav
                for t in ts:
                    if self.wait_keep is not None and (base, ep_id, t) not in self.wait_keep:
                        n_skip += 1                                    # visible: 사람이 안 보이는 대기 프레임 -> exclude 처럼
                        continue
                    for _ in range(self.wait_weight):
                        new_items.append((ep_id, data_path, video, h, p1, p2, instr, (t, t + 1), [WAIT_ACTION], None))
                        new_keys.append(scene_key)
                        n_wait += 1
        elif self.wait_mode == "exclude":
            pass
        else:
            assert False, f"unreachable wait_mode={self.wait_mode!r}"
        assert len(new_items) == len(new_keys)
        self.list_data_dict, self.scene_keys = new_items, new_keys
        return {"wait_frames": len(wait_set), "dropped": n_drop, "trimmed_turns": n_trim, "wait_samples": n_wait,
                "wait_frames_skipped": n_skip, "total": len(new_items)}
