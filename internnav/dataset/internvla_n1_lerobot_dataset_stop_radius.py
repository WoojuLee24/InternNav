"""NavPixelGoalDataset + 반경 STOP 라벨 (stop_radius_m).

기본 로더(`internvla_n1_lerobot_dataset.NavPixelGoalDataset`)는 STOP 샘플을 에피소드마다 마지막 프레임
(actions_len-1, actions_len) 하나만 만들고 stop_weight 번 복사한다. 끝 직전 프레임은 건너뛰고(start == actions_len-1),
goal_len < 3 인 픽셀 샘플도 버리므로 마지막 ~0.5 m 구간에는 라벨이 없다 (프레임 간격 ≈ 0.25 m).
→ 모델이 "도착했는데 멈추지 않는" 쪽으로 치우친다 (노은역 hunav 열린 루프: 도착 프레임 STOP 14/37).

stop_radius_m > 0 이면, 아래 두 조건을 모두 만족하는 프레임 f 를 sample_step 과 상관없이 전부 STOP 샘플로 넣는다.
  (a) 마지막 프레임까지의 xy 거리 (poses_<h>cm_<p>deg 의 [:2, 3]) <= stop_radius_m
  (b) f 부터 끝까지 남은 action (부모와 같은 시프트 `actions = item['actions'][1:] + [0]` 의 actions[f:L-1]) 이 전진(1)뿐
      — 아직 돌아야 하는 프레임에 STOP 을 붙이지 않는다.
  - 튜플 형식은 기존 stop 과 같다: (..., (f, f+1), 0, None), stop_weight 번 복사.
  - 같은 프레임에서 시작하는 기존 pixel/turn 샘플은 뺀다 (같은 사진에 정답 둘 금지).
  - 마지막 프레임 STOP 은 기존 그대로 둔다 (f < L-1 만 추가).
stop_radius_m == 0 (기본) 이면 아무것도 바꾸지 않는다 → 부모와 샘플 목록이 같다.

부모가 샘플 목록을 다 만든 뒤 `list_data_dict` 와 `scene_keys` 를 같이(1:1) 고친다 (WAIT 데이터셋과 같은 방식).
반경 STOP 샘플은 목록 끝에 붙는다 (학습 sampler 가 섞고, val split 은 scene 단위라 순서와 무관).

클래스 (trainer 가 이름 표로 고른다, internvla_n1_stop_radius_trainer.py):
  STOP_RADIUS_DATASETS["none"] = NavPixelGoalStopRadiusDataset      (NavPixelGoalDataset 상속)
  STOP_RADIUS_DATASETS["wait"] = NavPixelGoalWaitStopRadiusDataset  (NavPixelGoalWaitDataset 상속, wait_mode 와 함께)
"""

import os

import numpy as np

import internnav.dataset.internvla_n1_lerobot_dataset as _d
from internnav.dataset.internvla_n1_lerobot_dataset import (
    NavPixelGoalDataset,
    data_list,
    get_annotations_from_lerobot_data,
    rank0_print,
)
from internnav.dataset.internvla_n1_wait_lerobot_dataset import NavPixelGoalWaitDataset

FORWARD = 1
STOP = 0
LENGTH_BINS = (("0-10", 0, 10), ("10-30", 10, 30), ("30-60", 30, 60), ("60+", 60, float("inf")))  # 경로 길이(m) 구간 — 통계용


def length_bin(length_m: float) -> str:
    for name, lo, hi in LENGTH_BINS:
        if lo <= length_m < hi:
            return name
    assert False, f"unreachable length_m={length_m!r}"


def radius_stop_frames(actions: list, poses: list, stop_radius_m: float) -> list:
    """(a) 마지막 프레임까지 xy 거리 <= r, (b) actions[f:L-1] 이 전부 전진 인 프레임 f (f < L-1). actions 는 시프트된 것."""
    L = len(actions)
    if stop_radius_m <= 0 or L < 2:
        return []
    xy = np.asarray(poses, dtype=np.float64)[:, :2, 3]
    assert len(xy) == L, (len(xy), L)
    d = np.linalg.norm(xy - xy[L - 1], axis=1)
    out = []
    for f in range(L - 2, -1, -1):        # 끝에서부터: 전진이 아닌 action 을 만나면 그 앞은 (b) 불가
        if actions[f] != FORWARD:
            break
        if d[f] <= stop_radius_m:
            out.append(f)
    return sorted(out)


def path_length_m(poses: list) -> float:
    xy = np.asarray(poses, dtype=np.float64)[:, :2, 3]
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def episode_radius_info(data_args, stop_radius_m: float) -> dict:
    """{(video, ep_id, instruction): (data_path, height, pitch_1, pitch_2, [반경 STOP 프레임], 경로 길이 m)} — 부모와 같은 목록·경로 규칙."""
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
            poses = item[f"poses_{height}cm_{pitch_2}deg"]
            fs = radius_stop_frames(actions, poses, stop_radius_m)
            out[(item["video"], item["id"], item["instructions"])] = (data_path, height, pitch_1, pitch_2, fs, path_length_m(poses))
    return out


def apply_stop_radius(ds, data_args) -> dict:
    """ds.list_data_dict / ds.scene_keys 를 고친다. stop_radius_m == 0 이면 아무것도 안 한다."""
    r = float(getattr(data_args, "stop_radius_m", 0.0))
    assert r >= 0, f"stop_radius_m={r}"
    if r == 0:
        return {"stop_radius_m": 0.0, "radius_frames": 0, "replaced": 0, "radius_stop_samples": 0, "total": len(ds.list_data_dict)}
    assert not ds.pixel_goal_only, "stop_radius_m 은 pixel_goal_only=False 에서만 (STOP 샘플이 없는 설정)"
    info = episode_radius_info(data_args, r)
    radius_set = {(k[0], k[1], k[2], f) for k, v in info.items() for f in v[4]}

    new_items, new_keys, n_replaced = [], [], 0
    for item, key in zip(ds.list_data_dict, ds.scene_keys):
        ep_id, data_path, video, h, p1, p2, instr, (s, e), action, pose = item
        is_stop = pose is None and action == STOP
        is_wait = pose is None and isinstance(action, list) and action and action[0] == 4  # WAIT 샘플은 그대로
        if not is_stop and not is_wait and (video, ep_id, instr, s) in radius_set:   # 반경 STOP 프레임의 pixel/turn
            n_replaced += 1
            continue
        new_items.append(item)
        new_keys.append(key)

    n_stop = 0
    for (video, ep_id, instr), (data_path, h, p1, p2, fs, _) in sorted(info.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        scene = video[len(data_path):].lstrip("/").split("/")[0]
        scene_key = f"{data_path.rstrip('/').split('/')[-1]}/{scene}"
        for f in fs:
            for _ in range(ds.stop_weight):
                new_items.append((ep_id, data_path, video, h, p1, p2, instr, (f, f + 1), STOP, None))
                new_keys.append(scene_key)
                n_stop += 1
    assert len(new_items) == len(new_keys)
    ds.list_data_dict, ds.scene_keys = new_items, new_keys
    return {"stop_radius_m": r, "radius_frames": len(radius_set), "replaced": n_replaced,
            "radius_stop_samples": n_stop, "total": len(new_items)}


class NavPixelGoalStopRadiusDataset(NavPixelGoalDataset):
    def __init__(self, tokenizer, data_args):
        # 부모 __init__ 은 `super(NavPixelGoalDataset, self)` 로 모듈 전역 이름을 찾는다. trainer 가 그 이름을 이 클래스로
        # 바꿔 끼우므로, 부모 초기화 동안만 원래 클래스로 되돌린다 (NavPixelGoalWaitDataset 과 같은 이유).
        cur = _d.NavPixelGoalDataset
        _d.NavPixelGoalDataset = NavPixelGoalDataset
        try:
            super().__init__(tokenizer, data_args)
        finally:
            _d.NavPixelGoalDataset = cur
        self.stop_radius_stats = apply_stop_radius(self, data_args)
        rank0_print(f"[stop_radius] {self.stop_radius_stats}")


class NavPixelGoalWaitStopRadiusDataset(NavPixelGoalWaitDataset):
    def __init__(self, tokenizer, data_args):
        super().__init__(tokenizer, data_args)   # WAIT 처리까지 끝난 목록 (전역 이름 되돌리기는 부모가 한다)
        self.stop_radius_stats = apply_stop_radius(self, data_args)
        rank0_print(f"[stop_radius] {self.stop_radius_stats}")


STOP_RADIUS_DATASETS = {"none": NavPixelGoalStopRadiusDataset, "wait": NavPixelGoalWaitStopRadiusDataset}
