"""실제 이산 궤적에서 VLM 입력 프레임을 고른다.

b0 원본(gate1 keyframes)은 waypoint 위치에 카메라를 직접 놓고 렌더했다:
  start  : 시작 위치, start_rotation
  turn_i : waypoint i (구간 방향이 turn_deg 이상 바뀌는 곳), **새 구간 방향**을 향함
  goal   : 마지막 waypoint, 도착 방향을 향함
여기서는 InternNav 학습/평가와 같은 조건의 이미지를 쓰기 위해, 걸어간 궤적의 프레임 중 위 의미에 맞는 것을 고른다:
  start  : frame 0
  turn_i : waypoint i 에 가장 가까이 온 프레임 이후, 처음으로 다음 action 이 전진인 프레임 (= 회전을 마치고 새 방향을 향함)
  goal   : 마지막 프레임
waypoint 는 시작 기준 로컬 좌표(릴리스 pose 규약: x 전방, y 좌)로 바꿔 pose 의 위치와 비교한다.
"""

import numpy as np

from route_words import diff, heading


def waypoints_local(ep: dict) -> np.ndarray:
    """reference_path (habitat world, y-up) -> 시작 기준 로컬 (x 전방, y 좌) (N,2)."""
    x, y, z, w = ep['start_rotation']
    yaw = 2.0 * np.arctan2(y, w)                       # y 축 회전
    fwd = np.array([-np.sin(yaw), -np.cos(yaw)])       # (x, z) 성분, R_y(yaw) @ (0,0,-1)
    left = np.array([-np.cos(yaw), np.sin(yaw)])       # R_y(yaw) @ (-1,0,0)
    p0 = np.asarray(ep['start_position'])[[0, 2]]
    P = np.asarray(ep['reference_path'])[:, [0, 2]] - p0
    return np.stack([P @ fwd, P @ left], -1)


def select_keyframes(ep: dict, pose: np.ndarray, action: np.ndarray, turn_deg: float) -> list:
    """[{'frame', 'label', 'waypoint_idx'}] (프레임 순, 중복 제거)."""
    path = ep['reference_path']
    xy = pose[:, :2, 3]
    wl = waypoints_local(ep)
    T = len(xy)
    out = [{'frame': 0, 'label': 'start', 'waypoint_idx': 0}]
    cursor = 0
    prev_h = None
    for i in range(len(path) - 1):
        h = heading(path[i], path[i + 1])
        if prev_h is not None and abs(diff(prev_h, h)) >= turn_deg:
            d = np.linalg.norm(xy[cursor:] - wl[i], axis=1)
            arrive = cursor + int(np.argmin(d))
            f = arrive
            while f + 1 < T and action[f + 1] != 1:   # 회전이 끝날 때까지 (다음 action 이 전진이면 멈춤)
                f += 1
            out.append({'frame': f, 'label': f'turn_{i}', 'waypoint_idx': i})
            cursor = arrive
        prev_h = h
    out.append({'frame': T - 1, 'label': 'goal', 'waypoint_idx': len(path) - 1})
    seen, uniq = set(), []
    for f in sorted(out, key=lambda x: x['frame']):
        if f['frame'] not in seen:
            seen.add(f['frame'])
            uniq.append(f)
    return uniq


def add_approach_frames(sel: list, pose: np.ndarray, ap: dict) -> list:
    """도착 직전 프레임: 궤적을 따라 끝까지 남은 거리가 dist_m 에 가장 가까운 프레임 (이미 고른 프레임이면 건너뜀).
    goal 프레임 바로 앞에 놓인다 (프레임 순서 정렬)."""
    xy = pose[:, :2, 3]
    seg = np.r_[np.linalg.norm(np.diff(xy, axis=0), axis=1), 0.0]
    remain = np.cumsum(seg[::-1])[::-1] - seg  # frame i 에서 끝까지 남은 거리
    have = {f['frame'] for f in sel}
    extra = []
    for d in ap['dist_m']:
        f = int(np.argmin(np.abs(remain - d)))
        if f in have or f >= len(xy) - 1 or remain[f] < 0.5 * d:  # 경로가 너무 짧으면 넣지 않는다
            continue
        have.add(f)
        extra.append({'frame': f, 'label': f'approach_{d:g}m', 'waypoint_idx': None, 'caption': ap['caption']})
    return sorted(sel + extra, key=lambda x: x['frame'])


def subsample(frames: list, k: int) -> list:
    """b0 와 같은 규칙: 처음·끝 유지, 가운데 균등."""
    if len(frames) <= k:
        return frames
    return [frames[round(i * (len(frames) - 1) / (k - 1))] for i in range(k)]
