"""reference_path -> 이산 action 생성기 (val 처럼 action 이 없는 split 용).

후보 (P1-A 에서 릴리스 train action 과 대조):
- `greedy_wp`  : habitat GreedyGeodesicFollower 로 reference_path 정점을 차례로 따라감 (정점마다 goal_radius).
                 habitat-lab ShortestPathFollower 와 같은 구현 (make_greedy_follower). InternNav evaluator 도 이것을 쓴다.
- `greedy_goal`: 마지막 정점으로 바로 감 (중간 정점 무시).
- `greedy_dense`: 정점 사이를 `dense_m` 간격으로 보간한 점을 차례로 따라감.

모두 sim 위에서 agent.act 로 실제로 걷는다 (allow_sliding 반영, 렌더 없음).
반환: action 리스트, 릴리스 규약 (`[-1, a1, a2, ...]`, STOP 없음).
"""

import numpy as np

from habitat_sim.errors import GreedyFollowerError
from habitat_sim.nav import GreedyGeodesicFollower

ACTION_ID = {'move_forward': 1, 'turn_left': 2, 'turn_right': 3}


def densify(path, step_m: float):
    out = []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return out


def goals_for(method: str, reference_path, dense_m: float = 0.5):
    if method == 'greedy_wp':
        return list(reference_path[1:])
    elif method == 'greedy_goal':
        return [reference_path[-1]]
    elif method == 'greedy_dense':
        return densify(reference_path, dense_m)
    else:
        assert False, f'unreachable method={method!r}'


def generate_actions(rig_sim, episode: dict, method: str, goal_radius: float, max_steps: int = 500,
                     dense_m: float = 0.5) -> dict:
    """rig_sim(RigSim) 위에서 episode 의 start 부터 follower 를 돌린다. 끝나면 agent 는 마지막 위치에 있다."""
    rig_sim.set_start(episode['start_position'], episode['start_rotation'])
    agent = rig_sim.agent
    follower = GreedyGeodesicFollower(rig_sim.sim.pathfinder, agent, goal_radius=goal_radius, stop_key='stop',
                                      forward_key='move_forward', left_key='turn_left', right_key='turn_right')
    actions, errors, truncated = [-1], 0, False
    for g in goals_for(method, episode['reference_path'], dense_m):
        goal = np.asarray(g, dtype=np.float32)
        try:
            while True:
                if len(actions) >= max_steps:
                    truncated = True
                    break
                a = follower.next_action_along(goal)
                if a is None or a == 'stop':
                    break
                agent.act(a)
                actions.append(ACTION_ID[a])
        except GreedyFollowerError:
            errors += 1
        if truncated:
            break
    return {'actions': actions, 'follower_errors': errors, 'truncated': truncated}
