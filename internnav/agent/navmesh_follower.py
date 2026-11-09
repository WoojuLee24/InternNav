"""S2 에이전트 pixel goal 추종 — navmesh 경로 서버(`scripts/eval/noeun_navmesh_server.py`)에 다음 동작을 묻는다.

habitat `_run_eval_system2` 의 ShortestPathFollower 자리. 에이전트 model_settings `follower: 'navmesh'` 일 때만 쓴다.
"""
import requests


class NavmeshFollower:
    def __init__(self, url, base_height, timeout_s=10.0):
        self.url = url.rstrip('/') + '/next'
        self.base_height = float(base_height)
        self.timeout_s = timeout_s
        self.last = None  # 마지막 서버 응답 (기록용)

    def next_action(self, pos, yaw, goal):
        r = requests.post(
            self.url,
            json={'pos': [float(v) for v in pos], 'yaw': float(yaw), 'goal': [float(v) for v in goal],
                  'base_height': self.base_height},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        self.last = r.json()
        return int(self.last['action'])
