"""flash(순간이동) + 바닥 높이 따라가기 — 다층 씬(계단)용.

VlnMoveByFlashCollisionController 는 z 를 바꾸지 않아 계단을 못 오르고, 충돌 판정(위에서 내려다본 depth)이
계단 단(riser)을 벽으로 본다. 여기서는 전진(action 1)만 PhysX raycast 로 다시 판정한다:
  1. 바닥: 목표 xy 에서 아래로 raycast. 지금 바닥보다 `max_step_up` 넘게 높거나 `max_step_down` 넘게 낮으면 막힘
     (벽 · 낭떠러지). 통과하면 base z = 목표 바닥 + `base_height`.
  2. 몸: 지금 위치 -> 목표 + `body_radius` 를 몸 좌·중·우 3 줄 x `probe_heights` 높이(두 바닥 중 높은 쪽 기준)로
     수평 raycast. 하나라도 맞으면 막힘. 높이를 높은 바닥 기준으로 재므로 계단 단은 막지 않는다.
지금 바닥도 base z 로 짐작하지 않고 raycast 로 잰다. 매 step(막힌 전진 · 회전 포함) 로봇을 바닥 + `base_height`,
upright(yaw 만)로 다시 놓는다 — 다시 놓지 않으면 막힌 동안 물리가 몸을 움직여 fall 판정이 난다.
로봇 자신의 collider 는 무시한다.
"""
import numpy as np
from internutopia.core.robot.articulation import ArticulationAction
from internutopia.core.robot.controller import BaseController

from .vln_move_by_flash_with_collision_controller import (
    VlnMoveByFlashCollisionController,
)

DEFAULTS = {
    'base_height': 1.05,  # H1 pelvis 높이 (vln_default_config robot_offset 과 같음)
    'max_step_up': 0.45,  # 0.25 m 전진 한 번에 오를 수 있는 높이 (계단 단 0.15~0.2 m x 2 + 여유)
    'max_step_down': 0.45,
    'body_radius': 0.15,  # 수평 raycast 를 몸 중심에서 좌우로 벌리는 폭 / 목표 앞 여유
    'probe_heights': (0.5, 0.9, 1.4),  # 높은 쪽 바닥 기준 수평 raycast 높이 [m]
}


@BaseController.register('VlnMoveByFlashFloorController')
class VlnMoveByFlashFloorController(VlnMoveByFlashCollisionController):
    def __init__(self, config, robot, scene) -> None:
        super().__init__(config=config, robot=robot, scene=scene)
        for k, v in DEFAULTS.items():
            setattr(self, k, getattr(config, k, None) if getattr(config, k, None) is not None else v)
        self._robot_root = None
        self.last_block_reason = ''

    def _hits(self, origin, direction, distance):
        """origin 에서 direction 으로 distance 안의 hit 들 (로봇 자신 제외), 거리순."""
        from omni.physx import get_physx_scene_query_interface

        if self._robot_root is None:
            self._robot_root = self.robot.articulation.prim.GetPath().pathString.rsplit('/', 1)[0]
        hits = []

        def report(hit):
            path = hit.rigid_body if hit.rigid_body else hit.collision
            if not str(path).startswith(self._robot_root):
                hits.append((hit.distance, np.array(hit.position)))
            return True

        get_physx_scene_query_interface().raycast_all(
            tuple(float(v) for v in origin), tuple(float(v) for v in direction), float(distance), report
        )
        return sorted(hits, key=lambda h: h[0])

    def floor_below(self, position):
        """position 아래 첫 바닥 z. 없으면 base z - base_height."""
        hits = self._hits(position, (0.0, 0.0, -1.0), self.base_height + self.max_step_down + 0.5)
        return hits[0][1][2] if hits else position[2] - self.base_height

    def probe_forward(self, position, target, floor_cur):
        """(통과 여부, 목표 base z, 막힌 이유)."""
        top = floor_cur + self.max_step_up + 0.05
        hits = self._hits((target[0], target[1], top), (0.0, 0.0, -1.0), self.max_step_up + self.max_step_down + 0.05)
        if not hits:
            return False, None, 'no_floor'
        floor_tgt = hits[0][1][2]
        if floor_tgt > floor_cur + self.max_step_up:
            return False, None, 'step_up'

        d = np.array([target[0] - position[0], target[1] - position[1], 0.0])
        dist = float(np.linalg.norm(d))
        d = d / max(dist, 1e-6)
        side = np.array([-d[1], d[0], 0.0])
        floor_hi = max(floor_cur, floor_tgt)
        for h in self.probe_heights:
            for s in (-self.body_radius, 0.0, self.body_radius):
                o = np.array([position[0], position[1], floor_hi + h]) + s * side
                hits = self._hits(o, d, dist + self.body_radius)
                if hits:
                    return False, None, f'body@{h:.2f}m side{s:+.2f} dist{hits[0][0]:.2f} at{np.round(hits[0][1], 2).tolist()}'
        return True, floor_tgt + self.base_height, ''

    def forward(self, action: int) -> ArticulationAction:
        self._collision_detected = False
        self.last_block_reason = ''
        positions, orientations = self.robot.articulation.get_world_pose()
        from omni.isaac.core.utils.rotations import (
            euler_angles_to_quat,
            quat_to_euler_angles,
        )

        positions = np.asarray(positions)
        new_position, new_rotation = self.get_new_position_and_rotation(positions, orientations, action)
        floor_cur = self.floor_below(positions)
        z = floor_cur + self.base_height
        if action == 1:
            ok, z_tgt, reason = self.probe_forward(positions, np.asarray(new_position), floor_cur)
            if ok:
                new_position, z = np.asarray(new_position), z_tgt
            else:
                self._collision_detected = True
                self.last_block_reason = reason
                print(f'[FLASH FLOOR] forward blocked: {reason}', flush=True)
                new_position = positions
        new_rotation = euler_angles_to_quat(np.array([0.0, 0.0, quat_to_euler_angles(new_rotation)[2]]))
        self.reset_robot_state(np.array([new_position[0], new_position[1], z]), new_rotation)
        return ArticulationAction()

    def get_obs(self):
        obs = super().get_obs()
        obs['block_reason'] = self.last_block_reason
        return obs
