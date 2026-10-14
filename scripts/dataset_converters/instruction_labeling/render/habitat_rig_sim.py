"""habitat-sim 렌더러 — InternData vln_ce 릴리스와 같은 카메라 rig 로 이산 action 을 재생한다.

릴리스 규약 (탐색으로 확인):
- action: -1(프레임0 센티널) / 1 전진 0.25 m / 2 좌 15° / 3 우 15°. sim.step 으로 navmesh 위를 걷는다
  (allow_sliding, 전진 거리가 0.2508~0.2524 로 흔들리고 높이가 변함 → 릴리스도 sim step 산물로 추정).
- `pose.<rig>`: **시작 기준 로컬 좌표**, x=시작 전방, y=시작 좌측, z=위(시작 바닥 기준, 카메라 높이 포함).
  카메라 회전은 OpenCV c2w (+x 오른쪽, +y 아래, +z 전방).
- 센서: 640x480, hfov 79, 높이 1.25, pitch 0 / 30 (아래).

사용법:
    sim = RigSim(cfg, scene_glb, gpu=0)
    sim.set_start(start_position, start_rotation_xyzw)
    obs = sim.observe()          # {rig: (rgb uint8 HxWx3, depth_m float32 HxW)}
    pose = sim.local_pose_cv(rig) # 4x4 (릴리스 pose.<rig> 와 같은 규약)
    sim.step(1)
"""

from pathlib import Path

import numpy as np
import quaternion  # noqa: F401  (np.quaternion 등록)

import habitat_sim

ACTION_NAMES = {1: 'move_forward', 2: 'turn_left', 3: 'turn_right'}
GL_TO_CV = np.diag([1.0, -1.0, -1.0])  # habitat(OpenGL) 카메라 축 -> OpenCV 카메라 축


def quat_xyzw_to_np(q_xyzw):
    x, y, z, w = q_xyzw
    return np.quaternion(w, x, y, z)


class RigSim:
    def __init__(self, cfg: dict, scene_glb: str, gpu: int = 0):
        cam, ag = cfg['camera'], cfg['agent']
        self.rigs = dict(cam['rigs'])            # rig -> pitch_down deg
        self.height_m = float(cam['height_m'])

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_glb)
        sim_cfg.gpu_device_id = int(gpu)
        sim_cfg.allow_sliding = bool(ag['allow_sliding'])
        sim_cfg.enable_physics = False

        specs = []
        for rig, pitch in self.rigs.items():
            for kind, stype in (('rgb', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH)):
                s = habitat_sim.CameraSensorSpec()
                s.uuid = f'{kind}.{rig}'
                s.sensor_type = stype
                s.resolution = [int(cam['height']), int(cam['width'])]
                s.hfov = float(cam['hfov'])
                s.position = [0.0, self.height_m, 0.0]
                s.orientation = [-np.deg2rad(float(pitch)), 0.0, 0.0]  # x축 음수 회전 = 아래
                specs.append(s)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = float(ag['height'])
        agent_cfg.radius = float(ag['radius'])
        agent_cfg.sensor_specifications = specs
        A, Act = habitat_sim.agent.ActionSpec, habitat_sim.agent.ActuationSpec
        agent_cfg.action_space = {
            'move_forward': A('move_forward', Act(amount=float(ag['forward_m']))),
            'turn_left': A('turn_left', Act(amount=float(ag['turn_deg']))),
            'turn_right': A('turn_right', Act(amount=float(ag['turn_deg']))),
        }
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        navmesh = Path(str(scene_glb)).with_suffix('.navmesh')
        if not navmesh.is_file():
            raise FileNotFoundError(f'navmesh 없음: {navmesh}')
        if not self.sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise RuntimeError(f'navmesh 로드 실패: {navmesh}')
        self.agent = self.sim.get_agent(0)
        self._origin = None  # (p0, B): 시작 위치, world->local 기저 (rows: fwd, left, up)

    def close(self):
        self.sim.close()

    def set_start(self, start_position, start_rotation_xyzw):
        st = habitat_sim.AgentState()
        st.position = np.asarray(start_position, dtype=np.float32)
        st.rotation = quat_xyzw_to_np(start_rotation_xyzw)
        self.agent.set_state(st)
        s = self.agent.get_state()
        R = quaternion.as_rotation_matrix(s.rotation)
        fwd = R @ np.array([0.0, 0.0, -1.0])
        left = R @ np.array([-1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        self._origin = (np.asarray(s.position, dtype=np.float64), np.stack([fwd, left, up]))

    def step(self, action: int):
        name = ACTION_NAMES.get(int(action))
        if name is None:
            raise ValueError(f'지원하지 않는 action {action}')
        self.sim.step(name)

    def agent_position(self) -> np.ndarray:
        return np.asarray(self.agent.get_state().position, dtype=np.float64)

    def observe(self) -> dict:
        o = self.sim.get_sensor_observations()
        return {rig: (np.ascontiguousarray(o[f'rgb.{rig}'][..., :3]), o[f'depth.{rig}'].astype(np.float32))
                for rig in self.rigs}

    def local_pose_cv(self, rig: str) -> np.ndarray:
        """센서의 시작 기준 로컬 OpenCV c2w (릴리스 `pose.<rig>` 규약)."""
        p0, B = self._origin
        ss = self.agent.get_state().sensor_states[f'rgb.{rig}']
        R_gl = quaternion.as_rotation_matrix(ss.rotation)
        T = np.eye(4)
        T[:3, :3] = B @ R_gl @ GL_TO_CV
        T[:3, 3] = B @ (np.asarray(ss.position, dtype=np.float64) - p0)
        return T
