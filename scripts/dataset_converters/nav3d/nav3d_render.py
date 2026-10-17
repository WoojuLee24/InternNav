"""habitat 에서 경로를 따라 1인칭 RGB·depth 를 렌더한다 (경로 확인용 — 데이터셋 렌더가 아님).

카메라: 바닥(경로 z) + `height_m`(vln_ce 125cm 리그 = 1.25 m), 방향 = 경로 진행 방향(앞뒤 `look_m` 평균),
`pitch_deg` 만큼 아래를 본다(0 = 수평, vln_ce 룩다운 = 30). hfov 79° (vln_ce 125cm 리그, fx 388.19 @ 640).

좌표: 우리(Z-up) → habitat(Y-up) = (x, z, −y) (`nu.ours_to_hab`). habitat 카메라는 −z 가 전방, +y 가 위.
센서 offset 을 0 으로 두고 agent 자세로 카메라를 놓는다(기본 offset [0, 1.5, 0] 을 두면 1.5 m 위에서 찍힌다 —
3dloader_vlnce/habitat_render.py 와 같은 함정).
"""

import numpy as np

import nav3d_utils as nu

HFOV_125CM = 79.0


def yaw_along(X, i, look_m=0.5):
    """경로 X (N,3) 의 i 번째 점에서 진행 방향 yaw (우리 좌표, +x 기준 반시계, rad)."""
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]
    j0 = int(np.searchsorted(L, L[i] - look_m))
    j1 = min(len(X) - 1, int(np.searchsorted(L, L[i] + look_m)))
    j0 = min(j0, i)
    if j1 == j0:
        j0, j1 = max(0, i - 1), min(len(X) - 1, i + 1)
    v = X[j1, :2] - X[j0, :2]
    return float(np.arctan2(v[1], v[0]))


def hab_rotation(yaw, pitch_deg):
    """우리 yaw(+x 기준) + 아래로 pitch -> habitat 카메라 회전 행렬(열 = 카메라 x, y, z 축의 habitat 방향)."""
    p = np.deg2rad(pitch_deg)
    f = np.array([np.cos(yaw) * np.cos(p), np.sin(yaw) * np.cos(p), -np.sin(p)])     # 전방 (우리)
    r = np.array([np.sin(yaw), -np.cos(yaw), 0.0])                                    # 오른쪽 (우리)
    u = np.cross(r, f)                                                                # 위 (우리)
    to_h = lambda v: np.array([v[0], v[2], -v[1]])
    return np.column_stack([to_h(r), to_h(u), -to_h(f)])        # habitat 카메라: +x 우, +y 위, −z 전방


class HabitatCam:
    def __init__(self, glb_path, width=640, height=480, hfov=HFOV_125CM, depth=False):
        import habitat_sim
        self._hs = habitat_sim
        bc = habitat_sim.SimulatorConfiguration()
        bc.scene_id = str(glb_path)
        bc.enable_physics = False
        specs = []
        for uuid, st in (('rgb', habitat_sim.SensorType.COLOR),) + ((('depth', habitat_sim.SensorType.DEPTH),) if depth else ()):
            s = habitat_sim.CameraSensorSpec()
            s.uuid, s.sensor_type, s.resolution, s.hfov = uuid, st, [height, width], hfov
            s.position, s.orientation = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            specs.append(s)
        ag = habitat_sim.AgentConfiguration()
        ag.sensor_specifications = specs
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(bc, [ag]))

    def render(self, p_floor, yaw, pitch_deg=0.0, height_m=1.25):
        """p_floor = 바닥 점 (우리 좌표). -> dict(rgb (H,W,3) uint8 [, depth])."""
        import quaternion
        st = self._hs.AgentState()
        st.position = nu.ours_to_hab(np.asarray(p_floor, float) + [0, 0, height_m]).astype(np.float32)
        st.rotation = quaternion.from_rotation_matrix(hab_rotation(yaw, pitch_deg))
        self.sim.get_agent(0).set_state(st, reset_sensors=True)
        o = self.sim.get_sensor_observations()
        out = {'rgb': np.asarray(o['rgb'])[:, :, :3].copy()}
        if 'depth' in o:
            out['depth'] = np.asarray(o['depth'], np.float32)
        return out

    def close(self):
        self.sim.close()


if __name__ == '__main__':
    # 회전 규약 확인 (habitat 불필요): yaw 0 = 우리 +x 를 봄 -> habitat 전방(−z 열) = (1, 0, 0)
    R = hab_rotation(0.0, 0.0)
    assert np.allclose(-R[:, 2], [1, 0, 0]) and np.allclose(R[:, 1], [0, 1, 0]), R
    R = hab_rotation(np.pi / 2, 0.0)                     # 우리 +y 를 봄 -> habitat (0, 0, −1)
    assert np.allclose(-R[:, 2], [0, 0, -1]), R
    R = hab_rotation(0.0, 30.0)                          # 30° 아래 -> habitat 전방 y 성분 = −sin30
    assert np.isclose(-R[1, 2], -0.5), R
    assert np.isclose(np.linalg.det(hab_rotation(1.0, 20.0)), 1.0)
    X = np.column_stack([np.linspace(0, 5, 51), np.zeros(51), np.zeros(51)])
    assert abs(yaw_along(X, 0)) < 1e-9 and abs(yaw_along(X, 50)) < 1e-9
    print('[nav3d_render] self-check OK')
