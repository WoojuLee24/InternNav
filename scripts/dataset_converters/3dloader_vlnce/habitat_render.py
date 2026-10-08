"""habitat에서 **임의 pose의 depth를 직접 렌더**한다 (G1c — 렌더러 간 gap 측정용).

## 왜 필요한가
우리는 Open3D로 `mp3d_n1/<scan>/*.obj`를 렌더하는데, 데이터셋은 habitat이 `mp3d_ce/mp3d/<scan>/<scan>.glb`
를 렌더해 만들었다. **에셋도 렌더러도 다르다.** "저장 depth와 우리 렌더가 일치한다"(G1)는 이미 확인했지만,
그것만으로는 "우리 렌더 == habitat 렌더"인지 알 수 없다 — 둘 다 저장 depth에 가까우면서 서로 다를 수 있다.
depth 범위를 넓히면(5→20 m) 먼 지오메트리가 들어와 차이가 커질 수 있어 범위별로 재야 한다.

## 좌표 변환 (직접 유도)
mesh(Z-up) ↔ habitat(Y-up):  `x_m = x_h`, `y_m = −z_h`, `z_m = y_h`
  → `M⁻¹` (mesh→habitat) = `(m_x, m_z, −m_y)` — `vlnce_align.hmap_translation`의 역과 일치.
카메라 규약: 우리 c2w는 **OpenCV**(+x 우, +y 하, +z 전방), habitat 센서는 **OpenGL**(+x 우, +y 상, −z 전방).
  → `C = diag(1, −1, −1)`
따라서 habitat 센서 회전 `R_h = M⁻¹ · R_mesh · C`, 위치 `t_h = M⁻¹ · t_mesh`.

## 검증 방법 (이 파일의 self-check가 아니라 호출부에서)
데이터셋이 habitat에서 나왔으므로 **habitat 렌더 ≈ 저장 depth**여야 한다. 그게 안 맞으면 위 변환이나
센서 설정(hfov/해상도)이 틀린 것이다 — 다른 결론을 내기 전에 이걸 먼저 통과시켜야 한다.

⚠️ 센서 offset을 0으로 두고 **agent 회전이 전부를 담당**하게 한다. habitat의 기본 센서는 눈높이
offset(`[0, 1.5, 0]`)이 있어 그대로 쓰면 카메라가 1.5 m 위에서 렌더된다.
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))

# mesh(Z-up) -> habitat(Y-up) 선형 변환. 위치·회전 모두 이걸 왼쪽에 곱한다.
M_INV = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]], dtype=np.float64)
# OpenCV 카메라 -> OpenGL 카메라 (y, z 뒤집기)
CV2GL = np.diag([1.0, -1.0, -1.0])


def hfov_from_fx(fx, width):
    """수평 화각[deg]. 저장 depth의 하드코딩 K(fx=388.19, w=640)면 79.0°가 나온다."""
    return float(np.degrees(2.0 * np.arctan(0.5 * width / float(fx))))


def mesh_c2w_to_habitat(c2w):
    """mesh 절대 c2w(OpenCV) -> (position (3,), rotation 3x3) habitat 센서 규약."""
    c2w = np.asarray(c2w, dtype=np.float64)
    return M_INV @ c2w[:3, 3], M_INV @ c2w[:3, :3] @ CV2GL


class HabitatDepthRenderer:
    """씬 하나를 올려두고 임의 pose에서 depth를 렌더한다. `with` 로 쓰거나 `close()` 호출.

    Simulator는 무거우므로(씬당 수 초) **씬당 1회** 만들고 재사용한다. dataloader worker에는 부적합 —
    이건 offline 대조 전용이다.
    """

    def __init__(self, glb_path, width=640, height=480, fx=388.19):
        import habitat_sim
        self._hs = habitat_sim
        bc = habitat_sim.SimulatorConfiguration()
        bc.scene_id = str(glb_path)
        bc.enable_physics = False
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = 'depth'
        spec.sensor_type = habitat_sim.SensorType.DEPTH
        spec.resolution = [height, width]
        spec.hfov = hfov_from_fx(fx, width)
        # 센서 offset을 0으로 — 기본값 [0,1.5,0]을 그대로 두면 카메라가 1.5 m 위에서 렌더된다.
        spec.position = [0.0, 0.0, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        agent = habitat_sim.AgentConfiguration()
        agent.sensor_specifications = [spec]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(bc, [agent]))

    def render(self, c2w_mesh):
        """mesh 절대 c2w -> depth (H,W) float32 [m]. 지오메트리 없는 픽셀은 0."""
        import quaternion
        pos, rot = mesh_c2w_to_habitat(c2w_mesh)
        st = self._hs.AgentState()
        st.position = pos.astype(np.float32)
        st.rotation = quaternion.from_rotation_matrix(rot)
        self.sim.get_agent(0).set_state(st)
        return np.asarray(self.sim.get_sensor_observations()['depth'], dtype=np.float32)

    def close(self):
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == '__main__':
    # 좌표 변환만 확인 (habitat 불필요). 왕복이 항등인지 + 알려진 값 대조.
    from vlnce_align import hmap_translation
    for p_hab in ([1.0, 0.2, -3.0], [-4.5, 1.7, 2.25]):
        m = hmap_translation(p_hab)                     # habitat -> mesh (기존 코드)
        back = M_INV @ np.asarray(m)                    # mesh -> habitat (여기)
        assert np.allclose(back, p_hab), f'왕복 실패 {p_hab} -> {m} -> {back}'
    # 회전: **수평으로 mesh +y를 보는** 카메라(= `synthesize_action_poses`가 만드는 자세)를 넣는다.
    # ⚠️ `c2w = I`로 검사하면 안 된다 — 그건 mesh +z(위)를 보는 카메라라 habitat에서도 위(+y)가 맞다.
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[1.0, 0.0, 0.0],    # cam +x(우)  = mesh +x
                            [0.0, 0.0, 1.0],    # cam +y(하)  = mesh -z
                            [0.0, -1.0, 0.0]])
    c2w[:3, 3] = [1.0, -3.0, 1.25]
    pos, rot = mesh_c2w_to_habitat(c2w)
    assert np.allclose(pos, [1.0, 1.25, 3.0]), f'위치 {pos}'
    fwd = rot @ np.array([0.0, 0.0, -1.0])              # habitat 센서 전방(-z)
    assert np.allclose(fwd, [0.0, 0.0, -1.0]), f'전방 {fwd}'   # mesh +y(전방) -> habitat -z
    up = rot @ np.array([0.0, 1.0, 0.0])                # habitat 센서 위(+y)
    assert np.allclose(up, [0.0, 1.0, 0.0]), f'위 {up}'        # mesh +z(위) -> habitat +y
    assert abs(hfov_from_fx(388.19, 640) - 79.0) < 0.1, hfov_from_fx(388.19, 640)
    print('[selfcheck] habitat_render 5/5 통과 (왕복 x2 · 위치 · 전방 · 위 · hfov 79.0°)')
