#!/usr/bin/env python3
"""
habitat-sim headless 렌더 스모크 테스트.

`import habitat_sim` 이 되는 것만으로는 설치가 끝났다고 볼 수 없다 (EGL/GPU 렌더가
실제로 도는지는 별개). 실제 mp3d scene 을 열어 RGB 한 장을 렌더해 저장한다.

사용:
  /workspace/isaaclab/_isaac_sim/python.sh docker/smoke_habitat_render.py
  ... --scene data/.../zsNo4HB9uLZ.glb --out logs/habitat_smoke.jpg
"""
import argparse
import os
import sys

DEFAULT_SCENE = (
    "data/InternData-N1-v0.5-mini/scene_data/mp3d/"
    "zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=DEFAULT_SCENE, help="mp3d .glb 경로")
    p.add_argument("--out", default=None, help="출력 jpg 경로 (기본: logs/habitat_smoke_<scene>.jpg)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--hfov", type=float, default=79.0)
    p.add_argument("--sensor-height", type=float, default=1.25, help="카메라 높이 (m)")
    args = p.parse_args()

    scene = os.path.abspath(args.scene)
    if not os.path.isfile(scene):
        print(f"✗ scene 파일이 없습니다: {scene}")
        return 1

    scene_name = os.path.splitext(os.path.basename(scene))[0]
    out = args.out or os.path.join("logs", f"habitat_smoke_{scene_name}.jpg")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import habitat_sim
    import numpy as np
    from PIL import Image

    print(f"habitat_sim v{habitat_sim.__version__}")
    print(f"scene: {scene}")

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [args.height, args.width]
    rgb_spec.position = [0.0, args.sensor_height, 0.0]
    rgb_spec.hfov = args.hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    try:
        # navmesh 가 있으면 실제로 서 있을 수 있는 지점에서 렌더한다
        if sim.pathfinder.is_loaded:
            start = sim.pathfinder.get_random_navigable_point()
            state = sim.get_agent(0).get_state()
            state.position = start
            sim.get_agent(0).set_state(state)
            print(f"navmesh OK, agent position: {np.round(start, 3).tolist()}")
        else:
            print("⚠ navmesh 미로드 — 기본 위치에서 렌더합니다")

        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]
    finally:
        sim.close()

    Image.fromarray(rgb.astype(np.uint8)).save(out, quality=92)
    nonblack = float((rgb.reshape(-1, 3).sum(axis=1) > 0).mean())
    print(f"saved: {out}  ({rgb.shape[1]}x{rgb.shape[0]}, non-black pixels {nonblack:.1%})")

    if nonblack < 0.01:
        print("✗ 렌더 결과가 거의 전부 검정입니다 — EGL/GPU 렌더가 동작하지 않았을 수 있습니다.")
        return 1
    print("✓ headless 렌더 성공")
    return 0


if __name__ == "__main__":
    sys.exit(main())
