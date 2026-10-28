"""Viewer module — deployment entry point (the live 3-panel visualizer, deploy-owned).

Runs `deploy.viewer_module.visualizer` (LEFT: the exact frame S2 processed + its pixel
goal | CENTER: live camera + action arrow + rates | RIGHT: the frame S1 computed on,
SYNCED, with the raw diffusion trajectories PROJECTED via the repo camera model and the
S1↔S2 goal-pixel agreement) with an out-of-band HealthBeacon (REP, port `viz_health`).
Pure observer: it only SUBs (cam:5605, viz:5607, s2_out:5602).

Backends (--view): tk (native window, needs DISPLAY) | mjpeg (browser at
http://<host>:8089/ — works headless/remote) | save (writes dbg/viz_live.png ~2 Hz).
Camera-model flags: --cam-hfov --cam-height --cam-pitch (defaults from VLN_CAM_* env).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import DeployConfig          # noqa: E402
from deploy.common.interface import HealthBeacon, install_ready_watch  # noqa: E402

READY_TOKEN = "[viz] cam SUB"     # printed once the three SUB sockets are connected


def main():
    cfg = DeployConfig.from_env()
    beacon = HealthBeacon("viz", cfg.ports["viz_health"])
    beacon.start()
    install_ready_watch(READY_TOKEN, beacon.mark_ready)
    from deploy.viewer_module import visualizer   # deploy-owned viewer (projection + sync)
    visualizer.main()


if __name__ == "__main__":
    main()
