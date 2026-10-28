"""Camera module — deployment entry point (V4L2 camera -> PUB rgb frames on `cam`:5605).

Wraps the canonical publisher (`quantization_kit.async_zmq.camera_publisher`) with an
out-of-band HealthBeacon (REP, port `cam_health`). Publishes square RGB u8 frames whose
size MUST equal the S2 module's --resize so S2 pixel goals map 1:1 onto the frame
(the manager keeps them in sync via the single --s2-resize knob). Flags pass through
(`run.sh --help`): --device N --size 336 --fps 30 --squash.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import DeployConfig          # noqa: E402
from deploy.common.interface import HealthBeacon, install_ready_watch  # noqa: E402

READY_TOKEN = "CAMERA READY"


def main():
    cfg = DeployConfig.from_env()
    beacon = HealthBeacon("cam", cfg.ports["cam_health"])
    beacon.start()
    install_ready_watch(READY_TOKEN, beacon.mark_ready)
    from quantization_kit.async_zmq import camera_publisher   # cv2 (headless build)
    camera_publisher.main()


if __name__ == "__main__":
    main()
