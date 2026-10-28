"""S2 module — deployment entry point (System-2: Qwen2.5-VL GGUF via llama-s2-runner, ~1 Hz).

Wraps the canonical S2 service (`quantization_kit.async_zmq.s2_service`) with an
out-of-band HealthBeacon (REP, port `s2_health`) so the manager can probe
liveness/readiness without touching the s2_in/s2_out data path. The beacon starts
BEFORE the heavy imports, so it answers ready:false throughout GGUF load. All CLI
flags pass through to the service unchanged (`run.sh --help`).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import DeployConfig          # noqa: E402  (light: no torch)
from deploy.common.interface import HealthBeacon, install_ready_watch  # noqa: E402

READY_TOKEN = "S2_SERVICE READY"


def main():
    cfg = DeployConfig.from_env()
    beacon = HealthBeacon("s2", cfg.ports["s2_health"])
    beacon.start()
    install_ready_watch(READY_TOKEN, beacon.mark_ready)
    from quantization_kit.async_zmq import s2_service   # heavy: torch + llama.cpp policy
    s2_service.main()


if __name__ == "__main__":
    main()
