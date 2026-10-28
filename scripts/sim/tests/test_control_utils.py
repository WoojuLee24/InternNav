import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control_utils import apply_deadzone  # noqa: E402


def test_deadzone_lifts_small_commands():
    v, w = apply_deadzone(0.05, 0.02, e_p=0.2, e_r=math.radians(10))
    assert v == 0.3 and w == 0.4


def test_deadzone_keeps_large_and_signs():
    v, w = apply_deadzone(0.4, -0.5, e_p=0.5, e_r=-1.0)
    assert v == 0.4 and w == -0.5
    v, w = apply_deadzone(0.1, 0.1, e_p=0.2, e_r=math.radians(-10))
    assert w == -0.4


def test_deadzone_zero_inside_tolerance():
    assert apply_deadzone(0.1, 0.1, e_p=0.01, e_r=math.radians(1)) == (0.0, 0.0)


def test_discrete_step_lead_uses_measured_speed():
    from control_utils import discrete_step

    # not moving yet (startup delay): keep commanding even though the command-based lead would stop
    v, w, done = discrete_step(0.0, 0.4, e_p=0.0, e_r=math.radians(12), v_meas=0.0, w_meas=0.0)
    assert w == 0.4 and not done
    # turning at 0.4 rad/s with 8 deg left (coast ~10 deg) -> stop now
    v, w, done = discrete_step(0.0, 0.2, e_p=0.0, e_r=math.radians(8), w_meas=0.4)
    assert w == 0.0 and done
    # walking at 0.35 m/s with 0.15 m left -> stop; 0.5 m left -> keep going
    assert discrete_step(0.3, 0.0, e_p=0.15, e_r=0.0, v_meas=0.35)[2]
    assert discrete_step(0.4, 0.0, e_p=0.5, e_r=0.0, v_meas=0.35)[0] == 0.4
