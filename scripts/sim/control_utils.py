"""Helpers for the discrete-action PID on the RBQ10 walking policy (pure python, unit-tested)."""
import math


def apply_deadzone(v, w, e_p, e_r, v_min=0.3, w_min=0.4, tol_p=0.05, tol_r=math.radians(3.0)):
    """Lift commands out of the locomotion dead zone (gdm uses rapp_vx_min_speed 0.3, align_w_min 0.4).

    Below those the sim's walking policy barely moves (2026-10-05: forward 0.25 m stalled at 0.002-0.04 m,
    15 deg turns ended 4-9 deg short). While an error is above tolerance, the command magnitude is at least the
    minimum; inside tolerance that axis is zeroed.
    """
    v = 0.0 if abs(e_p) < tol_p else math.copysign(max(abs(v), v_min), e_p)
    w = 0.0 if abs(e_r) < tol_r else math.copysign(max(abs(w), w_min), e_r)
    return v, w


def discrete_step(v, w, e_p, e_r, v_meas=0.0, w_meas=0.0, v_min=0.3, w_min=0.4, tol_p=0.05,
                  tol_r=math.radians(3.0), lead_s=0.45):
    """One PID tick for a discrete action with dead-zone lift and coast (lead) compensation.

    The walking policy reacts with a delay and keeps moving after a zero command (2026-10-05: 15 deg turns ended at
    ~27-30 deg, 0.25 m steps at ~0.39-0.44 m). An axis is stopped once its remaining error is within
    tol + |measured speed| * lead_s. The *measured* speed (pose difference) is used, not the command: right after
    the start the robot has not accelerated yet, and a command-based lead stopped 15 deg turns at ~4 deg.
    Returns (v, w, done).
    """
    v, w = apply_deadzone(v, w, e_p, e_r, v_min, w_min, tol_p, tol_r)
    if abs(e_p) <= tol_p + abs(v_meas) * lead_s:
        v = 0.0
    if abs(e_r) <= tol_r + abs(w_meas) * lead_s:
        w = 0.0
    return v, w, v == 0.0 and w == 0.0
