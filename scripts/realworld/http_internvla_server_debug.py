import argparse
import copy
import json
import math
import os
import queue
import statistics
import threading
import time
from datetime import datetime
import sys
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw, ImageFont
import cv2


# Add project path
# project_root = Path("/home/gdr/gd_vln/workspace/src/InternNav")
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformation import Calibration

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''
save_dir = 'vis_debug/http_internvla_server_debug'
os.makedirs(save_dir, exist_ok=True)
agent_lock = threading.Lock()
async_cache_lock = threading.Lock()  # Lock for async cached output
SERVER_MODE = "sync"

# Async infrastructure
import queue
import threading
import concurrent.futures

s2_request_queue = queue.Queue(maxsize=1)
s2_output_queue = queue.Queue(maxsize=1)
async_thread_running = False
s2_executor = None  # Thread pool for async S2

# Cached output from background loop
async_cached_trajectory = None
async_cached_action = None

# === Phase 3 Task #9: Temporal S2 caching ===
# Skip background agent.step() if the new frame is visually near-identical
# to the previously processed frame. Threshold is the cosine-similarity
# floor for skipping; 0.0 disables (always run).
temporal_cache_threshold = 0.0
temporal_cache_lock = threading.Lock()
_last_fingerprint = None  # numpy float32 vector, normalized

# === Phase 3 Task #9 v2 (I-046, I-047): Action-aware cache gate ===
# Without these guards, the cosine-only gate at thr>=0.92 systematically
# filters out frames where S2 would have produced discrete actions
# (stop/turn/look_down). Verified empirically: 0/28 fresh actions on bag
# 073623 vs ~50/50 control split. Two safeguards:
#   I-046: never replay the cache when the last fresh output was a discrete
#          action. Action frames are decision points; staleness there is a
#          safety failure mode.
#   I-047: bound max-hold time. Force a fresh S2 every _max_hold_frames
#          regardless of similarity, so the cache cannot lock onto one
#          trajectory output indefinitely.
_last_fresh_was_action = False
_consecutive_skip_count = 0
_max_hold_frames = 10  # Forced refresh threshold; tuned per experiment
# I-046 runtime toggle for ablation study (Gate 3d).
# True = default (action-aware); False = disabled for ablation condition 3.
_action_aware_enabled = True

# I-049: Adaptive max_hold (Gate 3e)
# In stable scenes (bag 073623), I-047 monopolizes all forced bypasses and
# resets I-046 after each (one-shot semantics). This causes V>0.10 on stable
# bags. Adaptive max_hold tracks recent similarity variance and decreases
# max_hold when scenes are stable, allowing natural diversity to emerge.
# Window of recent cosine similarities for adaptation.
_adaptive_max_hold_enabled = False  # Off by default; enable via endpoint
_similarity_window = []             # Rolling window of recent sim values
_ADAPTIVE_WINDOW_SIZE = 20          # Frames to average over
_ADAPTIVE_MAX_HOLD_MIN = 3          # Floor: never lower than 3 forced per cycle
_ADAPTIVE_MAX_HOLD_MAX = 25         # Ceiling: never exceed 25 frames between refresh
# Variance thresholds: below low_var → stable scene → reduce max_hold
# above high_var → dynamic scene → increase max_hold
_ADAPTIVE_VAR_LOW = 0.0002          # Below this: very stable → max_hold=5
_ADAPTIVE_VAR_HIGH = 0.001          # Above this: dynamic → max_hold=15

# I-051: Trajectory-Length-Adaptive Max Hold (Gate 3h)
# When S2 produces a N-waypoint trajectory, the plan has N*multiplier frames
# of semantic "plan consumption" before it can go stale.  Setting max_hold
# dynamically to min(cap, N*multiplier) couples hold duration to plan richness:
# short plans refresh often, long plans stay cached longer — both correctly.
_traj_adaptive_hold_enabled = False
_traj_adaptive_hold_multiplier = 7   # frames to hold per cached waypoint
_traj_adaptive_hold_cap = 50         # hard upper cap to prevent unbounded hold

# I-052: Request-Count-Adaptive Hold (Gate 3i, I-051v2)
# Counts HTTP responses that served the current cached trajectory.
# Each serve represents one robot control step consuming the plan.
# Force S2 refresh after _serve_hold_threshold consecutive serve cycles.
# Unlike I-051, this is independent of trajectory length and works correctly
# for fixed-length model outputs (e.g., InternVLA-N1's 33-waypoint decoder).
_serve_count_bypass_enabled = False
_serve_hold_threshold = 15          # force refresh after N serve cycles
_traj_serve_count = 0               # HTTP responses serving current cached traj
_serve_count_lock = threading.Lock()

# I-053: EMA Scene Fingerprint (Gate 3k)
# Current system: compare fp_current vs fp_at_last_cache_miss only.
# EMA variant: maintain an exponential moving average of ALL frame fingerprints
# (including skipped frames). The EMA tracks the "recent scene baseline" rather
# than a snapshot, reducing sensitivity to transient lighting/motion artifacts
# and adapting to slow scene drift without triggering I-047 forced refreshes.
#
# Skip criterion: cosine_sim(fp_current, _ema_fingerprint) >= tau
# (identical semantics to current; only the reference changes)
#
# alpha controls EMA adaptation speed:
#   alpha=0.0 → equivalent to current point-fingerprint (no update on skip)
#   alpha=0.1 → slow drift tracking (~10 frames to adapt fully)
#   alpha=0.5 → fast adaptation (~2 frames to adapt)
# The key difference from current: EMA updates on EVERY frame, including skips.
_ema_fingerprint_enabled = False
_ema_fingerprint_alpha = 0.15       # EMA smoothing factor; tuned per experiment
_ema_fingerprint = None             # running EMA vector (same shape as _last_fingerprint)
_ema_fp_lock = threading.Lock()

# I-055: Transition-Reset EMA (Gate 3m correction of I-053)
# I-053 design flaw: slow EMA update after forced bypasses (I-046/I-047/I-052) causes
# a cascade — EMA lags ~1/α frames behind the new scene, triggering natural misses
# at each lag frame (EMA still represents the old scene). Fix: hard-reset EMA to current
# frame after any forced bypass, so post-bypass comparison is against the current scene.
# During skip sequences, EMA still slowly tracks drift (same as I-053).
# _ema_transition_reset is a flag on the existing _ema_fingerprint_enabled path.
_ema_transition_reset = False           # True = I-055 (TR-EMA); False = I-053 (standard EMA)

# I-203: EMA Warm-Up Acceleration (Gate 8)
# After each TR-EMA reset (forced bypass), the EMA needs ~1/α frames to represent
# the new scene accurately. At α=0.10, this is ~10 frames → cold-start penalty.
# Fix: use α_warm (higher) for the first warmup_frames after each reset, then
# drop back to α_prod. Per-bypass warm-up closes the convergence gap after EVERY
# scene transition, not just server start.
_ema_warmup_enabled = False
_ema_warmup_frames = 10         # frames to use α_warm after each TR-EMA reset
_ema_warmup_alpha = 0.5         # fast-converge α (vs production α=0.10)
_ema_frames_since_reset = 0     # frames elapsed since last TR-EMA hard-reset
_ema_warmup_lock = threading.Lock()

# I-054: Similarity Slope Detector — Predictive Refresh (Gate 3l)
# All previous cache mechanisms are reactive: they trigger a refresh AFTER the
# similarity drops below τ or the hold count exceeds H_max.
# I-054 is predictive: by tracking the temporal derivative of the similarity
# signal, it fires a pre-emptive S2 run when the similarity is DECLINING RAPIDLY,
# anticipating cache invalidation before it occurs.
#
# Skip decision: if sim >= τ BUT d(sim)/dt < -_slope_threshold → predictive bypass
#   d(sim)/dt estimated as (sim[t] - sim[t-k]) / k  over the last _slope_window frames
#
# Impact: catches doorway/turn transitions 2-3 frames earlier than the reactive τ gate,
# reducing the probability that stale cache is served at safety-critical decision points.
# Allows higher τ (more permissive skip) while maintaining V ≤ 0.10.
_slope_predict_enabled = False
_slope_threshold = 0.015           # |Δsim / frame| trigger level; tuned by Gate 3l
_slope_window = 3                  # frames over which to estimate slope
_similarity_history = []           # rolling buffer of recent similarity scores
_slope_history_lock = threading.Lock()

# I-204: Cosine Similarity Variance Gating (Gate 9)
# Slope predict (I-054) catches monotone declining similarity (smooth transitions).
# Variance gating catches OSCILLATORY similarity (robot at doorways, looking left/right,
# or hovering between two distinct scenes). High std(sim) over W frames indicates the
# scene content is unstable even if the current sim >= τ.
#
# Bypass condition: sim >= τ AND std(sim_history[-W:]) > σ_threshold → force S2
# Mechanistically orthogonal to I-054 (slope) and I-047 (max_hold).
_var_gate_enabled = False
_var_gate_sigma = 0.03       # std(sim) threshold; tuned by Gate 9 sweep
_var_gate_window = 5         # frames of sim history to compute std over
_var_gate_lock = threading.Lock()

# I-206: Action-Streak Trajectory Recovery (Gate 11)
# Hypothesis: when S2 produces K consecutive action-type outputs (no trajectories),
# the robot is in an action-only regime and obstacle-avoidance quality degrades.
# Fix: enter "trajectory recovery mode" — reduce effective max_hold to _traj_recovery_hold
# (default 3) to force frequent S2 refreshes, increasing the probability of receiving
# a trajectory output. Exit recovery immediately upon any trajectory output.
# This is orthogonal to I-046 (next-frame action reflex) and I-047 (max-hold bound):
# I-206 provides a sustained pressure for trajectory outputs over multiple S2 calls,
# while I-046 fires once per action and I-047 has a fixed-period cadence.
# Unlike Gate 3h (fixed decoder length → always 33) this uses the ACTUAL output type.
_traj_recovery_enabled = False
_traj_recovery_k = 5            # action streak length that triggers recovery
_traj_recovery_hold = 3         # max_hold override during recovery (force refresh every 3 frames)
_in_traj_recovery = False       # True = currently in recovery mode
_fresh_output_history = []      # rolling list of recent fresh output types: 'T' or 'A'
_traj_recovery_lock = threading.Lock()

# I-058 (Gate 3p): Odometry-Progress Hold
# Force S2 when robot has traveled > threshold meters since last S2 run.
# Provides a SPATIAL invalidation signal orthogonal to I-047 (temporal) and I-052 (count).
# Requires client to send odom=[x, y, theta] in request JSON.
_odom_progress_enabled = False
_odom_progress_threshold = 0.5    # meters; swept by Gate 3p
_current_odom = None              # [x, y, theta] from latest HTTP request
_last_s2_odom = None              # [x, y, theta] at last completed S2 run
_odom_lock = threading.Lock()

# I-202 (Gate 7): Optical Flow Cache Invalidation
# Force S2 refresh when mean optical flow magnitude exceeds a threshold,
# catching rapid robot motion (turns, stops) that visual embeddings miss.
# The cosine-similarity gate misses scene changes where the embedding
# space is locally flat (e.g., first few frames of a turn). Optical flow
# directly measures pixel displacement and fires immediately on motion onset.
_flow_bypass_enabled = False
_flow_magnitude_threshold = 20.0   # mean pixels/frame at 80x60 resolution
_prev_flow_frame = None             # last grayscale 80x60 frame for flow
_flow_frame_lock = threading.Lock()


def _check_flow_bypass(rgb, threshold):
    """I-202: compute mean optical flow vs previous frame. Returns True if
    magnitude exceeds threshold. Always updates _prev_flow_frame."""
    global _prev_flow_frame
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        small = cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA)
    except Exception:
        return False
    with _flow_frame_lock:
        prev = _prev_flow_frame
        _prev_flow_frame = small
    if prev is None:
        return False
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev, small, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
        return mag > threshold
    except Exception:
        return False


def _image_fingerprint(rgb):
    """32x32 grayscale, raw pixel values 0-255 (no normalization).
    Returns a 1D float32 vector or None if input is unusable.
    Used with MAD-based similarity which is far more discriminative for
    natural video than L2-normalized cosine (cosine on smooth scenes is
    ~0.97+ even between visually distinct frames, making cosine useless
    for caching gates here)."""
    if rgb is None:
        return None
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32).flatten()
    except Exception:
        return None

def _cosine_sim(a, b):
    """Now MAD-based similarity in [0, 1].
    Identical frames -> 1.0. Pure-noise pair -> ~0.5.
    Natural video consecutive frames typically 0.94-0.99 (steady scene)
    or 0.85-0.95 (active motion). Threshold around 0.97 is a sensible
    skip floor for static scenes; below that the cache should refresh."""
    if a is None or b is None:
        return 0.0
    mad = float(np.abs(a - b).mean()) / 255.0
    return max(0.0, 1.0 - mad)

# ===== ASYNC METRICS INSTRUMENTATION =====
# ===== DUAL-SYSTEM PERFORMANCE METRICS =====
# 
# System Architecture:
# - System 2 (S2): Language planner - outputs coordinates OR discrete actions
# - System 1 (S1): Trajectory generator - generates trajectories from coordinates
#
# Speed Metrics (in Hz - frames/second):
# - S2 Speed: How fast the language model can process frames
# - S1 Speed: How fast trajectories can be generated (when coordinates received)  
# - Joint Speed: Combined throughput (S1 + S2 processing)
#
# Quality Metrics:
# - Trajectory Ratio: % of outputs that are trajectories (from coords)
# - Discrete Ratio: % of outputs that are discrete actions

ASYNC_BACKGROUND_INFERENCE = True  # TRUE fully decoupled: background runs step(), HTTP returns cached immediately

async_metrics = {
    "start_time": time.time(),

    "http_requests": 0,
    "http_cache_hits": 0,
    "http_cache_misses": 0,
    "total_http_latency": 0.0,

    # System 2 (planner) counters
    "s2_runs": 0,
    "s2_time_total": 0.0,
    "s2_coords_outputs": 0,
    "s2_discrete_outputs": 0,
    "s2_coord_time_total": 0.0,
    "s2_discrete_time_total": 0.0,

    # System 1 (trajectory generator) counters
    "s1_runs": 0,

    # Joint counters
    "total_action_time": 0.0,
    "total_requests": 0,

    # Background loop diagnostics
    "background_s2_runs": 0,
    "total_s2_time": 0.0,

    # Phase 3 Task #9: Temporal S2 caching diagnostics
    "temporal_cache_skips": 0,
    "temporal_cache_threshold": 0.0,

    # Fresh-only output type counters (only incremented when S2 actually ran,
    # not on cache replays). Lets us tell if cached outputs match the fresh
    # output distribution or if traj_ratio is biased by cache replay.
    "fresh_traj_outputs": 0,
    "fresh_action_outputs": 0,

    # I-046 / I-047 forced-fresh counters (cache bypass diagnostics)
    "action_aware_bypasses": 0,   # I-046: forced fresh because last was action
    "max_hold_bypasses": 0,       # I-047: forced fresh because skip count >= max

    # Gate 3c (I-010): cold-start pre-fetch diagnostics
    "waiting_responses": 0,       # HTTP responses returned before any cache hit
    "pre_warm_frames_queued": 0,  # synthetic frames queued at server start

    # Phase 3X (I-111/I-112): per-request response-type sequence for offline proxy analysis
    "response_sequence": [],      # list of 'T', 'W', or action-index per HTTP request
}
async_metrics_lock = threading.Lock()


def reset_dual_metrics():
    global _last_fresh_was_action, _consecutive_skip_count, _last_fingerprint, _similarity_window
    global _ema_fingerprint, _in_traj_recovery, _fresh_output_history
    with temporal_cache_lock:
        _last_fresh_was_action = False
        _consecutive_skip_count = 0
        _last_fingerprint = None
        _similarity_window.clear()  # I-049: reset adaptive window on metrics reset
    with _ema_fp_lock:
        _ema_fingerprint = None   # I-053: reset EMA on metrics reset
    with _slope_history_lock:
        _similarity_history.clear()  # I-054: reset slope history on metrics reset
    with async_metrics_lock:
        async_metrics["start_time"] = time.time()
        async_metrics["http_requests"] = 0
        async_metrics["http_cache_hits"] = 0
        async_metrics["http_cache_misses"] = 0
        async_metrics["total_http_latency"] = 0.0
        async_metrics["s2_runs"] = 0
        async_metrics["s2_time_total"] = 0.0
        async_metrics["s2_coords_outputs"] = 0
        async_metrics["s2_discrete_outputs"] = 0
        async_metrics["s2_coord_time_total"] = 0.0
        async_metrics["s2_discrete_time_total"] = 0.0
        async_metrics["s1_runs"] = 0
        async_metrics["s1_time_total"] = 0.0
        async_metrics["total_action_time"] = 0.0
        async_metrics["total_requests"] = 0
        async_metrics["background_s2_runs"] = 0
        async_metrics["total_s2_time"] = 0.0
        async_metrics["temporal_cache_skips"] = 0
        async_metrics["fresh_traj_outputs"] = 0
        async_metrics["fresh_action_outputs"] = 0
        async_metrics["action_aware_bypasses"] = 0
        async_metrics["max_hold_bypasses"] = 0
        async_metrics["waiting_responses"] = 0
        async_metrics["traj_lengths_sum"] = 0
        async_metrics["traj_lengths_count"] = 0
        async_metrics["serve_count_bypasses"] = 0
        async_metrics["slope_predict_bypasses"] = 0
        async_metrics["odom_progress_bypasses"] = 0
        async_metrics["flow_bypasses"] = 0      # I-202: optical flow triggers
        async_metrics["var_gate_bypasses"] = 0  # I-204: cosine variance gating
        async_metrics["traj_recovery_activations"] = 0  # I-206: action-streak recovery triggers
        async_metrics["response_sequence"] = []  # Phase 3X: reset per-run sequence log
        # pre_warm_frames_queued is NOT reset here — it's a server-lifetime counter
    with _traj_recovery_lock:
        _fresh_output_history.clear()
        _in_traj_recovery = False

def async_continuous_loop():
    """Background thread: continuously runs step() and caches output
    
    This achieves TRUE async:
    - Loop runs step() in background
    - Caches trajectory for HTTP to return immediately
    - HTTP never waits for inference!
    """
    global async_cached_trajectory, async_cached_action, async_thread_running

    instruction = "Exit door. Turn left and go straight until you find fire extinguisher. Then stop."
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    global _last_fingerprint, _last_fresh_was_action, _consecutive_skip_count, _max_hold_frames
    global _ema_fingerprint, _traj_serve_count, _similarity_history, _ema_transition_reset
    global _current_odom, _last_s2_odom
    global _ema_frames_since_reset, _var_gate_enabled, _var_gate_sigma, _var_gate_window
    global _in_traj_recovery, _fresh_output_history
    while async_thread_running:
        try:
            # Wait for new frame data from queue (non-blocking with timeout)
            try:
                image, depth = s2_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # === Phase 3 Task #9 v2: Action-aware temporal cache gate ===
            # I-046: never skip if last fresh output was a discrete action
            # I-047: never skip more than _max_hold_frames consecutively
            # I-049: adaptive max_hold based on scene stability (Gate 3e)
            # else: standard MAD-based similarity gate
            with temporal_cache_lock:
                threshold = temporal_cache_threshold
                last_was_action = _last_fresh_was_action
                skip_count = _consecutive_skip_count
                max_hold = _max_hold_frames
                action_aware_on = _action_aware_enabled
                adaptive_on = _adaptive_max_hold_enabled
                traj_adaptive_on = _traj_adaptive_hold_enabled
                traj_adaptive_multiplier = _traj_adaptive_hold_multiplier
                traj_adaptive_cap = _traj_adaptive_hold_cap
            with _serve_count_lock:
                serve_count = _traj_serve_count
                serve_bypass_on = _serve_count_bypass_enabled
                serve_threshold = _serve_hold_threshold
            with _ema_fp_lock:
                ema_on = _ema_fingerprint_enabled
                ema_alpha = _ema_fingerprint_alpha
                ema_tr_reset = _ema_transition_reset
            # I-203: resolve effective alpha (warm-up schedule)
            with _ema_warmup_lock:
                warmup_on = _ema_warmup_enabled
                warmup_n = _ema_warmup_frames
                warmup_alpha = _ema_warmup_alpha
                frames_since_reset = _ema_frames_since_reset
            if warmup_on and ema_on and frames_since_reset < warmup_n:
                effective_alpha = warmup_alpha
            else:
                effective_alpha = ema_alpha
            with _slope_history_lock:
                slope_on = _slope_predict_enabled
                slope_thr = _slope_threshold
                slope_win = _slope_window
            with _odom_lock:
                odom_progress_on = _odom_progress_enabled
                odom_progress_thr = _odom_progress_threshold
                cur_odom = _current_odom[:] if _current_odom is not None else None
                last_s2_pos = _last_s2_odom[:] if _last_s2_odom is not None else None
            flow_on = _flow_bypass_enabled
            flow_thr = _flow_magnitude_threshold
            with _var_gate_lock:
                var_on = _var_gate_enabled
                var_sigma = _var_gate_sigma
                var_win = _var_gate_window
            # I-206: read recovery state; override max_hold if in trajectory recovery
            with _traj_recovery_lock:
                traj_rec_on = _traj_recovery_enabled
                in_traj_recovery = _in_traj_recovery
                traj_rec_hold = _traj_recovery_hold
            if traj_rec_on and in_traj_recovery:
                max_hold = traj_rec_hold
            was_forced_bypass = False
            was_i046_bypass = False  # I-050: track bypass type for flag propagation
            was_i047_bypass = False
            was_i052_bypass = False
            was_i054_bypass = False
            was_i058_bypass = False
            was_flow_bypass = False
            if threshold > 0.0:
                # I-049: compute adaptive max_hold if enabled
                if adaptive_on and len(_similarity_window) >= _ADAPTIVE_WINDOW_SIZE:
                    sim_var = float(np.var(_similarity_window[-_ADAPTIVE_WINDOW_SIZE:]))
                    if sim_var < _ADAPTIVE_VAR_LOW:
                        # Very stable scene — reduce max_hold to force more diversity
                        adaptive_mh = _ADAPTIVE_MAX_HOLD_MIN
                    elif sim_var > _ADAPTIVE_VAR_HIGH:
                        # Dynamic scene — relax max_hold
                        adaptive_mh = _ADAPTIVE_MAX_HOLD_MAX
                    else:
                        # Interpolate linearly between min and 10 (static default)
                        t = (sim_var - _ADAPTIVE_VAR_LOW) / (_ADAPTIVE_VAR_HIGH - _ADAPTIVE_VAR_LOW)
                        adaptive_mh = int(_ADAPTIVE_MAX_HOLD_MIN + t * (10 - _ADAPTIVE_MAX_HOLD_MIN))
                    max_hold = adaptive_mh

                forced = None
                if action_aware_on and last_was_action:
                    forced = "action_aware_bypasses"  # I-046
                    was_i046_bypass = True
                elif skip_count >= max_hold:
                    forced = "max_hold_bypasses"  # I-047
                    was_i047_bypass = True
                elif flow_on and _check_flow_bypass(image, flow_thr):
                    forced = "flow_bypasses"  # I-202: optical flow motion onset
                    was_flow_bypass = True
                elif serve_bypass_on and serve_count >= serve_threshold:
                    forced = "serve_count_bypasses"  # I-052
                    was_i052_bypass = True
                    with _serve_count_lock:
                        _traj_serve_count = 0
                elif odom_progress_on and cur_odom is not None and last_s2_pos is not None:
                    dx = cur_odom[0] - last_s2_pos[0]
                    dy = cur_odom[1] - last_s2_pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist >= odom_progress_thr:
                        forced = "odom_progress_bypasses"  # I-058
                        was_i058_bypass = True
                if forced is not None:
                    was_forced_bypass = True
                    # Forced refresh: skip the similarity check entirely
                    with async_metrics_lock:
                        async_metrics[forced] = async_metrics.get(forced, 0) + 1
                    fp = _image_fingerprint(image)
                    if fp is not None:
                        _last_fingerprint = fp
                        # I-053/I-055: update EMA on forced frames
                        # I-055 (TR-EMA): hard-reset to current frame to prevent post-bypass cascade
                        # I-053 (standard EMA): slow update (original behavior)
                        # I-203: after TR-EMA reset, restart warm-up counter
                        if ema_on:
                            with _ema_fp_lock:
                                if _ema_fingerprint is None or ema_tr_reset:
                                    _ema_fingerprint = fp.copy()
                                    with _ema_warmup_lock:
                                        _ema_frames_since_reset = 0
                                else:
                                    _ema_fingerprint = (1 - effective_alpha) * _ema_fingerprint + effective_alpha * fp
                                    with _ema_warmup_lock:
                                        _ema_frames_since_reset += 1
                    with temporal_cache_lock:
                        _consecutive_skip_count = 0
                else:
                    fp = _image_fingerprint(image)
                    if fp is not None:
                        # I-053: update EMA on every frame (including would-be skips)
                        if ema_on:
                            with _ema_fp_lock:
                                if _ema_fingerprint is None:
                                    _ema_fingerprint = fp.copy()
                                    with _ema_warmup_lock:
                                        _ema_frames_since_reset = 0
                                else:
                                    _ema_fingerprint = (1 - effective_alpha) * _ema_fingerprint + effective_alpha * fp
                                    with _ema_warmup_lock:
                                        _ema_frames_since_reset += 1
                        # I-053: choose reference fingerprint (EMA or point)
                        ref_fp = _ema_fingerprint if (ema_on and _ema_fingerprint is not None) else _last_fingerprint
                    else:
                        ref_fp = None
                    if fp is not None and ref_fp is not None:
                        sim = _cosine_sim(fp, ref_fp)
                        # I-054: update slope history on every frame
                        if slope_on:
                            with _slope_history_lock:
                                _similarity_history.append(sim)
                                if len(_similarity_history) > slope_win * 4:
                                    del _similarity_history[:-slope_win * 2]
                        # I-049: track similarity for adaptive max_hold window
                        if adaptive_on:
                            _similarity_window.append(sim)
                            if len(_similarity_window) > _ADAPTIVE_WINDOW_SIZE * 2:
                                del _similarity_window[:-_ADAPTIVE_WINDOW_SIZE]
                        if sim >= threshold:
                            # I-054: predictive bypass — if similarity is DECLINING rapidly,
                            # fire a pre-emptive S2 run even though sim >= τ
                            if slope_on:
                                with _slope_history_lock:
                                    hist = _similarity_history
                                if len(hist) >= slope_win + 1:
                                    slope = (hist[-1] - hist[-1 - slope_win]) / slope_win
                                    if slope < -slope_thr:
                                        was_i054_bypass = True
                                        was_forced_bypass = True
                                        with async_metrics_lock:
                                            async_metrics["slope_predict_bypasses"] = (
                                                async_metrics.get("slope_predict_bypasses", 0) + 1)
                                        with temporal_cache_lock:
                                            _consecutive_skip_count = 0
                                        # Fall through to S2 run
                            # I-204: variance gate — high std(sim) over last W frames
                            # indicates oscillatory scene even though sim >= τ right now
                            was_var_bypass = False
                            if not was_i054_bypass and var_on:
                                with _slope_history_lock:
                                    vh = _similarity_history[-var_win:] if len(_similarity_history) >= var_win else []
                                if len(vh) >= var_win:
                                    sim_std = statistics.stdev(vh)
                                    if sim_std > var_sigma:
                                        was_var_bypass = True
                                        was_forced_bypass = True
                                        with async_metrics_lock:
                                            async_metrics["var_gate_bypasses"] = (
                                                async_metrics.get("var_gate_bypasses", 0) + 1)
                                        with temporal_cache_lock:
                                            _consecutive_skip_count = 0
                            if not was_i054_bypass and not was_var_bypass:
                                with async_metrics_lock:
                                    async_metrics["temporal_cache_skips"] = async_metrics.get("temporal_cache_skips", 0) + 1
                                with temporal_cache_lock:
                                    _consecutive_skip_count += 1
                                s2_request_queue.task_done()
                                continue
                    # First frame, or below threshold → process and update point fingerprint
                    if fp is not None:
                        _last_fingerprint = fp
                    with temporal_cache_lock:
                        _consecutive_skip_count = 0

            t0 = time.time()

            # Run step() in background - keep behavior consistent with sync path
            with agent_lock:
                look_down = False
                dual_output = agent.step(
                    image,
                    depth,
                    camera_pose,
                    instruction,
                    intrinsic=args.camera_intrinsic,
                    look_down=look_down,
                )
                if dual_output.output_action is not None and dual_output.output_action == [5]:
                    look_down = True
                    dual_output = agent.step(
                        image,
                        depth,
                        camera_pose,
                        instruction,
                        intrinsic=args.camera_intrinsic,
                        look_down=look_down,
                    )
                
                # Cache the output for HTTP
                fresh_was_traj = False
                fresh_was_action = False
                with async_cache_lock:
                    if dual_output.output_action is not None:
                        async_cached_action = dual_output.output_action
                        async_cached_trajectory = None  # Clear trajectory when action
                        fresh_was_action = True
                    elif dual_output.output_trajectory is not None:
                        async_cached_trajectory = dual_output.output_trajectory.tolist()
                        async_cached_action = None  # Clear action when trajectory
                        fresh_was_traj = True
                        # I-052: reset serve count when cache is refreshed
                        with _serve_count_lock:
                            _traj_serve_count = 0

            # I-058: record odom at S2 completion so odom-progress-hold knows last planned position
            with _odom_lock:
                if _current_odom is not None:
                    _last_s2_odom = _current_odom[:]

            # I-051: trajectory-length-adaptive max_hold
            # After trajectory is cached, update max_hold based on plan richness.
            # Executed outside async_cache_lock to avoid contention.
            if fresh_was_traj:
                traj_len = len(async_cached_trajectory) if async_cached_trajectory else 0
                with async_metrics_lock:
                    async_metrics["traj_lengths_sum"] = async_metrics.get("traj_lengths_sum", 0) + traj_len
                    async_metrics["traj_lengths_count"] = async_metrics.get("traj_lengths_count", 0) + 1
                if traj_adaptive_on and traj_len > 0:
                    new_mh = min(traj_adaptive_cap, traj_len * traj_adaptive_multiplier)
                    with temporal_cache_lock:
                        _max_hold_frames = new_mh
                        _consecutive_skip_count = 0

            t1 = time.time()

            # Update metrics
            with async_metrics_lock:
                async_metrics["background_s2_runs"] = async_metrics.get("background_s2_runs", 0) + 1
                async_metrics["total_s2_time"] = async_metrics.get("total_s2_time", 0.0) + (t1 - t0)
                if fresh_was_traj:
                    async_metrics["fresh_traj_outputs"] = async_metrics.get("fresh_traj_outputs", 0) + 1
                elif fresh_was_action:
                    async_metrics["fresh_action_outputs"] = async_metrics.get("fresh_action_outputs", 0) + 1

            # I-050 (Gate 3f): track last fresh output type for next-iteration gate.
            # I-046 bypass: one-shot semantics — reset flag so it doesn't re-trigger.
            # I-047 bypass: propagate actual output type — if S2 produced an action
            #   during an I-047 cycle, I-046 should fire on the next frame to refresh.
            #   This fixes the Gate 3d/3e bug where I-047 monopolized stable-scene
            #   bypasses and suppressed I-046 entirely (AA=0, V=0.17 on bag 073623).
            with temporal_cache_lock:
                if not _action_aware_enabled:
                    _last_fresh_was_action = False
                elif was_i046_bypass:
                    _last_fresh_was_action = False   # I-046 one-shot: reset after firing
                else:
                    _last_fresh_was_action = fresh_was_action  # I-047 or natural crossing

            # I-206: action-streak tracking → trajectory recovery mode
            # Append fresh output type, check if streak threshold crossed.
            # Recovery mode reduces effective max_hold → forces frequent S2 refresh
            # → increases probability of getting a trajectory output for obstacle avoidance.
            traj_recovery_activated = False
            with _traj_recovery_lock:
                if _traj_recovery_enabled:
                    _fresh_output_history.append('T' if fresh_was_traj else 'A')
                    if len(_fresh_output_history) > _traj_recovery_k:
                        _fresh_output_history.pop(0)
                    action_count = _fresh_output_history.count('A')
                    if action_count >= _traj_recovery_k and not _in_traj_recovery:
                        _in_traj_recovery = True
                        traj_recovery_activated = True
                    elif fresh_was_traj and _in_traj_recovery:
                        _in_traj_recovery = False
            if traj_recovery_activated:
                with async_metrics_lock:
                    async_metrics["traj_recovery_activations"] = (
                        async_metrics.get("traj_recovery_activations", 0) + 1)

            s2_request_queue.task_done()
            
        except Exception as e:
            print(f"[Async Loop] Error: {e}")
            try:
                s2_request_queue.task_done()
            except ValueError:
                pass
            time.sleep(0.1)

def run_s2_background(image, depth, camera_pose, instruction, intrinsic):
    """Feed input to background queue and return immediately"""
    try:
        # Non-blocking put - if queue full, drop the old frame and add new one
        try:
            s2_request_queue.put_nowait((image, depth))
        except queue.Full:
            try:
                s2_request_queue.get_nowait()
                s2_request_queue.put_nowait((image, depth))
            except queue.Empty:
                s2_request_queue.put((image, depth), timeout=0.01)
    except Exception as e:
        print(f"[S2 Background] Queue error: {e}")

def ensure_async_thread():
    """Ensure background S2 thread is running"""
    global s2_executor, async_thread_running, async_continuous_thread
    if not ASYNC_BACKGROUND_INFERENCE:
        return
    if s2_executor is None:
        async_thread_running = True
        s2_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="S2_")
        async_continuous_thread = threading.Thread(target=async_continuous_loop, daemon=True)
        async_continuous_thread.start()
        print("[Server] Started async continuous loop")

def stop_async_loop():
    """Stop the background async loop"""
    global async_thread_running, s2_executor
    async_thread_running = False
    if s2_executor:
        s2_executor.shutdown(wait=False)
        s2_executor = None
    print("[Server] Stopped async loop")

async_thread = None
SERVER_OPT_FLAGS = {
    "kv_cache": False,
    "tensorrt": False,
    "quantization": False,
    "vision_cache": False,
    "methods": [],
}


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    try:
        start_time = time.time()

        image_file = request.files['image']
        depth_file = request.files['depth']
        json_data = request.form['json']
        data = json.loads(json_data)

        try:
            image = Image.open(image_file.stream)
            image = image.convert('RGB')
            image = np.asarray(image)
        except Exception as e:
            print(f"[Server] Skip frame: cannot read image - {e}")
            return jsonify({'status': 'waiting'})

        try:
            depth = Image.open(depth_file.stream)
            depth = depth.convert('I')
            depth = np.asarray(depth)
            depth = depth.astype(np.float32) / 10000.0
        except Exception as e:
            print(f"[Server] Skip frame: cannot read depth - {e}")
            return jsonify({'status': 'waiting'})
        print(f"read http data cost {time.time() - start_time}")

        camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        #instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
        #instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
        # instruction = "Stop. Just stop. Move forward one step. Move forward two step. Turn right and stop."
        # instruction = "Go straight along the walkway and turn right at the crosswalk. Go straight to the end of the crosswalk and stop."
        # instruction = "Go straight along the walkway until you see a crosswalk. Go straight again until you see a second crosswalk. Turn right at the second crosswalk and go straight to the end of the crosswalk. Stop at the end of the crosswalk."
        instruction = "Exit door. Turn left and go straight until you find fire extinguisher. Then stop."
        policy_init = data['reset']
        req_mode = data.get('mode', SERVER_MODE)
        if req_mode != 'sync':
            print(f"[Server] Requested mode '{req_mode}' not implemented yet; using sync")
        req_opts = data.get('optimizations', {})
        if req_opts:
            print(f"[Server] Received optimization request (scaffold): {req_opts}")
        if policy_init:
            start_time = time.time()
            idx = 0
            output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
            os.makedirs(output_dir, exist_ok=True)
            print("init reset model!!!")
            with agent_lock:
                agent.reset()
            reset_dual_metrics()

        idx += 1

        # Read generation params from client
        client_temperature = data.get('temperature', 1.0)
        client_repetition_penalty = data.get('repetition_penalty', 1.0)
        
        look_down = False
        t0 = time.time()
        dual_sys_output = {}

        with agent_lock:
            dual_sys_output = agent.step(
                image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down,
                temperature=client_temperature, repetition_penalty=client_repetition_penalty
            )
            if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
                look_down = True
                dual_sys_output = agent.step(
                    image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down,
                    temperature=client_temperature, repetition_penalty=client_repetition_penalty
                )

        t1 = time.time()
        generate_time = t1 - t0
        print(f"dual sys step time: {generate_time}")

        # 클라이언트에서 보낸 idx 추출
        image_id = data.get('idx', 0)
        filename = f"frame_{image_id:05d}"
        
        has_action = dual_sys_output.output_action is not None
        has_trajectory = dual_sys_output.output_trajectory is not None

        # Quality-first policy: prefer trajectory when both are present.
        send_trajectory = has_trajectory
        send_action = (not send_trajectory) and has_action

        # ===== UPDATE METRICS (for both sync and async) =====
        with async_metrics_lock:
            async_metrics["total_requests"] = async_metrics.get("total_requests", 0) + 1
            async_metrics["http_requests"] = async_metrics.get("http_requests", 0) + 1
            async_metrics["total_http_latency"] = async_metrics.get("total_http_latency", 0.0) + generate_time

            if send_trajectory:
                async_metrics["s2_coords_outputs"] = async_metrics.get("s2_coords_outputs", 0) + 1
                async_metrics["s1_runs"] = async_metrics.get("s1_runs", 0) + 1
            elif send_action:
                async_metrics["s2_discrete_outputs"] = async_metrics.get("s2_discrete_outputs", 0) + 1

            async_metrics["s2_runs"] = async_metrics.get("s2_runs", 0) + 1
            async_metrics["s2_time_total"] = async_metrics.get("s2_time_total", 0.0) + generate_time
            async_metrics["total_action_time"] = async_metrics.get("total_action_time", 0.0) + generate_time
        
        # image_id = int(time.time() * 1000)
        # filename = f"rec_{image_id}.jpg"

        json_output = {}
        if send_trajectory:
            json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
            if dual_sys_output.output_pixel is not None:
                json_output['pixel_goal'] = dual_sys_output.output_pixel
                # annotate_image(image_id, image, 'traj', dual_sys_output.output_trajectory.tolist(), dual_sys_output.output_pixel, save_dir, filename)
            else:
                # annotate_image(image_id, image, 'traj_cached_latent', dual_sys_output.output_trajectory.tolist(), dual_sys_output.output_pixel, save_dir, filename)
                pass
        elif send_action:
            json_output['discrete_action'] = dual_sys_output.output_action
            # annotate_image(image_id, image, agent.llm_output, dual_sys_output.output_trajectory, dual_sys_output.output_pixel, save_dir, filename)
        else:
            json_output['status'] = 'waiting'

        # print(f"json_output {json_output}")
        return jsonify(json_output)
    except Exception as e:
        print(f"[Server] eval_dual exception: {repr(e)}")
        return jsonify({'status': 'waiting', 'error': str(e)})


@app.route("/eval_dual_async", methods=['POST'])
def eval_dual_async():
    """TRUE async endpoint - background continuously runs step(), HTTP returns cached immediately
    
    Architecture (TRUE async):
    - Background thread: continuously runs step() in a loop, populates cache
    - HTTP thread: returns cached output IMMEDIATELY, never waits
    - This achieves TRUE async decoupling!
    """
    global idx, output_dir, start_time, async_thread_running, async_cached_trajectory, async_cached_action
    global _traj_serve_count, _current_odom
    try:
        start_time = time.time()

        image_file = request.files['image']
        depth_file = request.files['depth']
        json_data = request.form['json']
        data = json.loads(json_data)

        try:
            image = Image.open(image_file.stream)
            image = image.convert('RGB')
            image = np.asarray(image)
        except Exception as e:
            print(f"[Server] Skip frame: cannot read image - {e}")
            return jsonify({'status': 'waiting'})

        try:
            depth = Image.open(depth_file.stream)
            depth = depth.convert('I')
            depth = np.asarray(depth)
            depth = depth.astype(np.float32) / 10000.0
        except Exception as e:
            print(f"[Server] Skip frame: cannot read depth - {e}")
            return jsonify({'status': 'waiting'})

        camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        instruction = "Exit door. Turn left and go straight until you find fire extinguisher. Then stop."
        
        # I-058: store current odometry for odom-progress-hold check
        odom_val = data.get('odom', None)
        if odom_val is not None and len(odom_val) >= 2:
            with _odom_lock:
                _current_odom = list(odom_val[:3])

        policy_init = data.get('reset', False)
        req_mode = data.get('mode', 'async')
        
        if policy_init:
            idx = 0
            with agent_lock:
                agent.reset()
                async_cached_trajectory = None
                async_cached_action = None
            reset_dual_metrics()
            ensure_async_thread()
            return jsonify({'status': 'reset'})
        
        idx += 1
        
        if req_mode == 'start_async':
            ensure_async_thread()
            return jsonify({'status': 'started'})
        
        if req_mode == 'stop_async':
            stop_async_loop()
            return jsonify({'status': 'stopped'})
        
        ensure_async_thread()
        
        t_http_start = time.time()

        # ===== TRUE ASYNC: queue frame for background, return cache immediately =====
        # agent.step() is NEVER called in the HTTP thread.
        # The background thread (async_continuous_loop) owns all inference.
        run_s2_background(image, depth, camera_pose, instruction, args.camera_intrinsic)

        with async_cache_lock:
            cached_traj = async_cached_trajectory
            cached_action = async_cached_action

        t_http_end = time.time()
        http_latency = t_http_end - t_http_start

        # ===== UPDATE METRICS =====
        with async_metrics_lock:
            async_metrics["http_requests"] = async_metrics.get("http_requests", 0) + 1
            async_metrics["total_requests"] = async_metrics.get("total_requests", 0) + 1
            async_metrics["total_http_latency"] = async_metrics.get("total_http_latency", 0.0) + http_latency
            async_metrics["total_action_time"] = async_metrics.get("total_action_time", 0.0) + http_latency
            if cached_traj is not None:
                async_metrics["s2_coords_outputs"] = async_metrics.get("s2_coords_outputs", 0) + 1
            elif cached_action is not None:
                async_metrics["s2_discrete_outputs"] = async_metrics.get("s2_discrete_outputs", 0) + 1

        if cached_traj is not None:
            # I-052: count HTTP responses serving current cached trajectory
            with _serve_count_lock:
                _traj_serve_count += 1
            json_output = {'trajectory': cached_traj}
        elif cached_action is not None:
            # Reset serve count when action is served (plan consumed to decision point)
            with _serve_count_lock:
                _traj_serve_count = 0
            json_output = {'discrete_action': cached_action}
        else:
            # Gate 3c (I-010): track cold-start "waiting" responses
            with async_metrics_lock:
                async_metrics["waiting_responses"] = async_metrics.get("waiting_responses", 0) + 1
            json_output = {'status': 'waiting'}

        # Phase 3X (I-111/I-112): log response type for offline sequence analysis
        with async_metrics_lock:
            seq = async_metrics.get("response_sequence")
            if seq is not None and len(seq) < 10000:
                if cached_traj is not None:
                    seq.append('T')
                elif cached_action is not None:
                    act = cached_action
                    seq.append(act[0] if isinstance(act, list) and act else str(act))
                else:
                    seq.append('W')

        return jsonify(json_output)
    except Exception as e:
        import traceback
        print(f"[Server] eval_dual_async exception: {repr(e)}")
        traceback.print_exc()
        return jsonify({'status': 'waiting', 'error': str(e)})


@app.route("/reset_metrics", methods=['GET'])
def reset_metrics_endpoint():
    """Reliably reset all experiment metrics without needing a valid image."""
    reset_dual_metrics()
    return jsonify({'status': 'reset', 'start_time': async_metrics.get('start_time', 0)})


@app.route("/async_metrics", methods=['GET'])
def get_async_metrics():
    """Get async performance metrics
    
    Metrics designed for clear audience understanding:
    
    SYSTEM 2 (S2) - Language Planner:
    - S2 req_hz: How many S2 inferences per second
    - S2 latency: Time per S2 inference (in ms)
    
    SYSTEM 1 (S1) - Trajectory Generator:
    - S1 req_hz: How many trajectory generations per second
    - S1 latency: Time per trajectory generation (in ms)
    
    JOINT (S1 + S2) - Combined:
    - Joint req_hz: Total HTTP requests processed per second
    - Joint latency: Total time from request to response (in ms)
    
    Quality:
    - Trajectory ratio: % of outputs that are trajectories (high is better)
    - Discrete ratio: % of outputs that are discrete actions
    """
    with async_metrics_lock:
        m = async_metrics.copy()
        start_time = async_metrics.get("start_time", time.time())
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    if elapsed < 0.1:
        elapsed = 1.0  # Prevent division by zero
    
    # ===== SYSTEM 2 (Language Planner) Metrics =====
    s2_runs = m.get("s2_runs", 0)
    s2_time_total = m.get("s2_time_total", 0.0)
    # In async mode, sync s2_runs is always 0; fall back to background thread counters.
    bg_runs = m.get("background_s2_runs", 0)
    bg_time = m.get("total_s2_time", 0.0)
    effective_runs = s2_runs if s2_runs > 0 else bg_runs
    effective_time = s2_time_total if s2_runs > 0 else bg_time

    if effective_runs > 0:
        s2_latency_s = effective_time / effective_runs  # seconds per S2 inference
        s2_latency_ms = s2_latency_s * 1000             # ms per S2 inference
        s2_req_hz = effective_runs / elapsed             # S2 inferences per second
    else:
        s2_latency_ms = 0.0
        s2_req_hz = 0.0
    
    # ===== SYSTEM 1 (Trajectory Generator) Metrics =====
    s1_runs = m.get("s1_runs", 0)
    s1_time_total = m.get("s1_time_total", 0.0)
    
    if s1_runs > 0:
        s1_req_hz = s1_runs / elapsed          # S1 generations per second
    else:
        s1_req_hz = 0.0

    # NOTE: Without internal instrumentation in agent.step(), S1 latency cannot be
    # isolated precisely from S2 latency. Keep this explicit for audience clarity.
    s1_latency_ms = None
    
    # ===== JOINT (Combined S1+S2) Metrics =====
    total_requests = m.get("total_requests", 0)
    total_action_time = m.get("total_action_time", 0.0)
    
    if total_requests > 0:
        joint_latency_s = total_action_time / total_requests  # seconds per request
        joint_latency_ms = joint_latency_s * 1000        # ms per request
        joint_req_hz = total_requests / elapsed            # requests per second
    else:
        joint_latency_ms = 0.0
        joint_req_hz = 0.0
    
    # ===== QUALITY Metrics =====
    s2_coords = m.get("s2_coords_outputs", 0)
    s2_discrete = m.get("s2_discrete_outputs", 0)
    total_outputs = s2_coords + s2_discrete
    
    if total_outputs > 0:
        trajectory_ratio = (s2_coords / total_outputs) * 100  # %
        discrete_ratio = (s2_discrete / total_outputs) * 100   # %
    else:
        trajectory_ratio = 0.0
        discrete_ratio = 0.0
    
    # Build response
    metrics = {
        # System 2 - Language Planner
        "s2_req_hz": round(s2_req_hz, 2),
        "s2_latency_ms": round(s2_latency_ms, 2),
        
        # System 1 - Trajectory Generator
        "s1_req_hz": round(s1_req_hz, 2),
        "s1_latency_ms": s1_latency_ms,
        
        # Joint - Combined System
        "joint_req_hz": round(joint_req_hz, 2),
        "joint_latency_ms": round(joint_latency_ms, 2),
        
        # Quality
        "trajectory_ratio": round(trajectory_ratio, 1),
        "discrete_ratio": round(discrete_ratio, 1),
        
        # Raw counts
        "total_requests": total_requests,
        "s2_runs": s2_runs,
        "s1_runs": s1_runs,
        "trajectories": s2_coords,
        "discrete_actions": s2_discrete,
        "elapsed_seconds": round(elapsed, 1),
        "background_s2_runs": m.get("background_s2_runs", 0),
        "async_background_enabled": ASYNC_BACKGROUND_INFERENCE,

        # Phase 3 Task #9: Temporal caching diagnostics
        "temporal_cache_threshold": temporal_cache_threshold,
        "temporal_cache_skips": m.get("temporal_cache_skips", 0),
        "temporal_cache_skip_ratio": round(
            m.get("temporal_cache_skips", 0) /
            max(1, m.get("temporal_cache_skips", 0) + m.get("background_s2_runs", 0)) * 100, 1
        ),

        # Fresh-only output type ratio (over actual S2 inference calls only).
        # Compare against trajectory_ratio: if they match, the cache replay is
        # representative; if they diverge, traj_ratio is biased by hold time.
        "fresh_traj_outputs": m.get("fresh_traj_outputs", 0),
        "fresh_action_outputs": m.get("fresh_action_outputs", 0),
        "fresh_trajectory_ratio": round(
            m.get("fresh_traj_outputs", 0) /
            max(1, m.get("fresh_traj_outputs", 0) + m.get("fresh_action_outputs", 0)) * 100, 1
        ),

        # I-046 / I-047 forced-fresh diagnostics
        "action_aware_bypasses": m.get("action_aware_bypasses", 0),
        "max_hold_bypasses": m.get("max_hold_bypasses", 0),
        "max_hold_frames": _max_hold_frames,
        "action_aware_enabled": _action_aware_enabled,
        # I-049: adaptive max_hold (Gate 3e)
        "adaptive_max_hold_enabled": _adaptive_max_hold_enabled,
        "adaptive_sim_window_size": len(_similarity_window),
        "adaptive_sim_variance": float(np.var(_similarity_window[-_ADAPTIVE_WINDOW_SIZE:])) if len(_similarity_window) >= _ADAPTIVE_WINDOW_SIZE else None,

        # I-052: request-count-adaptive hold (Gate 3i)
        "serve_count_bypass_enabled": _serve_count_bypass_enabled,
        "serve_hold_threshold": _serve_hold_threshold,
        "traj_serve_count": _traj_serve_count,
        "serve_count_bypasses": m.get("serve_count_bypasses", 0),

        # I-051: trajectory-length-adaptive max_hold (Gate 3h)
        "traj_adaptive_hold_enabled": _traj_adaptive_hold_enabled,
        "traj_adaptive_hold_multiplier": _traj_adaptive_hold_multiplier,
        "traj_adaptive_hold_cap": _traj_adaptive_hold_cap,
        "avg_trajectory_length": round(
            m.get("traj_lengths_sum", 0) / max(1, m.get("traj_lengths_count", 1)), 2
        ),

        # Gate 3c (I-010): cold-start diagnostics
        "waiting_responses": m.get("waiting_responses", 0),
        "pre_warm_frames_queued": m.get("pre_warm_frames_queued", 0),

        # I-053/I-055: EMA scene fingerprint (Gate 3k/3m)
        "ema_fingerprint_enabled": _ema_fingerprint_enabled,
        "ema_fingerprint_alpha": _ema_fingerprint_alpha,
        "ema_fingerprint_initialized": _ema_fingerprint is not None,
        "ema_transition_reset": _ema_transition_reset,

        # I-054: similarity slope predictive refresh (Gate 3l)
        "slope_predict_enabled": _slope_predict_enabled,
        "slope_threshold": _slope_threshold,
        "slope_window": _slope_window,
        "slope_predict_bypasses": m.get("slope_predict_bypasses", 0),

        # I-058: odometry-progress hold (Gate 3p)
        "odom_progress_enabled": _odom_progress_enabled,
        "odom_progress_threshold": _odom_progress_threshold,
        "odom_progress_bypasses": m.get("odom_progress_bypasses", 0),

        # I-202: optical flow cache invalidation (Gate 7)
        "flow_bypass_enabled": _flow_bypass_enabled,
        "flow_magnitude_threshold": _flow_magnitude_threshold,
        "flow_bypasses": m.get("flow_bypasses", 0),

        # I-203 (Gate 8): EMA warm-up acceleration
        "ema_warmup_enabled": _ema_warmup_enabled,
        "ema_warmup_frames": _ema_warmup_frames,
        "ema_warmup_alpha": _ema_warmup_alpha,

        # I-204 (Gate 9): cosine similarity variance gating
        "var_gate_enabled": _var_gate_enabled,
        "var_gate_sigma": _var_gate_sigma,
        "var_gate_window": _var_gate_window,
        "var_gate_bypasses": m.get("var_gate_bypasses", 0),

        # I-206 (Gate 11): action-streak trajectory recovery
        "traj_recovery_enabled": _traj_recovery_enabled,
        "traj_recovery_k": _traj_recovery_k,
        "traj_recovery_hold": _traj_recovery_hold,
        "in_traj_recovery": _in_traj_recovery,
        "traj_recovery_activations": m.get("traj_recovery_activations", 0),

        # Phase 3X (I-111/I-112): per-request response-type sequence
        "response_sequence": m.get("response_sequence", []),

        # Gate 6a: runtime max_new_tokens (S2 inference latency control)
        "max_new_tokens": agent.max_new_tokens if agent is not None else args.max_new_tokens,
    }

    return jsonify(metrics)


@app.route("/set_temporal_threshold", methods=['POST', 'GET'])
def set_temporal_threshold_endpoint():
    """Set the temporal cache cosine-similarity threshold without restarting.
    GET ?threshold=0.95  or  POST {threshold: 0.95}
    threshold=0.0 disables the cache (always run agent.step())."""
    global temporal_cache_threshold, _last_fingerprint
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        new_t = float(data.get('threshold', 0.0))
    else:
        new_t = float(request.args.get('threshold', 0.0))
    new_t = max(0.0, min(1.0, new_t))
    with temporal_cache_lock:
        temporal_cache_threshold = new_t
        _last_fingerprint = None  # reset so first frame after change always processes
    with async_metrics_lock:
        async_metrics["temporal_cache_threshold"] = new_t
    return jsonify({'status': 'ok', 'temporal_cache_threshold': new_t})


@app.route("/set_max_hold_frames", methods=['POST', 'GET'])
def set_max_hold_frames_endpoint():
    """I-047: configure the max number of consecutive cache skips before
    forcing a fresh S2 inference. Lower = more frequent forced refresh
    (safer but less compute saving). Higher = looser staleness bound.
    GET ?frames=10  or  POST {frames: 10}. 0 disables the bound."""
    global _max_hold_frames, _consecutive_skip_count
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        new_n = int(data.get('frames', 10))
    else:
        new_n = int(request.args.get('frames', 10))
    new_n = max(0, min(10000, new_n))
    with temporal_cache_lock:
        _max_hold_frames = new_n
        _consecutive_skip_count = 0
    return jsonify({'status': 'ok', 'max_hold_frames': new_n})


@app.route("/set_action_aware", methods=['POST', 'GET'])
def set_action_aware_endpoint():
    """Gate 3d ablation: toggle I-046 action-aware cache bypass on/off.
    GET ?enabled=true|false. When false, the cache never forces an S2 refresh
    after an action output — only similarity threshold + max_hold apply."""
    global _action_aware_enabled, _last_fresh_was_action
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    with temporal_cache_lock:
        _action_aware_enabled = enabled
        if not enabled:
            _last_fresh_was_action = False
    return jsonify({'status': 'ok', 'action_aware_enabled': enabled})


@app.route("/set_adaptive_max_hold", methods=['POST', 'GET'])
def set_adaptive_max_hold_endpoint():
    """I-049 (Gate 3e): Toggle adaptive max_hold based on scene stability.
    When enabled, max_hold automatically adjusts: low in stable scenes (so I-046
    can fire), high in dynamic scenes (to avoid excess S2 invocations).
    ?enabled=true|false
    """
    global _adaptive_max_hold_enabled, _similarity_window
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    with temporal_cache_lock:
        _adaptive_max_hold_enabled = enabled
        if not enabled:
            _similarity_window.clear()
    return jsonify({'status': 'ok', 'adaptive_max_hold_enabled': enabled})


@app.route("/set_trajectory_adaptive_hold", methods=['POST', 'GET'])
def set_trajectory_adaptive_hold_endpoint():
    """I-051 (Gate 3h): Toggle trajectory-length-adaptive max_hold.
    When enabled, each time a new trajectory is cached, max_hold is updated to
    min(cap, len(trajectory) * multiplier).  Long plans → longer hold; short
    plans → shorter hold.  This couples staleness budget to semantic plan richness.
    GET ?enabled=true|false&multiplier=7&cap=50"""
    global _traj_adaptive_hold_enabled, _traj_adaptive_hold_multiplier, _traj_adaptive_hold_cap
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    multiplier = int(request.args.get('multiplier', _traj_adaptive_hold_multiplier))
    cap = int(request.args.get('cap', _traj_adaptive_hold_cap))
    multiplier = max(1, min(100, multiplier))
    cap = max(1, min(1000, cap))
    with temporal_cache_lock:
        _traj_adaptive_hold_enabled = enabled
        _traj_adaptive_hold_multiplier = multiplier
        _traj_adaptive_hold_cap = cap
    return jsonify({
        'status': 'ok',
        'traj_adaptive_hold_enabled': enabled,
        'traj_adaptive_hold_multiplier': multiplier,
        'traj_adaptive_hold_cap': cap,
    })


@app.route("/set_serve_count_hold", methods=['POST', 'GET'])
def set_serve_count_hold_endpoint():
    """I-052 (Gate 3i): Toggle request-count-adaptive hold.
    When enabled, force S2 refresh after _serve_hold_threshold consecutive
    HTTP responses serving the same cached trajectory.  This measures plan
    consumption by robot control steps rather than background-loop skip count,
    making it independent of model trajectory length.
    GET ?enabled=true|false&threshold=15"""
    global _serve_count_bypass_enabled, _serve_hold_threshold, _traj_serve_count
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    threshold = int(request.args.get('threshold', _serve_hold_threshold))
    threshold = max(1, min(10000, threshold))
    with _serve_count_lock:
        _serve_count_bypass_enabled = enabled
        _serve_hold_threshold = threshold
        _traj_serve_count = 0
    return jsonify({
        'status': 'ok',
        'serve_count_bypass_enabled': enabled,
        'serve_hold_threshold': threshold,
    })


@app.route("/set_ema_fingerprint", methods=['POST', 'GET'])
def set_ema_fingerprint_endpoint():
    """I-053 (Gate 3k) / I-055 (Gate 3m): Toggle EMA scene fingerprint.
    I-053: standard EMA reference — slow update on ALL frames including forced bypasses.
    I-055 (transition_reset=true): Transition-Reset EMA — hard-reset EMA to current frame
    after each forced bypass (I-046/I-047/I-052), preventing the post-bypass lag cascade.
    GET ?enabled=true|false&alpha=0.15&transition_reset=false"""
    global _ema_fingerprint_enabled, _ema_fingerprint_alpha, _ema_fingerprint, _ema_transition_reset
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    alpha = float(request.args.get('alpha', _ema_fingerprint_alpha))
    alpha = max(0.01, min(1.0, alpha))
    tr_str = request.args.get('transition_reset', 'false').lower()
    tr_reset = tr_str in ('1', 'true', 'yes', 'on')
    with _ema_fp_lock:
        _ema_fingerprint_enabled = enabled
        _ema_fingerprint_alpha = alpha
        _ema_transition_reset = tr_reset
        _ema_fingerprint = None  # reset EMA on config change
    return jsonify({
        'status': 'ok',
        'ema_fingerprint_enabled': enabled,
        'ema_fingerprint_alpha': alpha,
        'ema_transition_reset': tr_reset,
    })


@app.route("/set_slope_predict", methods=['POST', 'GET'])
def set_slope_predict_endpoint():
    """I-054 (Gate 3l): Toggle similarity-slope predictive refresh.
    When enabled, fires a pre-emptive S2 run when the similarity score is
    declining rapidly (d(sim)/dt < -threshold), even if sim >= tau.
    This anticipates cache invalidation before it occurs (doorway transitions,
    turns) rather than reacting after the threshold is crossed.
    GET ?enabled=true|false&threshold=0.015&window=3"""
    global _slope_predict_enabled, _slope_threshold, _slope_window
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    threshold = float(request.args.get('threshold', _slope_threshold))
    threshold = max(0.001, min(0.5, threshold))
    window = int(request.args.get('window', _slope_window))
    window = max(1, min(20, window))
    with _slope_history_lock:
        _slope_predict_enabled = enabled
        _slope_threshold = threshold
        _slope_window = window
        _similarity_history.clear()
    return jsonify({
        'status': 'ok',
        'slope_predict_enabled': enabled,
        'slope_threshold': threshold,
        'slope_window': window,
    })


@app.route("/set_max_new_tokens", methods=['POST', 'GET'])
def set_max_new_tokens_endpoint():
    """Gate 6a (I-200): Change max_new_tokens for S2 inference at runtime.
    Reduces model generation length → lower S2 latency at cost of possible output truncation.
    GET ?tokens=32  (range: 8–128)"""
    global agent
    tokens = int(request.args.get('tokens', 80))
    tokens = max(8, min(128, tokens))
    old = agent.max_new_tokens if agent is not None else -1
    if agent is not None:
        agent.max_new_tokens = tokens
    return jsonify({'status': 'ok', 'max_new_tokens': tokens, 'previous': old})


@app.route("/set_odom_progress_hold", methods=['POST', 'GET'])
def set_odom_progress_hold_endpoint():
    """I-058 (Gate 3p): Toggle odometry-progress hold.
    Forces a fresh S2 run whenever the robot has traveled >= threshold meters
    since the last completed S2. Orthogonal to temporal (I-047) and count (I-052) signals.
    GET ?enabled=true|false&threshold=0.5"""
    global _odom_progress_enabled, _odom_progress_threshold, _last_s2_odom
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    threshold = float(request.args.get('threshold', _odom_progress_threshold))
    threshold = max(0.05, min(10.0, threshold))
    _odom_progress_enabled = enabled
    _odom_progress_threshold = threshold
    with _odom_lock:
        _last_s2_odom = None  # reset so first frame always runs S2
    return jsonify({
        'status': 'ok',
        'odom_progress_enabled': enabled,
        'odom_progress_threshold': threshold,
    })


@app.route("/set_flow_bypass", methods=['POST', 'GET'])
def set_flow_bypass_endpoint():
    """I-202 (Gate 7): Toggle optical flow cache invalidation.
    Forces S2 refresh when mean frame-to-frame flow magnitude > threshold (80x60 px).
    GET ?enabled=true|false&threshold=20.0"""
    global _flow_bypass_enabled, _flow_magnitude_threshold, _prev_flow_frame
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    threshold = float(request.args.get('threshold', _flow_magnitude_threshold))
    threshold = max(0.1, min(200.0, threshold))
    _flow_bypass_enabled = enabled
    _flow_magnitude_threshold = threshold
    with _flow_frame_lock:
        _prev_flow_frame = None  # reset frame reference on config change
    return jsonify({
        'status': 'ok',
        'flow_bypass_enabled': enabled,
        'flow_magnitude_threshold': threshold,
    })


@app.route("/set_ema_warmup", methods=['POST', 'GET'])
def set_ema_warmup_endpoint():
    """I-203 (Gate 8): EMA warm-up acceleration.
    After each TR-EMA reset, use alpha_warm for the first warmup_frames frames,
    then drop to the production alpha. Accelerates post-bypass EMA convergence.
    GET ?enabled=true|false&warmup_frames=10&alpha_warm=0.5"""
    global _ema_warmup_enabled, _ema_warmup_frames, _ema_warmup_alpha, _ema_frames_since_reset
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    warmup_frames = int(request.args.get('warmup_frames', _ema_warmup_frames))
    warmup_frames = max(1, min(100, warmup_frames))
    alpha_warm = float(request.args.get('alpha_warm', _ema_warmup_alpha))
    alpha_warm = max(0.1, min(1.0, alpha_warm))
    with _ema_warmup_lock:
        _ema_warmup_enabled = enabled
        _ema_warmup_frames = warmup_frames
        _ema_warmup_alpha = alpha_warm
        _ema_frames_since_reset = 0  # reset counter on config change
    return jsonify({
        'status': 'ok',
        'ema_warmup_enabled': enabled,
        'ema_warmup_frames': warmup_frames,
        'ema_warmup_alpha': alpha_warm,
    })


@app.route("/set_var_gate", methods=['POST', 'GET'])
def set_var_gate_endpoint():
    """I-204 (Gate 9): Cosine similarity variance gating.
    Forces S2 refresh when std(sim_history[-W:]) > sigma, catching oscillatory
    scenes where current sim >= τ but recent sim history is unstable.
    GET ?enabled=true|false&sigma=0.03&window=5"""
    global _var_gate_enabled, _var_gate_sigma, _var_gate_window
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    sigma = float(request.args.get('sigma', _var_gate_sigma))
    sigma = max(0.001, min(1.0, sigma))
    window = int(request.args.get('window', _var_gate_window))
    window = max(2, min(30, window))
    with _var_gate_lock:
        _var_gate_enabled = enabled
        _var_gate_sigma = sigma
        _var_gate_window = window
    return jsonify({
        'status': 'ok',
        'var_gate_enabled': enabled,
        'var_gate_sigma': sigma,
        'var_gate_window': window,
    })


@app.route("/set_traj_recovery", methods=['POST', 'GET'])
def set_traj_recovery_endpoint():
    """I-206 (Gate 11): Action-streak trajectory recovery mode.
    When K consecutive fresh S2 outputs are action-type, enter recovery mode:
    max_hold is reduced to recovery_hold (default 3) to force frequent refreshes
    until a trajectory output is received. Maximizes fresh_traj_ratio for obstacle avoidance.
    GET ?enabled=true|false&k=5&hold=3"""
    global _traj_recovery_enabled, _traj_recovery_k, _traj_recovery_hold
    enabled_str = request.args.get('enabled', 'true').lower()
    enabled = enabled_str in ('1', 'true', 'yes', 'on')
    k = int(request.args.get('k', _traj_recovery_k))
    k = max(1, min(50, k))
    hold = int(request.args.get('hold', _traj_recovery_hold))
    hold = max(1, min(15, hold))
    with _traj_recovery_lock:
        _traj_recovery_enabled = enabled
        _traj_recovery_k = k
        _traj_recovery_hold = hold
        _fresh_output_history.clear()
        _in_traj_recovery = False
    return jsonify({
        'status': 'ok',
        'traj_recovery_enabled': enabled,
        'traj_recovery_k': k,
        'traj_recovery_hold': hold,
    })


def annotate_image(idx, image, llm_output, trajectory, pixel_goal, output_dir, filename):
    import matplotlib
    matplotlib.use('Agg')  # 반드시 pyplot을 import하기 전에 실행해야 합니다.
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if 'look_down' not in filename:
        filename = f'{filename}_z'
    image = Image.fromarray(image)#.save(f'rgb_{idx}.png')
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    text_content = []
    text_content.append(f"Frame    Id  : {idx}")
    text_content.append(f"Actions      : {llm_output}" )
    max_width = 0
    total_height = 0
    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = 26
        max_width = max(max_width, text_width)
        total_height += text_height

    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding

    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

    text_color = 'white'
    y_position = box_y + padding
    
    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = 26
        y_position += text_height
    image = np.array(image)
    
    # Draw trajectory visualization in the top-right corner using matplotlib
    if trajectory is not None and len(trajectory) > 0:
        img_height, img_width = image.shape[:2]
        
        # Window parameters
        window_size = 200  # Window size in pixels
        window_margin = 0  # Margin from edge
        window_x = img_width - window_size - window_margin
        window_y = window_margin
        
        # Extract trajectory points
        traj_points = []
        for point in trajectory:
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                traj_points.append([float(point[0]), float(point[1])])
        
        if len(traj_points) > 0:
            traj_array = np.array(traj_points)
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            fig.patch.set_alpha(0.6)  # Semi-transparent background
            fig.patch.set_facecolor('gray')
            ax.set_facecolor('lightgray')
            
            # Plot trajectory
            # Coordinate system: x-axis points up, y-axis points left
            # Origin at bottom center
            ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')
            
            # Mark start point (green) and end point (red)
            ax.plot(y_coords[0], x_coords[0], 'go', markersize=6, label='Start')
            ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=6, label='End')
            
            # Mark origin
            ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=2, label='Origin')
            
            # Set axis labels
            ax.set_xlabel('Y (left +)', fontsize=8)
            ax.set_ylabel('X (up +)', fontsize=8)
            ax.invert_xaxis()
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
            
            # Add legend
            ax.legend(fontsize=6, loc='upper right')
            
            # Adjust layout
            plt.tight_layout(pad=0.3)
            
            # Convert matplotlib figure to numpy array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            # plot_img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            # plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # plt.close(fig)

            # Patch for RGBA to RGB conversion
            # 1. 버퍼로부터 RGBA(4채널) 데이터를 가져옵니다.
            rgba_buffer = canvas.buffer_rgba()
            plot_img = np.frombuffer(rgba_buffer, dtype=np.uint8)
            
            # 2. 4채널 모양으로 먼저 reshape 합니다.
            # (height, width, 4) 형태로 복원
            width, height = canvas.get_width_height()
            plot_img = plot_img.reshape((height, width, 4))
            
            # 3. OpenCV를 이용해 RGBA를 RGB로 변환합니다. (Alpha 채널 제거)
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)
            
            # 메모리 해제
            plt.close(fig)

            # Resize plot to fit window
            plot_img = cv2.resize(plot_img, (window_size, window_size))
            
            # Overlay plot on image
            image[window_y:window_y+window_size, window_x:window_x+window_size] = plot_img
    
    if pixel_goal is not None:
        cv2.circle(image, (pixel_goal[1], pixel_goal[0]), 5, (255, 0, 0), -1)
    image = Image.fromarray(image).convert('RGB')
    image.save(f'{output_dir}/{filename}.jpg')
    # to numpy array
    return np.array(image)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-w-NavDP")
    parser.add_argument("--resize_w", type=int, default=256)
    parser.add_argument("--resize_h", type=int, default=256)
    parser.add_argument("--num_history", type=int, default=1)
    parser.add_argument("--plan_step_gap", type=int, default=12)
    parser.add_argument("--mode", type=str, default="sync", choices=["sync", "async"],
                        help="Execution mode. async is scaffold-only for now.")
    parser.add_argument("--kv-cache", action="store_true", help="Enable KV-cache optimization (scaffold flag).")
    parser.add_argument("--tensorrt", action="store_true", help="Enable TensorRT optimization (scaffold flag).")
    parser.add_argument("--quantization", action="store_true", help="Enable quantization optimization (scaffold flag).")
    parser.add_argument("--quant-method", type=str, default="dynamic", choices=["dynamic", "static", "qat"],
                        help="Quantization method (safe mode currently supports dynamic CPU fallback).")
    parser.add_argument("--tensorrt-engine", type=str, default="",
                        help="Path to TensorRT engine (optional, safe fallback if unavailable).")
    parser.add_argument("--vision-cache", action="store_true", help="Enable vision-cache optimization (scaffold flag).")
    parser.add_argument("--max-new-tokens", type=int, default=80,
                        help="Max new tokens for language generation.")
    parser.add_argument("--require-flash-attn", action="store_true", default=True,
                        help="Require FlashAttention-2 at runtime (enabled by default).")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 matmul/cudnn where supported.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for language model generation.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for language model generation.")
    parser.add_argument("--method", action="append", default=[],
                        help="Additional optimization method tag (repeatable, scaffold only).")
    parser.add_argument("--calib", type=str, default="/home/gdr/gd_vln/workspace/src/InternNav/scripts/realworld/calib/calib_scout.txt",
                        help="Path to calibration file (e.g. calib/calib_scout.txt)")
    parser.add_argument("--port", type=int, default=5802,
                        help="HTTP server port (default 5802). Use a different port for parallel experiments.")
    parser.add_argument("--pre-warm-frames", type=int, default=0,
                        help="Gate 3c (I-010): queue N synthetic S2 frames at startup to pre-populate "
                             "the async cache before first real request. Eliminates cold-start 'waiting' "
                             "responses. Default 0 (off). Recommend 3 for rosbag experiments.")
    args = parser.parse_args()

    SERVER_MODE = args.mode
    if args.mode == 'async':
        print(f"[Server] Using async mode")
    SERVER_OPT_FLAGS = {
        "kv_cache": bool(args.kv_cache),
        "tensorrt": bool(args.tensorrt),
        "quantization": bool(args.quantization),
        "quant_method": str(args.quant_method),
        "tensorrt_engine": str(args.tensorrt_engine),
        "tf32": bool(args.tf32),
        "vision_cache": bool(args.vision_cache),
        "methods": list(args.method),
    }
    if any([args.kv_cache, args.tensorrt, args.quantization, args.vision_cache, len(args.method) > 0]):
        print(f"[Server] Optimization flags enabled (scaffold only): {SERVER_OPT_FLAGS}")

    calib = Calibration(args.calib)
    args.camera_intrinsic = np.array([
        [calib.f_u, 0.0,      calib.c_u, 0.0],
        [0.0,       calib.f_v, calib.c_v, 0.0],
        [0.0,       0.0,      1.0,       0.0],
        [0.0,       0.0,      0.0,       1.0],
    ])
    print(f"[Server] Loaded calib: {args.calib}")
    print(f"[Server] camera_intrinsic fx={calib.f_u:.2f} fy={calib.f_v:.2f} cx={calib.c_u:.2f} cy={calib.c_v:.2f}")
    agent = InternVLAN1AsyncAgent(args)
    # agent.step(
    #     np.zeros((480, 640, 3)),
    #     np.zeros((480, 640)),
    #     np.eye(4),
    #     "hello",
    #     args.camera_intrinsic,
    # )
    # Warmup can trigger CUDA asserts on some checkpoints/runtime combos; skip by default.
    if os.environ.get("INTERNNAV_SERVER_WARMUP", "0") == "1":
        agent.step(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640)),
            np.eye(4),
            "hello",
            args.camera_intrinsic,
        )
        agent.reset()

    # Gate 3c (I-010): cold-start pre-fetch
    # Run N synchronous agent.step() calls before starting the async thread.
    # This warms CUDA JIT compilation (~3s first inference → ~230ms thereafter)
    # without seeding the cache: results are discarded, async thread starts clean.
    if args.mode == 'async' and args.pre_warm_frames > 0:
        synthetic_image = np.zeros((480, 640, 3), dtype=np.uint8)
        synthetic_depth = np.zeros((480, 640), dtype=np.float32)
        synthetic_pose = np.eye(4)
        print(f"[Server] Gate 3c pre-warm: running {args.pre_warm_frames} synchronous warm-up inferences...")
        for i in range(args.pre_warm_frames):
            try:
                agent.step(synthetic_image, synthetic_depth, synthetic_pose,
                           "pre-warm", args.camera_intrinsic)
                print(f"[Server] Gate 3c pre-warm: frame {i+1}/{args.pre_warm_frames} done")
            except Exception as e:
                print(f"[Server] Gate 3c pre-warm: frame {i+1} error (non-fatal): {e}")
        # Reset agent state: clears KV cache + history contaminated by zero-frame inputs.
        # CUDA compiled kernels remain cached (warmup preserved), only Python-level
        # state is cleared so the first real frame gets an unbiased S2 output.
        agent.reset()
        with async_metrics_lock:
            async_metrics["pre_warm_frames_queued"] = args.pre_warm_frames
        print(f"[Server] Gate 3c pre-warm: CUDA warmed + agent reset (clean state)")
        # Async thread starts NOW, after prewarm — CUDA hot, agent state clean.
    if args.mode == 'async':
        ensure_async_thread()

    app.run(host='0.0.0.0', port=args.port)
