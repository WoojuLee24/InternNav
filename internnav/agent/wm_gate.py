"""Failure-aware S2 invocation gate.

Uses the WM head's prediction error as a proxy for S1 plan confidence.
Fires S2 only when uncertainty exceeds a calibrated threshold (RL²-VLA analog).
Provides a Pareto frontier between SR and S2-invocation-rate.
"""
import collections

import torch
import torch.nn.functional as F


class FailureAwareGate:
    """Decides per-step whether to invoke S2 (the slow VLM planner).

    Algorithm:
    - Maintains a rolling buffer of visual feature vectors from rgb_model
    - On each step, queries the WM head for prediction error (uncertainty)
    - If uncertainty > threshold OR forced_interval steps have elapsed → fire S2
    - threshold is calibrated with conformal coverage guarantee

    Args:
        wm_head: PredictiveWorldModelHead instance
        threshold: prediction error threshold (calibrate offline with conformal method)
        forced_interval: max steps before forcing S2 regardless (safety backstop)
        buffer_len: how many past frames to keep for uncertainty estimation
    """

    def __init__(self, wm_head, threshold=0.05, forced_interval=30, buffer_len=8):
        self.wm_head = wm_head
        self.threshold = threshold
        self.forced_interval = forced_interval
        self.buffer_len = buffer_len
        self._feat_buffer = collections.deque(maxlen=buffer_len)
        self._steps_since_s2 = 0
        self._total_steps = 0
        self._s2_calls = 0

    def reset(self):
        self._feat_buffer.clear()
        self._steps_since_s2 = 0
        self._total_steps = 0
        self._s2_calls = 0

    def update_features(self, visual_feats):
        """Call each step with current rgb_model features (B, N, 384)."""
        self._feat_buffer.append(visual_feats.detach().cpu())
        self._total_steps += 1

    def should_fire_s2(self, force=False):
        """Returns True if S2 should be invoked this step."""
        if force:
            self._steps_since_s2 = 0
            self._s2_calls += 1
            return True

        if self._steps_since_s2 >= self.forced_interval:
            self._steps_since_s2 = 0
            self._s2_calls += 1
            return True

        # Not enough history yet — fire S2 on first step only
        if len(self._feat_buffer) < max(self.wm_head.horizons) + 1:
            if self._steps_since_s2 == 0:
                self._s2_calls += 1
                return True
            self._steps_since_s2 += 1
            return False

        uncertainty = self._compute_uncertainty()
        self._steps_since_s2 += 1

        if uncertainty is not None and uncertainty.item() > self.threshold:
            self._steps_since_s2 = 0
            self._s2_calls += 1
            return True
        return False

    def _compute_uncertainty(self):
        feat_seq = list(self._feat_buffer)
        device = next(self.wm_head.parameters()).device
        dtype = next(self.wm_head.parameters()).dtype

        current = feat_seq[-1].to(device=device, dtype=dtype).unsqueeze(0)  # (1, N, 384)
        errors = []
        for i, horizon in enumerate(self.wm_head.horizons):
            if len(feat_seq) > horizon:
                actual = feat_seq[-horizon].to(device=device, dtype=dtype).unsqueeze(0)
                with torch.no_grad():
                    preds = self.wm_head(current)  # (1, len(horizons), N, 384)
                err = F.mse_loss(preds[:, i].float(), actual.float())
                errors.append(err)
        if not errors:
            return None
        return torch.stack(errors).mean()

    @property
    def invocation_rate(self):
        if self._total_steps == 0:
            return 0.0
        return self._s2_calls / self._total_steps

    def calibrate_threshold(self, val_uncertainties, coverage=0.9):
        """Conformal calibration: set threshold at (1-coverage) quantile of failure uncertainties.

        val_uncertainties: list of (uncertainty_scalar, did_fail: bool) from a calibration run
        Returns: calibrated threshold float
        """
        fail_uncertainties = sorted(u for u, failed in val_uncertainties if failed)
        if not fail_uncertainties:
            return self.threshold
        idx = max(0, int(len(fail_uncertainties) * (1 - coverage)))
        self.threshold = fail_uncertainties[idx]
        return self.threshold
