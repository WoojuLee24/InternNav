"""Compare _bev_scalar (old path) vs _bev_batch (new path).

_bev_scalar  : old for_s1 internals — direct bev_chw / occupancy call, scalar pitch/height only.
_bev_batch   : new _bev_batch — same logic but accepts scalar OR [N] tensor pitch/height.

Inputs are drawn from torch.rand so results are deterministic given a fixed seed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from internnav.model.utils.visual_input_provider import BEVProcessor

DEVICE = 'cpu'

# ---------------------------------------------------------------------------
# Reference implementation (old for_s1 core, scalar-only)
# ---------------------------------------------------------------------------

def _bev_scalar(proc: BEVProcessor, rgb_flat: torch.Tensor, depth_flat: torch.Tensor,
                pitch: float, image_type: str = 'rgb', cam_height: float = None) -> torch.Tensor:
    """Exact replica of the pre-refactor for_s1 body.

    Old code:
        bev = self.bev_chw(rgb_flat, depth_flat, depth_in_meters=True, cam_pitch_deg=cam_pitch_deg)
    (occ mode was not present in for_s1; added here for completeness via occupancy())
    """
    if image_type == 'occ':
        bev_hw = proc.occupancy(depth_flat, depth_in_meters=True,
                                cam_pitch_deg=pitch, rgb=rgb_flat,
                                cam_height=cam_height)
        return bev_hw.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    return proc.bev_chw(rgb_flat, depth_flat, depth_in_meters=True,
                        cam_pitch_deg=pitch, cam_height=cam_height)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_proc(**kw):
    return BEVProcessor(
        fx=388.19, fy=388.19, cx=160.0, cy=120.0,
        cam_height=1.25, cam_pitch_deg=0.0,
        ref_width=320, ref_height=240,
        bev_size=32, bev_range=5.0,
        depth_scale=1.0,
        device=DEVICE,
        **kw,
    )


def _rand_inputs(N, H, W, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    rgb   = torch.rand(N, H, W, 3, generator=g)
    depth = torch.rand(N, H, W, generator=g) * 5.0  # metres
    return rgb, depth


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_rgb_scalar_pitch_height():
    """Scalar pitch + height: _bev_scalar == _bev_batch, exact."""
    N, H, W = 4, 16, 16
    proc = make_proc()
    rgb, depth = _rand_inputs(N, H, W)

    for pitch in [0.0, 15.0, 30.0, 45.0]:
        for height in [0.60, 1.25]:
            ref = _bev_scalar(proc, rgb, depth, pitch=pitch, cam_height=height)
            out = proc._bev_batch(rgb, depth, pitch, 'rgb', cam_height=height)
            diff = (ref - out).abs().max().item()
            assert diff == 0.0, f"pitch={pitch} height={height}: max_diff={diff}"

    print("PASS  rgb mode, scalar pitch+height: _bev_scalar == _bev_batch  (max_diff=0)")


def test_occ_scalar_pitch_height():
    """occ mode, scalar pitch + height."""
    N, H, W = 4, 16, 16
    proc = make_proc()
    rgb, depth = _rand_inputs(N, H, W, seed=7)

    for pitch in [0.0, 30.0]:
        for height in [0.60, 1.25]:
            ref = _bev_scalar(proc, rgb, depth, pitch=pitch, image_type='occ', cam_height=height)
            out = proc._bev_batch(rgb, depth, pitch, 'occ', cam_height=height)
            diff = (ref - out).abs().max().item()
            assert diff == 0.0, f"pitch={pitch} height={height}: max_diff={diff}"

    print("PASS  occ mode, scalar pitch+height: _bev_scalar == _bev_batch  (max_diff=0)")


def test_rgb_tensor_pitch_height_matches_scalar_loop():
    """[N] tensor pitch/height in _bev_batch == per-sample _bev_scalar calls."""
    N, H, W = 3, 16, 16
    proc = make_proc()
    rgb, depth = _rand_inputs(N, H, W, seed=99)

    pitches = torch.tensor([0.0, 15.0, 45.0])
    heights = torch.tensor([1.25, 0.60, 1.25])

    # Reference: N separate scalar calls
    ref_chunks = []
    for i in range(N):
        ref_chunks.append(
            _bev_scalar(proc, rgb[i:i+1], depth[i:i+1],
                        pitch=pitches[i].item(), cam_height=heights[i].item())
        )
    ref = torch.cat(ref_chunks, dim=0)

    # New: single _bev_batch call with [N] tensors
    out = proc._bev_batch(rgb, depth, pitches, 'rgb', cam_height=heights)

    diff = (ref - out).abs().max().item()
    assert diff == 0.0, f"max_diff={diff}"
    print(f"PASS  rgb mode, [N] tensor pitch+height == scalar loop  (max_diff=0)")


def test_occ_tensor_pitch_height_matches_scalar_loop():
    """[N] tensor pitch/height, occ mode."""
    N, H, W = 3, 16, 16
    proc = make_proc()
    rgb, depth = _rand_inputs(N, H, W, seed=13)

    pitches = torch.tensor([0.0, 30.0, 45.0])
    heights = torch.tensor([1.25, 1.25, 0.60])

    ref_chunks = []
    for i in range(N):
        ref_chunks.append(
            _bev_scalar(proc, rgb[i:i+1], depth[i:i+1],
                        pitch=pitches[i].item(), image_type='occ', cam_height=heights[i].item())
        )
    ref = torch.cat(ref_chunks, dim=0)

    out = proc._bev_batch(rgb, depth, pitches, 'occ', cam_height=heights)

    diff = (ref - out).abs().max().item()
    assert diff == 0.0, f"max_diff={diff}"
    print(f"PASS  occ mode, [N] tensor pitch+height == scalar loop  (max_diff=0)")


def test_output_shapes():
    """Output shapes are [N, 3, bev_size, bev_size] for both modes."""
    N, H, W = 2, 16, 16
    proc = make_proc()
    rgb, depth = _rand_inputs(N, H, W)

    for mode in ('rgb', 'occ'):
        out = proc._bev_batch(rgb, depth, 30.0, mode, cam_height=1.25)
        assert out.shape == (N, 3, 32, 32), f"mode={mode}: shape={out.shape}"

    print("PASS  output shapes: [N, 3, 32, 32] for rgb and occ")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("_bev_scalar (old)  vs  _bev_batch (new)")
    print("=" * 60)
    test_output_shapes()
    test_rgb_scalar_pitch_height()
    test_occ_scalar_pitch_height()
    test_rgb_tensor_pitch_height_matches_scalar_loop()
    test_occ_tensor_pitch_height_matches_scalar_loop()
    print("=" * 60)
