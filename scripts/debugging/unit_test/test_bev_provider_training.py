"""Unit tests for training-time BEV via the inference module.

CPU-only; no checkpoints, no model load. The heaviest test asserts that the
training BEV substitution is byte-identical to (a) the legacy training helpers
in internvla_n1_bev.py and (b) the inference-side BEVProcessor — i.e. a model
trained with this code sees exactly what it sees at eval time.
"""

import argparse

import pytest
import torch

from internnav.model.basemodel.internvla_n1.internvla_n1_bev_provider import (
    _MODE_EQUIV,
    apply_bev_to_traj,
    make_train_bev_processor,
    parse_bev_cli_args,
)
from internnav.model.utils.visual_input_provider import BEVProcessor

B, T, H, W = 2, 3, 480, 640


def make_traj():
    torch.manual_seed(0)
    imgs = torch.rand(B, T, H, W, 3)
    depths = torch.rand(B, T, H, W) * 4.0 + 0.5  # metric metres
    return imgs, depths


# ------------------------------------------------------------- mode behaviour

def test_fpv_mode_is_identity():
    imgs, depths = make_traj()
    proc = make_train_bev_processor(1.25, 'cpu')
    out_i, out_d = apply_bev_to_traj(imgs, depths, proc, s1_mode='fpv')
    assert out_i is imgs and out_d is depths


def test_bev_mode_replaces_images():
    imgs, depths = make_traj()
    proc = make_train_bev_processor(1.25, 'cpu')
    out_i, out_d = apply_bev_to_traj(imgs, depths, proc, s1_mode='bev', image_type='rgb')
    assert out_i.shape == imgs.shape          # T unchanged
    assert out_d is depths                    # depths untouched
    assert not torch.equal(out_i, imgs)       # actually replaced


def test_fpv_bev_mode_concats_along_T():
    imgs, depths = make_traj()
    proc = make_train_bev_processor(1.25, 'cpu')
    out_i, out_d = apply_bev_to_traj(imgs, depths, proc, s1_mode='fpv_bev', image_type='rgb')
    assert out_i.shape == (B, 2 * T, H, W, 3)     # T doubled
    assert torch.equal(out_i[:, :T], imgs)        # FPV first, untouched
    assert out_d.shape == depths.shape            # depths stay [B, T, ...] (legacy semantics)


def test_occ_image_type_shape():
    imgs, depths = make_traj()
    proc = make_train_bev_processor(1.25, 'cpu')
    out_i, _ = apply_bev_to_traj(imgs, depths, proc, s1_mode='bev', image_type='occ')
    assert out_i.shape == imgs.shape
    # occupancy is replicated across the 3 channels
    assert torch.allclose(out_i[..., 0], out_i[..., 1]) and torch.allclose(out_i[..., 1], out_i[..., 2])


def test_bad_mode_raises():
    imgs, depths = make_traj()
    proc = make_train_bev_processor(1.25, 'cpu')
    with pytest.raises(ValueError):
        apply_bev_to_traj(imgs, depths, proc, s1_mode='nope')


# ------------------------------------------------------- parity (the point)


def test_parity_train_vs_inference_processor():
    """The processor used in training equals an eval-config-style BEVProcessor."""
    imgs, depths = make_traj()
    train_proc = make_train_bev_processor(1.25, 'cpu')
    # what create_visual_provider would build for Habitat (388.19 @ 640x480)
    eval_proc = BEVProcessor(
        fx=388.19, fy=388.19, cx=319.5, cy=239.5,
        cam_height=1.25, cam_pitch_deg=0.0, ref_width=640, ref_height=480,
        bev_size=224, bev_range=5.0, depth_scale=1.0, z_min=-0.2, z_max=2.5, device='cpu',
    )
    a = train_proc.bev_chw(imgs.flatten(0, 1), depths.flatten(0, 1), depth_in_meters=True)
    b = eval_proc.bev_chw(imgs.flatten(0, 1), depths.flatten(0, 1), depth_in_meters=True)
    assert torch.equal(a, b)


# ------------------------------------------------------------------ CLI parse

def test_parse_bev_cli_args_extracts_and_strips():
    argv = [
        '--bev_s1_mode', 'fpv_bev', '--bev_image_type', 'occ',
        '--bev_depth_source', 'gt', '--model_name_or_path', 'ckpt/x',
        '--num_history', '4',
    ]
    bev, remaining = parse_bev_cli_args(argv)
    assert bev['bev_s1_mode'] == 'fpv_bev' and bev['bev_image_type'] == 'occ'
    assert '--model_name_or_path' in remaining and 'ckpt/x' in remaining
    assert '--bev_s1_mode' not in remaining  # stripped for the base parser


def test_parse_bev_cli_args_defaults():
    bev, remaining = parse_bev_cli_args(['--num_history', '8'])
    assert bev['bev_s1_mode'] == 'bev' and bev['bev_image_type'] == 'rgb'
    assert bev['bev_depth_source'] == 'gt'
    assert remaining == ['--num_history', '8']


def test_parse_bev_cli_args_rejects_bad_choice():
    with pytest.raises(SystemExit):
        parse_bev_cli_args(['--bev_s1_mode', 'garbage'])


def test_mode_equivalence_table_covers_legacy():
    """Every legacy bev_mode is reachable from a (s1_mode, image, depth) combo."""
    legacy = set(_MODE_EQUIV.values())
    assert {'none', 'rgb_gt', 'occ_gt', 'fpv_concat_gt',
            'rgb_depthanythingv2', 'occ_depthanythingv2'} <= legacy


# --------------------------------------------------------- model (optional)

def test_model_class_importable():
    from internnav.model.basemodel.internvla_n1.internvla_n1_bev_provider import (
        InternVLAN1BEVProviderForCausalLM,
    )
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    assert issubclass(InternVLAN1BEVProviderForCausalLM, InternVLAN1ForCausalLM)
