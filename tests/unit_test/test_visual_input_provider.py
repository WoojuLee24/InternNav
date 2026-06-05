"""Unit tests for internnav/model/utils/visual_input_provider.py (BEV injection).

CPU-only — no simulator, no checkpoints. Heavier registry checks (agent /
evaluator subclasses) are skipped automatically when their deps are missing.
"""

import numpy as np
import pytest
import torch

from internnav.model.utils.depth_rgb_to_bev_torch import depth_rgb_to_bev
from internnav.model.utils.visual_input_provider import (
    BEVFeatureProvider,
    BEVImageProvider,
    BEVProcessor,
    FPVProvider,
    S1VisualInput,
    create_visual_provider,
)

H, W = 48, 64
FX, FY, CX, CY = 58.5, 58.5, (W - 1) / 2.0, (H - 1) / 2.0


def make_processor(**kwargs):
    defaults = dict(
        fx=FX, fy=FY, cx=CX, cy=CY,
        cam_height=1.25, cam_pitch_deg=0.0,
        ref_width=W, ref_height=H,
        bev_size=224, bev_range=5.0,
        depth_scale=1.0, z_min=-0.2, z_max=2.5,
        device='cpu',
    )
    defaults.update(kwargs)
    return BEVProcessor(**defaults)


def synthetic_frame():
    rgb = torch.rand(1, H, W, 3)
    depth = torch.rand(1, H, W) * 4.0 + 0.5  # 0.5–4.5 m
    return rgb, depth


# ---------------------------------------------------------------- BEVProcessor

def test_bev_chw_shape_and_range():
    proc = make_processor()
    rgb, depth = synthetic_frame()
    bev = proc.bev_chw(rgb, depth, depth_in_meters=True)
    assert bev.shape == (1, 3, 224, 224)
    assert float(bev.min()) >= 0.0 and float(bev.max()) <= 1.0


def test_bev_chw_matches_direct_call():
    """BEVProcessor must reproduce depth_rgb_to_bev with the same params."""
    proc = make_processor()
    rgb, depth = synthetic_frame()
    got = proc.bev_chw(rgb, depth, depth_in_meters=True)
    want = depth_rgb_to_bev(
        depth, rgb,
        cam_height=1.25, cam_pitch_deg=0.0,
        fx=FX, fy=FY, cx=CX, cy=CY,
        bev_range=5.0, bev_size=224,
        depth_scale=1.0, z_min=-0.2, z_max=2.5,
    )
    assert torch.allclose(got, want)


def test_bev_geometry_single_point():
    """A single point 2 m straight ahead lands in the expected BEV cell."""
    proc = make_processor()
    rgb = torch.zeros(1, H, W, 3)
    depth = torch.zeros(1, H, W)
    # principal-pixel ray ≈ (0, 0, 1); place a red point at depth 2 m
    v, u = int(round(CY)), int(round(CX))
    depth[0, v, u] = 2.0
    rgb[0, v, u] = torch.tensor([1.0, 0.0, 0.0])
    bev = proc.bev_chw(rgb, depth, depth_in_meters=True)
    # world (X=2, Y≈0, Z=1.25) → i = 112 - 2*22.4 ≈ 67, j ≈ 112
    region = bev[0, :, 64:70, 110:115]
    assert float(region[0].max()) > 0.5, "red point missing from expected BEV region"
    assert float(bev[0, 0].sum()) == pytest.approx(float(region[0].sum())), "colour leaked outside expected cell"


def test_bev_depth_scale_equivalence():
    """raw depth × depth_scale must equal pre-scaled metric depth."""
    proc = make_processor(depth_scale=10.0)
    rgb, depth_m = synthetic_frame()
    raw = depth_m / 10.0
    a = proc.bev_chw(rgb, raw, depth_in_meters=False)   # applies ×10
    b = proc.bev_chw(rgb, depth_m, depth_in_meters=True)
    assert torch.allclose(a, b)


def test_intrinsics_rescaling():
    """Same scene at 2× resolution (intrinsics auto-scaled) → similar BEV."""
    proc = make_processor()
    rgb, depth = synthetic_frame()
    rgb2 = rgb.permute(0, 3, 1, 2)
    rgb2 = torch.nn.functional.interpolate(rgb2, scale_factor=2, mode='nearest').permute(0, 2, 3, 1)
    depth2 = torch.nn.functional.interpolate(depth.unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1)
    bev1 = proc.bev_chw(rgb, depth, depth_in_meters=True)
    bev2 = proc.bev_chw(rgb2, depth2, depth_in_meters=True)
    # occupancy patterns should overlap heavily (not exact: 4× the points)
    occ1, occ2 = (bev1.sum(1) > 0), (bev2.sum(1) > 0)
    inter = (occ1 & occ2).float().sum()
    assert float(inter / occ1.float().sum().clamp(min=1)) > 0.9


def test_occupancy_shape():
    proc = make_processor()
    _, depth = synthetic_frame()
    occ = proc.occupancy(depth, depth_in_meters=True)
    assert occ.shape == (1, 224, 224)
    assert float(occ.min()) >= 0.0 and float(occ.max()) <= 1.0


def test_for_s2_returns_pil():
    proc = make_processor()
    rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    depth = np.random.rand(H, W).astype(np.float32) * 4.0 + 0.5
    img = proc.for_s2(rgb, depth, depth_in_meters=True)
    from PIL import Image
    assert isinstance(img, Image.Image)
    assert img.size == (224, 224)
    # [H, W, 1] depth must also be accepted
    img2 = proc.for_s2(rgb, depth[..., None], depth_in_meters=True)
    assert isinstance(img2, Image.Image)


def test_for_s1_layout_and_dtype():
    proc = make_processor()
    rgb = torch.rand(1, 2, 224, 224, 3, dtype=torch.float32)
    depth = torch.rand(1, 2, 224, 224, 1) * 4.0 + 0.5
    out = proc.for_s1(rgb, depth)
    assert out.shape == (1, 2, 224, 224, 3)
    assert out.dtype == rgb.dtype and out.device == rgb.device


# ------------------------------------------------------------------ providers

def test_fpv_provider_is_noop():
    p = FPVProvider()
    assert p.s1_mode == 'fpv' and p.s2_mode == 'fpv'
    s1 = p.get_s1_input(torch.rand(1, 2, 8, 8, 3), torch.rand(1, 2, 8, 8, 1))
    assert s1.images is None and s1.features is None and s1.depths is None
    assert p.get_s2_extra(None, None, is_lookdown=True) == []
    p.reset()
    p.set_goal(None, None)  # hooks must not raise


def test_bev_image_provider_s1_bev_mode():
    p = BEVImageProvider(make_processor(), s1_mode='bev')
    rgb = torch.rand(1, 2, 224, 224, 3)
    depth = torch.rand(1, 2, 224, 224, 1) * 4.0 + 0.5
    s1 = p.get_s1_input(rgb, depth)
    assert s1.images is not None and s1.images.shape == rgb.shape
    assert s1.depths is None  # depths stay FPV (rgb_gt training convention)
    assert not torch.equal(s1.images, rgb)  # actually replaced


def test_bev_image_provider_s1_fpv_bev_mode():
    """fpv_bev: BEV concat along T (fpv_concat_gt convention), depths duplicated."""
    p = BEVImageProvider(make_processor(), s1_mode='fpv_bev')
    rgb = torch.rand(1, 2, 224, 224, 3)
    depth = torch.rand(1, 2, 224, 224, 1) * 4.0 + 0.5
    s1 = p.get_s1_input(rgb, depth)
    assert s1.images.shape == (1, 4, 224, 224, 3)
    assert s1.depths.shape == (1, 4, 224, 224, 1)
    assert torch.equal(s1.images[:, :2], rgb)            # FPV frames first, untouched
    assert torch.equal(s1.depths[:, :2], s1.depths[:, 2:])  # BEV frames reuse FPV depth
    bev_only = BEVImageProvider(make_processor(), s1_mode='bev').get_s1_input(rgb, depth)
    assert torch.equal(s1.images[:, 2:], bev_only.images)  # tail frames are the BEV stack


def test_bev_image_provider_s1_fpv_mode():
    p = BEVImageProvider(make_processor(), s1_mode='fpv')
    s1 = p.get_s1_input(torch.rand(1, 2, 16, 16, 3), torch.rand(1, 2, 16, 16, 1))
    assert s1.images is None and s1.depths is None  # identical to original path


def test_bev_image_provider_s1_non_tensor_fallback():
    p = BEVImageProvider(make_processor())
    s1 = p.get_s1_input(np.zeros((H, W, 3)), np.zeros((H, W, 1)))
    assert s1.images is None  # legacy sync path falls back to FPV


def test_bev_image_provider_s2_gating():
    p = BEVImageProvider(make_processor(), s2_depth_in_meters=True)
    rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    depth = np.random.rand(H, W).astype(np.float32) + 0.5
    assert p.get_s2_extra(rgb, depth, is_lookdown=False) == []
    extra = p.get_s2_extra(rgb, depth, is_lookdown=True)
    assert len(extra) == 1
    # 'fpv' mode → no extra images; 'bev' mode → BEV returned (replaces FPV at call site)
    assert BEVImageProvider(make_processor(), s2_mode='fpv').get_s2_extra(rgb, depth, is_lookdown=True) == []
    p_bev = BEVImageProvider(make_processor(), s2_mode='bev', s2_depth_in_meters=True)
    assert len(p_bev.get_s2_extra(rgb, depth, is_lookdown=True)) == 1


def test_bev_image_provider_rejects_bad_mode():
    with pytest.raises(ValueError):
        BEVImageProvider(make_processor(), s1_mode='nope')
    with pytest.raises(ValueError):
        BEVImageProvider(make_processor(), s2_mode='bev_only')


def test_bev_feature_provider_is_stub():
    p = BEVFeatureProvider(make_processor())
    with pytest.raises(NotImplementedError):
        p.get_s1_input(torch.rand(1, 2, 8, 8, 3), torch.rand(1, 2, 8, 8, 1))


# -------------------------------------------------------------------- factory

def test_factory_default_is_fpv():
    assert isinstance(create_visual_provider({}, device='cpu'), FPVProvider)
    assert isinstance(create_visual_provider({'visual_provider': 'fpv'}, device='cpu'), FPVProvider)


def test_factory_bev_image_with_config():
    cfg = {
        'visual_provider': 'bev_image',
        'bev_fx': FX, 'bev_fy': FY, 'bev_cx': CX, 'bev_cy': CY,
        'bev_ref_width': W, 'bev_ref_height': H,
        'bev_cam_height': 1.25, 'bev_cam_pitch_deg': 30.0,
        'bev_depth_scale': 10.0,
        'bev_s1_mode': 'fpv_bev', 'bev_s2_mode': 'fpv',
        'bev_s1_pitch_deg': 60.0,
    }
    p = create_visual_provider(cfg, device='cpu')
    assert isinstance(p, BEVImageProvider)
    assert p.s1_mode == 'fpv_bev' and p.s2_mode == 'fpv'
    assert p.s1_pitch_deg == 60.0
    assert p.processor.cam_pitch_deg == 30.0
    assert p.processor.depth_scale == 10.0


def test_factory_mode_defaults_and_legacy_bools():
    # defaults: s1='bev', s2='fpv_bev'
    p = create_visual_provider({'visual_provider': 'bev_image'}, device='cpu')
    assert p.s1_mode == 'bev' and p.s2_mode == 'fpv_bev'
    # legacy booleans map to default-mode / 'fpv'
    p = create_visual_provider({'visual_provider': 'bev_image', 'bev_s1': False, 'bev_s2': False}, device='cpu')
    assert p.s1_mode == 'fpv' and p.s2_mode == 'fpv'
    # mode keys win over legacy booleans
    p = create_visual_provider(
        {'visual_provider': 'bev_image', 'bev_s1': False, 'bev_s1_mode': 'fpv_bev'}, device='cpu'
    )
    assert p.s1_mode == 'fpv_bev'


def test_factory_accepts_namespace():
    import argparse
    ns = argparse.Namespace(visual_provider='bev_image', bev_fx=FX, bev_fy=FY, bev_cx=CX, bev_cy=CY)
    assert isinstance(create_visual_provider(ns, device='cpu'), BEVImageProvider)


def test_factory_bev_feature_and_unknown():
    p = create_visual_provider({'visual_provider': 'bev_feature'}, device='cpu')
    assert isinstance(p, BEVFeatureProvider)
    with pytest.raises(ValueError):
        create_visual_provider({'visual_provider': 'nope'}, device='cpu')


# ------------------------------------------------------- registries (optional)

def test_agent_registry_entry():
    pytest.importorskip("transformers")
    pytest.importorskip("gym")
    try:
        import internnav.agent.internvla_n1_agent_bev  # noqa: F401
    except Exception as exc:  # missing heavy deps (flash_attn, etc.)
        pytest.skip(f"agent deps unavailable: {exc}")
    from internnav.agent.base import Agent

    assert 'internvla_n1_bev' in Agent.agents


def test_evaluator_registry_entry():
    pytest.importorskip("habitat")
    try:
        import internnav.habitat_extensions.vln.habitat_vln_evaluator_bev  # noqa: F401
    except Exception as exc:
        pytest.skip(f"habitat evaluator deps unavailable: {exc}")
    from internnav.evaluator import Evaluator

    assert 'habitat_vln_bev' in Evaluator.evaluators
