"""BEV Fusion: Depth -> Bird's-Eye View -> 4 CLIP tokens for NexusNav S2.

NexusNav contribution C12. Converts the agent's depth observation to a
128x128 bird's-eye view occupancy map and encodes it to 4 visual tokens
prepended to S2's context. Gives S2 top-down spatial awareness of corridors
and obstacles. First BEV integration in VLM + continuous VLN (CVPR 2026).

Input:  depth_image (H, W) -- metric depth in meters, from DepthAnythingV2
Output: bev_tokens (B, 4, token_dim) -- tokens in Qwen2.5-VL embedding space

Usage:
    bev = BEVFusion(token_dim=3584)  # 3584 = Qwen2.5-VL-7B hidden size
    tokens = bev(depth_image, agent_height=1.5, vfov_deg=60.0)
    # Prepend tokens to S2 input_embeds before generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_to_bev(
    depth: torch.Tensor,
    agent_height: float = 1.5,
    vfov_deg: float = 60.0,
    bev_size: int = 128,
    bev_range_m: float = 8.0,
) -> torch.Tensor:
    """Project depth image to a BEV occupancy map via pinhole camera model.

    For each pixel (u, v) with depth d, compute (x, z) in agent-centric
    coordinates and mark the corresponding BEV cell as occupied.

    Args:
        depth: (H, W) -- metric depth in meters
        agent_height: camera height above floor in meters
        vfov_deg: vertical field of view in degrees
        bev_size: output BEV resolution (bev_size x bev_size)
        bev_range_m: forward range the BEV covers in meters

    Returns:
        (bev_size, bev_size) float tensor with values in {0, 1}
    """
    H, W = depth.shape[-2], depth.shape[-1]
    device = depth.device

    vfov = vfov_deg * np.pi / 180.0
    fy = H / (2.0 * np.tan(vfov / 2.0))
    cx, cy = W / 2.0, H / 2.0

    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )

    d = depth.float().squeeze()
    valid = (d > 0.1) & (d < bev_range_m * 1.5)

    x_cam = (u_grid - cx) / fy * d
    y_cam = (v_grid - cy) / fy * d
    z_cam = d

    # World coords: y = height above floor
    y_world = agent_height - y_cam

    # Keep obstacles at approximately agent torso/head height
    height_mask = (y_world > 0.2) & (y_world < 2.2)
    obstacle_mask = valid & height_mask

    bev = torch.zeros(bev_size, bev_size, device=device)
    bev_x = ((x_cam[obstacle_mask] / bev_range_m + 0.5) * bev_size).long()
    bev_z = ((z_cam[obstacle_mask] / bev_range_m) * bev_size).long()

    in_bounds = (bev_x >= 0) & (bev_x < bev_size) & (bev_z >= 0) & (bev_z < bev_size)
    bev[bev_z[in_bounds], bev_x[in_bounds]] = 1.0

    return bev


class BEVEncoder(nn.Module):
    """Lightweight CNN: 128x128 BEV occupancy map -> 4 spatial tokens."""

    def __init__(self, token_dim: int = 3584, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),    # 64x64
            nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),   # 32x32
            nn.GELU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 16x16
            nn.GELU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2), # 8x8
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),                  # 2x2 = 4 spatial positions
        )
        self.token_proj = nn.Linear(256, token_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)

    def forward(self, bev_map: torch.Tensor) -> torch.Tensor:
        """
        bev_map: (B, bev_size, bev_size)
        returns: (B, n_tokens, token_dim)
        """
        if bev_map.dim() == 2:
            bev_map = bev_map.unsqueeze(0)
        x = bev_map.unsqueeze(1)                   # (B, 1, H, W)
        x = self.cnn(x)                             # (B, 256, 2, 2)
        x = x.flatten(2).transpose(1, 2)            # (B, 4, 256)
        tokens = self.token_proj(x)                 # (B, 4, token_dim)
        return tokens + self.pos_embed


class BEVFusion(nn.Module):
    """Full BEV fusion module: depth image -> 4 LLM-space tokens (C12).

    Args:
        token_dim: LLM hidden dim (3584 for Qwen2.5-VL-7B)
        n_tokens: number of BEV tokens to produce
        bev_size: BEV map resolution
        bev_range_m: spatial range in meters
    """

    def __init__(
        self,
        token_dim: int = 3584,
        n_tokens: int = 4,
        bev_size: int = 128,
        bev_range_m: float = 8.0,
    ):
        super().__init__()
        self.bev_size = bev_size
        self.bev_range_m = bev_range_m
        self.encoder = BEVEncoder(token_dim=token_dim, n_tokens=n_tokens)

    def forward(
        self,
        depth: torch.Tensor,
        agent_height: float = 1.5,
        vfov_deg: float = 60.0,
    ) -> torch.Tensor:
        """
        depth: (H, W) or (B, H, W) -- metric depth in meters
        returns: (B, 4, token_dim) -- BEV tokens in LLM embedding space
        """
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        B = depth.shape[0]
        bev_maps = torch.stack([
            depth_to_bev(depth[i], agent_height, vfov_deg, self.bev_size, self.bev_range_m)
            for i in range(B)
        ])  # (B, bev_size, bev_size)
        return self.encoder(bev_maps)  # (B, 4, token_dim)
