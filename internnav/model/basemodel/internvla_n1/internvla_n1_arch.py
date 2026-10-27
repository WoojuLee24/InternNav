from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

LatentEmbSize = 768
MODEL_PATH_TO = "checkpoints"


def build_navdp(navdp_cfg, memory_size):
    from .navdp import NavDP_Policy_DPT_CriticSum_DAT

    navdp = NavDP_Policy_DPT_CriticSum_DAT(memory_size=memory_size, navdp_version=0.1)
    navdp.load_model()
    return navdp


def build_traj_dit(config):
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    from .nextdit_crossattn_traj import NextDiTCrossAttn, NextDiTCrossAttnConfig

    dit = NextDiTCrossAttn(NextDiTCrossAttnConfig(latent_embedding_size=LatentEmbSize))
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    return dit, noise_scheduler


def build_depthanythingv2(config):
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import (
        DepthAnythingV2,
    )

    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    DAv2_model = DepthAnythingV2(**model_configs['vits'])
    DAv2_model.load_state_dict(
        torch.load(f'{MODEL_PATH_TO}/depth_anything_v2_metric_hypersim_vits.pth', map_location="cpu"),
        assign=True,
    )  # download from https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
    rgb_model = DAv2_model.pretrained

    return rgb_model


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class MemoryEncoder(nn.Module):
    def __init__(self, hidden_size=384, num_heads=6, num_layers=3, max_len=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.memory_pos = nn.Parameter(torch.randn(max_len, hidden_size))

    def forward(self, memory, memory_mask=None):
        """
        memory: (B, N, C)
        memory_mask: (B, N)
        """
        B, N, C = memory.shape
        pos = self.memory_pos[:N, :].unsqueeze(0).expand(B, -1, -1)  # (B, N, C)
        memory = memory + pos
        encoded_memory = self.encoder(memory, src_key_padding_mask=memory_mask)
        return encoded_memory


class QFormer(nn.Module):
    def __init__(self, num_query=32, hidden_size=768, num_layers=3, num_heads=12):
        super().__init__()
        self.num_query = num_query
        self.hidden_size = hidden_size

        self.query_tokens = nn.Parameter(torch.randn(num_query, hidden_size))
        self.query_pos = nn.Parameter(torch.randn(num_query, hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.visual_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, visual_feats, visual_attn_mask=None):
        B = visual_feats.size(0)

        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_tokens = query_tokens + self.query_pos.unsqueeze(0)

        out = self.decoder(query_tokens, visual_feats, memory_key_padding_mask=visual_attn_mask)
        return out


class PredictiveWorldModelHead(nn.Module):
    """Predicts future visual latents at horizon steps from current features (NavForesee-style).

    Training: auxiliary loss L_pred = MSE(predicted, actual_future.detach())
    Inference: uncertainty = mean prediction error → S2 firing gate signal.

    Input: current visual features (B, N, feat_dim) from rgb_model intermediate layers
    Output: predicted latents at each horizon, shape (B, len(horizons), N, feat_dim)
    """

    def __init__(self, feat_dim=384, hidden_dim=512, horizons=(2, 4)):
        super().__init__()
        self.horizons = horizons
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feat_dim),
            )
            for _ in horizons
        ])

    def forward(self, visual_feats):
        # visual_feats: (B, N, feat_dim)
        encoded = self.feat_encoder(visual_feats)  # (B, N, hidden_dim)
        predictions = []
        for head in self.horizon_heads:
            predictions.append(head(encoded))  # (B, N, feat_dim)
        return torch.stack(predictions, dim=1)  # (B, len(horizons), N, feat_dim)

    def compute_uncertainty(self, visual_feats_seq):
        """Compute per-step prediction error for the S2 gate signal.

        visual_feats_seq: list of (B, N, feat_dim) tensors, length >= max(horizons)+1
        Returns: uncertainty scalar (B,) or None if not enough history.
        """
        if len(visual_feats_seq) < max(self.horizons) + 1:
            return None
        current = visual_feats_seq[-1]
        errors = []
        for i, horizon in enumerate(self.horizons):
            if len(visual_feats_seq) > horizon:
                actual_future = visual_feats_seq[-horizon]
                pred = self.forward(current)[:, i]
                error = F.mse_loss(pred, actual_future.detach(), reduction='none').mean(dim=(-1, -2))
                errors.append(error)
        if not errors:
            return None
        return torch.stack(errors, dim=0).mean(dim=0)


class LACPModule(nn.Module):
    """Latent Anticipatory Causal Predictor (LACP).

    2M-parameter module that predicts future WM-head uncertainty from current
    DiT action latents. Makes the failure gate anticipatory (fires before failure)
    rather than reactive (fires after detecting prediction error).

    NexusNav contribution C11. First anticipatory VLN failure gate (CVPR 2026).

    Input:  action_features (B, T, input_dim) — DiT latent before traj_dit forward
    Output: anticipated_uncertainty (B, T) — predicted WM error at t+1, values >= 0
    Training loss: MSE(predicted, actual_wm_uncertainty.detach())
    """

    def __init__(self, input_dim: int = 384, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        """
        action_features: (B, T, input_dim)
        returns: anticipated_uncertainty (B, T) — values >= 0
        """
        x = self.input_proj(action_features)   # (B, T, d_model)
        x = self.transformer(x)                 # (B, T, d_model)
        out = self.out_proj(x).squeeze(-1)      # (B, T)
        return F.softplus(out)                  # ensure >= 0


class InternVLAN1MetaModel:
    def __init__(self, config):
        super(InternVLAN1MetaModel, self).__init__(config)
        if hasattr(config, "system1"):
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))

            if 'nextdit' in config.system1:
                self.traj_dit, self.noise_scheduler = build_traj_dit(config)
                self.action_encoder = nn.Linear(3, 384, bias=True)
                self.pos_encoding = SinusoidalPositionalEncoding(384)
                self.action_decoder = nn.Linear(384, 3, bias=True)
                self.cond_projector = nn.Sequential(
                    nn.Linear(3584, LatentEmbSize), nn.GELU(approximate="tanh"), nn.Linear(LatentEmbSize, LatentEmbSize)
                )

                if 'async' in config.system1:
                    self.rgb_model = build_depthanythingv2(config)
                    self.memory_encoder = MemoryEncoder()
                    self.rgb_resampler = QFormer()
                    self.wm_head = PredictiveWorldModelHead(feat_dim=384, hidden_dim=512, horizons=(2, 4))
                    self.imag_token_proj = nn.Linear(384, LatentEmbSize)
                    self.lacp = LACPModule(input_dim=384, d_model=256, n_layers=2, n_heads=4)

            elif 'navdp' in config.system1:
                if 'async' in config.system1:
                    self.navdp = build_navdp(config, memory_size=2)
            else:
                raise NotImplementedError

    def initialize_vision_modules(self, model_args):
        if 'nextdit' in model_args.system1:
            self.traj_dit, self.noise_scheduler = build_traj_dit(model_args)
            self.action_encoder = nn.Linear(3, 384, bias=True)
            self.pos_encoding = SinusoidalPositionalEncoding(384)
            self.action_decoder = nn.Linear(384, 3, bias=True)

            self.cond_projector = nn.Sequential(
                nn.Linear(3584, LatentEmbSize), nn.GELU(approximate="tanh"), nn.Linear(LatentEmbSize, LatentEmbSize)
            )

            if 'async' in model_args.system1:
                self.rgb_model = build_depthanythingv2(model_args)
                self.memory_encoder = MemoryEncoder()
                self.rgb_resampler = QFormer()
                self.wm_head = PredictiveWorldModelHead(feat_dim=384, hidden_dim=512, horizons=(2, 4))
                self.imag_token_proj = nn.Linear(384, LatentEmbSize)
                self.lacp = LACPModule(input_dim=384, d_model=256, n_layers=2, n_heads=4)
        elif 'navdp' in model_args.system1:
            if 'async' in model_args.system1:
                self.navdp = build_navdp(model_args, memory_size=2)
        else:
            raise NotImplementedError

        self.config.system1 = model_args.system1
        self.config.n_query = model_args.n_query
        existing_lq = getattr(self, 'latent_queries', None)
        if existing_lq is None or existing_lq.shape[1] != model_args.n_query:
            print(f"random initiation the latent_queries !!! (n_query={model_args.n_query})")
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))


class InternVLAN1MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_mm_projector(self):
        return self.get_model().mm_projector

    def get_n_query(self):
        return self.get_model().config.n_query

    def get_system1_type(self):
        return self.get_model().config.system1

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
