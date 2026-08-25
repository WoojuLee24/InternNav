"""Qwen2.5-VL ViT patch embedding 구현 선택 ('conv' | 'gemm' | 'channels_last').

## 왜 필요한가

`Qwen2_5_VisionPatchEmbed.forward`는 flat한 `pixel_values` (N, 1176)을
`(N, in_channels, temporal_patch_size, patch_size, patch_size)`로 reshape한 뒤
**커널 크기가 입력 전체와 동일한** `Conv3d(bias=None)`를 돌린다 (출력 1x1x1).
슬라이딩할 자리가 없으므로 정의상 dense linear map이다:

    out[n, c] = sum_{i,j,k,l} x[n, i, j, k, l] * w[c, i, j, k, l]
              = (x.flatten(1) @ w.flatten(1).T)[n, c]

그런데 PyTorch는 **저정밀(bf16/fp16) 3D conv를 contiguous(NCDHW)로 받으면 cuDNN에 보내지 않고**
ATen `slow_conv_dilated3d` fallback으로 떨어뜨린다 (cuDNN의 텐서코어 3D conv 커널이 NDHWC를
요구하기 때문). 그 fallback은 **배치 차원을 순차 루프로 돈다.** 이 모델은 패치를 배치 차원에
넣으므로 루프가 수만 번 돌고, 각 회차는 1176->1280 GEMV 하나(3 MFLOP)라 커널 런치 지연에 묶인다.

## 실측 (H200, torch 2.9.0+cu130, bf16, N=34,480 = stage1 bs4의 micro-batch 패치 수)

    mode            backend           forward      현행 대비   forward 출력
    conv            SlowDilated3d    1258.13 ms        --       기준
    channels_last   Cudnn               4.30 ms      293x       다름 (maxdiff 1.56e-2)
    gemm            (conv 아님)          0.153 ms     8222x      비트 동일 (torch.equal=True)

fwd+bwd: conv 2885.6 ms / gemm 1.216 ms (2373x). conv의 wgrad는 fp64 정확해 대비 13.3% 오차
(bf16 누산기로 N회 순차 누적), gemm은 0.17%로 bf16 저장 하한에 도달.
전체 학습 step end-to-end: 18.043 s -> 7.793 s (2.32x), peak 메모리 동일(51.7 GiB).

버전 무관: transformers 4.49.0/4.51.0/4.57.1 소스 동일, torch 2.4.0/2.9.0 모두
`CUDNN_TENSOR_DTYPES`에 bf16 없음, `cudnn.enabled` on/off 무영향.

분석 전문: `.claude/memory/understanding_qwen25vl_patch_embed_conv3d_bottleneck.md`

## 설계

두 대안 모두 **파라미터를 교체하지 않는다** -> state_dict / checkpoint / ZeRO 파티셔닝 / resume 무영향:
  - `gemm`          : `proj.weight`를 `reshape()`(view)로 2D로 보고 GEMM 한 번. 파라미터 객체 그대로.
  - `channels_last` : **입력만** channels_last_3d로 변환. Parameter 레이아웃을 바꾸면 ZeRO-2의
                      flatten과 충돌할 수 있으므로 절대 건드리지 않는다. 입력만 변환해도
                      backend가 `Cudnn`으로 바뀜을 실측 확인.
"""

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed

PATCH_EMBED_IMPLS = ("conv", "gemm", "channels_last")


class LinearPatchEmbed(Qwen2_5_VisionPatchEmbed):
    """Conv3d == dense GEMM. `forward`만 override, 나머지는 부모 그대로."""

    def forward(self, hidden_states):
        # (embed_dim, in_channels*temporal_patch_size*patch_size^2) view — 복사 없음.
        # gradient는 원래 5D 파라미터로 그대로 흐른다.
        weight = self.proj.weight.reshape(self.embed_dim, -1)
        return hidden_states.view(-1, weight.shape[1]).to(weight.dtype) @ weight.t()


class ChannelsLastPatchEmbed(Qwen2_5_VisionPatchEmbed):
    """Conv3d를 유지하되 입력만 channels_last_3d로 -> PyTorch가 cuDNN 경로를 택한다.

    `conv`와 forward 출력이 **다르다** (cuDNN 커널의 누적 순서가 달라서, maxdiff ~1.56e-2).
    `gemm`보다 28배 느리다. conv 의미론을 유지한 대조군 용도.
    """

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        ).to(dtype=target_dtype)
        hidden_states = hidden_states.contiguous(memory_format=torch.channels_last_3d)
        return self.proj(hidden_states).view(-1, self.embed_dim)


def _check_equivalent(patch_embed) -> None:
    """Conv3d == GEMM 등가 전제조건. 하나라도 깨지면 교체하면 안 된다."""
    conv = patch_embed.proj
    assert isinstance(patch_embed, Qwen2_5_VisionPatchEmbed), type(patch_embed)
    assert conv.bias is None, "bias가 있으면 단순 GEMM과 다르다"
    # 커널이 reshape된 입력 전체를 덮어야 한다 (sliding 없음 -> stride 무관)
    expected_kernel = (patch_embed.temporal_patch_size, patch_embed.patch_size, patch_embed.patch_size)
    assert tuple(conv.kernel_size) == expected_kernel, f"{tuple(conv.kernel_size)} != {expected_kernel}"
    assert tuple(conv.padding) == (0, 0, 0), tuple(conv.padding)
    assert tuple(conv.dilation) == (1, 1, 1), tuple(conv.dilation)
    assert conv.groups == 1, conv.groups
    assert conv.weight.shape == (patch_embed.embed_dim, patch_embed.in_channels, *expected_kernel), conv.weight.shape


def apply_patch_embed_impl(model, impl: str = "conv"):
    """`model.visual.patch_embed`의 forward 구현만 교체한다 (파라미터는 그대로).

    instance의 `__class__`만 바꾸므로 `isinstance(pe, Qwen2_5_VisionPatchEmbed)`가 계속 참이고
    모듈 트리/파라미터 이름이 전혀 바뀌지 않는다. 중복 호출에 안전하다.

    impl='conv' (기본)이면 아무것도 하지 않는다 -> 기존 동작 완전 보존.
    """
    if impl == "conv":
        return model

    patch_embed = model.visual.patch_embed
    base_cls = (LinearPatchEmbed, ChannelsLastPatchEmbed)
    if isinstance(patch_embed, base_cls):  # 이미 적용됨
        return model
    _check_equivalent(patch_embed)

    if impl == "gemm":
        patch_embed.__class__ = LinearPatchEmbed
    elif impl == "channels_last":
        patch_embed.__class__ = ChannelsLastPatchEmbed
    else:
        assert False, f"unreachable patch_embed_impl={impl!r} (expected one of {PATCH_EMBED_IMPLS})"
    return model
