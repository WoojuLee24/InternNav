"""`--patch_embed_impl`이 모델 입출력/동작을 어떻게 바꾸는지(또는 안 바꾸는지) 검증한다.

V1: `model.visual(...)` 출력이 적용 전/후 `torch.equal`, state_dict 키/shape 불변
V2: 1-step loss와 파라미터별 grad 비교.
    flash-attn backward는 dq를 atomics로 누적해 비결정적이므로 **대조군**(같은 설정 반복)으로
    노이즈 바닥을 먼저 재고, 그걸 넘는 차이만 변경 효과로 판정한다.

기대 결과:
  gemm          : forward/loss 비트 동일. wgrad만 달라짐(정확해 방향으로 — conv가 13.3% 오차,
                  gemm이 0.17%). 방향 차이 7.3도 = mini-batch 노이즈 84.7도의 1/11.6.
  channels_last : forward부터 달라짐 (maxdiff ~1.56e-2, cuDNN 커널의 누적 순서 차이).

weight 값은 등가성 판정에 무관하므로(선형성은 값과 무관) config로부터 랜덤 초기화한다
-> 16GB 다운로드 불필요. 실제 체크포인트로 확인하려면 --no-random-init --model-path를 준다.

    /usr/bin/python scripts/train_eval/qwenvl_train/baseline/verify_patch_embed_impl.py --impl gemm
    /usr/bin/python scripts/train_eval/qwenvl_train/baseline/verify_patch_embed_impl.py --impl channels_last
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))))

from transformers import AutoConfig, AutoProcessor  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VLForConditionalGeneration,
)

from internnav.model.basemodel.internvla_n1.patch_embed_impl import apply_patch_embed_impl  # noqa: E402

IMG_ID = 151655


def build(args):
    cfg = AutoConfig.from_pretrained(args.model_path)
    cfg._attn_implementation = cfg.vision_config._attn_implementation = "flash_attention_2"
    cfg.use_cache = False
    if args.random_init:
        torch.manual_seed(0)
        with torch.device("meta"):
            model = Qwen2_5_VLForConditionalGeneration(cfg)
        model = model.to_empty(device="cuda").to(torch.bfloat16)
        for p in model.parameters():  # to_empty는 uninitialized -> 재현 가능한 값으로 채운다
            with torch.no_grad():
                p.normal_(0.0, 0.02)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, config=cfg, torch_dtype=torch.bfloat16).cuda()
    model.train()
    return model, cfg


def make_batch(args, cfg):
    """stage1 실측 분포와 동일: 384x384 9장 + lookdown 640x480 1장, text 107 tok."""
    proc = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=3136, max_pixels=313600, use_fast=False).image_processor
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (384, 384, 3), dtype=np.uint8) for _ in range(9)]
    imgs.append(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
    ps = proc(images=imgs, return_tensors="pt")
    pv1, thw1 = ps["pixel_values"], ps["image_grid_thw"]
    vtok = int((thw1.prod(-1) // proc.merge_size ** 2).sum())
    seq = vtok + 107
    bs = args.batch_size
    input_ids = torch.full((bs, seq), 1000, dtype=torch.long).cuda()
    input_ids[:, :vtok] = IMG_ID
    labels = input_ids.clone()
    labels[:, :vtok] = -100
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(seq).view(1, 1, seq).expand(3, bs, seq).cuda().contiguous(),
        labels=labels,
        pixel_values=torch.cat([pv1] * bs, 0).to("cuda", torch.bfloat16),
        image_grid_thw=torch.cat([thw1] * bs, 0).cuda(),
    ), vtok, seq


def sig(model):
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def vit_out(model, batch):
    with torch.no_grad():
        return model.visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"]).clone()


def loss_and_grads(model, batch):
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return loss.detach().clone(), grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--random-init", action="store_true", default=True,
                    help="config만 읽어 랜덤 초기화 (기본). --no-random-init으로 실제 weight 로드")
    ap.add_argument("--no-random-init", dest="random_init", action="store_false")
    ap.add_argument("--batch-size", type=int, default=2, help="1-step 비교용 (기본 2로 메모리 절약)")
    ap.add_argument("--impl", choices=["gemm", "channels_last"], default="gemm",
                    help="검증할 구현 (기준은 항상 'conv')")
    args = ap.parse_args()

    model, cfg = build(args)
    batch, vtok, seq = make_batch(args, cfg)
    print(f"batch: bs={args.batch_size} seq={seq} vtok={vtok} "
          f"vit_patches={batch['pixel_values'].shape[0]}", flush=True)

    ok = True

    # ---------------- V1 ----------------
    sig_before = sig(model)
    out_before = vit_out(model, batch)
    l_before, g_before = loss_and_grads(model, batch)

    apply_patch_embed_impl(model, args.impl)

    pe = model.visual.patch_embed
    print(f"\n[V1] impl={args.impl}  patch_embed class : {type(pe).__name__}  "
          f"isinstance(Qwen2_5_VisionPatchEmbed)={isinstance(pe, Qwen2_5_VisionPatchEmbed)}")
    same_sig = sig(model) == sig_before
    print(f"[V1] state_dict 키/shape 불변 : {same_sig}  (keys={len(sig_before)})")
    print(f"[V1] proj.weight is_contiguous : {pe.proj.weight.is_contiguous()}  (ZeRO-2 flatten 안전성)")
    ok &= same_sig and isinstance(pe, Qwen2_5_VisionPatchEmbed) and pe.proj.weight.is_contiguous()

    out_after = vit_out(model, batch)
    eq = torch.equal(out_before, out_after)
    mad = (out_before.float() - out_after.float()).abs().max().item()
    print(f"[V1] visual() 출력 torch.equal : {eq}   max_abs_diff={mad:.3e}")
    # gemm은 비트 동일이 기대값. channels_last는 cuDNN 커널이라 달라지는 것이 정상.
    if args.impl == "gemm":
        ok &= eq
    else:
        print(f"       (channels_last는 forward가 달라지는 것이 예상 동작 — 실패로 세지 않음)")

    # ---------------- V2 ----------------
    # flash-attn backward는 dq를 atomics로 누적해 기본적으로 비결정적이다.
    # 따라서 "변경 전 vs 변경 후"만 비교하면 변경 효과와 run-to-run 노이즈를 구분할 수 없다.
    # 대조군(g_ctrl = 변경 없이 한 번 더)을 두고 노이즈 바닥값을 먼저 측정한다.
    l_after, g_after = loss_and_grads(model, batch)
    leq = torch.equal(l_before, l_after)
    print(f"\n[V2] loss  before={l_before.item():.8f}  after={l_after.item():.8f}  equal={leq}")
    if args.impl == "gemm":
        ok &= leq

    assert set(g_before) == set(g_after), "grad 파라미터 집합이 달라졌다"

    def rel(a, b):
        d = (a.float() - b.float()).abs().max().item()
        scale = max(a.float().abs().max().item(), 1e-30)
        return d, d / scale

    diff_change = {n: rel(g_before[n], g_after[n]) for n in g_before}
    n_eq = sum(1 for n in g_before if torch.equal(g_before[n], g_after[n]))
    print(f"[V2] 변경 전/후 grad 비트 동일 : {n_eq}/{len(g_before)}")

    print("\n[V2-control] 같은 설정으로 한 번 더 backward (변경 무관 비결정성 측정)")
    l_ctrl, g_ctrl = loss_and_grads(model, batch)
    diff_noise = {n: rel(g_after[n], g_ctrl[n]) for n in g_after}
    n_eq_ctrl = sum(1 for n in g_after if torch.equal(g_after[n], g_ctrl[n]))
    print(f"[V2-control] 동일 설정 반복 시 grad 비트 동일 : {n_eq_ctrl}/{len(g_after)}"
          f"   -> 비결정성 바닥 max_rel={max(v[1] for v in diff_noise.values()):.3e}")

    # 변경 효과가 비결정성 바닥을 넘는 파라미터만 진짜 차이로 본다
    real = [(n, diff_change[n][0], diff_change[n][1], diff_noise[n][1])
            for n in g_before if diff_change[n][1] > max(diff_noise[n][1] * 4, 1e-9)]
    real.sort(key=lambda x: -x[2])
    print(f"\n[V2] 비결정성 바닥을 넘는 파라미터 : {len(real)}/{len(g_before)}")
    for n, d, r, nz in real[:10]:
        print(f"       {n}\n         change max_abs={d:.3e} rel={r:.3e}   (noise rel={nz:.3e})")

    bf16_eps = 2 ** -8  # bf16 mantissa 7bit -> 상대오차 ~3.9e-3
    over = [x for x in real if x[2] > bf16_eps]
    print(f"\n[V2] bf16 반올림 수준(rel<{bf16_eps:.1e})을 넘는 파라미터 : {len(over)}")

    # 판정 기준은 mode마다 다르다.
    #   gemm          : forward/loss가 비트 동일해야 하고, wgrad는 patch_embed 하나만 달라져야 한다.
    #                   (그 하나는 conv가 fp64 정확해 대비 13.3% 틀린 것을 gemm이 0.17%로 고치는 것이며,
    #                    방향 차이 7.3도 = mini-batch 노이즈 84.7도의 1/11.6.)
    #   channels_last : cuDNN 커널이라 forward부터 달라지고 그것이 LLM 전체로 전파된다.
    #                   "기존 동작 보존"을 원하면 이 mode를 쓰면 안 된다 — 대조군 용도.
    over_names = {n for n, _, _, _ in over}
    if args.impl == "gemm":
        expected = {"visual.patch_embed.proj.weight"}
        unexpected = over_names - expected
        print(f"[V2] gemm 기대: patch_embed 하나만 초과 -> 실제 초과 {sorted(over_names)}")
        ok &= not unexpected
        verdict = ("forward/loss 비트 동일 + wgrad는 patch_embed 하나만 변화(정확해 방향)"
                   if ok else f"예상 밖 파라미터가 변했다: {sorted(unexpected)}")
    else:
        print(f"[V2] channels_last는 forward부터 달라지므로 grad 광범위 변화가 예상 동작이다 "
              f"({len(over)}/{len(g_before)} 초과)")
        verdict = "cuDNN 경로 전환 -> forward/loss/grad 모두 변화 (대조군 용도, 기존 동작 보존 아님)"

    print(f"\n{'PASS' if ok else 'FAIL'} (impl={args.impl}): {verdict}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
