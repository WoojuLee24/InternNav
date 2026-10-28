"""Append the InternVLA-N1 latent query tokens <|latent_0..n-1|> to a Qwen2.5-VL text GGUF (llama.cpp).

llama-s2-runner extracts traj_latents by decoding the string "<|latent_0|>...<|latent_3|>" after the S2 answer and
reading the last-layer hidden states (tools/mtmd/s2-runner.cpp extract_latent). That equals HF
InternVLAN1ForCausalLM.generate_latents (latent_queries appended as input embeddings after prompt + answer) iff the
embedding rows of those tokens ARE the trained latent_queries. This script injects them verbatim:

  tokenizer.ggml.tokens / token_type (+ scores)  += "<|latent_i|>" (USER_DEFINED, like the original file)
  token_embd.weight rows                          += latent_queries[i]   (written in the tensor's own type; F16/F32/BF16)
  output.weight rows                              += zeros               (lm_head has no trained rows; never sampled)

Input must have unquantized token_embd/output (run on the f16 GGUF, before llama-quantize; keep them F16 there with
--token-embedding-type f16 --output-tensor-type f16 so the latent rows stay exact).

  python quantization_kit/tools/gguf_add_latent_tokens.py text-f16.gguf latent_queries.npy text-f16-latent.gguf
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

NP = {gguf.GGMLQuantizationType.F16: np.float16, gguf.GGMLQuantizationType.F32: np.float32}


def main(src, lq_path, dst):
    lq = np.load(lq_path).reshape(-1, np.load(lq_path).shape[-1]).astype(np.float32)   # [n_query, n_embd]
    n_q = lq.shape[0]
    new_tokens = [f"<|latent_{i}|>" for i in range(n_q)]
    r = gguf.GGUFReader(src)
    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(dst, arch=arch)

    tok_f = r.fields[gguf.Keys.Tokenizer.LIST]
    n_vocab = len(tok_f.data)
    assert all(t not in new_tokens for t in tok_f.contents()), "latent tokens already present"
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == gguf.Keys.Tokenizer.LIST:
            val = list(val) + new_tokens
        elif field.name == gguf.Keys.Tokenizer.TOKEN_TYPE:
            val = list(val) + [int(gguf.TokenType.USER_DEFINED)] * n_q
        elif field.name == gguf.Keys.Tokenizer.SCORES:
            val = list(val) + [0.0] * n_q
        w.add_key_value(field.name, val, vt, sub_type=st)

    datas = {}
    for t in r.tensors:
        d = t.data
        if t.name in ("token_embd.weight", "output.weight"):
            assert t.tensor_type in NP, f"{t.name} is {t.tensor_type.name}: inject before quantizing"
            assert d.shape == (n_vocab, lq.shape[1]), f"{t.name} shape {d.shape} != ({n_vocab}, {lq.shape[1]})"
            extra = lq if t.name == "token_embd.weight" else np.zeros_like(lq)
            d = np.concatenate([d, extra.astype(NP[t.tensor_type])], 0)
            if t.name == "token_embd.weight":
                back = d[n_vocab:].astype(np.float32)
                print(f"[latent] {t.name}: +{n_q} rows ({t.tensor_type.name}); max |row - latent_queries| = "
                      f"{np.abs(back - lq).max():.3e}")
        datas[t.name] = (d, t.tensor_type)
        w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(datas[t.name][0], tensor_endianess=r.endianess)
    w.close()
    print(f"[latent] vocab {n_vocab} -> {n_vocab + n_q}: {new_tokens} -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
