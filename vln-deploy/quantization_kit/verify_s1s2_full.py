"""Full S2 -> S1 pipeline on the dataset: {HF bf16, GGUF} S2 x {PyTorch bf16, TensorRT FP16} S1.

Reuses the S2 replays of verify_s2_gguf.py (same VLN-CE frames, same fixed S2 schedule: S2 every 5th step, history
in between). Each S2 pixel-goal answer drives S1 on the S2 frame and the following 4 frames (S1 keeps running on
newer frames with the same latent until the next S2). The S1 goal image is the frame the pixel goal refers to
(look-down frame if S2 looked down). Every S1 call of a (episode, S2 step, frame) case starts from the same initial
noise in all four combinations, so the differences come only from S2 (latent/decision) and S1 (backend).
  ref  (gd_vln habitat image)  S1 = original checkpoint PyTorch bf16, 10 steps (agent)      -> out/s1s2_full_pt.npz
  trt  (isaacsim image)        S1 = deploy configuration: S1LiteModel + TRT FP16 repro engines (denoise + DINOv2 +
                               memory encoder/resampler), CPU PIL prep (--no-gpu-prep), 10 steps -> out/s1s2_full_trt.npz
  cmp  (any)                   all combos vs HF+PyTorch: decision agreement per S1 call, trajectory ADE/FDE, first
                               action agreement, latency                                     -> out/report_s1s2_full.json
  /usr/bin/python3 quantization_kit/verify_s1s2_full.py ref                                   # gd_vln, from vln-deploy/
  quantization_kit/convenv_x86/bin/python quantization_kit/verify_s1s2_full.py trt            # isaacsim, from vln-deploy/
  /usr/bin/python3 quantization_kit/verify_s1s2_full.py ref --noise_salt 1                   # optional noise baseline
  python3 quantization_kit/verify_s1s2_full.py cmp
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

QK = Path(__file__).resolve().parent
ROOT = QK.parent                      # vln-deploy
INTERNNAV = ROOT.parent
S2_OUT = QK / "verify_s2_gguf_out"
OUT = QK / "verify_s2_gguf_out"
GAP = 5                               # verify_s2_gguf.py schedule: S2 every 5th step
STEPS = 10


def cases_for(tag):
    """[(key, row_index, ep, s2_step, cur_step, looked_down)] for every S2 pixel-goal answer x frames until next S2."""
    sys.path.insert(0, str(QK))
    from verify_s2_gguf import episodes

    n_frames = {ep["id"]: len(ep["fpv"]) for ep in episodes()}
    rows = list(np.load(S2_OUT / f"{tag}.npy", allow_pickle=True))
    out = []
    for i, r in enumerate(rows):
        if r["latent"] is None:
            continue
        for k in range(GAP):
            cur = r["step"] + k
            if cur < n_frames[r["ep"]]:
                out.append((f"{r['ep']}|{r['step']}|{cur}", i, r["ep"], r["step"], cur, bool(r["looked_down"])))
    return rows, out


def frames():
    sys.path.insert(0, str(QK))
    from verify_s2_gguf import episodes, load

    fr = {ep["id"]: (ep["fpv"], ep["look"]) for ep in episodes()}
    return fr, load


SALT = ""                             # --noise_salt: a different (but fixed) initial noise per case


def noise_for(key):
    return np.random.default_rng(zlib.crc32((key + SALT).encode())).standard_normal((32, 32, 3)).astype(np.float32)


class FixedNoise:
    def __init__(self):
        self.cur = None

    def __call__(self, shape, generator=None, device=None, dtype=None, **kw):
        import torch

        return torch.from_numpy(self.cur).reshape(shape).to(device=device, dtype=dtype)


def wp_act(dp):
    from internnav.model.utils.vln_utils import traj_to_actions

    wp = np.asarray(traj_to_actions(dp.clone(), use_discrate_action=False), np.float32)
    act = [int(a) for a in traj_to_actions(dp.clone()) if a != 0][:4]
    return wp, act


def run_backend(a, s1_fn, tag_out):
    """S1 on every S2 source; one file per (backend, source): out/s1s2_full_<backend>__<source>.npz"""
    fr, load = frames()
    for src in [a.ref_tag] + a.test_tags:
        rows, cases = cases_for(src)
        wps, acts, keys, ms = [], [], [], []
        for key, i, ep, step, cur, looked in cases:
            fpv, look = fr[ep]
            goal = load(look[step]) if looked else load(fpv[step])
            wp, act, t = s1_fn(rows[i]["latent"].reshape(1, 4, -1).astype(np.float32), goal, load(fpv[cur]), noise_for(key))
            wps.append(wp), acts.append(json.dumps(act)), keys.append(key), ms.append(t)
        np.savez(OUT / f"s1s2_full_{tag_out}__{src}.npz", wp=np.stack(wps), act=np.array(acts), key=np.array(keys),
                 ms=np.array(ms))
        print(f"[{tag_out}] {src}: {len(cases)} S1 calls, p50 {np.median(ms[3:]):.1f} ms", flush=True)


def timed(fn):
    import torch

    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


def phase_ref(a):
    sys.path[:0] = [str(INTERNNAV), str(INTERNNAV / "scripts/sim"), str(ROOT), str(INTERNNAV / "src/diffusion-policy")]
    import torch
    from PIL import Image

    import internnav.model.basemodel.internvla_n1.internvla_n1 as n1
    from deploy.s1_module.service_hf import load_s1_only

    noise = FixedNoise()
    n1.randn_tensor = noise
    model = load_s1_only(a.ckpt, torch.device("cuda:0")).to(torch.bfloat16)
    pr = lambda x: np.array(Image.fromarray(x).resize((224, 224))) / 255  # noqa: E731  (agent S1 preprocessing)

    def s1(lat, goal, cur, nz):
        rgbs = torch.stack([torch.from_numpy(pr(goal)), torch.from_numpy(pr(cur))]).unsqueeze(0).cuda()
        depths = torch.zeros(1, 2, 224, 224, 1, dtype=torch.float64).cuda()
        noise.cur = nz
        with torch.inference_mode():
            dp, t = timed(lambda: model.generate_traj(torch.from_numpy(lat).cuda().to(torch.bfloat16), rgbs, depths,
                                                      num_inference_steps=STEPS))
        return (*wp_act(dp.float()), t)

    run_backend(a, s1, "pt" + (f"_salt{SALT}" if SALT else ""))


def phase_trt(a):
    sys.path[:0] = [str(ROOT), str(QK), str(INTERNNAV)]
    import torch

    import quantization_kit.s1_inference as s1i
    import quantization_kit.trt_s1 as trt_s1
    from quantization_kit.s1_inference import S1LiteModel, prep_images

    noise = FixedNoise()
    s1i.randn_tensor = noise
    trt_s1.randn_tensor = noise
    dev = "cuda:0"
    m = S1LiteModel(a.ckpt).load_s1_weights(a.ckpt).to(dev, torch.float32).eval()
    trt_s1.attach_trt_denoise(m, onnx_dir=a.onnx_dir, engine_dir=a.engine_dir, trt_cond=True)

    def s1(lat, goal, cur, nz):
        z = np.zeros(cur.shape[:2], np.float32)
        noise.cur = nz
        with torch.no_grad():
            def go():
                rgbs, depths = prep_images(goal, z, cur, z, dev, torch.float32)   # --no-gpu-prep (sim stack)
                return m.generate_traj(traj_latents=torch.from_numpy(lat).to(dev), images_dp=rgbs, depths_dp=depths,
                                       num_inference_steps=STEPS)
            dp, t = timed(go)
        return (*wp_act(dp.float()), t)

    run_backend(a, s1, "trt")


def first_yaw_deg(wp):
    """yaw (deg) of the first target the discretizer turns to (lookahead 4 from the start) -> distance of
    yaw/15 deg to the rounding boundary (x.5) tells how close the first discrete action is to flipping."""
    if np.linalg.norm(wp[0] - wp[-1]) <= 0.2:
        return None
    d = wp[min(4, len(wp) - 1)] - wp[0]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def phase_cmp(a):
    def load_combo(backend, src):
        f = OUT / f"s1s2_full_{backend}__{src}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        return {k: (w, json.loads(c)) for k, w, c in zip(d["key"], d["wp"], d["act"])}, d["ms"]

    s2 = {t: list(np.load(S2_OUT / f"{t}.npy", allow_pickle=True)) for t in [a.ref_tag] + a.test_tags}

    def decisions(src):   # per S1 slot (ep|s2_step|cur): "traj" or the discrete S2 action; final S2 text
        dec, txt, lat = {}, {}, {}
        for r in s2[src]:
            for k in range(GAP):
                s = f"{r['ep']}|{r['step']}|{r['step'] + k}"
                dec[s] = "traj" if r["latent"] is not None else json.dumps(r["action"])
                txt[s] = r["text"].split("  >  ")[-1]
                lat[s] = r["latent"]
        return dec, txt, lat

    ref_tab, _ = load_combo("pt", a.ref_tag)
    ref_dec, ref_txt, ref_lat = decisions(a.ref_tag)
    combos = {"HF+TRT": ("trt", a.ref_tag)}
    for t in a.test_tags:
        combos[f"{t}+PyTorch"] = ("pt", t)
        combos[f"{t}+TRT"] = ("trt", t)
    combos["HF+PyTorch(noise 변경)"] = ("pt_salt1", a.ref_tag)
    report = {}
    for name, (backend, src) in combos.items():
        got = load_combo(backend, src)
        if got is None:
            continue
        tab, ms = got
        dec, txt, lat = decisions(src)
        slots = [s for s in ref_dec if s in dec]
        traj = [s for s in slots if ref_dec[s] == "traj" and dec[s] == "traj"]
        ade = {s: float(np.linalg.norm(tab[s][0] - ref_tab[s][0], axis=-1).mean()) for s in traj}
        fde = {s: float(np.linalg.norm(tab[s][0][-1] - ref_tab[s][0][-1])) for s in traj}
        first_ok = {s: tab[s][1][:1] == ref_tab[s][1][:1] for s in traj}
        same_txt = [s for s in traj if txt[s] == ref_txt[s]]
        diff_txt = [s for s in traj if txt[s] != ref_txt[s]]
        cos = lambda x, y: float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y)))  # noqa: E731
        # first-action flips vs closeness of the reference first yaw to the 15 deg rounding boundary
        margin = {}
        for s in traj:
            y = first_yaw_deg(ref_tab[s][0])
            margin[s] = None if y is None else abs(abs(y / 15.0) - np.floor(abs(y / 15.0)) - 0.5) * 15.0
        flips = [s for s in traj if not first_ok[s]]
        report[name] = {
            "slots": len(slots),
            "1_s2_decision_mismatch_slots": sum(ref_dec[s] != dec[s] for s in slots),
            "2_traj_slots": len(traj),
            "2a_first_action_mismatch": len(flips),
            "2b_flip_ref_margin_deg (distance to 7.5/22.5 deg boundary)": [None if margin[s] is None else round(margin[s], 1) for s in flips],
            "2c_flip_ade_cm": [round(ade[s] * 100, 1) for s in flips],
            "command_agree(first action)": float(np.mean([(ref_dec[s] == dec[s]) and (ref_dec[s] != "traj" or first_ok[s]) for s in slots])),
            "traj_agree(ADE<5cm)": float(np.mean([(ref_dec[s] == dec[s]) and (ref_dec[s] != "traj" or ade[s] < 0.05) for s in slots])),
            "traj_agree(ADE<2cm)": float(np.mean([(ref_dec[s] == dec[s]) and (ref_dec[s] != "traj" or ade[s] < 0.02) for s in slots])),
            "ade_median_cm": float(np.median(list(ade.values())) * 100), "ade_p90_cm": float(np.percentile(list(ade.values()), 90) * 100),
            "fde_median_cm": float(np.median(list(fde.values())) * 100),
            "same_s2_text: n / first_agree / ade_median_cm / latent_cos": [
                len(same_txt), float(np.mean([first_ok[s] for s in same_txt])) if same_txt else None,
                float(np.median([ade[s] for s in same_txt]) * 100) if same_txt else None,
                float(np.median([cos(lat[s], ref_lat[s]) for s in same_txt])) if same_txt else None],
            "diff_s2_text: n / first_agree / ade_median_cm / latent_cos": [
                len(diff_txt), float(np.mean([first_ok[s] for s in diff_txt])) if diff_txt else None,
                float(np.median([ade[s] for s in diff_txt]) * 100) if diff_txt else None,
                float(np.median([cos(lat[s], ref_lat[s]) for s in diff_txt])) if diff_txt else None],
            "s1_ms_p50": float(np.median(ms[3:])),
        }
    margins_all = [m for m in (abs(abs(first_yaw_deg(w) / 15.0) - np.floor(abs(first_yaw_deg(w) / 15.0)) - 0.5) * 15.0
                               for w, _ in ref_tab.values() if first_yaw_deg(w) is not None)]
    report["ref_margin_distribution_deg"] = {"n": len(margins_all), "lt1": int(sum(m < 1 for m in margins_all)),
                                             "lt2": int(sum(m < 2 for m in margins_all)),
                                             "lt3": int(sum(m < 3 for m in margins_all))}
    (OUT / "report_s1s2_full.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ref", "trt", "cmp"])
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/InternVLA-N1-DualVLN"))
    ap.add_argument("--ref_tag", default="hf_bf16_384", help="S2 reference replay (verify_s2_gguf_out/<tag>.npy)")
    ap.add_argument("--test_tags", nargs="+", default=["hfp_gguf_q8", "hfp_gguf_f16", "hfp_gguf_q4"],
                    help="S2 compressed replays (verify_s2_gguf_out/<tag>.npy)")
    ap.add_argument("--onnx_dir", default=str(QK / "out/onnx"), help="[trt] S1 ONNX (engine rebuild source)")
    ap.add_argument("--engine_dir", default=str(QK / "out/trt_engines"), help="[trt] S1 TRT engines (<tag>_sm<cc>.engine)")
    ap.add_argument("--noise_salt", default="", help="[ref] other fixed initial noise -> out/s1s2_full_pt_salt<salt>.npz")
    a = ap.parse_args()
    global SALT
    SALT = a.noise_salt
    if a.phase == "ref":
        phase_ref(a)
    elif a.phase == "trt":
        phase_trt(a)
    elif a.phase == "cmp":
        phase_cmp(a)
    else:
        assert False, a.phase


if __name__ == "__main__":
    main()
