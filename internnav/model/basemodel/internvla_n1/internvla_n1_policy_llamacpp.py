"""Dual-system policy: System-2 on llama.cpp via the persistent `llama-s2-runner` CLI
(one subprocess, model loaded once), System-1 on the lite diffusion model. Drop-in for
InternVLAN1Agent's `self.policy` (s2_step / s1_step_latent / step_no_infer / reset / eval).

No HTTP, no two servers, no base64-over-the-wire: the runner does mtmd image + text generation
+ traj_latents extraction in-process (solves the embeddings-mode image limitation). Parity:
vision cosine 1.0 · text/pixel l2_median 0 (@448) · latent hidden-state cosine 0.999.

Pipe protocol (see tools/mtmd/s2-runner.cpp):
  RESET                                  -> OK
  TEXT  \t img1,img2,.. \t <prompt>      -> OK\t<text>     (<__media__> marks image slots)
  LATENT                                 -> OK\t<base64 float32[4*n_embd]>
"""
import base64
import itertools
import os
import re
import subprocess
import tempfile
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from internnav.model.utils.vln_utils import (
    S1Output,
    S2Output,
    chunk_token,
    traj_to_actions,
)

MEDIA = "<__media__>"      # mtmd default marker (must match mtmd_default_marker())
SYSTEM_PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? "
    "Please output the next waypoint's coordinates in the image. "
    "Please output STOP when you have successfully completed the task."
)
CONJUNCTIONS = [
    "you can see ", "in front of you is ", "there is ", "you can spot ",
    "you are toward the ", "ahead of you is ", "in your sight is ",
]
N_QUERY = 4


class InternVLAN1LlamaCppPolicy:
    def __init__(
        self,
        runner_bin,            # path to llama-s2-runner
        model_gguf,            # text-f16-latent.gguf (with latent tokens)
        mmproj_gguf,           # mmproj-f16.gguf
        lib_dir,               # build_x86/bin (for LD_LIBRARY_PATH)
        s1_model_path="checkpoints/InternVLA-N1-DualVLN",
        device="cuda:0",
        num_history=8,
        resize=448,            # 112-aligned; REQUIRED for llama.cpp<->HF parity
        max_new_tokens=128,
        ngl=99,
        n_ctx=8192,
        continuous_traj=True,
        s1_steps=5,            # diffusion denoise steps: 5 -> ~30 Hz on Thor (paper's S1 rate); 10 -> ~16 Hz, higher quality
        s1_dtype=torch.float32,
        use_trt_s1=False,      # native TensorRT FP16 denoise loop (~6-7x faster S1 on Jetson)
        kv_reuse=False,        # multi-turn KV: append only the new frame each step (cuts in-episode re-encode)
        lookdown_full_res=False,  # look_down frame at camera resolution (matches internvla_n1_policy.py:127-128)
        load_s1=True,          # build/load System-1 in-process; False for an S2-only service (async ZMQ split)
        runner_endpoint=None,  # if set (e.g. "tcp://127.0.0.1:5610"), talk to llama-s2-runner over ZMQ REP
        spawn_runner=True,     # spawn the runner process (vs. connect to an already-running --zmq endpoint)
    ):
        self.device = device
        self.num_history = num_history
        self.kv_reuse = kv_reuse
        self.lookdown_full_res = lookdown_full_res
        self.resize = resize
        self.continuous_traj = continuous_traj
        self.s1_steps = s1_steps
        self.s1_dtype = s1_dtype
        self.actions2idx = OrderedDict({"STOP": [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})
        self._tmp = tempfile.mkdtemp(prefix="s2runner_")
        self._img_n = 0

        env = dict(os.environ, LD_LIBRARY_PATH=lib_dir)
        runner_args = [runner_bin, "-m", model_gguf, "--mmproj", mmproj_gguf,
                       "-ngl", str(ngl), "-c", str(n_ctx), "-n", str(max_new_tokens),
                       # Disable llama.cpp's auto memory-fit. It loads the model just to probe device
                       # memory via cudaMemGetInfo, which CRASHES the runner under MPS (Tegra). We pin
                       # -ngl/-c explicitly so the fit is unnecessary; skipping it is safe and avoids the crash.
                       "--fit", "off",
                       "--temp", "0", "--repeat-penalty", "1.05", "--repeat-last-n", "-1"]
        self._runner_mode = "zmq" if runner_endpoint else "pipe"
        self._zmq = None
        self.proc = None
        if self._runner_mode == "pipe":
            # default: persistent subprocess over stdin/stdout (unchanged behaviour)
            self.proc = subprocess.Popen(
                runner_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                env=env, text=True, bufsize=1,
            )
        else:
            # async split: talk to the runner over ZMQ REP (--zmq); it can live in another process/host
            import zmq
            if spawn_runner:
                self.proc = subprocess.Popen(
                    runner_args + ["--zmq", runner_endpoint],
                    stderr=subprocess.PIPE, env=env, text=True,
                )
                self._wait_runner_listening()
            self._zmq = zmq.Context.instance().socket(zmq.REQ)
            self._zmq.connect(runner_endpoint)

        self.s1 = None
        if load_s1:
            from quantization_kit.s1_inference import S1LiteModel
            self.s1 = S1LiteModel(s1_model_path).load_s1_weights(s1_model_path).to(device, s1_dtype).eval()
            if use_trt_s1:
                from quantization_kit.trt_s1 import attach_trt_denoise
                attach_trt_denoise(self.s1)   # swaps generate_traj's per-step DiT to a native TRT FP16 engine
        self.reset()

    def _wait_runner_listening(self, timeout=300):
        """Block until the --zmq runner announces it is listening. FAIL FAST if it dies during load
        (e.g. a CUDA/MPS error) instead of returning 'ready' and then hanging forever on the first
        REQ to a dead endpoint."""
        import threading
        self._ready = threading.Event()
        self._runner_died = threading.Event()

        def _drain():
            for line in iter(self.proc.stderr.readline, ""):
                if "ZMQ REP listening" in line or "s2-runner: ready" in line:
                    self._ready.set()
            if not self._ready.is_set():      # hit EOF without ever listening -> the runner died
                self._runner_died.set()
            self._ready.set()                 # unblock the waiter regardless

        threading.Thread(target=_drain, daemon=True).start()
        if not self._ready.wait(timeout):
            raise RuntimeError("llama-s2-runner did not reach ZMQ-listening state in time")
        if self._runner_died.is_set() or self.proc.poll() is not None:
            rc = self.proc.poll()
            try: self.proc.kill()
            except Exception: pass
            raise RuntimeError(
                f"llama-s2-runner exited (rc={rc}) before listening -- it crashed during model load "
                f"(see its backtrace above; common cause: a CUDA/MPS error). Retry on a clean GPU, "
                f"or run without MPS (--s2-mps-pct 0).")

    # ---- runner transport (stdin/stdout pipe, or ZMQ REP) ----
    def _cmd(self, line):
        if self._zmq is not None:
            self._zmq.send_string(line)
            resp = self._zmq.recv_string()
        else:
            self.proc.stdin.write(line + "\n"); self.proc.stdin.flush()
            resp = self.proc.stdout.readline().rstrip("\n")
        if not resp.startswith("OK"):
            raise RuntimeError(f"s2-runner: {resp!r}")
        return resp[3:] if len(resp) > 2 else ""

    def _save(self, pil):
        p = os.path.join(self._tmp, f"i{self._img_n}.png"); self._img_n += 1
        pil.convert("RGB").save(p); return p

    # ---- policy interface ----
    def eval(self):
        return self

    def reset(self):
        self.rgb_list = []
        self.episode_idx = 0
        self.llm_output = ""
        self._convo_open = False
        try:
            self._cmd("RESET")
        except Exception:
            pass

    def parse_actions(self, output):
        regex = re.compile("|".join(re.escape(a) for a in self.actions2idx))
        return list(itertools.chain.from_iterable(self.actions2idx[m] for m in regex.findall(output)))

    @staticmethod
    def _parse_pixel(text):
        coord = [int(c) for c in re.findall(r"\d+", text)]
        return np.array([int(coord[1]), int(coord[0])]) if len(coord) >= 2 else None

    def _resize(self, rgb):
        return Image.fromarray(rgb).convert("RGB").resize((self.resize, self.resize))

    def step_no_infer(self, rgb, depth, pose):
        self.rgb_list.append(self._resize(rgb))
        self.episode_idx += 1

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        if look_down and self.lookdown_full_res:
            cur = Image.fromarray(rgb).convert("RGB")  # camera resolution, as in the unquantized policy
        else:
            cur = self._resize(rgb)
        if not look_down:
            self.rgb_list.append(cur)
            prompt = SYSTEM_PROMPT_TEMPLATE.replace("<instruction>.", instruction)
            if self.kv_reuse:
                # multi-turn KV: keep the runner's conversation open and append ONLY the new frame.
                # Prior frames stay in KV as history -> no per-step RESET, no history re-encode.
                images = [cur]
            else:
                self._cmd("RESET")           # fresh conversation each step (re-encodes sampled history)
                if self.episode_idx == 0:
                    hist_ids = []
                else:
                    hist_ids = sorted(np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist())
                    prompt += " These are your historical observations: " + (MEDIA + "\n") * len(hist_ids) + "."
                images = [self.rgb_list[i] for i in hist_ids] + [cur]
            self.episode_idx += 1
        else:
            assert self.llm_output != "", "look_down requires a previous output"
            images = [cur]               # look-down frame (not added to history); runner keeps multi-turn KV
            prompt = ""
        prompt += f" {CONJUNCTIONS[0]}{MEDIA}."

        img_paths = ",".join(self._save(im) for im in images)
        esc = prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", " ")
        self.llm_output = self._cmd(f"TEXT\t{img_paths}\t{esc}")

        out = S2Output()
        if bool(re.search(r"\d", self.llm_output)):       # pixel goal
            out.output_pixel = self._parse_pixel(self.llm_output)
            lat_b64 = self._cmd("LATENT")
            v = np.frombuffer(base64.b64decode(lat_b64), dtype=np.float32).reshape(N_QUERY, -1)
            out.output_latent = torch.from_numpy(v.copy()).unsqueeze(0)   # [1,4,H]
        else:                                              # discrete action(s)
            out.output_action = self.parse_actions(self.llm_output)
        return out

    @torch.no_grad()
    def s1_step_latent(self, rgb, depth, latent):
        latent = latent.to(self.device, self.s1_dtype)
        rgb = (rgb if torch.is_tensor(rgb) else torch.as_tensor(rgb)).to(self.device, self.s1_dtype)
        depth = torch.as_tensor(depth).to(self.device, self.s1_dtype) if depth is not None else None
        dp = self.s1.generate_traj(traj_latents=latent, images_dp=rgb, depths_dp=depth, num_inference_steps=self.s1_steps)
        actions = traj_to_actions(dp.clone()) if self.continuous_traj else chunk_token(dp[np.random.choice(dp.shape[0])])
        actions = [a for a in actions if a != 0]
        return S1Output(idx=actions[:4])

    def __del__(self):
        try:
            if getattr(self, "_zmq", None) is not None:
                try: self._zmq.send_string("QUIT"); self._zmq.recv_string()
                except Exception: pass
            elif getattr(self, "proc", None) is not None and self.proc.stdin is not None:
                self.proc.stdin.write("QUIT\n"); self.proc.stdin.flush()
            if getattr(self, "proc", None) is not None:
                self.proc.wait(timeout=5)
        except Exception:
            try: self.proc.kill()
            except Exception: pass
