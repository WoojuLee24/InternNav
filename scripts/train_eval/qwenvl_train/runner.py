"""Pure-Python train+eval engine for qwenvl_train experiments.

Why this exists
---------------
The old flow was ``train_x.sh`` -> ``eval_dual_system_mini_8gpu.sh`` -> ``torchrun``
(shell calling shell calling shell). That is undebuggable, and train/eval params
drifted apart (each layer re-declared values, and the eval default config did not
even exist on disk).

This module is the single, debuggable orchestrator — a pure engine. The config it
runs (the ``Params`` dataclass + the eval-config builders) lives in its own file,
``default_config.py``, and is selected with ``--config`` (mirroring
``scripts/eval/eval.py --config``). An experiment is described by one ``Params``
dataclass (the single source of truth); the SAME ``Params`` instance is used to:
  1. build the trainer ``torchrun`` command (``Params.train_argv``), and
  2. build the Habitat / h1 eval config (``default_config.build_eval_cfg``),
so training and evaluation can never disagree on ``num_history``, ``resize_*``,
``predict_step_num`` or ``num_future_steps``.

Usage::
    python runner.py                              # default --config (b4 baseline)
    python runner.py --config batch_size/b2_eff128.py

Only two subprocess calls are made (train, then eval) — both are ``torchrun``
(distributed training/eval needs the launcher; ``python file.py`` would run a single
process). Everything around them is plain Python you can step through with pdb.

This file is NEW and touches no existing code.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from typing import List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# The config (Params + eval-cfg builders) lives in its own file, separated out so
# this module stays a pure train+eval engine. runner imports Params for type hints
# and uses default_config.py as the default --config (the b4 baseline).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from default_config import Params  # noqa: E402


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _run_and_tee(cmd: List[str], log_path: Optional[str], env=None) -> int:
    """Run cmd, streaming combined stdout+stderr to console and optionally a log file
    (the Python equivalent of ``... 2>&1 | tee log``).

    Bytes are forwarded to the console RAW (no newline translation) so tqdm's ``\\r``
    progress updates overwrite the same line in place — the bar stays one line instead
    of one line per step. The log file gets the same stream with ``\\r`` normalized to
    ``\\n`` so it is greppable / tail-able.
    """
    import codecs

    print("[run] " + " ".join(cmd), flush=True)
    fp = open(log_path, "w") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # bytes mode (no text=): keep '\r' intact for tqdm
        )
        out = sys.stdout.buffer
        dec = codecs.getincrementaldecoder("utf-8")("replace")  # log only; tolerate split chars
        while True:
            chunk = proc.stdout.read(1024)
            if not chunk:
                break
            out.write(chunk)            # raw -> '\r' refreshes the progress bar in place
            out.flush()
            if fp:
                fp.write(dec.decode(chunk).replace("\r\n", "\n").replace("\r", "\n"))
                fp.flush()
        return proc.wait()
    finally:
        if fp:
            fp.close()


def run_train(p: Params, output_dir: str, run_name: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    master_port = p.master_port or _pick_port()
    launcher = [
        "torchrun",
        f"--nnodes={p.nnodes}",
        f"--nproc_per_node={p.nproc_per_node}",
        f"--node_rank={p.node_rank}",
        f"--master_addr={p.master_addr}",
        f"--master_port={master_port}",
        "internnav/trainer/internvla_n1_trainer.py",
    ]
    cmd = launcher + p.train_argv(output_dir, run_name)
    return _run_and_tee(cmd, os.path.join(output_dir, "train.log"))


def _pick_port() -> int:
    # deterministic-ish, avoids importing random just for a port
    return 20001 + (os.getpid() % 101)


def find_best_checkpoint(output_dir: str) -> Optional[str]:
    """Lowest best_metric across checkpoint-*/trainer_state.json (== shell logic)."""
    best, best_loss = None, float("inf")
    for f in glob.glob(os.path.join(output_dir, "checkpoint-*", "trainer_state.json")):
        try:
            loss = json.load(open(f)).get("best_metric")
        except Exception:
            continue
        if loss is not None and loss < best_loss:
            best_loss, best = loss, os.path.dirname(f)
    return best


def _prep_ckpt_aux_files(output_dir: str, ckpt: str, system2_ckpt: str) -> None:
    import shutil
    for src, name in [
        (os.path.join(output_dir, "preprocessor_config.json"), "preprocessor_config.json"),
        (os.path.join(system2_ckpt, "chat_template.json"), "chat_template.json"),
    ]:
        try:
            shutil.copy(src, os.path.join(ckpt, name))
        except Exception:
            pass


def _wandb_resume_env(base_env=None) -> dict:
    """Resume the training wandb run so eval metrics land in the same run."""
    env = dict(base_env or os.environ)
    runs = sorted(glob.glob(os.path.join(REPO_ROOT, "wandb", "run-*")), key=os.path.getmtime, reverse=True)
    if runs:
        base = os.path.basename(runs[0])  # run-<date>_<time>-<id>
        run_id = base.split("-", 2)[-1] if base.count("-") >= 2 else None
        if run_id:
            env["WANDB_RUN_ID"] = run_id
            env["WANDB_RESUME"] = "allow"
    return env


def run_eval(config_path: str, model_path: str, run_name: str, output_dir: str,
             eval_target: str = "habitat", nproc: int = 8, master_port: int = 2333) -> int:
    env = _wandb_resume_env()
    env["TRAIN_EVAL_TARGET"] = eval_target  # read by the experiment config.py
    cmd = [
        "torchrun", f"--nproc_per_node={nproc}", f"--master_port={master_port}",
        "scripts/eval/eval.py", "--config", config_path, "--quiet",
        "--model_path", model_path, "--wandb_run_name", run_name,
    ]
    return _run_and_tee(cmd, os.path.join(output_dir, "test.log"), env=env)


def train_and_eval(p: Params, exp_name: str, config_path: str, *,
                   eval_target: str = "habitat", do_train: bool = True, do_eval: bool = True,
                   checkpoints_root: str = "/home/irteam/data-vol2/checkpoints",
                   model_path: Optional[str] = None) -> None:
    """End-to-end driver. ``exp_name`` e.g. 'batch_size/b4_eff128_base'."""
    run_name = f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(checkpoints_root, run_name)

    if do_train:
        rc = run_train(p, output_dir, run_name)
        if rc != 0:
            print(f"[train] FAILED (exit {rc}); skipping eval.", file=sys.stderr)
            return

    if not do_eval:
        return

    # node_rank guard mirrors `if [ "${NODE_RANK}" = "0" ]` in the shell scripts
    if p.node_rank != 0:
        return

    best = model_path or find_best_checkpoint(output_dir)
    if not best:
        print("[eval] No best checkpoint found, skipping eval.")
        return
    print(f"[eval] target={eval_target}  checkpoint={best}")
    if do_train:
        _prep_ckpt_aux_files(output_dir, best, p.system2_ckpt)
    os.makedirs(output_dir, exist_ok=True)
    run_eval(config_path, best, run_name, output_dir, eval_target=eval_target, nproc=p.nproc_per_node)


# --------------------------------------------------------------------------- #
# CLI entrypoint: python runner.py --config <experiment config .py>
# --------------------------------------------------------------------------- #
# Mirrors the repo convention `python scripts/eval/eval.py --config <cfg.py>`.
# The SAME config file is read here for training PARAMS and passed to eval.py for
# eval_cfg, so train and eval can never diverge. Experiment configs live next to
# batch_size/default_config.py and override only the knobs they change.
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "default_config.py")


def _load_config(config_path: str):
    """Load an experiment config .py file and return (module, abs_path)."""
    import importlib.util

    config_path = os.path.abspath(config_path)
    parent = os.path.dirname(config_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)  # let the config `import default_config`
    spec = importlib.util.spec_from_file_location("exp_config_" + os.path.basename(config_path), config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, config_path


def _derive_exp_name(config_path: str) -> str:
    """e.g. .../batch_size/b2_eff128.py -> 'batch_size/b2_eff128'."""
    parent = os.path.basename(os.path.dirname(config_path))
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return f"{parent}/{stem}"


def main_cli() -> None:
    """`python runner.py --config <cfg.py>` — config-file driven train+eval."""
    ap = argparse.ArgumentParser(description="qwenvl_train train+eval driver (config-file based)")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="experiment config .py exposing PARAMS (+ optional EXP_NAME). "
                         "Defaults to batch_size/default_config.py (the b4 baseline).")
    ap.add_argument("--eval-target", choices=["h200", "5090", "h1"], default="h200",
                    help="h200/5090 = Habitat machine config; h1 = Isaac Sim (manual)")
    ap.add_argument("--no-train", action="store_true", help="skip training (eval an existing ckpt)")
    ap.add_argument("--no-eval", action="store_true", help="train only")
    ap.add_argument("--model-path", default=None, help="ckpt to eval when --no-train")
    ap.add_argument("--data-root", default=None, help="override dataset root")
    ap.add_argument("--checkpoints-root", default="/home/irteam/data-vol2/checkpoints")
    ap.add_argument("--print-train-argv", action="store_true", help="print resolved train flags and exit")
    args = ap.parse_args()

    mod, config_path = _load_config(args.config)
    params: Params = mod.PARAMS
    exp_name = getattr(mod, "EXP_NAME", None) or _derive_exp_name(config_path)

    if args.data_root:
        params = replace(params, data_root=args.data_root)

    if args.print_train_argv:
        print(" ".join(params.train_argv("<output_dir>", exp_name)))
        return

    train_and_eval(
        params, exp_name, config_path,
        eval_target=args.eval_target,
        do_train=not args.no_train,
        do_eval=not args.no_eval,
        checkpoints_root=args.checkpoints_root,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main_cli()
