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
H1_PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"

# The config (Params + eval-cfg builders) lives in its own file, separated out so
# this module stays a pure train+eval engine. runner imports Params for type hints
# and uses default_config.py as the default --config (the b4 baseline).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # make `internnav` importable without pip install -e .
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
    os.umask(0o000)  # child processes inherit this → new dirs/files get 777/666
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


def _kill_group(proc, term_wait: int = 15) -> None:
    """Kill the whole process group of `proc` (SIGTERM, then SIGKILL). `proc` must
    have been started with start_new_session=True so it is its own group leader,
    which lets us take down the eval's child tree (python.sh -> eval.py + Isaac)."""
    import signal
    import time

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < term_wait:
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _wait_gpu_free(gpu: Optional[str], threshold_mb: int = 2000, max_wait: int = 300) -> None:
    """Best-effort: block until GPU `gpu`'s used memory drops below threshold, so a
    relaunch doesn't OOM against the just-killed process's still-freeing memory."""
    import time

    if not gpu:
        time.sleep(10)
        return
    print(f"[watchdog] waiting for GPU {gpu} memory to free (<{threshold_mb} MiB)...", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                capture_output=True, text=True, timeout=15,
            )
            used = int(r.stdout.strip().splitlines()[0])
            if used < threshold_mb:
                print(f"[watchdog] GPU {gpu} free ({used} MiB).", flush=True)
                return
        except Exception:
            pass
        time.sleep(5)
    print(f"[watchdog] GPU {gpu} still busy after {max_wait}s; relaunching anyway.", flush=True)


def _run_with_watchdog(cmd: List[str], log_path: Optional[str], env, idle_limit: int,
                       max_noprogress: int = 3, gpu: Optional[str] = None) -> int:
    """Like _run_and_tee, but for the h1 (Isaac Sim) eval which intermittently HANGS
    inside env.step() (no output, GPU idle, process alive). Streams the child's output
    and, if nothing is emitted for `idle_limit` seconds, kills the process group and
    relaunches the SAME cmd — the eval auto-resumes (lmdb skips finished episodes;
    wandb resumes via <model_path>/wandb_run_id.txt). Gives up after `max_noprogress`
    consecutive relaunches that complete no new episode (guards an unrecoverable loop,
    e.g. a missing chat_template.json that makes S2 error every step).

    Returns 0 on the eval's own clean exit; the last non-zero rc (or 1) on give-up.
    """
    import codecs
    import select

    def _completed() -> int:
        if not log_path or not os.path.exists(log_path):
            return 0
        try:
            with open(log_path, "rb") as f:
                return f.read().count(b"finish: [trajectory_id")
        except OSError:
            return 0

    fp = open(log_path, "w") if log_path else None  # fresh once; appended across relaunches below
    os.umask(0o000)  # child dirs/files world-writable (matches _run_and_tee)
    out = sys.stdout.buffer
    noprogress = 0
    attempt = 0
    try:
        while True:
            attempt += 1
            before = _completed()
            print(f"[watchdog] launch #{attempt} (idle_limit={idle_limit}s): " + " ".join(cmd), flush=True)
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True,  # own group => killable tree
            )
            dec = codecs.getincrementaldecoder("utf-8")("replace")
            fd = proc.stdout.fileno()
            hung = False
            while True:
                # select returns empty ONLY if no output arrived for the full idle_limit
                # (a clean exit closes the pipe => fd is 'readable' => we detect EOF fast).
                ready, _, _ = select.select([fd], [], [], idle_limit)
                if not ready:
                    print(f"\n[watchdog] HANG: no output for {idle_limit}s; killing eval.", flush=True)
                    hung = True
                    _kill_group(proc)
                    break
                chunk = os.read(fd, 4096)
                if not chunk:
                    break  # EOF -> process exiting
                out.write(chunk)
                out.flush()
                if fp:
                    fp.write(dec.decode(chunk).replace("\r\n", "\n").replace("\r", "\n"))
                    fp.flush()
            rc = proc.wait()

            if not hung and rc == 0:
                print("[watchdog] eval finished cleanly.", flush=True)
                return 0

            _wait_gpu_free(gpu)
            after = _completed()
            if after > before:
                noprogress = 0
                print(f"[watchdog] progress {before}->{after} episodes; relaunching (resume).", flush=True)
            else:
                noprogress += 1
                why = "hang, no new episode" if hung else f"exit {rc}, no new episode"
                print(f"[watchdog] no progress ({why}); strike {noprogress}/{max_noprogress}.", flush=True)
                if noprogress >= max_noprogress:
                    print(f"[watchdog] giving up after {noprogress} no-progress relaunches.", flush=True)
                    return rc if rc != 0 else 1
    finally:
        if fp:
            fp.close()


def _run_in_process(script: str, argv: List[str], env_extra: Optional[dict] = None) -> int:
    """Run ``script`` (a ``__main__`` entry script) inside THIS Python process — no
    torchrun subprocess — so a VSCode debugger attached to runner.py hits breakpoints
    in the trainer / eval / model / dataset code directly. Single process only.
    """
    import runpy

    if env_extra:
        os.environ.update(env_extra)
    abs_script = os.path.join(REPO_ROOT, script)
    old_argv = sys.argv
    sys.argv = [abs_script] + argv
    print(f"[run:in-process] {script} " + " ".join(argv), flush=True)
    try:
        runpy.run_path(abs_script, run_name="__main__")
        return 0
    except SystemExit as e:  # scripts may call sys.exit(0) on success
        return int(e.code) if isinstance(e.code, int) else (0 if not e.code else 1)
    finally:
        sys.argv = old_argv


def run_train(p: Params, output_dir: str, run_name: str, in_process: bool = False, debugpy: str = None) -> int:
    os.makedirs(output_dir, exist_ok=True)
    master_port = p.master_port or _pick_port()
    argv = p.train_argv(output_dir, run_name)
    if debugpy:
        argv = ["--debugpy", debugpy] + argv
    if in_process:
        # single-process distributed env (mirrors torchrun --nproc_per_node=1 so
        # DeepSpeed / HF Trainer initialize the process group correctly)
        env = {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
               "MASTER_ADDR": p.master_addr, "MASTER_PORT": str(master_port),
               "WANDB_DIR": output_dir}
        return _run_in_process(p.trainer, argv, env_extra=env)
    launcher = [
        "torchrun",
        f"--nnodes={p.nnodes}",
        f"--nproc_per_node={p.nproc_per_node}",
        f"--node_rank={p.node_rank}",
        f"--master_addr={p.master_addr}",
        f"--master_port={master_port}",
        p.trainer,  # base trainer, or the BEV provider trainer when p.bev
    ]
    cmd = launcher + argv
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return _run_and_tee(cmd, os.path.join(log_dir, "train.log"),
                        env={**os.environ, "WANDB_DIR": output_dir})


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


def _backfill_chat_template(ckpt: str, system2_ckpt: str) -> None:
    """eval-only (--no-train) runs never call _prep_ckpt_aux_files, so a checkpoint
    that was trained via a train-only run (--no-eval) and evaluated later never gets
    chat_template.json — processor.apply_chat_template() then fails every step.
    chat_template.json is a fixed file (identical across every checkpoint in this repo;
    the actual per-mode prompt text lives in code, not this file), so backfilling it
    from system2_ckpt is always safe.
    """
    import shutil

    dst = os.path.join(ckpt, "chat_template.json")
    if os.path.exists(dst):
        return
    src = os.path.join(system2_ckpt, "chat_template.json")
    try:
        shutil.copy(src, dst)
        print(f"[eval] backfilled missing chat_template.json into {ckpt}", flush=True)
    except Exception as e:
        print(f"[eval] could not backfill chat_template.json from {src}: {e}", flush=True)


_WANDB_RUN_ID_FILE = "wandb_run_id.txt"


def _save_wandb_run_id(output_dir: str) -> None:
    """After training, save the wandb run ID to the checkpoint dir (on shared storage)."""
    runs = sorted(glob.glob(os.path.join(output_dir, "wandb", "run-*")), key=os.path.getmtime, reverse=True)
    if not runs:
        return
    base = os.path.basename(runs[0])
    run_id = base.split("-", 2)[-1] if base.count("-") >= 2 else None
    if run_id:
        with open(os.path.join(output_dir, _WANDB_RUN_ID_FILE), "w") as f:
            f.write(run_id)


def _wandb_resume_env(output_dir: str = None, base_env=None, override_run_id: str = None) -> dict:
    """Return env vars that make the subprocess resume an existing wandb run.

    Priority:
      1. override_run_id — caller already resolved a run ID (e.g. --wandb-new-run one-shot)
      2. <output_dir>/wandb_run_id.txt — written after training or first eval
    """
    env = dict(base_env or os.environ)
    run_id = override_run_id or None
    if not run_id and output_dir:
        id_file = os.path.join(output_dir, _WANDB_RUN_ID_FILE)
        if os.path.exists(id_file):
            run_id = open(id_file).read().strip() or None
    if run_id:
        env["WANDB_RUN_ID"] = run_id
        env["WANDB_RESUME"] = "allow"
        print(f"[wandb] resuming run {run_id}", flush=True)
    return env


def run_eval(config_path: str, model_path: str, run_name: str, output_dir: str,
             machine: str = "h200", nproc: int = 8, master_port: int = 2333,
             in_process: bool = False, debugpy: str = None,
             debug_dir: Optional[str] = None,
             eval_max_episodes: Optional[int] = None,
             headless: bool = False,
             flash_collision: Optional[str] = None,
             wandb_run_id: Optional[str] = None,
             wandb_new_run: bool = False,
             use_wandb: bool = True,
             watchdog: bool = False,
             watchdog_idle_min: int = 10) -> int:
    # h1 (Isaac Sim) eval: give each collision mode its own logs dir so `none` and
    # `stop` runs against the same --model-path don't share one lmdb/result/test log
    # (the lmdb key is trajectory_id_episode_id, NOT tagged by collision mode, so a
    # shared dir makes the second mode resume-skip everything and overwrites the
    # first's result_h1.json). wandb_run_id.txt lives at output_dir (above log_dir),
    # so it stays shared -> both modes still resume the SAME wandb run.
    # Other machines (habitat h200/5090) keep the plain "logs" dir, unchanged.
    if machine == "h1":
        log_dir = os.path.join(output_dir, f"logs_{flash_collision or 'none'}")
    else:
        log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not use_wandb:
        # smoke-test / debug run: create/resume NO wandb run at all (not even the
        # placeholder-run-for-its-ID below). WANDB_MODE=disabled also makes any
        # wandb.init() the evaluator itself calls a local no-op, belt and suspenders.
        env = dict(os.environ)
        env["WANDB_MODE"] = "disabled"
    else:
        id_file = os.path.join(output_dir, _WANDB_RUN_ID_FILE)
        one_shot_run_id = None  # set for --wandb-new-run (not saved to txt)

        if wandb_run_id:
            # --wandb-run-id: explicit run to resume; persist so future evals reuse it
            with open(id_file, "w") as f:
                f.write(wandb_run_id)
            print(f"[wandb] using specified run ID: {wandb_run_id}", flush=True)
        elif wandb_new_run:
            # --wandb-new-run: fresh run for this eval only; do NOT overwrite txt
            ckpt_name = os.path.basename(output_dir.rstrip('/'))
            new_run_name = f"new/{ckpt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                import wandb
                if wandb.run is None:
                    run = wandb.init(project="huggingface", name=new_run_name, dir=output_dir)
                    print(f"[wandb] new run created (one-shot): {run.url}", flush=True)
                    wandb.finish()
                    one_shot_run_id = run.id
            except Exception as e:
                print(f"[wandb] init failed: {e}", flush=True)
        elif not os.path.exists(id_file):
            # old checkpoint without txt: create run and save for future evals
            ckpt_name = os.path.basename(output_dir.rstrip('/'))
            new_run_name = f"original/{ckpt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                import wandb
                if wandb.run is None:
                    run = wandb.init(project="huggingface", name=new_run_name, dir=output_dir)
                    print(f"[wandb] new run created: {run.url}", flush=True)
                    wandb.finish()
                    with open(id_file, "w") as f:
                        f.write(run.id)
            except Exception as e:
                print(f"[wandb] init failed: {e}", flush=True)

        env = _wandb_resume_env(output_dir=output_dir, override_run_id=one_shot_run_id)

    env["TRAIN_EVAL_TARGET"] = machine  # read by the experiment config.py
    env["EVAL_OUTPUT_DIR"] = os.path.abspath(log_dir)
    env["WANDB_DIR"] = output_dir
    if debugpy:
        env["DEBUGPY"] = debugpy  # habitat_vln_evaluator.py checks DEBUGPY=='eval' after model load
    if debug_dir:
        env["BEV_DEBUG_DIR"] = debug_dir  # read by default_config.make_eval_cfg at import time
    if eval_max_episodes:
        env["EVAL_MAX_EPISODES"] = str(eval_max_episodes)  # read by default_config.make_eval_cfg at import time
    if headless:
        # eval.py subprocess re-imports the experiment config fresh, so a --headless
        # CLI flag set on runner.py's own Params object never reaches it directly —
        # same env-var relay as BEV_DEBUG_DIR/EVAL_MAX_EPISODES above.
        env["EVAL_HEADLESS"] = "1"
    if flash_collision:
        env["EVAL_FLASH_COLLISION"] = flash_collision  # same env-var relay; see default_config.build_h1_eval_cfg
    eval_argv = [
        "--config", config_path, "--quiet",
        "--model_path", model_path, "--wandb_run_name", run_name,
    ]
    if in_process:
        # eval.py runs fine as a single process on one GPU (no torchrun); --quiet
        # dropped so console logs are visible while stepping in the debugger.
        argv = [a for a in eval_argv if a != "--quiet"]
        return _run_in_process("scripts/eval/eval.py", argv, env_extra=env)
    log_file = os.path.join(log_dir, f"test_{machine}.log")
    if machine == "h1":
        # Isaac Sim does not support torchrun; run as a single process.
        python = H1_PYTHON if os.path.exists(H1_PYTHON) else sys.executable
        python_argv = [python, "-X", "frozen_modules=off"] if debugpy else [python]
        # Isaac Sim's bundled Python 3.11 enables frozen stdlib modules, which makes
        # pydevd silently miss breakpoints after --debugpy attach (see the "frozen
        # modules" warning pydevd itself prints); only disabled for debug runs.
        cmd = python_argv + ["scripts/eval/eval.py"] + eval_argv
        if watchdog:
            # h1 eval can freeze inside Isaac Sim's env.step(); auto-detect the stall
            # (no output for watchdog_idle_min minutes) and relaunch (resumes lmdb+wandb).
            gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0] or None
            return _run_with_watchdog(cmd, log_file, env, idle_limit=watchdog_idle_min * 60, gpu=gpu)
        return _run_and_tee(cmd, log_file, env=env)
    cmd = ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={master_port}",
           "scripts/eval/eval.py"] + eval_argv
    return _run_and_tee(cmd, log_file, env=env)


def train_and_eval(p: Params, exp_name: str, config_path: str, *,
                   machine: str = "h200", do_train: bool = True, do_eval: bool = True,
                   checkpoints_root: str = "/home/irteam/data-vol2/checkpoints",
                   model_path: Optional[str] = None, in_process: bool = False,
                   debugpy: str = None,
                   wandb_run_id: Optional[str] = None,
                   wandb_new_run: bool = False,
                   watchdog: bool = False,
                   watchdog_idle_min: int = 10) -> None:
    """End-to-end driver. ``exp_name`` e.g. 'batch_size/b4_eff128_base'.

    ``in_process=True`` runs train/eval in THIS process (no torchrun) for VSCode
    attach debugging; debug one phase at a time (--no-eval / --no-train), since
    running both in one process re-uses an already-initialized dist/CUDA state.
    """
    run_name = f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if p.debug_dir and (p.max_steps > 0 or not do_train):
        # smoke-style run (training smoke via max_steps>0, or an eval-only debug
        # run): skip checkpoints_root / the (possibly shared) model_path dir
        # entirely, write straight into the debug-image dir the caller asked for
        # (training saves nothing when max_steps>0 -- see internvla_n1_trainer.py;
        # eval-only runs would otherwise collide with other combos sharing the
        # same checkpoint's progress.json).
        output_dir = os.path.abspath(p.debug_dir)
    elif not do_train and model_path:
        output_dir = os.path.abspath(model_path)
    else:
        output_dir = os.path.join(checkpoints_root, run_name)

    if do_train:
        rc = run_train(p, output_dir, run_name, in_process=in_process,
                       debugpy=debugpy if debugpy == "trainer" else None)
        if rc != 0:
            print(f"[train] FAILED (exit {rc}); skipping eval.", file=sys.stderr)
            return
        _save_wandb_run_id(output_dir)

    if not do_eval:
        return

    # node_rank guard mirrors `if [ "${NODE_RANK}" = "0" ]` in the shell scripts
    if p.node_rank != 0:
        return

    eval_machine = getattr(sys.modules.get("default_config"), "EVAL_MACHINE", {})
    default_model_path = eval_machine.get(machine, {}).get("model_path")
    best = model_path or find_best_checkpoint(output_dir) or default_model_path
    if not best:
        print("[eval] No best checkpoint found, skipping eval.")
        return
    print(f"[eval] machine={machine}  checkpoint={best}")
    if do_train:
        _prep_ckpt_aux_files(output_dir, best, p.system2_ckpt)
    else:
        _backfill_chat_template(best, p.system2_ckpt)
    os.makedirs(output_dir, exist_ok=True)
    run_eval(config_path, best, run_name, output_dir, machine=machine,
             nproc=p.nproc_per_node, in_process=in_process, debugpy=debugpy,
             debug_dir=p.debug_dir, eval_max_episodes=p.eval_max_episodes,
             headless=p.headless, flash_collision=p.flash_collision,
             wandb_run_id=wandb_run_id, wandb_new_run=wandb_new_run,
             use_wandb=p.use_wandb, watchdog=watchdog, watchdog_idle_min=watchdog_idle_min)


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
    ap.add_argument("--machine", choices=["h200", "5090", "h1"], default="h200",
                    help="target machine: h200/5090 selects train+eval infra presets; h1 = Isaac Sim (single-process python.sh)")
    ap.add_argument("--no-train", action="store_true", help="skip training (eval an existing ckpt)")
    ap.add_argument("--no-eval", action="store_true", help="train only")
    ap.add_argument("--model-path", default=None, help="ckpt to eval when --no-train")
    ap.add_argument("--data-root", default=None, help="override dataset root")
    ap.add_argument("--nproc", type=int, default=None,
                    help="GPUs per node for train+eval torchrun (e.g. 1 for a single 5090)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="smoke-test override: stop training after N steps (disables mid-train eval/save/load-best) "
                         "and cap Habitat eval to N episodes instead of the full split")
    ap.add_argument("--checkpoints-root", default="/home/irteam/data-vol2/checkpoints")
    ap.add_argument("--headless", action="store_true",
                    help="h1/Isaac Sim eval only: run without the Isaac Sim GUI window")
    ap.add_argument("--flash-collision", choices=["stop", "reset", "none"], default=None,
                    help="h1/Isaac Sim eval only: collision handling during flash-move "
                         "('stop'=halt in place, 'reset'=episode failure, 'none'=no detection). "
                         "Default: stop (config default, unchanged if omitted).")
    ap.add_argument("--print-train-argv", action="store_true", help="print resolved train flags and exit")
    # --- VSCode debugging ---
    ap.add_argument("--in-process", action="store_true",
                    help="run train/eval in THIS process (no torchrun); forces nproc=1.")
    ap.add_argument("--debugpy", type=str, default=None, choices=["runner", "trainer", "eval"],
                    help="debugpy attach target — fixed ports: runner=5680, trainer=5681, eval=5679. "
                         "runner: pauses at start; trainer/eval: pauses after model load.")
    ap.add_argument("--debug-dir", default=None,
                    help="BEV debug image output dir (auto-set to output/bev_debug when --debugpy is given)")
    # --- wandb control ---
    ap.add_argument("--wandb-run-id", default=None,
                    help="resume a specific wandb run ID (saved to wandb_run_id.txt for future evals)")
    ap.add_argument("--wandb-new-run", action="store_true",
                    help="force a fresh wandb run for this eval (does NOT overwrite wandb_run_id.txt)")
    # --- watchdog (h1 / Isaac Sim eval only) ---
    ap.add_argument("--watchdog", action="store_true",
                    help="h1 eval only: if the eval freezes inside Isaac Sim's env.step() "
                         "(no output for --watchdog-idle-min minutes), kill it and relaunch; "
                         "the eval auto-resumes (lmdb skip-done + wandb run-id). Gives up after "
                         "3 consecutive relaunches with no newly completed episode.")
    ap.add_argument("--watchdog-idle-min", type=int, default=10,
                    help="minutes of no eval output that count as a hang (default 10).")
    args = ap.parse_args()

    if not (args.max_steps or args.debugpy or args.debug_dir):  # smoke tests / debug runs skip wandb entirely
        try:
            import wandb
            wandb.login(relogin=False)  # prompts once if not logged in; saves to ~/.netrc for subprocesses
        except Exception as e:
            print(f"[Warning] wandb login failed: {e}. Metrics will not be logged to wandb.")

    # Set TRAIN_EVAL_TARGET before _load_config so that make_eval_cfg in the config
    # module builds the correct eval config (e.g. h1 → Isaac Sim, not Habitat).
    os.environ["TRAIN_EVAL_TARGET"] = args.machine

    in_process = args.in_process
    if args.debugpy == "runner":
        import debugpy
        debugpy.listen(("0.0.0.0", 5680))
        print("[debug] Waiting for VSCode attach on port 5680 ...", flush=True)
        debugpy.wait_for_client()

    mod, config_path = _load_config(args.config)
    params: Params = mod.PARAMS
    exp_name = getattr(mod, "EXP_NAME", None) or _derive_exp_name(config_path)

    # Apply machine-specific train defaults from TRAIN_MACHINE (if defined in config module).
    # CLI overrides (--data-root, --nproc, --checkpoints-root) take precedence.
    checkpoints_root = args.checkpoints_root
    train_machine = getattr(mod, "TRAIN_MACHINE", None) or getattr(
        sys.modules.get("default_config"), "TRAIN_MACHINE", None
    )
    if train_machine and args.machine in train_machine:
        m = train_machine[args.machine]
        overrides = dict(data_root=m["data_root"], system2_ckpt=m["system2_ckpt"],
                         nproc_per_node=m["nproc_per_node"])
        for key in ("batch_size", "grad_accum_steps", "max_pixels"):
            if key in m:
                overrides[key] = m[key]
        params = replace(params, **overrides)
        if args.checkpoints_root == "/home/irteam/data-vol2/checkpoints":  # still at default
            checkpoints_root = m["checkpoints_root"]

    if args.data_root:
        params = replace(params, data_root=args.data_root)
    if args.nproc:
        params = replace(params, nproc_per_node=args.nproc)
    if args.headless:
        params = replace(params, headless=True)
    if args.flash_collision:
        params = replace(params, flash_collision=args.flash_collision)
    if args.max_steps:  # one smoke-test knob: caps train steps AND eval episodes to the same N
        params = replace(params, max_steps=args.max_steps, eval_max_episodes=args.max_steps)
    if in_process:
        params = replace(params, nproc_per_node=1)  # single process
    if args.debug_dir:  # standalone --debug-dir (no --debugpy needed)
        debug_dir = args.debug_dir
        if not args.debugpy:
            # --no-train -> eval-only run; --no-eval -> train-only run. (--debugpy
            # already splits train/eval below, off its own target, so skip here.)
            if args.no_train:
                debug_dir = os.path.join(debug_dir, 'eval_isaac' if args.machine == 'h1' else 'eval')
            elif args.no_eval:
                debug_dir = os.path.join(debug_dir, 'train')
        params = replace(params, debug_dir=debug_dir)
    if args.debugpy:
        base_dir = params.debug_dir or 'output/bev_debug'
        if args.debugpy == 'trainer':
            debug_dir = os.path.join(base_dir, 'train')
        elif args.debugpy == 'eval':
            debug_dir = os.path.join(base_dir, 'eval')
        else:
            debug_dir = base_dir
        params = replace(params, debug_dir=debug_dir)
    if args.max_steps or args.debugpy or args.debug_dir:
        # smoke-test / debugger / debug-image runs: no online wandb run at all
        # (matches the wandb.login() skip above) — see run_eval()'s use_wandb branch.
        params = replace(params, use_wandb=False)

    if args.print_train_argv:
        print(" ".join(params.train_argv("<output_dir>", exp_name)))
        return

    train_and_eval(
        params, exp_name, config_path,
        machine=args.machine,
        do_train=not args.no_train,
        do_eval=not args.no_eval,
        checkpoints_root=checkpoints_root,
        model_path=args.model_path,
        in_process=in_process,
        debugpy=args.debugpy,
        wandb_run_id=args.wandb_run_id,
        wandb_new_run=args.wandb_new_run,
        watchdog=args.watchdog,
        watchdog_idle_min=args.watchdog_idle_min,
    )


if __name__ == "__main__":
    main_cli()
