#!/usr/bin/env python
"""Verify the Python batch_size entrypoints produce the SAME trainer flags as the
original shell scripts in scripts/train/qwenvl_train/batch_size/.

Parses the trainer argument block out of each *.sh (resolving ${var} and $((expr))),
then diffs it against runner.Params.train_argv for the matching variant. Exits
non-zero on any mismatch.

    python scripts/train_eval/qwenvl_train/batch_size/verify_params.py
"""

import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import default_config as base  # noqa: E402
import runner  # noqa: E402

REPO_ROOT = runner.REPO_ROOT
SH_DIR = os.path.join(REPO_ROOT, "scripts", "train", "qwenvl_train", "batch_size")


def _load_exp_params(filename):
    """Load PARAMS from a real experiment config file (tests the actual files)."""
    mod, _ = runner._load_config(os.path.join(HERE, filename))
    return mod.PARAMS


# (shell file, PARAMS read from the matching Python config file)
VARIANTS = [
    ("b4_eff128_base.sh", _load_exp_params("b4_eff128_base.py")),
    ("b2_eff128.sh", _load_exp_params("b2_eff128.py")),
    ("b8_eff128.sh", _load_exp_params("b8_eff128.py")),
]

# keys whose values are paths/timestamps, not hyperparameters
IGNORE = {"output_dir", "run_name"}


# --------------------------------------------------------------------------- #
# minimal shell parsing
# --------------------------------------------------------------------------- #
def _parse_assignments(text):
    """Top-level `name=value` lines -> resolved dict (strings)."""
    vars_ = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]\w*)=(.*)$", line)  # no leading space => top-level
        if not m:
            continue
        name, raw = m.group(1), m.group(2).strip()
        vars_[name] = _resolve(raw, vars_)
    return vars_


def _resolve(s, vars_):
    """Resolve ${name:-default}, $((expr)), ${name}, $name against known vars.

    Anything unresolvable (e.g. RANDOM in a launcher port) is left as-is; those
    tokens belong to launcher vars, not the trainer flags we compare.
    """
    # ${name:-default} -> known value or the default (env vars are unset here)
    s = re.sub(r"\$\{(\w+):-(.*?)\}", lambda m: str(vars_.get(m.group(1), m.group(2))), s)

    def _arith(m):
        expr = m.group(1)
        ns = {k: int(v) for k, v in vars_.items() if re.fullmatch(r"-?\d+", str(v))}
        try:
            return str(int(eval(expr, {"__builtins__": {}}, ns)))  # noqa: S307 (trusted shell math)
        except Exception:
            return m.group(0)  # leave unresolved (launcher-only, not a trainer flag)

    s = re.sub(r"\$\(\((.*?)\)\)", _arith, s)
    s = re.sub(r"\$\{(\w+)\}", lambda m: str(vars_.get(m.group(1), m.group(0))), s)
    s = re.sub(r"\$(\w+)", lambda m: str(vars_.get(m.group(1), m.group(0))), s)
    return s


def _parse_trainer_flags(text, vars_):
    """Extract the internnav/trainer/...py argument block as a flag dict."""
    start = text.index("internnav/trainer/internvla_n1_trainer.py")
    block = text[start:]
    block = block[: block.index("2>&1")]
    block = block.split("internnav/trainer/internvla_n1_trainer.py", 1)[1]
    block = block.replace("\\\n", " ")              # join shell line continuations
    block = _resolve(block, vars_)                   # substitute ${var}/$((..))
    tokens = shlex.split(block)
    return _tokens_to_dict(tokens)


def _tokens_to_dict(tokens):
    out = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        key = t[2:]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = tokens[i + 1]
            i += 2
        else:
            out[key] = True  # bare flag, e.g. --bf16
            i += 1
    return out


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def _norm(key, val):
    """Normalize a flag value for semantic comparison."""
    if val is True:
        return True
    if key == "lr_scheduler_kwargs":
        import json
        return json.loads(val)
    try:
        return float(val)          # 1e-4 == 0.0001 etc.
    except (TypeError, ValueError):
        return val


def _compare(name, sh_flags, py_flags):
    keys = (set(sh_flags) | set(py_flags)) - IGNORE
    diffs = []
    for k in sorted(keys):
        a, b = sh_flags.get(k, "<missing>"), py_flags.get(k, "<missing>")
        if a == "<missing>" or b == "<missing>" or _norm(k, a) != _norm(k, b):
            diffs.append((k, a, b))
    return diffs


def _load_eval_cfg(path):
    """Load eval_cfg from a reference config the way scripts/eval/eval.py does."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ref_eval_cfg_" + os.path.basename(path), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.eval_cfg


def _flat_eval(cfg):
    """Flatten an EvalCfg to {model_settings.*, env.config_path, eval_settings.*}."""
    flat = {}
    for k, v in cfg.agent.model_settings.items():
        flat[f"model_settings.{k}"] = v
    flat["env.config_path"] = cfg.env.env_settings.get("config_path")
    for k, v in cfg.eval_settings.items():
        flat[f"eval_settings.{k}"] = v
    return flat


def _compare_eval(ref, gen, ignore):
    """Diff two flattened eval configs. predict_step_num absent in ref == default 32."""
    diffs = []
    for k in sorted(set(ref) | set(gen)):
        if k in ignore:
            continue
        a, b = ref.get(k, "<missing>"), gen.get(k, "<missing>")
        if k == "model_settings.predict_step_num" and a == "<missing>" and b == 32:
            continue  # evaluator default is 32 -> behaviourally identical
        if a != b:
            diffs.append((k, a, b))
    return diffs


def main():
    all_ok = True
    for sh_name, params in VARIANTS:
        text = open(os.path.join(SH_DIR, sh_name)).read()
        vars_ = _parse_assignments(text)
        sh_flags = _parse_trainer_flags(text, vars_)
        py_flags = _tokens_to_dict(params.train_argv("<output_dir>", "<run_name>"))

        diffs = _compare(sh_name, sh_flags, py_flags)
        status = "OK" if not diffs else "MISMATCH"
        print(f"[{status}] {sh_name}  (batch_size={params.batch_size}, "
              f"grad_accum={params.grad_accum_steps}, eval_steps={params.val_interval_steps})")
        for k, a, b in diffs:
            print(f"    {k}: shell={a!r}  python={b!r}")
            all_ok = False

    # A generated Habitat eval_cfg = machine infra (from the reference file) +
    # train model params (from PARAMS). We verify the two halves separately.
    REFS = {"h200": "habitat_dual_system_mini_h200_cfg.py",
            "5090": "habitat_dual_system_mini_5090_cfg.py"}
    # runtime-overridden / cosmetic keys (not parameters that affect results)
    IGNORE_EVAL = {"model_settings.model_path", "eval_settings.output_path", "eval_settings.wandb_run_name",
                   # Params에서 오는 키. 기준 config 파일들은 이 키가 생기기 전에 작성됐다.
                   # 기본값 'conv' = 기존 eval 동작이므로 부재/'conv'는 동등하다.
                   "model_settings.patch_embed_impl"}
    # model params that intentionally come from training, NOT the reference file
    TRAIN_SYNCED = {"model_settings.num_history", "model_settings.resize_w",
                    "model_settings.resize_h", "model_settings.predict_step_num"}

    # (1) machine infra (yaml, wandb, max_new_tokens, port, ...) must equal the ref file
    print("\n[eval-infra] generated eval_cfg infra == reference config file")
    for machine, ref_name in REFS.items():
        ref = _load_eval_cfg(os.path.join(REPO_ROOT, "scripts", "eval", "configs", ref_name))
        gen = base.build_habitat_eval_cfg(base.PARAMS, machine=machine)
        diffs = _compare_eval(_flat_eval(ref), _flat_eval(gen), IGNORE_EVAL | TRAIN_SYNCED)
        print(f"    [{'OK' if not diffs else 'MISMATCH'}] {machine} ({ref_name})")
        for k, a, b in diffs:
            print(f"        {k}: ref={a!r}  generated={b!r}")
            all_ok = False

    # (2) for batch_size the train params equal h200's -> generated == h200 byte-for-param
    print("\n[eval-full] generated h200 eval_cfg == h200 file (ALL params)")
    ref = _load_eval_cfg(os.path.join(REPO_ROOT, "scripts", "eval", "configs", REFS["h200"]))
    gen = base.build_habitat_eval_cfg(base.PARAMS, machine="h200")
    diffs = _compare_eval(_flat_eval(ref), _flat_eval(gen), IGNORE_EVAL)
    print(f"    [{'OK' if not diffs else 'MISMATCH'}] (model_path/output_path overridden; "
          f"predict_step_num=32 == evaluator default)")
    for k, a, b in diffs:
        print(f"        {k}: ref={a!r}  generated={b!r}")
        all_ok = False

    # (3) train<->eval sync: eval model params come from the same PARAMS
    print("\n[eval-sync] train PARAMS == eval model_settings (h200 / 5090 / h1)")
    for target in ("h200", "5090", "h1"):
        ms = base.build_eval_cfg(base.PARAMS, target).agent.model_settings
        pkey = "predict_step_nums" if target == "h1" else "predict_step_num"
        checks_full = {"num_history": base.PARAMS.num_history, "resize_w": base.PARAMS.resize_w,
                       "resize_h": base.PARAMS.resize_h, pkey: base.PARAMS.predict_step_num}
        bad = {k: (ms.get(k), v) for k, v in checks_full.items() if ms.get(k) != v}
        print(f"    [{'OK' if not bad else 'MISMATCH'}] {target}: " +
              ", ".join(f"{k}={ms.get(k)}" for k in checks_full))
        if bad:
            all_ok = False

    print("\n" + ("ALL PARAMETERS MATCH ✅" if all_ok else "PARAMETER MISMATCH ❌"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
