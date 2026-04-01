# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**InternNav** is an embodied navigation framework supporting Vision-Language Navigation (VLN), Visual Navigation (VN), and real-world robot deployment. It integrates multiple simulators (Habitat, Isaac Sim/InternUtopia) and supports models from simple baselines (Seq2Seq, CMA) to the dual-system InternVLA-N1 foundation model.

## Installation

```bash
pip install -e .                      # core only
pip install -e ".[habitat]"           # with Habitat simulator
pip install -e ".[isaac]"             # with Isaac Sim / InternUtopia
pip install -e ".[internvla_n1]"      # with InternVLA-N1 model
pip install -e ".[model]"             # with baseline models
```

The `src/diffusion-policy` path is appended to `sys.path` in several scripts; this is a third-party submodule in `third_party/`.

## Common Commands

**Evaluation:**
```bash
python scripts/eval/eval.py --config scripts/eval/configs/h1_cma_cfg.py
```
Config files live in `scripts/eval/configs/`. Each file must expose an `eval_cfg` attribute (an `EvalCfg` Pydantic object). Available configs cover CMA, RDP, Seq2Seq, InternVLA-N1, dual-system, object navigation, and dialog navigation variants.

**Training:**
```bash
python scripts/train/base_train/train.py  # default model: cma
```
Model selection is via `TrainCfg.model_name` (options: `cma`, `cma_plus`, `seq2seq`, `seq2seq_plus`, `rdp`, `navdp`).

**Real-world deployment (two terminals):**
```bash
# Terminal 1 - model inference server (Flask)
python scripts/realworld/http_internvla_server.py

# Terminal 2 - ROS2 robot client
python scripts/realworld/http_internvla_client.py
```
The debug variant `http_internvla_server_debug.py` (currently modified on branch `v0.1`) adds visualization to the server.

**Tests:**
```bash
pytest tests/                  # all tests
pytest tests/ -m slow          # slow tests only
pytest tests/ -m gpu           # GPU tests only (requires CUDA)
pytest tests/unit_test/        # unit tests only
```

**Code formatting:**
```bash
black --line-length 120 .
isort --profile black .
```

## Architecture

### Registry Pattern (used everywhere)

`Agent`, `Evaluator`, and `Env` all follow the same registry pattern:

```python
@Agent.register('my_agent')
class MyAgent(Agent): ...

agent = Agent.init(config)  # dispatches by config.model_name
```

The same applies to `Evaluator.register` / `Evaluator.init` (dispatched by `config.eval_type`) and `Env.register` / `Env.init` (dispatched by `config.env_type`).

### Policy Factory (`internnav/model/__init__.py`)

`get_policy(policy_name)` and `get_config(policy_name)` are the entry points for all model classes. Supported policy names: `CMA_Policy`, `CMA_CLIP_Policy`, `Seq2Seq_Policy`, `RDP_Policy`, `InternVLAN1_Policy`, `NavDP_Policy`.

### Config System (`internnav/configs/`)

Configs are Pydantic models. Eval configs are plain Python files that expose an `eval_cfg = EvalCfg(...)` object. `EvalCfg` composes `AgentCfg`, `EnvCfg`, `TaskCfg`, `EvalDatasetCfg`, and `eval_type` string. The `eval_type='vln_distributed'` path applies additional defaults via `get_config()` from `internnav/configs/evaluator/vln_default_config.py`.

### Dual-System InternVLA-N1 Agent (`internnav/agent/internvla_n1_agent.py`)

The main agent for InternVLA-N1 runs two systems concurrently via threading:
- **System 1 (S1)**: Visual navigation policy — runs at high frequency, outputs continuous trajectories.
- **System 2 (S2)**: Language-grounded planner — runs asynchronously in a background thread, outputs waypoints/subgoals.

Communication uses `S1Input`, `S1Output`, `S2Input`, `S2Output` dataclasses (defined in `internnav/model/utils/vln_utils.py`). Thread locks (`s2_input_lock`, `s2_output_lock`, `s2_agent_lock`) protect shared state. The real-world variant is `internnav/agent/internvla_n1_agent_realworld.py`.

### Evaluator Architecture

In `vln_distributed` mode, the evaluator separates model inference from environment stepping. `AgentClient` (in `internnav/utils/comm_utils/client.py`) communicates over HTTP with a separately-launched agent server (`scripts/eval/start_server.py`). This enables multi-GPU distributed evaluation.

### Training Pipeline

`scripts/train/base_train/train.py` uses HuggingFace `TrainingArguments` and custom `Trainer` subclasses (`CMATrainer`, `RDPTrainer`, `NavDPTrainer`). Datasets are in LeRobot format (e.g., `CMALerobotDataset`, `RDP_LerobotDataset`). Each model has a matching collate function.

## Key File Locations

| Purpose | Path |
|---|---|
| Agent registry + base | `internnav/agent/base.py` |
| InternVLA-N1 dual-system agent | `internnav/agent/internvla_n1_agent.py` |
| Policy factory | `internnav/model/__init__.py` |
| Evaluator base + registry | `internnav/evaluator/base.py` |
| Distributed VLN evaluator | `internnav/evaluator/vln_distributed_evaluator.py` |
| Env base + registry | `internnav/env/base.py` |
| Pydantic config schemas | `internnav/configs/` |
| Eval entry point | `scripts/eval/eval.py` |
| Train entry point | `scripts/train/base_train/train.py` |
| Real-world server | `scripts/realworld/http_internvla_server.py` |
| Real-world client (ROS2) | `scripts/realworld/http_internvla_client.py` |
| Eval config examples | `scripts/eval/configs/` |

## Supported Simulators / Environments

- `internutopia` — InternUtopia (Isaac Sim-based), used for VLN-PE with H1 humanoid robot
- `habitat` — Habitat, used for VLN-CE (discrete action) benchmarks
- `realworld_agilex` — real robot interface via ROS2

## Data Paths Convention

Evaluation configs reference paths like `data/scene_data/mp3d_pe`, `data/vln_pe/raw_data/r2r`, and USD robot assets at `data/Embodiments/...`. These paths are relative to the repo root and must be populated separately (see README and model zoo).

## Recent Development Log
<!-- AUTO-UPDATED by post-commit hook - DO NOT EDIT MANUALLY -->
_Last updated: 2026-04-01 11:54_

### [5574daff] 2026-04-01 — [feat] train ablation

### [9113f143] 2026-03-31 — [feat] 1. optimize distance calculation for dataloader, 2. optimize batch size and nw

### [690901cf] 2026-03-29 — [chore] add date

### [89976813] 2026-03-29 — [chore] edit run_name

### [3962b544] 2026-03-29 — [feat] system 2 validation added

### [9725fed4] 2026-03-27 — [chore] edit param

### [70add386] 2026-03-27 — find=False