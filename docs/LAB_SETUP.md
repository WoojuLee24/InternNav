# InternNav Research Lab Setup

_Branch: research/async-foundation_  
_Created: 2026-04-30_

## Overview

The lab is a 5-window tmux session that coordinates four independent agents and all experiment infrastructure in parallel. Each window has a clear role and its own panes. Windows are narrow enough to be readable on a standard terminal.

```
bash scripts/lab_session.sh          # launch and attach
tmux attach -t internav_lab          # re-attach after disconnect
```

---

## Window Map

| Key | Window | Role |
|-----|--------|------|
| `Ctrl-b 0` | `W0:agents` | Three AI agents — main researcher, literature/docs writer, bounded worker |
| `Ctrl-b 1` | `W1:server` | Docker container management + server log stream |
| `Ctrl-b 2` | `W2:experiment` | 4-pane experiment runner (matches `async_check_tmux.sh`) |
| `Ctrl-b 3` | `W3:monitor` | Live `/stats` polling, log tail, git watcher |
| `Ctrl-b 4` | `W4:paper` | Ablation plots, paper writing, INNOVATIONS_CATALOGUE |

---

## W0: agents — Three-pane AI collaboration

```
┌──────────────────────────────────┬──────────────────┐
│                                  │  W0·P1           │
│  W0·P0  Claude Main  (55%)       │  OpenCode        │
│  Branch: research/async-foundation  opencode-exp    │
│  Full context. Drives PLAN.md.   ├──────────────────┤
│  Assigns tasks to P2.            │  W0·P2           │
│                                  │  Claude Worker   │
└──────────────────────────────────┴──────────────────┘
```

### W0·P0 — Claude Code Main (this session)

**Start**: `claude --continue`

**Role**: Full-context orchestrator. Reads PLAN.md, drives gate-by-gate progression, monitors experiment results, assigns bounded tasks to W0·P2.

**Scope**: All files in `InternNav/`. Priority files:
- `scripts/realworld/http_internvla_server_debug.py` (Gate 0 fix)
- `internnav/agent/internvla_n1_agent_enhanced.py` (Gate 2+ work)
- `PLAN.md` (progress tracking)

**Monitoring other panes**: Claude Main can capture pane output with:
```bash
tmux capture-pane -t internav_lab:W3:monitor.0 -p | tail -20
tmux capture-pane -t internav_lab:W2:experiment.1 -p | tail -30
```

### W0·P1 — OpenCode (research writer)

**Start**: `cd ../InternNav-opencode && opencode`

**Worktree**: `../InternNav-opencode` on branch `opencode-exp` (isolated, no conflict with main)

**Paste this prompt into OpenCode on startup**:
```
You are a research literature and documentation agent for the InternNav project
— a dual-system VLN robot (S1 visual nav + S2 language planner, async decoupled).

Working directory: ../InternNav-opencode (branch: opencode-exp)

Step 1: Read PLAN.md to understand the research roadmap (Phase 0 → Phase 5).
Step 2: Read RESEARCH_TRACKING.md for current open questions and proven results.
Step 3: Search arxiv/Semantic Scholar for papers related to:
  - Speculative decoding for embodied AI or VLN
  - Adaptive inference scheduling in dual-system architectures
  - Action chunking (OpenVLA, PD-VLA, ACT)
  - Temporal feature caching in vision-language models
  - Knowledge distillation from large VLMs to smaller nav policies
For each paper: extract borrowable technique, rate relevance 1–5, map to InternNav component.
Append findings to RESEARCH_TRACKING.md.

Step 4: Begin writing INNOVATIONS_CATALOGUE.md — 50+ novel ideas, each with:
  - Title, category, hypothesis, expected gain, paper grounding, effort estimate.

DO NOT edit any .py files. DO NOT edit PLAN.md.
```

**Owned files (edit freely)**:
- `RESEARCH_TRACKING.md`
- `EXPERIMENTS_LOG.md`
- `INNOVATIONS_CATALOGUE.md`
- `docs/*.md`

**Do NOT edit**: `scripts/realworld/http_internvla_*.py`, `internnav/agent/*.py`, `PLAN.md`

### W0·P2 — Claude Worker (bounded tasks)

**Start**: `claude --continue` or `claude -p "<task>"`

**Paste this prompt on startup**:
```
You are a bounded Claude Code worker for InternNav.
Wait for an explicit task assignment from the main agent (W0:P0).
Each assignment will specify:
  - Exact files you may edit
  - The precise change needed
  - The success condition

When assigned:
1. Read only the specified files.
2. Make only the specified change — nothing more.
3. Run any test or curl check.
4. Print: TASK DONE: [what changed] | RESULT: [test output]

Never self-assign work. Never edit outside the specified file scope.
```

**How Claude Main assigns tasks to Claude Worker**: The Main agent writes a task message directly into W0·P2's pane:
```bash
tmux send-keys -t internav_lab:W0:agents.2 "Task: edit scripts/realworld/http_internvla_server_debug.py lines 437-448 only. Remove the agent.step() call. Keep lines 492-516. Success: curl http://localhost:5000/eval_dual_async returns in <50ms." Enter
```

---

## W1: server — Container management

```
┌──────────────────────┬───────────────────────────────┐
│  W1·P0               │  W1·P1                        │
│  Container ctrl      │  Server logs (docker logs -f) │
│  start/stop/health   │                               │
└──────────────────────┴───────────────────────────────┘
```

Container management scripts live at `~/VLNav/VLNav/scripts/model/`:

| Script | Action |
|--------|--------|
| `./scripts/model/start.sh` | Start `vlnav_internvla_server` container |
| `./scripts/model/stop.sh` | Stop and remove container |
| `./scripts/model/health.sh` | Poll `/health` endpoint until ready |
| `./scripts/model/exec.sh` | Interactive shell (sources ROS2 Jazzy inside) |
| `./scripts/model/commit.sh` | Snapshot container → `vlnav/internvla-server:snapshot` |
| `./scripts/model/logs.sh` | Follow container logs |

**Typical start sequence**:
```bash
cd ~/VLNav/VLNav
./scripts/model/start.sh
# wait ~30s for model weights to load
./scripts/model/health.sh
# then go to W2 and run experiment
```

---

## W2: experiment — 4-pane runner

Mirrors the layout of `scripts/realworld/async_check_tmux.sh`. Run all 4 in order.

```
┌──────────────────────┬──────────────────────┐
│  W2·P0               │  W2·P2               │
│  SERVER (in Docker)  │  BRIDGE (in Docker)  │
├──────────────────────┼──────────────────────┤
│  W2·P1               │  W2·P3               │
│  CLIENT (host ROS2)  │  ROSBAG (host ROS2)  │
└──────────────────────┴──────────────────────┘
```

### W2·P0 — Server (inside container)

```bash
# Step 1: shell into container
cd ~/VLNav/VLNav && ./scripts/model/exec.sh

# Step 2: inside container
cd /workspace/InternNav
/opt/venv/bin/python3 scripts/realworld/http_internvla_server_debug.py \
  --mode async --kv-cache --temperature 0.8 \
  --max-new-tokens 80 --resize_w 256 --resize_h 256 \
  --num_history 1 --plan_step_gap 12 \
  --device cuda:0 \
  --model_path checkpoints/InternVLA-N1-w-NavDP \
  --calib scripts/realworld/calib/calib_scout.txt
```

### W2·P1 — Client (host with ROS2)

```bash
source /opt/ros/jazzy/setup.bash
cd ~/VLNav/VLNav/workspaces/model/InternNav
python3.12 scripts/realworld/http_internvla_client_debug.py \
  --mode async --kv-cache --temperature 0.8 \
  --jpeg-quality 95 --depth-png-compress 6 \
  --calib scripts/realworld/calib/calib_scout.txt
```

**Watch for**: `[HTTP] <N> ms` → must be `< 20ms` for Gate 0 to pass.

### W2·P2 — Bridge (inside container)

```bash
cd ~/VLNav/VLNav && ./scripts/model/exec.sh
# inside:
cd /workspace/InternNav && python3.12 scripts/realworld/scout_bridge.py
```

### W2·P3 — Rosbag (host with ROS2)

```bash
source /opt/ros/jazzy/setup.bash
BAG=../rosbag/my_camera_bag_20260317_073623   # swap for other bags
ros2 bag play $BAG --rate 0.5
```

**Gate 0 bags** (run all 3 to pass):
- `my_camera_bag_20260317_061841`
- `my_camera_bag_20260317_063047`
- `my_camera_bag_20260317_073623`

**Or use async_check_tmux.sh** (creates its own tmux session; switch back with `Ctrl-b s`):
```bash
NO_ATTACH=1 KV_CACHE=1 TEMPERATURE=0.8 \
  bash scripts/realworld/async_check_tmux.sh \
  ../rosbag/my_camera_bag_20260317_073623 gate0_bag3
```

---

## W3: monitor — Three auto-running monitors

```
┌──────────────────────┬──────────────────────────────┐
│  W3·P0               │  W3·P2                       │
│  Live /stats poll    │  Git watcher (both repos)    │
│  (every 3s)          │                              │
├──────────────────────┤                              │
│  W3·P1               │                              │
│  Log tail (client.log│                              │
└──────────────────────┴──────────────────────────────┘
```

All three monitors auto-start when the lab session is created.

**W3·P0** polls `localhost:8087/stats` every 3 seconds and shows key fields: `s2_hz`, `s2_avg_ms`, `trajectory_ratio`, `background_s2_runs`.

**W3·P1** tails the most recent `test_data/*/client.log` files.

**W3·P2** runs `git status --short` on both `research/async-foundation` and `opencode-exp` every 10 seconds.

---

## W4: paper — Ablation plots and writing

```bash
# Generate all ablation plots from experiment stats
python scripts/viz/plot_ablations.py

# Specific control variable
python scripts/viz/plot_ablations.py --var temperature

# Time series for one experiment
python scripts/viz/plot_ablations.py --timeseries bag073623_kv-on_temp0.8

# Quality–speed tradeoff scatter
python scripts/viz/plot_ablations.py --tradeoff --color-by temperature
```

Plots save to `figures/ablations/` as both `.pdf` (300 DPI) and `.png`.

**Parse client.log → structured stats**:
```bash
python scripts/realworld/stats_recorder.py parse \
  test_data/<run_dir>/client.log \
  --tag bag073623_kv-on_temp0.8 \
  --bag-id 073623 --kv-cache --temperature 0.8

python scripts/realworld/stats_recorder.py summary stats/bag073623_kv-on_temp0.8/

python scripts/realworld/stats_recorder.py compare stats/exp_a/ stats/exp_b/
```

---

## File Ownership (no-conflict zones)

| Agent | Owned files | Prohibited |
|-------|------------|------------|
| Claude Main (W0:P0) | All `.py` files, `PLAN.md`, `internnav/` | — |
| OpenCode (W0:P1) | `RESEARCH_TRACKING.md`, `EXPERIMENTS_LOG.md`, `INNOVATIONS_CATALOGUE.md`, `docs/` | `scripts/realworld/http_*.py`, `internnav/`, `PLAN.md` |
| Claude Worker (W0:P2) | Only files explicitly assigned by Main | Everything not assigned |
| Monitor (W3) | Read-only | — |

---

## Gate Progression (quick reference)

```
Gate 0 — joint_latency_ms < 20ms, trajectory_ratio ≥ 50%, joint_req_hz ≥ 10Hz
  → Fix: http_internvla_server_debug.py remove agent.step() call at line ~437
  → Verify: 3 bags (061841, 063047, 073623), rate=0.5, KV=ON, temp=0.8

Gate 1 — trajectory_ratio ≥ 63% at best temperature
  → Sweep: temp ∈ {0.70, 0.75, 0.80, 0.85}

Gate 2 — enhanced agent no regression vs Gate 1 baseline
  → Connect internvla_n1_agent_enhanced.py via --agent-type flag

Gates 3a/3b/3c — adaptive gap / temporal cache / speculative pre-fetch
  → Run in parallel after Gate 2

Gate 4 — trajectory_ratio correlates with SPL/SR (r > 0.7)
Gate 5 — S2→S1 distillation (S2 used ≤ 20% of frames)
```

Full detail: `PLAN.md`
