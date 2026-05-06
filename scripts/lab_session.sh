#!/usr/bin/env bash
# ============================================================================
# InternNav Research Lab Session — 5-window tmux layout
# ============================================================================
#
# WINDOWS  (Ctrl-b 0..4 to switch)
#   W0  agents      Claude Main (left) | OpenCode (top-right) | Claude Worker (bot-right)
#   W1  server      Container ctrl (left) | Server logs (right)
#   W2  experiment  Server | Bridge | Client | Rosbag  (2×2 grid)
#   W3  monitor     Live /stats | Log tail | Git watch
#   W4  paper       Writing, plots, notes
#
# USAGE
#   bash scripts/lab_session.sh          # launch and attach
#   NO_ATTACH=1 bash scripts/lab_session.sh  # create without attaching
#   SESSION=lab2 bash scripts/lab_session.sh
#
# RE-ATTACH AFTER DISCONNECT
#   tmux attach -t internav_lab
#
# KEY BINDINGS
#   Ctrl-b 0..4   switch window
#   Ctrl-b arrows move between panes
#   Ctrl-b q      show pane numbers, press number to jump
#   Ctrl-b z      zoom/unzoom current pane
#   Ctrl-b d      detach (session keeps running)
# ============================================================================

SESSION="${SESSION:-internav_lab}"
INTERNAV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLNAV_ROOT="$(cd "$INTERNAV_DIR/../../.." && pwd)"
WORKTREE_DIR="$(dirname "$INTERNAV_DIR")/InternNav-opencode"
SERVER_PORT="5802"

# ── colors ─────────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[0;33m'; C='\033[0;36m'; NC='\033[0m'

# ── guard: already running ─────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo -e "${G}Session '$SESSION' already running.${NC}"
    echo "  Attach : tmux attach -t $SESSION"
    echo "  Kill   : tmux kill-session -t $SESSION"
    exit 0
fi

# ── worktree: create if missing ────────────────────────────────────────────────
if [ ! -d "$WORKTREE_DIR" ]; then
    cd "$INTERNAV_DIR"
    git worktree add "$WORKTREE_DIR" -b opencode-exp 2>/dev/null \
        || git worktree add "$WORKTREE_DIR" opencode-exp 2>/dev/null \
        || echo "[warn] Could not create worktree — OpenCode pane will open in INTERNAV_DIR"
fi

# ── capture terminal size so detached session has real dimensions ──────────────
# This is the fix for "size missing": new-session -d has no terminal,
# so we must pass explicit dimensions.
COLS=$(tput cols  2>/dev/null || echo 220)
ROWS=$(tput lines 2>/dev/null || echo 50)

# ── helper: send text to a pane (no trailing Enter unless given) ───────────────
# Usage: pane_send  agents.0  "text"       # type text, no Enter
#        pane_send  agents.0  "cmd" Enter  # type cmd and press Enter
pane_send() {
    local target="$1"; shift
    tmux send-keys -t "${SESSION}:${target}" "$@"
}

# ── helper: set pane title ─────────────────────────────────────────────────────
pane_title() {
    tmux select-pane -t "${SESSION}:$1" -T "$2"
}

# ── helper: run cmd in pane and press Enter ────────────────────────────────────
pane_run() {
    tmux send-keys -t "${SESSION}:$1" "$2" Enter
}

# ── helper: write a header block then a blank prompt ──────────────────────────
# Uses a temp file to avoid quoting/escaping issues with heredocs in send-keys
header_file=$(mktemp /tmp/lab_header_XXXXXX.sh)
trap 'rm -f "$header_file"' EXIT

write_header() {
    # write_header  <target_pane>  <color_code>  <lines...>
    local target="$1" color="$2"; shift 2
    {
        echo "clear"
        echo "printf '${color}'"
        for line in "$@"; do
            # escape single quotes inside the line
            safe="${line//\'/\'\\\'\'}"
            echo "echo '$safe'"
        done
        echo "printf '\\033[0m'"
        echo ""
    } > "$header_file"
    pane_run "$target" "bash $header_file"
}

# ==============================================================================
# W0: agents  — 3-pane layout
# ==============================================================================
tmux new-session -d -s "$SESSION" -n "agents" -c "$INTERNAV_DIR" \
     -x "$COLS" -y "$ROWS"

# Enable pane border titles (best-effort — silently ignore if tmux too old)
tmux set-option -t "$SESSION" pane-border-status top      2>/dev/null || true
tmux set-option -t "$SESSION" pane-border-format \
     " #[bold]#{pane_title}#[default] "                   2>/dev/null || true

# Layout: split right column off, then split that vertically
#   Before: [P0 full]
#   After split-h: [P0 left 58%] [P1 right 42%]  ← P1 is active (new pane)
#   After split-v on P1: [P0] [P1 top] [P2 bottom]
tmux split-window -h -t "${SESSION}:agents" \
     -c "$INTERNAV_DIR" -l "42%"
# P1 is now active (right column). Split it vertically.
tmux split-window -v -t "${SESSION}:agents" \
     -c "${WORKTREE_DIR:-$INTERNAV_DIR}" -l "50%"
# Now: P0=Claude Main, P1=OpenCode (top-right), P2=Claude Worker (bot-right)

pane_title "agents.0" "CLAUDE:MAIN  research/async-foundation"
pane_title "agents.1" "OPENCODE  opencode-exp"
pane_title "agents.2" "CLAUDE:WORKER  bounded tasks"

# ── W0·P0: Claude Main ────────────────────────────────────────────────────────
pane_run "agents.0" "clear"
pane_run "agents.0" "printf '\\033[1;36m'"
pane_run "agents.0" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "agents.0" "echo '  W0·P0  CLAUDE CODE — MAIN RESEARCH AGENT'"
pane_run "agents.0" "echo '  Branch : research/async-foundation'"
pane_run "agents.0" "echo '  Role   : Full-context orchestrator. Drives all PLAN.md gates.'"
pane_run "agents.0" "echo '           Assigns bounded tasks to W0·P2 (Claude Worker).'"
pane_run "agents.0" "echo '  Scope  : All files. Priority: http_*_server_debug.py + enhanced agent.'"
pane_run "agents.0" "echo '  Gate   : #1 — Remove agent.step() from /eval_dual_async (~line 437)'"
pane_run "agents.0" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "agents.0" "printf '\\033[0m'"
pane_run "agents.0" "echo '  START: claude --continue'"
pane_run "agents.0" "echo"

# ── W0·P1: OpenCode research writer ───────────────────────────────────────────
pane_run "agents.1" "clear"
pane_run "agents.1" "printf '\\033[1;33m'"
pane_run "agents.1" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "agents.1" "echo '  W0·P1  OPENCODE — RESEARCH WRITER'"
pane_run "agents.1" "echo '  Branch : opencode-exp (worktree)'"
pane_run "agents.1" "echo '  OWNS   : RESEARCH_TRACKING.md'"
pane_run "agents.1" "echo '           EXPERIMENTS_LOG.md'"
pane_run "agents.1" "echo '           INNOVATIONS_CATALOGUE.md'"
pane_run "agents.1" "echo '           docs/*.md'"
pane_run "agents.1" "echo '  SKIP   : scripts/realworld/http_*.py'"
pane_run "agents.1" "echo '           internnav/agent/*.py  PLAN.md'"
pane_run "agents.1" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "agents.1" "printf '\\033[0m'"
pane_run "agents.1" "echo"
pane_run "agents.1" "echo 'Paste this prompt into OpenCode on startup:'"
pane_run "agents.1" "echo '--- OPENCODE PROMPT ---'"
pane_run "agents.1" "cat $INTERNAV_DIR/.claude/opencode_prompt.txt 2>/dev/null || echo '(see docs/LAB_SETUP.md > W0-P1 section)'"
pane_run "agents.1" "echo '-----------------------'"
pane_run "agents.1" "echo"
pane_run "agents.1" "echo 'START: cd ${WORKTREE_DIR:-$INTERNAV_DIR} && opencode'"

# ── W0·P2: Claude Worker ──────────────────────────────────────────────────────
pane_run "agents.2" "clear"
pane_run "agents.2" "printf '\\033[1;35m'"
pane_run "agents.2" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "agents.2" "echo '  W0·P2  CLAUDE CODE — WORKER (bounded tasks)'"
pane_run "agents.2" "echo '  Role : Execute isolated tasks assigned by Main.'"
pane_run "agents.2" "echo '         Only edits files explicitly assigned.'"
pane_run "agents.2" "echo '         Reports: TASK DONE: <change> | RESULT: <output>'"
pane_run "agents.2" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "agents.2" "printf '\\033[0m'"
pane_run "agents.2" "echo"
pane_run "agents.2" "echo 'Waiting for task assignment from W0·P0 (Claude Main).'"
pane_run "agents.2" "echo 'START: claude --continue  (or: claude -p \"<task>\"'"
pane_run "agents.2" "echo"

# ==============================================================================
# W1: server — 2-pane layout
# ==============================================================================
tmux new-window -t "${SESSION}" -n "server" -c "$VLNAV_ROOT"
tmux split-window -h -t "${SESSION}:server" -c "$VLNAV_ROOT" -l "60%"
# P0=container ctrl (left 40%), P1=server logs (right 60%)

pane_title "server.0" "CONTAINER  start/stop/health"
pane_title "server.1" "SERVER LOGS  docker logs -f"

pane_run "server.0" "clear"
pane_run "server.0" "printf '\\033[1;32m'"
pane_run "server.0" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "server.0" "echo '  W1·P0  DOCKER CONTAINER MANAGEMENT'"
pane_run "server.0" "echo '  Container : vlnav_internvla_server  Port: $SERVER_PORT'"
pane_run "server.0" "echo '  Scripts   : (run from $VLNAV_ROOT)'"
pane_run "server.0" "echo '    ./scripts/model/start.sh   — start'"
pane_run "server.0" "echo '    ./scripts/model/stop.sh    — stop'"
pane_run "server.0" "echo '    ./scripts/model/health.sh  — poll /health'"
pane_run "server.0" "echo '    ./scripts/model/exec.sh    — interactive shell'"
pane_run "server.0" "echo '    ./scripts/model/commit.sh  — snapshot image'"
pane_run "server.0" "echo '    ./scripts/model/logs.sh    — follow logs'"
pane_run "server.0" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "server.0" "printf '\\033[0m'"
pane_run "server.0" "echo"
pane_run "server.0" "echo 'Container status:'"
pane_run "server.0" "docker ps --filter 'name=vlnav_internvla_server' --format '  {{.Names}}: {{.Status}}' 2>/dev/null || echo '  (docker not accessible)'"

pane_run "server.1" "clear"
pane_run "server.1" "printf '\\033[1;32m'"
pane_run "server.1" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "server.1" "echo '  W1·P1  SERVER LOGS'"
pane_run "server.1" "echo '    ./scripts/model/logs.sh         — follow container logs'"
pane_run "server.1" "echo '    watch -n2 curl localhost:$SERVER_PORT/async_metrics  — live stats'"
pane_run "server.1" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "server.1" "printf '\\033[0m'"
pane_run "server.1" "echo"
pane_run "server.1" "docker logs --tail 30 vlnav_internvla_server 2>/dev/null || echo '(container not running)'"

# ==============================================================================
# W2: experiment — 2×2 grid (mirrors async_check_tmux.sh layout)
# ==============================================================================
tmux new-window -t "${SESSION}" -n "experiment" -c "$INTERNAV_DIR"
# Build 2×2: split right half, then split each column vertically
#   P0 (top-left) | P1 (top-right)
#   P2 (bot-left) | P3 (bot-right)
tmux split-window -h -t "${SESSION}:experiment" -c "$INTERNAV_DIR" -l "50%"
# Active pane is now P1 (right). Select P0 (left) and split it.
tmux select-pane  -t "${SESSION}:experiment.0"
tmux split-window -v -t "${SESSION}:experiment" -c "$INTERNAV_DIR" -l "50%"
# P0=top-left, P2=bot-left (active). Now select P1 and split it.
tmux select-pane  -t "${SESSION}:experiment.1"
tmux split-window -v -t "${SESSION}:experiment" -c "$INTERNAV_DIR" -l "50%"
# P0=top-left, P1=top-right, P2=bot-left, P3=bot-right

pane_title "experiment.0" "SERVER  (inside Docker)"
pane_title "experiment.1" "BRIDGE  (inside Docker)"
pane_title "experiment.2" "CLIENT  (host ROS2)"
pane_title "experiment.3" "ROSBAG  (host ROS2)"

# W2·P0: server inside container
pane_run "experiment.0" "clear"
pane_run "experiment.0" "printf '\\033[1;31m'"
pane_run "experiment.0" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "experiment.0" "echo '  W2·P0  INTERNAV SERVER  (run inside Docker container)'"
pane_run "experiment.0" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "experiment.0" "printf '\\033[0m'"
pane_run "experiment.0" "echo '  Step 1 — shell into container:'"
pane_run "experiment.0" "echo '    cd $VLNAV_ROOT && ./scripts/model/exec.sh'"
pane_run "experiment.0" "echo"
pane_run "experiment.0" "echo '  Step 2 — inside container, start server:'"
pane_run "experiment.0" "echo '    cd /workspace/InternNav'"
pane_run "experiment.0" "echo '    /opt/venv/bin/python3 scripts/realworld/http_internvla_server_debug.py \\'"
pane_run "experiment.0" "echo '      --mode async --kv-cache --temperature 0.8 \\'"
pane_run "experiment.0" "echo '      --max-new-tokens 80 --resize_w 256 --resize_h 256 \\'"
pane_run "experiment.0" "echo '      --num_history 1 --plan_step_gap 12 --device cuda:0 \\'"
pane_run "experiment.0" "echo '      --model_path checkpoints/InternVLA-N1-w-NavDP \\'"
pane_run "experiment.0" "echo '      --calib scripts/realworld/calib/calib_scout.txt'"

# W2·P1: bridge inside container
pane_run "experiment.1" "clear"
pane_run "experiment.1" "printf '\\033[1;31m'"
pane_run "experiment.1" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "experiment.1" "echo '  W2·P1  SCOUT BRIDGE  (inside Docker — start AFTER server)'"
pane_run "experiment.1" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "experiment.1" "printf '\\033[0m'"
pane_run "experiment.1" "echo '  Step 1:  cd $VLNAV_ROOT && ./scripts/model/exec.sh'"
pane_run "experiment.1" "echo '  Step 2:  cd /workspace/InternNav'"
pane_run "experiment.1" "echo '           python3.12 scripts/realworld/scout_bridge.py'"

# W2·P2: client on host
pane_run "experiment.2" "clear"
pane_run "experiment.2" "printf '\\033[1;31m'"
pane_run "experiment.2" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "experiment.2" "echo '  W2·P2  HTTP CLIENT  (host with ROS2 — start AFTER server + bridge)'"
pane_run "experiment.2" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "experiment.2" "printf '\\033[0m'"
pane_run "experiment.2" "echo '  source /opt/ros/jazzy/setup.bash'"
pane_run "experiment.2" "echo '  cd $INTERNAV_DIR'"
pane_run "experiment.2" "echo '  python3.12 scripts/realworld/http_internvla_client_debug.py \\'"
pane_run "experiment.2" "echo '    --mode async --kv-cache --temperature 0.8 \\'"
pane_run "experiment.2" "echo '    --jpeg-quality 95 --depth-png-compress 6 \\'"
pane_run "experiment.2" "echo '    --calib scripts/realworld/calib/calib_scout.txt'"
pane_run "experiment.2" "echo"
pane_run "experiment.2" "echo '  Watch for: [HTTP] <N> ms  → must be <20ms for Gate 0'"
pane_run "experiment.2" "source /opt/ros/jazzy/setup.bash 2>/dev/null && echo '  ROS2 Jazzy: ready' || echo '  ROS2 not on host PATH'"

# W2·P3: rosbag on host
pane_run "experiment.3" "clear"
pane_run "experiment.3" "printf '\\033[1;31m'"
pane_run "experiment.3" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "experiment.3" "echo '  W2·P3  ROSBAG PLAYBACK  (host ROS2 — start LAST)'"
pane_run "experiment.3" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "experiment.3" "printf '\\033[0m'"
pane_run "experiment.3" "echo '  source /opt/ros/jazzy/setup.bash'"
pane_run "experiment.3" "echo '  ros2 bag play ../rosbag/my_camera_bag_20260317_073623 --rate 0.5'"
pane_run "experiment.3" "echo"
pane_run "experiment.3" "echo '  Gate 0 bags (run all 3 to pass):'"
pane_run "experiment.3" "echo '    my_camera_bag_20260317_061841'"
pane_run "experiment.3" "echo '    my_camera_bag_20260317_063047'"
pane_run "experiment.3" "echo '    my_camera_bag_20260317_073623  (sync baseline ref)'"
pane_run "experiment.3" "echo"
pane_run "experiment.3" "echo '  Or run full async_check_tmux.sh from W0 pane (NO_ATTACH=1 set):'"
pane_run "experiment.3" "echo '    KV_CACHE=1 TEMPERATURE=0.8 bash scripts/realworld/async_check_tmux.sh'"
pane_run "experiment.3" "source /opt/ros/jazzy/setup.bash 2>/dev/null && echo '  ROS2 Jazzy: ready' || echo '  ROS2 not on host PATH'"

# ==============================================================================
# W3: monitor — 3-pane layout (all auto-start)
# ==============================================================================
tmux new-window -t "${SESSION}" -n "monitor" -c "$INTERNAV_DIR"
tmux split-window -h -t "${SESSION}:monitor" -c "$INTERNAV_DIR" -l "50%"
tmux select-pane  -t "${SESSION}:monitor.0"
tmux split-window -v -t "${SESSION}:monitor" -c "$INTERNAV_DIR" -l "45%"
# P0=metrics(top-left), P2=logs(bot-left), P1=git(right)

pane_title "monitor.0" "METRICS  /stats poll every 3s"
pane_title "monitor.1" "GIT  both repos watch"
pane_title "monitor.2" "LOGS  experiment tail"

# W3·P0: live metrics — poll /stats every 3s
pane_run "monitor.0" "while true; do clear; printf '\\033[1;36m%s\\033[0m  /async_metrics\\n' \"\$(date '+%H:%M:%S')\"; curl -sf http://localhost:$SERVER_PORT/async_metrics 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); keys=['total_requests','joint_req_hz','joint_latency_ms','trajectory_ratio','background_s2_runs','elapsed_seconds']; [print(f'  {k:35s}: {d[k]}') for k in keys if k in d]\" 2>/dev/null || echo '  server offline (port $SERVER_PORT)'; sleep 3; done"

# W3·P1: git watcher
pane_run "monitor.1" "while true; do clear; printf '\\033[1;33m=== research/async-foundation ===\\033[0m\\n'; git -C '$INTERNAV_DIR' status --short 2>/dev/null | head -10; echo; printf '\\033[1;33m=== opencode-exp ===\\033[0m\\n'; git -C '${WORKTREE_DIR:-$INTERNAV_DIR}' status --short 2>/dev/null | head -8 || echo '  (no worktree)'; echo; printf '\\033[1;33m=== recent commits ===\\033[0m\\n'; git -C '$INTERNAV_DIR' log --oneline -5 2>/dev/null; sleep 10; done"

# W3·P2: log tail
pane_run "monitor.2" "echo 'Waiting for experiment logs in test_data/...'; while true; do LATEST=\$(ls -t $INTERNAV_DIR/test_data/*/client.log 2>/dev/null | head -1); if [ -n \"\$LATEST\" ]; then echo \"Tailing: \$LATEST\"; tail -f \"\$LATEST\"; else sleep 4; fi; done"

# ==============================================================================
# W4: paper — 1 pane
# ==============================================================================
tmux new-window -t "${SESSION}" -n "paper" -c "$INTERNAV_DIR"

pane_title "paper.0" "PAPER  writing / plots"
pane_run "paper.0" "clear"
pane_run "paper.0" "printf '\\033[1;34m'"
pane_run "paper.0" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "paper.0" "echo '  W4  PAPER / WRITING / ABLATION PLOTS'"
pane_run "paper.0" "echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'"
pane_run "paper.0" "printf '\\033[0m'"
pane_run "paper.0" "echo '  Parse log → stats:'"
pane_run "paper.0" "echo '    python scripts/realworld/stats_recorder.py parse test_data/.../client.log --tag <name>'"
pane_run "paper.0" "echo"
pane_run "paper.0" "echo '  Generate all ablation plots:'"
pane_run "paper.0" "echo '    python scripts/viz/plot_ablations.py'"
pane_run "paper.0" "echo '    python scripts/viz/plot_ablations.py --var temperature'"
pane_run "paper.0" "echo '    python scripts/viz/plot_ablations.py --tradeoff'"
pane_run "paper.0" "echo"
pane_run "paper.0" "echo '  Trajectory plots:'"
pane_run "paper.0" "echo '    python scripts/viz/plot_trajectories.py --tag <name>'"
pane_run "paper.0" "echo"
pane_run "paper.0" "echo '  Gate check:'"
pane_run "paper.0" "echo '    bash scripts/gate_check.sh 0   # latency + ratio'"
pane_run "paper.0" "echo '    bash scripts/gate_check.sh 1   # temperature sweep results'"
pane_run "paper.0" "echo"
pane_run "paper.0" "echo '  Key files: RESEARCH_TRACKING.md  EXPERIMENTS_LOG.md  INNOVATIONS_CATALOGUE.md'"
pane_run "paper.0" "echo"

# ==============================================================================
# Final: focus W0·P0 and attach
# ==============================================================================
tmux select-window -t "${SESSION}:agents"
tmux select-pane   -t "${SESSION}:agents.0"

echo ""
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${G}  InternNav Lab Session '${SESSION}' ready  (${COLS}×${ROWS})${NC}"
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
printf "  ${C}Ctrl-b 0${NC}  agents      Claude Main | OpenCode | Claude Worker\n"
printf "  ${C}Ctrl-b 1${NC}  server      Container ctrl | Server logs\n"
printf "  ${C}Ctrl-b 2${NC}  experiment  Server | Bridge | Client | Rosbag\n"
printf "  ${C}Ctrl-b 3${NC}  monitor     Live /stats | Logs | Git watch  (auto-running)\n"
printf "  ${C}Ctrl-b 4${NC}  paper       Plots | Writing | Gate check\n"
echo ""
echo "  Detach:    Ctrl-b d"
echo "  Re-attach: tmux attach -t $SESSION"
echo "  Kill:      tmux kill-session -t $SESSION"
echo ""
echo -e "  ${Y}Recommended start order:${NC}"
echo "  1. W1 → start.sh → health.sh (wait for model load ~30s)"
echo "  2. W2 → exec.sh in P0+P1, then client in P2, then bag in P3"
echo "  3. W0·P0 → claude --continue  (continues this session)"
echo ""

if [ -z "${NO_ATTACH:-}" ]; then
    tmux attach -t "$SESSION"
fi
