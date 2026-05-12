---
allowed-tools: Read, Bash(cat *), Bash(grep*), Bash(git log*), Bash(git branch*), Bash(docker exec*), Bash(docker ps*), Bash(date*)
description: Gate management — design a new gate script, check gate status, parse gate results. Usage: /gate design [name] | /gate status | /gate parse [log_dir]
---

## Current Gate State

**Pipeline**:
!`head -12 docs/GATE_STATUS.md 2>/dev/null`

**Container**:
!`docker ps --format "{{.Names}}: {{.Status}}" 2>/dev/null | grep vlnav || echo "[not running]"`

**Queued innovations** (from PLAN.md Phase 3X/6):
!`grep -E "Phase 3X|Phase 6|I-1[01][0-9]|I-06[0-9]|I-02[0-9]" PLAN.md 2>/dev/null | head -15`

---

## Task

`$ARGUMENTS` — one of: `design [gate_name]`, `status`, `parse [log_dir]`, `next`

---

### Mode: `design [name]`

Generate a complete gate script following the established pattern.
Use existing gate scripts as templates (e.g., gate3n_production_stack.sh, gate3q_variance_analysis.sh).

Requirements for every gate script:
1. `set -e` + source ROS2
2. Server start with production flags (--mode async --temperature 0.75 --kv-cache --pre-warm-frames 3)
3. Configure via REST API (not flags) for runtime mutability
4. Run all 3 bags: 073623, 061841, 063047 at rate=0.5
5. Save metrics.json per bag to `/tmp/gate[name]/[bag]_[tag]/`
6. Terminal sentinel: `echo "GATE[NAME]_DONE"` at very end
7. Inline Python analysis with pairwise Cramér's V vs Gate 3n NOCACHE baseline
8. LaTeX table output for paper

Gate naming: next available letter after 3q is 3r, 3s, etc. Or 4a for Phase 4 gates.

The script must test innovations NOT already covered by Gates 3b–3q.
**Do not re-test**: τ, MH, AA, TR-EMA, slope — these are in production.
Focus on: offline metrics (I-111–I-116), action tokens (I-020), speculative prefetch (I-060).

---

### Mode: `status`

Check all active gate experiments:
1. Any containers with active gate scripts (docker exec to check /tmp/gate*)
2. Recent gate commits (git log --oneline --grep=gate)
3. What gate to run next (per PLAN.md Phase 3X and 6)

---

### Mode: `parse [log_dir]`

Parse gate results from a completed experiment directory.
1. Load all metrics.json files
2. Compute Cramér's V vs NOCACHE baseline
3. Report: Skip%, V per bag, V_max, OP counts, pass/fail
4. Generate LaTeX table rows
5. Compare to Gate 3n production stack (the current gold standard)

---

### Mode: `next`

Given current state (15% complete, Gate 4 blocked), recommend the next gate to run.
Priority: does it unblock Gate 4, or does it test a genuinely new mechanism?
Output: gate name, innovation ID, hypothesis, estimated run time, expected result.
