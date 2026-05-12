---
allowed-tools: Read, Bash(cat *), Bash(git log*), Bash(git diff*), Bash(git branch*), Bash(date*), Bash(docker ps*), Bash(docker exec*)
description: Daily research standup — gate status, proven results, next experiment. Usage: /daily
---

## Daily Context

**Today**: !`date '+%Y-%m-%d %A'`
**Branch**: !`git branch --show-current`

**Commits since yesterday**:
!`git log --oneline --since="25 hours ago" 2>/dev/null || git log --oneline -5`

**Container status**:
!`docker ps --format "{{.Names}}: {{.Status}}" 2>/dev/null | grep vlnav || echo "[no container running]"`

**Active experiments**:
!`docker exec vlnav_internvla_server bash -c "ls /tmp/gate3* /tmp/gate4* /tmp/gate5* /tmp/gate6* 2>/dev/null | tail -8" 2>/dev/null || echo "[none]"`

**Gate pipeline**:
!`head -10 docs/GATE_STATUS.md 2>/dev/null`

---

## Task

Run the daily research standup for InternNav dual-system VLN project.

### Yesterday (Evidence)
From git log and container — what was actually run/committed/measured?

### Today (Hypothesis)
What is the single most valuable experiment to run today?
Frame as: "Hypothesis: if we do X, metric Y changes by Z because mechanism M."
Link to specific PLAN.md phase and innovation ID (e.g., I-111).

### Progress Check
**We are at ~15% of the overall plan.** Answer:
1. Is Phase 4 (Habitat + R2R) still blocked? If yes, what's the unblock path today?
2. Is there a Phase 3X offline metric (I-111–I-116) that can substitute Gate 4?
3. Are we doing 80% research (new hypotheses, new experiments) vs 20% engineering (code fixes)?

### Blockers
Specific blockers with specific unblock actions (not vague).

### Next Action
One concrete command to run or one file to create. Not a list — one thing.
