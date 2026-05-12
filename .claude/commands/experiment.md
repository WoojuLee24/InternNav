---
allowed-tools: Read, Bash(cat *), Bash(git log*), Bash(git branch*), Bash(date*), Bash(grep*), Bash(docker exec*)
description: Design, log, and compare experiments — structured hypothesis testing for InternNav. Usage: /experiment design | /experiment log | /experiment compare | /experiment status
---

## Current Experiment State

**Branch**: !`git branch --show-current`
**Date**: !`date '+%Y-%m-%d'`

**Production baseline (Gate 3n)**:
- τ=0.92, MH=15, AA, TR-EMA α=0.10, slope δ_s=0.010
- V_max=0.0791, skip=91%, hz=11.8–12.2, noise floor V=0.007

**Recent gate results**:
!`head -15 docs/GATE_STATUS.md 2>/dev/null`

**Recent commits**:
!`git log --oneline -8`

---

## Task

`$ARGUMENTS` — one of: `design`, `log [results]`, `compare`, `status`, `next`

---

### Mode: `design` (or describing a new experiment)

Produce a complete experiment spec following the gate protocol.
Every experiment must go beyond the current Phase 3 stack — do not re-test already-validated configs.
**Priority**: Does this experiment unblock Gate 4, improve Phase 5, or address a novel mechanism?

```
## Experiment: [NAME] — Gate [N]
**Date**: [today]
**Innovation**: [I-XXX from INNOVATIONS_CATALOGUE.md]
**Phase**: [which PLAN.md phase]

### Hypothesis
If [CHANGE], then [METRIC] will [IMPROVE] by [AMOUNT] because [MECHANISM].

### What makes this novel
[Why this isn't covered by existing Gates 3b–3q]

### Independent Variable
[Exactly what changes — code path, parameter, endpoint]

### Dependent Variables
| Metric | Gate 3n Baseline | Expected | Pass Threshold |
|--------|-----------------|---------|---------------|
| V_max (3 bags) | 0.0791 | ≤ ? | ≤ 0.10 |
| skip% | 91% | ? | ≥ 88% |
| [new metric] | — | ? | ? |

### Exact Commands
```bash
# Start server in container
docker exec vlnav_internvla_server bash -c "..."
# Configure via API
curl "http://localhost:5802/set_[endpoint]?..."
# Run bags
```

### Success Criteria
[Specific thresholds, all must pass]

### If it fails
[What the failure would tell us scientifically]
```

---

### Mode: `log`

Format results and append to EXPERIMENTS_LOG.md:
```
## Result: [EXPERIMENT_NAME] — [DATE] — Gate [N] — [PASS/FAIL]
**Innovation**: [I-XXX]

| Metric | Baseline | Result | Delta | Status |
|--------|----------|--------|-------|--------|

**Key finding**: [What was learned beyond just pass/fail]
**Impact on production stack**: [Add / No change / Explains prior anomaly]
**Next**: [What this result suggests]
**Commit**: [git hash]
```

---

### Mode: `status`

Summarize:
1. **Proven (in production)**: Gate 3n stack items
2. **Promising**: marginal gates (3o, 3p) and what they need to pass cleanly
3. **Failed** (and what each teaches): I-049, I-051, I-052, I-053
4. **Critical gap**: What's blocking Phase 4 and Phase 5
5. **Untouched** (85/101 innovations): which category is highest leverage next

---

### Mode: `next`

Given the current state (~15% complete, Gate 4 blocked), recommend the **single highest-leverage experiment** to run today.
Must be runnable with current container setup (no Habitat required, OR specify Habitat setup as the task itself).
Frame as: Innovation ID + hypothesis + exact commands + pass criteria.

---

### Mode: `compare`

Side-by-side table of all gate results from GATE_STATUS.md + EXPERIMENTS_LOG.md.
Group by: (a) in production stack, (b) failed with finding, (c) marginal, (d) not yet tried.
Highlight the path to Gate 4.
