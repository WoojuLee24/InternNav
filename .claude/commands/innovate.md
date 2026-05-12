---
allowed-tools: Read, WebSearch, Bash(cat *), Bash(grep*), Bash(git log*)
description: Research ideation — generate novel, implementable research directions grounded in current bottlenecks and literature. Usage: /innovate [focus area]
---

## System State

**Production stack (Gate 3n)**:
!`grep -A5 "Production Stack" PLAN.md 2>/dev/null | head -8`

**What has been tried (Gates 3b–3q)**:
!`grep "^| 3" PLAN.md 2>/dev/null | head -20`

**Untouched categories** (85/101 innovations queued):
- Inference Efficiency (I-002–I-009: action tokens, speculative decoding, KV compression)
- Speculative Execution (I-060–I-065)
- Knowledge Distillation (I-070–I-079)
- Offline Evaluation Proxy (I-111–I-116) ← CRITICAL — unblocks Gate 4

**Current blockers**:
!`grep -E "BLOCKED|blocked|⛔" PLAN.md | head -8`

---

## Task

You are a senior researcher in embodied AI and VLN. InternNav runs S1 (fast visual policy) + S2 (async LLM planner) on a real Scout robot.

**Focus area**: `$ARGUMENTS` (if blank: prioritize what unblocks Gate 4 or Phase 5)

**Context you must know**:
- Phase 3 (temporal cache stack) is complete: 91% skip, V_max=0.0791, SNR=19.67×
- Phase 4 (Habitat benchmark) is blocked — no Habitat install
- Phase 5 (S2→S1 distillation) is the MAIN CONTRIBUTION but needs Gate 4
- We are at ~15% of planned work; the high-value innovations haven't started

Generate **5 novel research directions**, each with:

#### [N]. [TITLE] — Priority: [P0=unblocks everything | P1=high value | P2=nice to have]

**Problem**: Which specific blocker or gap does this address?

**Core idea**: The novel mechanism in 2–3 sentences. Be concrete, not vague.

**Connection to literature**: Specific paper or technique (speculative decoding, distillation, etc.)

**Implementation**:
```
File: [exact file path]
Change: [what to add/modify, ~lines]
Effort: [hours/days]
```

**Measurable gate criterion**:
- What metric improves? By how much?
- Pass/fail threshold?

**Why publishable**: Would this be a contribution at ICLR / ICRA / CoRL 2027?

---

After generating 5 directions, add:

### Recommended Next Experiment
Given that we're at 15% completion and Gate 4/5 are blocked, what single experiment
would give the most research leverage today? Frame as a gate: hypothesis + metric + threshold.
