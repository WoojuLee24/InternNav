---
allowed-tools: Read, WebFetch, WebSearch, Bash(cat *), Bash(grep*), Bash(date*)
description: Track a research paper — extract core ideas, rate relevance to InternNav, map to codebase components, append to RESEARCH_TRACKING.md. Usage: /paper arxiv:2401.12345 or /paper "NavGPT: explicit reasoning in VLN"
---

## Existing Research Tracker
!`cat RESEARCH_TRACKING.md 2>/dev/null || echo "[RESEARCH_TRACKING.md not yet created]"`

## InternNav Component Map (for connection mapping)
- `internnav/agent/internvla_n1_agent.py` — S1/S2 dual-system coordination, threading, locks
- `internnav/agent/internvla_n1_agent_realworld.py` — real-world deployment variant
- `scripts/realworld/http_internvla_server*.py` — inference server, async pipeline
- `scripts/realworld/http_internvla_client*.py` — ROS2 client, control loop
- `internnav/model/` — model architecture, policy classes
- `internnav/evaluator/` — evaluation methodology, metrics
- `internnav/env/` — simulator environments

---

## Task

**Input**: `$ARGUMENTS` — arxiv ID (e.g. `2401.12345`), URL, or paper title.

### Step 1 — Fetch
- If arxiv ID: fetch `https://arxiv.org/abs/[ID]` and extract title, authors, abstract, key contributions
- If title: WebSearch for the paper, find abstract
- If URL: fetch directly

### Step 2 — Analyze for InternNav
Answer these specific questions:
1. **Core contribution** (1 sentence): What is fundamentally new?
2. **Problem match**: Which exact problem in InternNav does this address?
   - S2 latency / async scheduling?
   - S1/S2 communication bandwidth?
   - Trajectory quality vs speed tradeoff?
   - VLN benchmark performance?
   - Real-world deployment robustness?
   - Other?
3. **Borrowable technique**: What specific mechanism could we directly adapt?
4. **Implementation target**: Which file/class would this change?
5. **Estimated impact**: Would this likely improve `trajectory_ratio`? `s2_req_hz`? Navigation success rate?

### Step 3 — Rate Relevance (1–5)
- **5**: Directly applicable, implement next sprint
- **4**: Strong match, worth prototyping this week
- **3**: Relevant context, read carefully when planning next experiment
- **2**: Weak connection, background knowledge
- **1**: Tangential, file for general awareness

### Step 4 — Append to RESEARCH_TRACKING.md
Add an entry in this exact format at the top of the Papers section:

```
### [TITLE] — Relevance: X/5
**Date logged**: YYYY-MM-DD
**Source**: [arxiv link or URL]
**Authors**: [first author et al., year]
**Core idea**: [1 sentence]
**Borrowable technique**: [specific mechanism]
**Target component**: [file or module]
**Expected impact**: [which metrics improve and why]
**Status**: `[ ] queued` | `[ ] reading` | `[x] analyzed` | `[ ] prototyping` | `[ ] implemented`
**Notes**: 
```

After appending, confirm the entry was added and suggest whether this should be implemented before or after the current async experiment concludes.
