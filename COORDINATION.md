# Multi-Session Coordination Board

> **Every Claude session and agent fork MUST read this file at the start of work.**
> **Every session MUST update this file when it claims, completes, or abandons work.**
> Keep entries short — this is a coordination surface, not documentation.

_Format: update the `_last_updated` line and the relevant row. Do NOT rewrite the whole file._
_last_updated: 2026-10-22 20:10 UTC by session_01ARwduxfwfZ4FyxfSkDxPoN_

---

## Branch Strategy

| Branch | Purpose | Owner |
|--------|---------|-------|
| `dev` | Main integration — merge target | all (merge only) |
| `dev/vln-ce` | VLN-CE training + eval configs | session A |
| `dev/nexusnav-s2` | S2 modules, NexusNav policy | session B |
| `dev/paper` | NexusNav + ChronoNav paper drafts | session C |
| `dev/chrononav` | ChronoNav annotation pipeline (v280+) | session D |

**Current branch:** `dev` (single — all work committed here for now).
When multiple sessions run in parallel, create feature branches and open a PR to `dev`.

---

## File Ownership (live)

A file is "owned" until its session pushes to `dev`. Claim before touching.

| File / Directory | Owner Session | Status | Notes |
|-----------------|---------------|--------|-------|
| `internnav/agent/latent_action.py` | session_01ARw | ✅ done | VQ-VAE 256-token, 22 unit tests pass |
| `internnav/agent/latent_reasoning.py` | session_01ARw | ✅ done | ThoughtCell N=4, S2ThoughtCache |
| `internnav/agent/memory/` | session_01ARw | ✅ done | TaskMemory, WorldMemory, BodyMemory |
| `internnav/agent/s2_kv_cache.py` | session_01ARw | ✅ done | S2KVCache, VisionEmbeddingCache τ=0.92 |
| `internnav/agent/world_predictor.py` | session_01ARw | ✅ done | Prophet-style online learning |
| `internnav/model/basemodel/nexusnav/` | session_01ARw | ✅ done | NexusNavNet, NexusNavS2Heads, NexusNavModelConfig |
| `internnav/model/__init__.py` | session_01ARw | ✅ done | NexusNav_Policy registered |
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` | session_01ARw | ✅ done | 4-bit/8-bit device_map fix |
| `paper/nexusnav_v3.tex` | session_01ARw | 🔄 draft | 8-page CVPR draft, needs measured results |
| `paper/nexusnav_refs.bib` | session_01ARw | 🔄 draft | 22 refs |
| `scripts/eval/configs/habitat_*_vln_ce_mini_*.py` | session_01ARw | ✅ done | 4 model eval configs |
| `scripts/train/base_train/configs/*_vln_ce_mini.py` | session_01ARw | ✅ done | 4 model training configs |
| `scripts/train/base_train/start_vln_ce_train.sh` | session_01ARw | ✅ done | per-model launch script |
| `scripts/cloud/` | session_01ARw | ✅ done | Dockerfile, PyTorchJob YAML, ZeRO-2 config |
| `internnav/configs/` | FREE | — | claim before touching |
| `internnav/evaluator/` | FREE | — | claim before touching |
| `internnav/agent/internvla_n1_agent.py` | FREE | — | claim before touching |

---

## Running Containers (live)

| Container | Model | GPU | Status | Log |
|-----------|-------|-----|--------|-----|
| `vln_ce_mini_eval_4bit` | DualVLN 4-bit | all | ▶ RUNNING (ep 0/10) | `/mnt/nvme0/vln_habitat/logs/vln_ce_mini_dualvln_4bit.log` |

**Completed containers (results):**
| Container | Model | Result |
|-----------|-------|--------|
| `mini_eval_final` | NavDP | S2=7529ms, 0.13Hz — baseline measured |
| `mini_eval_v3` | DualVLN/ChronoNav/NexusNav | PARTIAL — depth_anything path fix applied |

---

## Work Queue (ordered by priority)

### IN PROGRESS
- [ ] VLN-CE mini eval: DualVLN 4-bit (10 eps) — `vln_ce_mini_eval_4bit`
- [ ] Push 5 commits to origin/dev (blocked by auto-mode, user must run manually)

### NEXT (sequential — don't start until prior done)
- [ ] Collect DualVLN VLN-CE baseline SR/SPL/NE
- [ ] Run ChronoNav mini eval (use `habitat_vln_ce_mini_chrononav_cfg.py`, `load_in_4bit: True`)
- [ ] Run NexusNav-V3 mini eval (use `habitat_vln_ce_mini_nexusnav_cfg.py`)
- [ ] Fill real numbers into `paper/nexusnav_v3.tex` ablation table
- [ ] v280 reannotation (blocked on vLLM server at 10.77.32.231 restart)

### PARALLEL-SAFE (any session can start independently)
- [ ] ChronoNav: integrate NexusNav S2 modules into ChronoNav agent loop
- [ ] NexusNav: wire LatentReasoner + S2ThoughtCache into inference path (`internvla_n1_agent.py`)
- [ ] OmniNav architecture diagram (update for NexusNav)
- [ ] Naver Cloud NavDP training (requires local smoke test to pass first)

---

## Mini 4-Model Comparison — Running Log

**Goal:** measure real S2 latency (ms), S2 Hz, and SR on VLN-CE val_unseen mini (10 eps) for 4 models.

| Model | Container | S2 latency | S2 Hz | VLN-CE SR | Notes |
|-------|-----------|-----------|-------|-----------|-------|
| NavDP | `mini_eval_final` | 7529ms | 0.13 Hz | — | S1 only (VLN-PE model) |
| DualVLN (4-bit) | `vln_ce_mini_eval_4bit` | — | — | **RUNNING** | Habitat, 10 eps |
| ChronoNav | pending | — | — | pending | need DualVLN result first |
| NexusNav-V3 | pending | — | — | pending | need ChronoNav result first |

**Three-session history:**
- Session 1: mini_eval_final ran NavDP (7529ms), DualVLN/ChronoNav/NexusNav FAILED (depth_anything path)
- Session 2: depth_anything fix via CWD symlink applied, mini_eval_v3 container ran — partial results
- Session 3 (current): Habitat VLN-CE eval configured (4-bit for 12GB GPU), running now

---

## Session Handoff Protocol

When finishing a session:
1. Update **Running Containers** and **Work Queue** tables above
2. Commit `COORDINATION.md` with session ID in the commit message
3. Push to `dev` (or tell user to push if blocked)
4. Update `memory/project_state.md` with current best results
5. Write 2-line handoff note below:

### Latest Handoff Note
**From:** session_01ARwduxfwfZ4FyxfSkDxPoN (2026-10-22)
**To next session:** DualVLN 4-bit eval running in `vln_ce_mini_eval_4bit`. When done, collect SR/SPL, then launch ChronoNav and NexusNav-V3 evals sequentially. 5 commits ready to push to origin/dev (user must run `git push`). NexusNav paper draft at `paper/nexusnav_v3.tex` — needs real numbers. 3-hour revisit loop armed.

---

## Architecture Invariants (never change without updating this)

| Invariant | Value |
|-----------|-------|
| VisionEmbeddingCache EMA τ | 0.92 |
| LatentReasoner N steps | 4 |
| ActionTokenVocabulary vocab_size | 256 |
| ActionTokenVocabulary embed_dim | 64 |
| NexusNavS2Heads hidden→thought_dim | 4096→256 |
| LoRA r / alpha | 32 / 64 |
| VLN-CE action space | STOP=0, FORWARD=1, LEFT=2, RIGHT=3 |
| Hardstop: SR<62% at ≥200 eps | kill + result.json |
| Hardstop: SR≥62% at ≥500 eps | cap at 500 eps |
| GPU rule (this machine) | single GPU → `--gpus all` |
