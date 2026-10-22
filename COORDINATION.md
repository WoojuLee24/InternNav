# Multi-Session Coordination Board

> **Every Claude session and agent fork MUST read this file at the start of work.**
> **Every session MUST update this file when it claims, completes, or abandons work.**
> Keep entries short — this is a coordination surface, not documentation.

_Format: update the `_last_updated` line and the relevant row. Do NOT rewrite the whole file._
_last_updated: 2026-10-22 20:10 UTC by session_01ARwduxfwfZ4FyxfSkDxPoN_

---

## Worktree Map (ISOLATED — use these for parallel sessions)

| Worktree path | Branch | Purpose | Session |
|---------------|--------|---------|---------|
| `/home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev` | `dev` | Main integration (MERGE TARGET) | coordinator |
| `/mnt/nvme0/worktrees/vln-ce` | `wt/vln-ce` | VLN-CE eval, training configs, eval monitoring | session A |
| `/mnt/nvme0/worktrees/nexusnav-s2` | `wt/nexusnav-s2` | S2 module wiring into agent inference path | session B |
| `/mnt/nvme0/worktrees/paper` | `wt/paper` | NexusNav + ChronoNav paper writing | session C |
| `/mnt/nvme0/worktrees/wiring` | `wt/wiring` | VisionEmbeddingCache evaluator wiring | session D |

**How to use:**
1. New session: pick a worktree that matches your work. Edit files ONLY in that worktree's directory.
2. When done: `git -C <worktree> commit -m "..."` then `git -C <worktree> push origin <branch>`
3. Coordinator session: review PRs, `git merge wt/<branch>` into `dev`, push `dev`.
4. To get another session's finished work: `git -C <your-worktree> merge origin/dev` after they push.

**Share findings without conflicts**: read OMNAV_MASTER.md from `dev` branch (it gets updated when worktrees merge). Each worktree has its own copy of all files — edits don't touch `dev` until merged.

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
| `vln_ce_mini_s2only` | DualVLN S2-only 4-bit | all | ▶ RUNNING (ep 0/10) | `/mnt/nvme0/vln_habitat/logs/vln_ce_mini_s2only_4bit.log` |

**Completed containers (results):**
| Container | Model | Result |
|-----------|-------|--------|
| `mini_eval_final` | NavDP | S2=7529ms, 0.13Hz — baseline measured |
| `mini_eval_v3` | DualVLN/ChronoNav/NexusNav | PARTIAL — depth_anything path fix applied |
| `vln_ce_mini_eval_4bit` | DualVLN 4-bit | CRASHED — memory_encoder dtype mismatch in generate_traj |

---

## Work Queue (ordered by priority)

### IN PROGRESS
- [x] VLN-CE mini eval: DualVLN S2-only 4-bit — DONE: SR=90%, SPL=77.4%, NE=4.50 (10 eps mini)
- [ ] VLN-CE mini eval: ChronoNav 4-bit — `vln_ce_mini_chrononav` LAUNCHING NOW
- [ ] Push 9+ commits to origin/dev (user must run: `! git -C /home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev push origin dev`)

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
| DualVLN S2-only (4-bit) | `vln_ce_mini_s2only` | 1517ms | 0.66 Hz | **SR=90% SPL=77.4** ✅ | mode=system2, 10 eps, 9/10 success |
| DualVLN dual_system (4-bit) | pending | — | — | pending | need fp16 cast fix or cloud GPU |
| ChronoNav | pending | — | — | pending | after s2only result |
| NexusNav-V3 | pending | — | — | pending | after ChronoNav |

**Session history:**
- Session 1: mini_eval_final ran NavDP (7529ms), DualVLN/ChronoNav/NexusNav FAILED (depth_anything path)
- Session 2: depth_anything fix via CWD symlink applied, mini_eval_v3 container ran — partial results
- Session 3: 4-bit BnB fix applied — crashed on memory_encoder dtype in generate_traj (dual_system)
- Session 4 (3hr revisit): Switched to mode=system2 (avoids generate_traj) — running cleanly

---

## Session Handoff Protocol

When finishing a session:
1. Update **Running Containers** and **Work Queue** tables above
2. Commit `COORDINATION.md` with session ID in the commit message
3. Push to `dev` (or tell user to push if blocked)
4. Update `memory/project_state.md` with current best results
5. Write 2-line handoff note below:

### Latest Handoff Note
**From:** session_01ARwduxfwfZ4FyxfSkDxPoN (2026-10-22 — 3hr revisit)
**To next session:** S2-only eval running in `vln_ce_mini_s2only` (10 eps, mode=system2). Root fix: `memory_encoder` in dual_system mode is incompatible with 4-bit BnB (F.multi_head_attention_forward bypasses dequant hook). Fix applied in `habitat_vln_evaluator.py`: post-load cast `memory_encoder/dit_model/depth_anything` to fp16. S2-only avoids this entirely. When `vln_ce_mini_s2only` completes (check `/mnt/nvme0/vln_habitat/logs/vln_ce_mini_s2only_4bit.log`), collect SR/SPL/NE, then launch `habitat_vln_ce_mini_chrononav_cfg.py`. 9+ commits ready to push to origin/dev (user must run `! git -C /home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev push origin dev`).

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
