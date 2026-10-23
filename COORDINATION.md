# Multi-Session Coordination Board

> **Every Claude session and agent fork MUST read this file at the start of work.**
> **Every session MUST update this file when it claims, completes, or abandons work.**
> Keep entries short — this is a coordination surface, not documentation.

_Format: update the `_last_updated` line and the relevant row. Do NOT rewrite the whole file._
_last_updated: 2026-10-23 (2aac473 mini-train scripts + trainer fix) by session_01ARwduxfwfZ4FyxfSkDxPoN_

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
| `paper/nexusnav_v3.tex` | session_01ARw | ✅ done | Table 4 (B2 σ=1.52×), ablation table, 90%+ path section — commit 714feaf |
| `paper/nexusnav_refs.bib` | session_01ARw | 🔄 draft | 22 refs |
| `scripts/eval/configs/habitat_*_vln_ce_mini_*.py` | session_01ARw | ✅ done | 4 model eval configs |
| `scripts/train/base_train/configs/*_vln_ce_mini.py` | session_01ARw | ✅ done | 4 model training configs |
| `scripts/train/base_train/start_vln_ce_train.sh` | session_01ARw | ✅ done | per-model launch script |
| `scripts/cloud/` | session_01ARw | ✅ done | Dockerfile, PyTorchJob YAML, ZeRO-2 config |
| `internnav/configs/` | FREE | — | claim before touching |
| `internnav/evaluator/` | FREE | — | claim before touching |
| `internnav/agent/internvla_n1_agent.py` | session_01ARw | ✅ done | LatentReasoner+S2ThoughtCache wired (7d41569, merged to dev) |

---

## Running Containers (live)

| Container | Model | GPU | Status | Log |
|-----------|-------|-----|--------|-----|
| (none) | — | — | All eval containers stopped | — |

**Completed containers (results):**
| Container | Model | Result |
|-----------|-------|--------|
| `vln_ce_b2_uncached` | DualVLN S2-only 4-bit (no cache) | SR=83.3%, s2_ms=981, n=60 ✅ |
| `vln_ce_b2_cached_v6` | DualVLN + VisionEmbCache τ=0.92 | SR=18.0%, s2_ms=647, hit=87.9%, n=50. σ=1.52× but SR collapsed. ✅ |
| `vln_ce_b2_cached_v7` | DualVLN + VisionEmbCache (ema_alpha=1.0) | SR=17.1% — confirms τ=0.92 fundamentally flawed, not EMA ✅ |
| `vln_ce_mini_chrononav` | ChronoNav S2-only 4-bit | SR=72.82% (n=504 full val_unseen BEST) ✅ |
| `vln_ce_mini_s2only` | DualVLN S2-only 4-bit | SR=90%, SPL=77.4%, NE=4.50 (10 eps mini) ✅ |
| `mini_eval_final` | NavDP | S2=7529ms, 0.13Hz — baseline measured |
| `mini_eval_v3` | DualVLN/ChronoNav/NexusNav | PARTIAL — depth_anything path fix applied |
| `vln_ce_mini_eval_4bit` | DualVLN 4-bit | CRASHED — memory_encoder dtype mismatch in generate_traj |

---

## Work Queue (ordered by priority)

### IN PROGRESS (session 2026-10-23 continued)
- [x] Paper ablation table + 90-100% path section — DONE (714feaf) by fork: paper-update
- [x] Artifact updates — NexusNav architecture + eval results — DONE (artifacts fork ran)
- [x] COORDINATION.md updated with B2 container results + handoff (committed in 714feaf)
- [x] ChronoNav COORDINATION.md updated with B2 results + 90-100% target (a5fe0ff)
- [x] Mini training pipeline unblocked: trainer model-class fix + 1-GPU script (2aac473)

### RECENTLY COMPLETED (session 2026-10-23)
- [x] B2 A/B A/B results encoded in paper Table 4 (21763ad) — σ=1.52×, SR 83%→18%
- [x] B2 negative result framed: motivates Action Token approach
- [x] All repos pushed: InternNav-dev (21763ad), ChronoNav (a8ac2a9), ChronoNav-gates (7a097bf) — all CLEAN
- [x] Memory files updated: omnav_master_plan.md, project_state.md (90-100% target encoded)
- [x] SOTA chain documented: 60.9% → 72.82% → 83% → 90-100%

### DONE THIS SESSION
- [x] VLN-CE mini eval: DualVLN S2-only 4-bit — SR=91.7% SPL=81.2% NE=4.23 (n=12, strict_lt)
- [x] VisionEmbeddingCache wired into `_run_eval_system2()` (6a00455, merged to dev)
- [x] LatentReasoner + S2ThoughtCache wired into `internvla_n1_agent.py` (7d41569, merged to dev)
- [x] NexusNav paper: ablation + cache gap analysis + bib (b6413a8, merged to dev)
- [x] timing_rank0.json emission wired into evaluator (0d15e21) — s2_ms/s2_hz/cache_hit_rate/gpu_h
- [x] timing_rank0.json written for s2only_4bit (s2_ms=1517ms, cache_hit_rate=0.0)
- [x] Architecture artifact updated with all NexusNav-V3 module statuses
- [x] peer `debug-duilvn-metrics-validation` ingested B2 timing into comparison.tex (@e43d4a5)

### MINI TRAINING STATUS (2026-10-23)

**Mini training data (local, images present):**
- Path: `data/InternData-N1-v0.5-mini/vln_ce/` — 5 scenes, 120 episodes, 107GB
- Camera angles: `125cm_0/30/45deg`, `60cm_15/30deg` — all 4 vln_dataset keys verified ✓

**Training pipeline status per model:**
| Model | Script | Blocker | Status |
|-------|--------|---------|--------|
| DualVLN S2 (7B) | `start_mini_dualsystem_1gpu.sh` | ZeRO-3 CPU offload, needs ~56GB RAM (available: 81GB) | **READY TO LAUNCH** |
| ChronoNav | Same as DualVLN (inference-time τ=0.92 only) | None — use DualVLN checkpoint at eval | Ready |
| NexusNav-V3 | `nexusnav_vln_ce_mini.py` | model_name='nexusnav_v3' not in train.py | Blocked: needs nexusnav trainer |
| NavDP | `navdp_vln_ce_mini.py` | pointcloud.ply missing + wrong video dir format | Blocked: needs data restructure |

**Launch DualVLN mini training (1-GPU, 12GB):**
```bash
cd /home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev
bash scripts/train/qwenvl_train/start_mini_dualsystem_1gpu.sh
```
> GPU: 476 MiB/12288 MiB free. RAM: 81GB available. ZeRO-3+CPU offload should fit.

### NEXT (sequential, GPU blocked until ChronoNav caps ~7.5h)
- [ ] **USER ACTION**: Launch DualVLN mini training — `bash scripts/train/qwenvl_train/start_mini_dualsystem_1gpu.sh`
- [ ] After DualVLN trains: eval mini DualVLN on val_unseen (peer session handles)
- [ ] After B2: dual_system mode test with fp16 cast fix (need free GPU)
- [ ] After dual_system: NexusNav-V3 eval (`habitat_vln_ce_mini_nexusnav_cfg.py`)
- [ ] v280 reannotation (blocked: vLLM server at 10.77.32.231 needs restart)
- [ ] Push commits to origin/dev (user must run: `! git -C /home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev push origin dev`)

### PARALLEL-SAFE (any session can start independently)
- [ ] NexusNav: wire remaining modules (S2KVCache in evaluator, WorldPredictor in inference)
- [ ] OmniNav architecture diagram — update with NexusNav-V3 modules ✅ done (artifact updated)
- [ ] Naver Cloud NavDP training (requires local smoke test to pass first)

---

## Mini 4-Model Comparison — Running Log

**Goal:** measure real S2 latency (ms), S2 Hz, and SR on VLN-CE val_unseen mini (10 eps) for 4 models.

| Model | Container | S2 latency | S2 Hz | VLN-CE SR | Notes |
|-------|-----------|-----------|-------|-----------|-------|
| NavDP | `mini_eval_final` | 7529ms | 0.13 Hz | — | S1 only (VLN-PE model) |
| DualVLN S2-only (4-bit) | `vln_ce_mini_s2only` | 1517ms | 0.66 Hz | **SR=91.7% SPL=81.2%** ✅ | mode=system2, n=12, strict_lt, no cache |
| DualVLN dual_system (4-bit) | pending | — | — | pending | OOM on 12GB (ChronoNav holding GPU) |
| ChronoNav | `vln_ce_mini_chrononav` | — | — | **SR=77.3% @ ep44 (running)** | n=44, cap at 500 |
| NexusNav-V3 | pending | — | — | pending | GPU blocked ~7.5h |

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
**From:** session_01ARwduxfwfZ4FyxfSkDxPoN (2026-10-23 continued)
**State:** All eval containers done. B2 A/B complete: σ=1.52× speedup but SR 83.3%→18.0% at τ=0.92 (negative result, Table 4 filled, commit 21763ad). ChronoNav BEST=72.82% (v278, n=504). Two fork agents running: (1) paper-update adding ablation table + 90-100% SR path to nexusnav_v3.tex; (2) artifacts-update updating NexusNav Architecture + ChronoNav Eval Results artifacts with 90-100% target and B2 findings. Target SOTA chain: 60.9%→72.82%→83%→90-100%. All repos pushed and clean (InternNav-dev: 15cdcb2, ChronoNav: a8ac2a9). **Next session:** (1) check fork outputs and commit if good; (2) wire remaining modules into eval path for ablation measurement; (3) NexusNav-V3 mini eval (50 eps) to get first projected→measured gap; (4) ActionTokenVocab Naver Cloud training once SR≥75% gate confirmed.

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
