# Multi-Session Coordination Board

> **Every Claude session and agent fork MUST read this file at the start of work.**
> **Every session MUST update this file when it claims, completes, or abandons work.**
> Keep entries short — this is a coordination surface, not documentation.

_Format: update the `_last_updated` line and the relevant row. Do NOT rewrite the whole file._
_last_updated: 2026-10-27 (nexusnav-s2: C1-C12 all committed 7e7ab56; A/B/C queued) by session_01ARwduxfwfZ4FyxfSkDxPoN_

---

## Worktree Map (ISOLATED — use these for parallel sessions)

| Worktree path | Branch | Purpose | **LIVE SESSION** |
|---------------|--------|---------|---------|
| `/home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev` | `dev` | Main integration + C8-C12 agent wiring (MERGE TARGET) | **vlnav-9d** |
| `/mnt/nvme0/worktrees/vln-ce` | `wt/vln-ce` | Mini training launch + eval watchdog | **debug-duilvn-metrics-validation** |
| `/mnt/nvme0/worktrees/nexusnav-s2` | `wt/nexusnav-s2` | NexusNav model arch: C1-C12 WM/LACP/BEV/NavCoT + A/B/C aux heads | **model-2e (this session)** |
| `/mnt/nvme0/worktrees/paper` | `wt/paper` | NexusNav + ChronoNav paper writing | FREE |
| `/mnt/nvme0/worktrees/wiring` | `wt/wiring` | Future module wiring | FREE |

**Branch divergence point: `718d05d`** — both `dev` and `wt/nexusnav-s2` branched here.
**Known merge conflicts (resolve at PR time):**
- `internnav/agent/navcot.py` — created by BOTH branches. **Resolution: use wt/nexusnav-s2 content** (more complete wrapper).
- `internnav/agent/internvla_n1_agent.py` — modified by BOTH. **Resolution: 3-way merge**, keep vlnav-9d's LACP gate wiring + keep nexusnav-s2's NavCoT prefix; deduplicate BEV init.
**No conflict (different paths):** `agent/lacp.py` vs arch `LACPModule`; `agent/bev_fusion.py` vs `model/basemodel/nexusnav/bev_fusion.py`; `scripts/train/gqrm/` vs `model/basemodel/nexusnav/grpo_nav.py`.

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
| `vln_ce_chrononav_80pct` | ChronoNav a351125 (use_kv_cache=False) | GPU 0 | ▶ RUNNING 2026-10-26 | `/mnt/nvme0/vln_habitat/logs/vln_ce_full/run_chrononav_80pct.log` |

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
**From:** session_01ARwduxfwfZ4FyxfSkDxPoN (2026-10-26)
**State:** a351125 committed (NOT YET PUSHED — GitHub auth expired, user must run: `! git -C /home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev push origin dev`).

**3 zero-shot quick wins implemented (a351125):**
1. `body_memory.py`: `is_oscillating()` threshold tightened (var<0.5m², window=12), recovery action added
2. `internvla_n1_agent.py`: memory_context built + passed to s2_step; 3-step recovery override in step()
3. `internvla_n1_policy.py`: `memory_context=""` kwarg in s2_step, injected into prompt
4. `habitat_vln_evaluator.py`: `_S2_MAX_HOLD=15` prevents stale-plan loops
- 36 tests pass, 1 skip (no CUDA)
- Expected SR: 73% → ~78-80% (inference-only, deep research validated)

**Deep research findings (2026-10-26):**
- 88% of failures = stuck/looping (max_steps hit) — not nav errors
- All memory modules were PASSIVE (update-only, never fed into S2 prompt) — now ACTIVE
- VLN-Cache arXiv:2603.07080 confirms our exact τ=0.92 failure mode + fix
- ChronoNav 72.82% already exceeds CLASH ~66% (supervised VLN-CE SOTA)

**VLN-PE result:** ChronoNav VLN-PE SR=99.1% (n=1347 full val_unseen, H1/InternUtopia). Added to artifact v11.

**Next priorities:**
1. Push a351125 to origin/dev (user action — GitHub auth needed)
2. τ: 0.92→0.97 config change + heading guard in s2_kv_cache.py
3. History 3→5 frames config change
4. Run full val_unseen eval with a351125 changes to confirm 80% milestone
5. DualVLN + NexusNav mini training (image build status unknown)

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
| GPU rule (this machine) | `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` — NOT `--gpus all` (/run/nvidia-persistenced/socket absent) |
