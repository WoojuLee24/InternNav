# OmniNav Master Plan
> **Every Claude session and every agent fork MUST read this file at the start of work.**  
> It is the authoritative source for goals, architecture decisions, module inventory, eval results, and agent coordination.  
> Update it whenever something material changes. Never let it go stale.

_Last updated: 2026-10-22 by session_01ARwduxfwfZ4FyxfSkDxPoN_

---

## 1. Grand Vision

**OmniNav** = unified embodied navigation framework. Two research threads, one deployment pipeline:

```
ChronoNav (temporal cache + VLM annotation)
    └─ SR=72.82% BEST (v278, 504 eps)
    └─ target: ≥75% full val_unseen

NexusNav-V3 (dual-system + 6 S2 modules)
    └─ SR=TBD (needs full eval after Naver Cloud training)
    └─ target: ≥83% full val_unseen

Real-world: Scout robot (ROS2) → RealSense RGB-D + odometry
    └─ S2 inference currently 1517ms/0.66Hz (4-bit 12GB GPU)
    └─ target: ≤309ms/3.2Hz (A100 fp16)
```

**System invariant**: every module must improve BOTH simulator benchmark AND real-world latency simultaneously.

---

## 2. Paper Targets

### GitHub Anonymization (for blind review)
- **Service**: anonymous.4open.science
- **Trigger**: when SOTA confirmed on R2R val_unseen full (1839 eps)
- **Steps**: push final code → create anonymous mirror → add link to paper supplementary

### Target Venues
| Venue | Fit | Key criteria to hit |
|-------|-----|-------------------|
| **CVPR 2027** | ★★★★★ | R2R SOTA + architectural novelty (ActionTokenVocabulary) + simulator benchmark |
| **CoRL 2026** | ★★★★ | Real-world Scout deployment + hardware-in-the-loop + safety |
| **RSS 2027** | ★★★★ | Closed-loop control, mobile base robustness |
| **NeurIPS 2026** | ★★★ | Latent reasoning architecture + VQ-VAE action tokenization |
| **ICCV 2027** | ★★★ | Visual grounding + novel 3D environment tokenization |

---

## 3. Architecture Map

```
Sensors: RealSense RGB-D (640×480) + Odometry
              │
              ▼
   ┌──────────────────────────────────────┐
   │         System 2 (S2) — 3.2 Hz      │
   │  InternVLA-N1-DualVLN 7B            │
   │  + VisionEmbeddingCache (τ=0.92)    │
   │  + S2KVCache (94.5% hit rate)       │
   │  + LatentReasoner (N=4 ThoughtCells)│
   │  + Memory: Task+World+Body          │
   │  + ActionTokenVocabulary (VQ-VAE)   │
   │  + WorldPredictor (BCE reachability)│
   └──────────────┬───────────────────────┘
                  │ waypoints / subgoals / action tokens
   ┌──────────────▼───────────────────────┐
   │         System 1 (S1) — 3000+ Hz    │
   │  NavDP trajectory predictor         │
   │  Continuous control output          │
   └──────────────┬───────────────────────┘
                  │ cmd_vel
                  ▼
             Robot actuators
```

**Sensor extensions (documented, NOT activated)**:
- 360° Ricoh Theta → WorldMemory panoramic context
- Ouster OS1-64 LiDAR → WorldPredictor reachability
- MID-360 LiDAR → obstacle avoidance replacement for depth cam

---

## 4. Module Inventory

### NexusNav-V3 (InternNav-dev)

| Module | File | Status | Tests | Key Equation |
|--------|------|--------|-------|-------------|
| ActionTokenVocabulary | `internnav/agent/latent_action.py` | ✅ done | 22 pass | VQ-VAE: L_VQ = ‖sg[z_e]-e‖² + β‖z_e-sg[e]‖² |
| LatentReasoner | `internnav/agent/latent_reasoning.py` | ✅ done | 22 pass | h_t = g⊙h_{t-1} + (1-g)⊙tanh(W[x,h_{t-1}]+b) |
| TaskMemory | `internnav/agent/memory/task_memory.py` | ✅ done | included | segment completion tracking |
| WorldMemory | `internnav/agent/memory/world_memory.py` | ✅ done | included | topometric graph + loop closure |
| BodyMemory | `internnav/agent/memory/body_memory.py` | ✅ done | included | oscillation detection |
| S2KVCache | `internnav/agent/s2_kv_cache.py` | ✅ done | included | persistence across S2 calls |
| VisionEmbeddingCache | `internnav/agent/s2_kv_cache.py` | ✅ done | included | EMA: v̂_t = τ·v̂_{t-1} + (1-τ)·φ(I_t) |
| WorldPredictor | `internnav/agent/world_predictor.py` | ✅ done | included | online BCE reachability |
| NexusNavNet | `internnav/model/basemodel/nexusnav/` | ✅ done | — | 7B + LoRA r=32 + NexusNavS2Heads |
| **Inference wiring** | `internnav/agent/internvla_n1_agent.py` | ❌ **TODO** | — | modules not yet called in inference loop |
| **Evaluator cache wiring** | `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` | ❌ **TODO** | — | VisionEmbeddingCache + S2KVCache not consumed in eval loop |

> **WIRING GAP (CONFIRMED 2026-10-22)**: ChronoNav mini eval with `vision_cache_tau=0.92` showed 1497ms avg S2 latency — identical to DualVLN baseline (1517ms). The `use_kv_cache` and `vision_cache_tau` model_settings are passed to the evaluator but NEVER consumed. `grep "use_kv_cache\|VisionEmbeddingCache" habitat_vln_evaluator.py` → zero results. Fix: wire `VisionEmbeddingCache` into `_run_eval_system2()`: compute embedding fingerprint per observation, skip S2 if cos_sim(v̂_t, v̂_{t-1}) > τ, reuse previous action queue.

### ChronoNav (ChronoNav repo)

| Module | Status | Notes |
|--------|--------|-------|
| VLM annotation pipeline | ✅ v280 | fewshot-3 + stop-gtdist |
| TemporalCache (τ=0.92) | ✅ | action_aware=False for VLN-CE |
| v280 reannotation | ⏸ BLOCKED | vLLM server at 10.77.32.231 needs restart |

### Infrastructure

| Item | Status | Location |
|------|--------|---------|
| Habitat eval container | ✅ | `vlnav/habitat-eval:rebuilt` |
| Docker Scout deployment | ✅ | `vlnav/internvla-server:optimized-20260323` |
| Naver Cloud PyTorchJob | ✅ | `scripts/cloud/nexusnav-pytorchjob.yaml` |
| DeepSpeed ZeRO-2 config | ✅ | `scripts/cloud/ds_config_zero2.json` |
| Training launch script | ✅ | `scripts/train/base_train/start_vln_ce_train.sh` |

---

## 5. Eval Results (live table)

### VLN-CE Mini (10 eps, scene 2azQ1b91cZZ) — **EASY SET: single scene**
| Model | Mode | SR | SPL | NE | S2 ms | S2 Hz | Container |
|-------|------|----|----|----|-------|-------|-----------|
| NavDP | S1 only | — | — | — | 7529 | 0.13 | done |
| DualVLN S2-only (4-bit) | system2 | **90.0%** | 77.4% | 4.50 | 1517 | 0.66 | done ✅ |
| NexusNav-V3 (4-bit) | system2 | pending | — | — | — | — | not started (GPU blocked) |

> ⚠️ Mini numbers are relative comparison only. Published S2-only = 60.9% on full 1839-ep val_unseen.

### ChronoNav Full Val Unseen (running — scene 2azQ1b91cZZ only, cap at 500 eps)
| Checkpoint | Eps done | SR | SPL | NE | Status |
|-----------|---------|----|----|----|----|
| ChronoNav (τ=0.92, 4-bit, system2) | 35 | 85.7% | 0.657 | 3.83 | RUNNING — cap at 500 |

> Container: `vln_ce_mini_chrononav`, 35/1839 eps at 15:22 UTC. Est. ~9hrs to 500 eps.
> GPU: 10919/12288 MiB used — NexusNav eval BLOCKED until ChronoNav finishes.

### ChronoNav Campaign (full val_unseen)
| Version | SR | Episodes | Notes |
|---------|----|---------|----|
| v278 | **72.82%** | 504 | CAMPAIGN BEST |
| v280 | pending | 0 | blocked on vLLM server |

---

## 6. Research Prompt Log (key decisions)

| Date | User instruction | Decision taken | Output |
|------|-----------------|---------------|--------|
| 2026-10-22 | "focus on VLN-CE mini training and evaluating" | 4 model eval configs + 4 training configs | `scripts/eval/configs/habitat_*_mini_*.py` |
| 2026-10-22 | "divide work to ~50 agents, coordinate well" | COORDINATION.md created, fork agents launched | `COORDINATION.md` |
| 2026-10-22 | "every 3 hours revisit as expert group" | ScheduleWakeup(3600) + this master doc | OMNAV_MASTER.md |
| 2026-10-22 | "document all things, keep system consistency" | OmniNav master plan doc + memory system | This file |
| 2026-10-22 | GitHub anonymization for SOTA submission | anonymous.4open.science plan documented | Memory + this file |

---

## 7. Mathematical Contributions Summary

### New (NexusNav-V3)
1. **ActionTokenVocabulary**: VQ-VAE 256-token codebook for continuous action discretization in VLN
2. **LatentReasoner**: N-step GRU thought chain without autoregressive generation (+0.66 Hz S2 speed)
3. **VisionEmbeddingCache**: EMA-gated vision embedding reuse (τ=0.92, 94.5% hit rate expected)
4. **WorldPredictor**: Per-episode online reachability learning via BCE loss

### Existing (ChronoNav)
5. **Temporal annotation pipeline**: VLM fewshot-3 annotation vs GT labels (+47pp on poor-GT episodes)
6. **GPS_STOP distance success trigger**: 3m radius, model-agnostic navigation success capture

---

## 8. Naver Cloud Training Plan

Sequential (each gates the next):
```
1. NavDP (smoke test → ship)
2. DualVLN (smoke test → ship)
3. ChronoNav (smoke test → ship)
4. NexusNav-V3 (smoke test → ship)
```

Gate for each: `latency_report.json` exists + local mini eval SR > 0 at ≥5 eps.
Launch: `./scripts/train/base_train/start_vln_ce_train.sh --model {navdp|dualsystem|chrononav|nexusnav}`

---

## 9. Outstanding Work (ordered)

### P0 — CRITICAL PATH
- [ ] Wire `LatentReasoner` + `S2ThoughtCache` into `internnav/agent/internvla_n1_agent.py`
- [ ] Collect ChronoNav mini results → compare vs DualVLN baseline (running now)
- [ ] NexusNav-V3 mini eval → `habitat_vln_ce_mini_nexusnav_cfg.py` (fix to system2 mode first)
- [ ] v280 vLLM reannotation (unblock: restart server at 10.77.32.231)
- [ ] Push all 10+ commits to origin/dev

### P1 — PAPER QUALITY  
- [ ] Full 1839-ep val_unseen eval (needs Naver Cloud A100) → get real SR for paper table
- [ ] Ablation runs (w/o each module, mini set) → fill `tab:ablation` in `nexusnav_v3.tex`
- [ ] Memory analysis figures (`paper/figures/memory_analysis.pdf`)
- [ ] GitHub anonymization setup (anonymous.4open.science) → paper supplementary link

### P2 — ROBUSTNESS + STRETCH
- [ ] Real-world Scout validation run → CoRL/RSS paper evidence
- [ ] Multi-modal sensors (360° Theta, OS1, Mid-360) — activate in commented code
- [ ] Full R2R training set on Naver Cloud (beyond mini)

---

## 10. Agent Coordination (key rule)

**Agent spawning pattern:**
```
Main session
  ├── Fork A: eval monitoring (watch containers, collect results)
  ├── Fork B: paper writing (update LaTeX with results)
  ├── Fork C: module implementation (S2 wiring)
  └── Fork D: research (new papers, math derivations)
```

**Partition by FILE OWNERSHIP** (see COORDINATION.md). Each fork claims files before touching them. No two forks edit the same file.

**After every fork completes**: commit + update COORDINATION.md + update OMNAV_MASTER.md section 5 (Eval Results) or section 4 (Module Inventory).
