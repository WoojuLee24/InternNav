# Eval metrics system: schema, decision metrics, label validation — result (2026-09-30)

Plan: `~/.claude/plans/1-7-fluttering-raven.md`. Previous step: `eval_raw_output_dashboard_result.md`.

## Changes (every new feature is off by default; with it off, behavior is unchanged)

| Item | Files | Content |
|---|---|---|
| 1+ raw validation | `scripts/eval_dashboard/validate_raw.py` | Checks an eval log dir: completeness, raw==progress (exact), last-frame d2g==NE, TL recompute, **aggregate == result row (bit-exact, reproducing the gather order)**, run.json meta, navigable ratio on the same floor, decision counters, plus a PNG. |
| 4 label rules | `internnav/habitat_extensions/vln/label_rules.py`, `.claude/memory/understanding_s2_label_generation.md` | pixel-goal projection (confirmed), empirical look-ahead rule, dataset-local↔habitat coordinate transform, habitat camera projection |
| 5 reference validation | `scripts/eval_dashboard/validate_reference_labels.py` → `logs/label_validation/` | train/val × 125/60cm, dataset rule / ShortestPathFollower / geodesic path vs labels. **Report: https://claude.ai/artifact/B7G88Uxeog7R2uFNzn5fhX** |
| 6 schema | `internnav/evaluator/metrics_schema.py`, guard in `calc_metrics` (habitat) and after `finalize_all_results` (Isaac), `Params.eval_metrics_schema` | Parallel recompute + exact comparison. Results go to a separate file, `schema_check_<machine>.jsonl`, so result rows and wandb keys stay as they were. |
| 7a train/val | `internnav/trainer/decision_metrics.py`, `internvla_n1_metrics_trainer.py` (entry script patches `_base.Trainer`), `Params.decision_metrics` | teacher-forced STOP P/R/F1, type_acc, turn_exact, coord_err_px/le10, per-type CE → `train/dec_*`, `eval/dec_*`, `val_metrics/step*_rank*.jsonl` |
| 7b online | `eval_recorder.py` (`_decide`), `Params.eval_decision_metrics`, dashboard frame strip | Per S2 call: exact 3 m STOP oracle, reference SPF action, reference pixel goal → `decision` on the frame, `dec_*` on progress/result rows |
| Other | `eval_recorder.py` | Several S2 texts before one move are kept, joined with ` \| ` (previously only the last one survived). The map height is the most common 0.5 m floor bin, not the median (the median landed mid-stairs). progress rows gain `rank`. |
| Front-view frames (2026-10-01) | `eval_recorder.py` (`_capture`, `_keep_frames`), `Params.eval_raw_frames/_scale/_sample`, dashboard front-view panel, `/api/frame` route, static builds (copied / data URIs) | When save_raw is on, front-view JPEGs at 320×240 (scale 0.5) under `raw/<stamp>/frames/<scene>_<ep>/<i>.jpg`. **Default policy `fail` (success == 0 = not stopped within 3 m)**. `fail+sample` also keeps a fixed 5% control group of successes; `all` and `none` are also available. Look-down (tilted) frames are excluded. `validate_raw` checks the policy against the saved files. Estimated size: about 10 KB per frame × ~60 per failed episode × ~45% failures ≈ 0.5 GB per eval. |
| Collisions (2026-10-01) | `eval_recorder.py` (per-frame `collision`, episode `collisions`, `collision_aggregate`), `metrics_schema` (CR/CFSR compared too), dashboard (CR/CFSR columns, Coll column and filter, timeline markers, strip) | Uses the habitat `Collisions` measure (forward-move collisions only). With Isaac's definitions, **CR** = total collisions / total steps and **CFSR** = share of episodes that succeed with 0 collisions; these appear as `crs_all` / `cfsrs_all` in the result row. `validate_raw` checks collisions == the number of collision frames. |
| Verification config | `baseline_ablation/dualvln_stage1_full_gemm_check.py` | GEMM stage1 plus the 3 new flags turned on (smoke tests only; logs go to a separate slug) |

## Label validation (61 scenes × 2 episodes)
- **STOP: accurate.** Every dataset STOP is within 3 m of the goal; SPF agrees on 97–100% of them.
- **pixel-goal rule: reference.** Where a label exists, the rule also finds a goal 95–99% of the time; the same k 55–69%; within 10 px 68–77%.
- **SPF action / geodesic pixel: weak reference.** SPF agrees on GT forward only 49–51%; the geodesic pixel is within 30 px 40–50% of the time.
- Therefore the online metrics use key names that mark them: STOP (`dec_stop_precision/recall`) is exact, and anything with `_ref` in the key is reference only.

## Verification
- Unit tests (`pytest tests/unit_test/`): 4 passed.
  - schema bit-exact parity with the real `calc_metrics` and the real `ResultLogger`, including NaN/inf/duplicate rows and 3-rank lmdb, plus mismatch detection.
  - `DecisionMetricsTrainer` on CPU: `train/dec_*`, `eval_dec_*`, and a val JSONL with `sample_idx`.
- Self-checks pass: `eval_recorder`, `label_rules` (habitat projection == dataset projection), `decision_metrics` (plus spans checked with the real Qwen tokenizer), `data`.
- GPU-free end-to-end run (`check_recorder.py` + `validate_raw.py`): ALL PASS, including the bit-exact aggregate, same-floor navigability 1.000, and decision counters == raw records.
- Default path: `--print-train-argv` md5 unchanged (`f9013cfa…`); with the flags off, the trainer and the result rows are unchanged.

## Commands for the user (GPU needed; run when a node is free)

1. Real-model raw + schema + decision smoke test (3 episodes):
```
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline_ablation/dualvln_stage1_full_gemm_check.py --machine h200 --no-train --model-path /home/irteam/data-vol2/checkpoints/baseline_ablation/dualvln_stage1_full_gemm_20260825_041953/checkpoint-40000 --max-steps 3
```
   - `--max-steps 3` evaluates 3 episodes with no wandb. Check the results with:
```
python3 scripts/eval_dashboard/validate_raw.py /home/irteam/data-vol2/checkpoints/baseline_ablation/dualvln_stage1_full_gemm_20260825_041953/checkpoint-40000/logs/baseline_ablation_dualvln_stage1_full_gemm_check --expected 3
```
   - Also confirm that `schema_check_h200.jsonl` shows `"status": "pass"`, and that `decision` records carry `ref_pixel` values. This is the first test with a real camera and depth.
2. Dashboard: `python scripts/eval_dashboard/app.py --root /home/irteam/data-vol2/checkpoints --port 8088` (workstation)
3. wandb ckpt curve: `python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline_ablation/dualvln_stage1_full_gemm.py --machine h200 --no-train --model-path /home/irteam/data-vol2/checkpoints/baseline_ablation/dualvln_stage1_full_gemm_20260825_041953 --eval-ckpts 35000` → wandb run `lin686gd`: plot `test/*` against `test/ckpt_step`.
4. 7a training smoke test (20 steps, prints `train/dec_*`): `python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline_ablation/dualvln_stage1_full_gemm_check.py --machine h200 --no-eval --max-steps 20`

## Not verified
- Real-model eval, including `_ref_pixel` with a real camera and depth (the harness has no sensors, so `ref_pixel` is None there).
- The GPU run of 7a (only the CPU test with a toy model was done).
- The Isaac schema on a real h1 eval (tested with a synthetic lmdb only).
- In dual (stage2), STOP/turn samples are absent, so `train/dec_*` only has pixel/coord metrics.
- The eval `dec_*` counters may include duplicates from accelerate's padding (up to world×batch). The JSONL is exact.
