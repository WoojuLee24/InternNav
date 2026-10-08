# InternData-N1 VLN-CE S2 label generation (2026-09-30)

Report: https://claude.ai/artifact/B7G88Uxeog7R2uFNzn5fhX · code: `internnav/habitat_extensions/vln/label_rules.py`, `scripts/eval_dashboard/validate_reference_labels.py` · raw results: `logs/label_validation/`.

## Label types (`internvla_n1_lerobot_dataset.py`)

The first assistant token decides the label type:

| First token | Meaning | Follow-up |
|---|---|---|
| `↓` | pixel goal | a look-down image turn, then the answer `"x y"` (x = column, y = row, 640×480) |
| `STOP` | stop | appears **only at the last frame of an episode**, oversampled ×5 |
| `↑←→` | turn | used when there is no pixel goal and the next action is not forward |

- With `pixel_goal_only=True` (stage 2 / dual) there are no STOP or turn samples, so dual training cannot measure STOP P/R. Stage 1 (`pixel_goal_only=False`) can.
- Actions: forward is 0.25 m, a turn is 15°. The loader shifts actions (`actions[1:] + [0]`), so `actions[t]` is the action taken after frame t.

## Pixel goal
- **Projection (confirmed):** the goal is the floor point under the camera at frame t+k+1, projected into frame t's look-down camera.
  - Camera: fx = fy = 388.2 at 125 cm (about 79° FOV), 465.5 at 60 cm (about 69°); c = ((W−1)/2, (H−1)/2).
  - Label reproduction error has a median of 0.35 px; 92–95% of labels are within 2 px.
- **Choice of k (not identified):** the paper describes it as "the farthest visible point". The empirical rule is in `label_rules.EMPIRICAL_RULE`: planar distance ≤ 3.25 m, image margin 40/48/30 px, depth margin ≥ 5 cm, stop at the first failure, k ≥ 3.
  - Frame agreement (label present or absent, plus the same k) is 55–69%.
  - Where a label exists, the rule also finds a goal 95–99% of the time.
  - Pixel error is within 10 px for 68–77% of frames.
  - When k differs, the rule almost always picks a point farther than the label. So the original visibility test is stricter (probably a mesh raycast).

## Coordinates
- Dataset pose: camera-to-world, OpenCV camera, world z-up, **episode-local frame** (starts at the origin facing +x).
- habitat = `start_position + R(start_rotation[x,y,z,w]) @ [-y, z, -x]`. The start matches exactly and the end lands 0.02–0.26 m from the goal. Heading = atan2 of the camera forward vector; the conversion error is 0°.

## Validation results (61 scenes × 2 episodes, trainer split with val = last 10% of scenes)
- **STOP:** every dataset STOP is within 3 m of the goal (median 0.05–0.11 m). ShortestPathFollower returns STOP at 97–100% of GT STOP frames. **The online 3 m oracle STOP metric is accurate.**
- **ShortestPathFollower vs GT action:** 60–64% agreement overall. Forward agrees only 49–51%: SPF turns toward the final goal, while the GT route follows waypoints. **Treat it as a reference only.**
- **Geodesic-path pixel goal vs label:** 40–50% within 30 px (median 30–40 px). **A weak reference only.**

## Side findings (not fixed)
- system2 mode back-projection hardcodes a 30° pitch (`habitat_vln_evaluator.py` around line 1010). It is correct with the ld30 yaml, which is 2×15°, but wrong with ld60.
- The visualization draws `pixel_goal=[row,col]` as `(x,y)`, so the dot appears transposed.

## Rule v2 (2026-10-05) — `label_rules.reference_pixel_goal_v2` (v1 unchanged)

Facts found from the label points themselves:
- Path length is ≤ 3.25 m; float32 accumulation gives 3.25005…, so the check uses a 0.01 tolerance.
- Every label point is unoccluded at its own pixel (depth ≥ point distance).
- When a turn in place repeats a position, the label takes the first frame at that position.

Rule: the farthest point satisfying these conditions, with no contiguity stop.

Held-out results (61 scenes × episodes 3–4), pixel within 10 px, v1 → v2:

| Camera | train | val |
|---|---|---|
| 125cm | 66.8 → 75.2% | 63.5 → 70.5% |
| 60cm | 75.1 → 79.3% | 72.2 → 76.9% |

Frame match improves by +8.9 / +6.2 pt at 125cm and is roughly flat at 60cm (+0.9 / −0.4).

Tried without improvement:
- navmesh straight-line traversability (27% of label points fail it)
- 7×7 patch depth
- floor-segment visibility
- body-height (0.5 m) visibility
- stopping right before a turn (much worse)

Results: `logs/label_validation_v2/`, `logs/label_validation_v2_heldout/`; the report (artifact B7G88Uxeog7R2uFNzn5fhX) was updated.

## Rule v3 (2026-10-05) — learned candidate selector (`label_rules.reference_pixel_goal_v3`, v1/v2 unchanged)

How it works:
- Candidates are the frames satisfying the v2 conditions.
- A gradient-boosted classifier picks among them; if the score is below the threshold, the result is "no pixel goal".
- Fit with `scripts/eval_dashboard/fit_label_rule_v3.py`; models are saved at `logs/label_validation_v3/v3_model_<setting>.pkl`.
- Offline only: it needs the GT trajectory, actions and depth.

Held-out results (episodes 3–4), frame match / pixel ≤10 px, v2 → v3:

| Camera | train | val |
|---|---|---|
| 125cm | 62.7/75.2 → **88.3/88.2%** | 61.3/70.5 → **85.9/85.1%** |
| 60cm | 65.5/79.3 → **82.5/86.0%** | 63.1/76.9 → **80.0/82.7%** |

Important features, in order:
1. farthest-first rank
2. depth margin (how clearly the point is visible)
3. angle between the heading at t and the candidate direction
4. path length
5. action right after the candidate (frames right before a turn are avoided)
6. position within the turn-in-place duplicate group

So "the farthest visible point" is the base principle, and visibility clarity, view direction and avoiding pre-turn frames act together on top of it.

**3D mesh ray:** the scene .glb was ray-cast with open3d/Embree (z-up → y-up rotation; median error against the depth PNG 0–1 mm). Agent-to-agent line of sight, camera to point, and footprint-ring visibility were all tested and none helped: +0.2 pt as a rule, 88.2% vs 88.0% as v3 features. The dataset depth is rendered from the same mesh, so it carries the same information.

Results: `logs/label_validation_v3/`, `logs/label_validation_v3_heldout/`; report artifact B7G88Uxeog7R2uFNzn5fhX v3.
