"""dev subset 고정: val_unseen 에서 trajectory 100 개 (scene 비율 유지, seed 0). 빠른 반복용.

    /usr/bin/python scripts/dataset_converters/instruction_labeling/eval/make_dev_subset.py
"""
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
N, SEED = 100, 0
raw = json.load(gzip.open(REPO / 'data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz'))['episodes']
by_scene = defaultdict(set)
for e in raw:
    by_scene[Path(e['scene_id']).stem].add(e['trajectory_id'])
total = sum(len(v) for v in by_scene.values())
rng = random.Random(SEED)
picked = []
for s in sorted(by_scene):
    k = max(1, round(N * len(by_scene[s]) / total))
    picked += rng.sample(sorted(by_scene[s]), min(k, len(by_scene[s])))
picked = sorted(picked)
out = {'split': 'val_unseen', 'n_trajectories': len(picked), 'seed': SEED,
       'n_episodes': sum(1 for e in raw if e['trajectory_id'] in set(picked)), 'trajectory_ids': picked}
(Path(__file__).parent / 'dev_subset.json').write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != 'trajectory_ids'})
