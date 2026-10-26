"""GEPA 결과(best.json) 를 instruction_labeling configs/style.yaml 의 writer.<key> 로 저장한다 (annotate.py --set style.writer=<key> 로 쓰게).

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/save_writer.py --run v4-c3-gemma --key gepa-v4-c3-gemma
이미 같은 key 가 있으면 내용이 같을 때만 통과하고, 다르면 멈춘다 (덮어쓰지 않음).
"""
import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
STYLE = REPO / 'scripts/dataset_converters/instruction_labeling/configs/style.yaml'

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--run', required=True, help='logs/gepa_labeling/gepa/<run>/best.json')
ap.add_argument('--key', required=True, help='style.yaml writer.<key>')
a = ap.parse_args()
run = REPO / 'logs/gepa_labeling/gepa' / a.run
best = json.load(open(run / 'best.json'))
seed = json.load(open(run / 'seed.json'))
cands = json.load(open(run / 'candidates.json'))
have = yaml.safe_load(open(STYLE))['writer']
if a.key in have:
    assert have[a.key] == {k: best[k] for k in ('style_instruction', 'facts_header')}, f'writer.{a.key} 가 이미 있고 내용이 다르다 (덮어쓰지 않음)'
    print(f'writer.{a.key} 이미 같음')
else:
    s0, s1 = cands[0]['val_score'], max(c['val_score'] for c in cands)
    s = open(STYLE).read().rstrip('\n') + '\n'
    s += (f"  {a.key}:  # logs/gepa_labeling/gepa/{a.run}/best.json ({seed.get('_name')} 에서 출발, 사실 검사기 {seed.get('_fact_checker')}; "
          f"고르기용 경로 100: {s0:.3f} -> {s1:.3f})\n")
    for k in ('style_instruction', 'facts_header'):
        s += f'    {k}: {json.dumps(best[k], ensure_ascii=False)}\n'
    open(STYLE, 'w').write(s)
    assert yaml.safe_load(open(STYLE))['writer'][a.key] == {k: best[k] for k in ('style_instruction', 'facts_header')}
    print(f'writer.{a.key} 저장 ({s0:.3f} -> {s1:.3f})')
