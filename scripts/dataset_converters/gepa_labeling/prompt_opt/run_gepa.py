"""3단계 실행: 손잡이 조건 프롬프트(style_instruction) 와 사실 메모 머리말(facts_header) 을 GEPA 로 고친다.

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/run_gepa.py --run v1
    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/run_gepa.py --run smoke --smoke
    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/run_gepa.py --run v2-joint --set route.facts=memo-v1 --set style.knobs=v2 --set delivery=joint3-v1
출력: logs/gepa_labeling/gepa/<run>/ — seed.json, best.json (사람이 읽는 최종 글), candidates.json (후보·검증 점수), GEPA 상태, 사례 사진 모음
데이터: train 의 scene 을 나눠 고치기(train) / 고르기(val) 경로를 뽑는다. val_seen 은 렌더 이미지가 없고, val_unseen 은 쓰지 않는다.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scorer import CFG, REPO  # noqa: E402  (instruction_labeling/label 을 sys.path 에 넣는다)

import config as il_config  # noqa: E402
from sources import items  # noqa: E402
from adapter import NavLabelAdapter, reflection_lm  # noqa: E402


def split_paths(D: dict, smoke: bool) -> tuple:
    by = collections.defaultdict(list)
    for it in items('train'):
        by[it['ep']['trajectory_id']].append(it)
    paths = [{'traj': t, 'scene': v[0]['scene'], 'items': sorted(v, key=lambda x: x['sib_idx'])} for t, v in by.items() if len(v) >= 3]
    scenes = sorted({p['scene'] for p in paths})
    rng = random.Random(D['seed'])
    rng.shuffle(scenes)
    val_sc = set(scenes[:D['val_scenes']])
    tr = [p for p in paths if p['scene'] not in val_sc]
    va = [p for p in paths if p['scene'] in val_sc]
    rng.shuffle(tr)
    rng.shuffle(va)
    n_tr, n_va = (5, 5) if smoke else (D['n_train_paths'], D['n_val_paths'])
    return tr[:n_tr], va[:n_va], sorted(val_sc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--smoke', action='store_true', help='경로 5/5, 예산 20 으로 한 번 (동작 확인용)')
    ap.add_argument('--set', action='append', default=None, help='instruction_labeling 설정 (기본: configs/gepa.yaml sets). 예: --set delivery=joint3-v1')
    a = ap.parse_args()
    from gepa import optimize
    from openai import OpenAI
    G = CFG['gepa']
    sets = a.set if a.set else CFG['sets']
    cfg = il_config.load(CFG['method'], sets)
    client = OpenAI(base_url=cfg['vllm']['base_url'], api_key=cfg['vllm']['api_key'])
    run_dir = REPO / 'logs/gepa_labeling/gepa' / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = {'style_instruction': cfg['style']['knobs']['instruction'], 'facts_header': cfg['route']['facts']['header']}
    (run_dir / 'seed.json').write_text(json.dumps(dict(seed, _sets=sets, _name=cfg['name']), ensure_ascii=False, indent=1))
    tr, va, val_scenes = split_paths(G['data'], a.smoke)
    (run_dir / 'data.json').write_text(json.dumps({'train': [p['traj'] for p in tr], 'val': [p['traj'] for p in va], 'val_scenes': val_scenes}))
    print(f'train paths {len(tr)}, val paths {len(va)} (val scenes {val_scenes})', flush=True)
    res = optimize(
        seed_candidate=seed, trainset=tr, valset=va,
        adapter=NavLabelAdapter(cfg, client, run_dir),
        reflection_lm=reflection_lm(client, cfg['vllm']['model']),
        reflection_prompt_template=G['reflection_templates'],
        candidate_selection_strategy=G['candidate_selection_strategy'],
        reflection_minibatch_size=G['reflection_minibatch_size'],
        max_metric_calls=20 if a.smoke else G['max_metric_calls'],
        run_dir=str(run_dir / 'gepa_state'), seed=G['seed'], display_progress_bar=False, raise_on_exception=False)
    best = res.best_candidate
    (run_dir / 'best.json').write_text(json.dumps(best, ensure_ascii=False, indent=1))
    cands = [{'candidate': c, 'val_score': s} for c, s in zip(res.candidates, res.val_aggregate_scores)]
    (run_dir / 'candidates.json').write_text(json.dumps(cands, ensure_ascii=False, indent=1))
    print(f'best val {max(res.val_aggregate_scores):.3f} (seed {res.val_aggregate_scores[0]:.3f}), candidates {len(cands)}')
    print(json.dumps(best, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
