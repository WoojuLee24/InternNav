"""0단계 검증: 사람이 직접 판정할 train 문장 N 개를 뽑고 (sheet), 채운 뒤 gemma 판정과의 일치도를 잰다 (agree).

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/style_types/human_sheet.py sheet
    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/style_types/human_sheet.py agree
sheet 출력: <out_dir>/<human.file> — 두 칸을 채운다.
  goal_only   : yes = 어디로 가는지만 말하고 가는 길은 없음 / no = 길을 조금이라도 설명함
  granularity : step = 회전 명령이 중심 / landmark = 지나는 물체·방이 중심 / mixed = 비슷하게 섞임
"""
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract import CFG, REPO  # noqa: E402


def kappa(a, b, cats):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    mode = sys.argv[1]
    H = CFG['human']
    out = REPO / CFG['out_dir']
    path = out / H['file']
    if mode == 'sheet':
        rows = [json.loads(l) for l in open(out / 'train_rules.jsonl')]
        pick = random.Random(H['seed']).sample(rows, H['n'])
        assert not path.exists(), f'이미 있다 (덮어쓰지 않음): {path}'
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode_id', 'text', 'goal_only', 'granularity'])
            for r in pick:
                w.writerow([r['episode_id'], r['text'], '', ''])
        print(f'{path} ({len(pick)}) — goal_only 칸에 yes / no, granularity 칸에 step / mixed / landmark')
    elif mode == 'agree':
        hum = [r for r in csv.DictReader(open(path)) if r['goal_only'].strip()]
        J = {json.loads(l)['episode_id']: json.loads(l) for l in open(out / 'train_judge.jsonl')}
        R = {json.loads(l)['episode_id']: json.loads(l) for l in open(out / 'train_rules.jsonl')}
        h = [r['goal_only'].strip().lower() for r in hum]
        for name in ['a', 'b']:
            g = [J[int(r['episode_id'])][name]['goal_only'] for r in hum]
            print(f'goal_only 사람 vs gemma {name}: 일치 {100 * sum(x == y for x, y in zip(h, g)) / len(h):.1f} %, kappa {kappa(h, g, ["yes", "no"]):.2f} (n={len(h)})')
        hg = [r for r in hum if r['granularity'].strip()]
        if hg:
            def share_bin(x):
                return 'step' if x >= 0.6 else 'landmark' if x <= 0.4 else 'mixed'
            h2 = [r['granularity'].strip().lower() for r in hg]
            g2 = [share_bin(R[int(r['episode_id'])]['step_share']) for r in hg]
            print(f'granularity 사람 vs 규칙 비율: 일치 {100 * sum(x == y for x, y in zip(h2, g2)) / len(h2):.1f} %, kappa {kappa(h2, g2, ["step", "mixed", "landmark"]):.2f} (n={len(h2)})')
    else:
        assert False, f'unreachable mode={mode!r}'


if __name__ == '__main__':
    main()
