# relabel / relabel_half 전 결과 수집 -> json (run · eval · SR/SPL/OSR/NE · 학습 step · 마지막 20 step 평균 loss)
# 사용: python3 docs/reports/collect_relabel_results.py docs/reports/relabel_ablation_results.json
import glob, json, os, re, sys
ROOT = '/home/irteam/data-vol2/checkpoints'
rows = []
for grp in ('relabel', 'relabel_half'):
    for run in sorted(glob.glob(f'{ROOT}/{grp}/*/')):
        run = run.rstrip('/'); name = os.path.basename(run)
        m = re.match(r'dualvln_mini(?:_half)?_s(\d)\.train\.(.+?)_eval\.(.+?)(?:_internvla-n1-system2)?_(\d{8}_\d{6})$', name)
        if not m:
            rows.append({'group': grp, 'run': name, 'parse': 'fail'}); continue
        stage, tl, te_run, stamp = m.groups()
        ts = os.path.join(run, 'trainer_state.json')
        loss = step = None
        if os.path.exists(ts):
            st = json.load(open(ts)); lh = [h for h in st.get('log_history', []) if 'loss' in h]
            if lh: loss = round(sum(h['loss'] for h in lh[-20:]) / len(lh[-20:]), 3); step = lh[-1].get('step')
        trained = os.path.exists(os.path.join(run, 'model.safetensors.index.json'))
        for res in sorted(glob.glob(f'{run}/logs/*/result_h200.json')):
            slug = os.path.basename(os.path.dirname(res))
            em = re.search(r'_eval\.(.+?)_internvla-n1-system2$', slug) or re.search(r'_eval\.(.+?)$', slug)
            ev = em.group(1) if em else slug
            lines = [json.loads(l) for l in open(res) if l.strip()]
            r = lines[-1]
            rows.append({'group': grp, 'stage': int(stage), 'train': tl, 'eval': ev, 'stamp': stamp, 'run': name, 'trained': trained,
                         'SR': round(100 * r['sucs_all'], 1), 'SPL': round(100 * r['spls_all'], 1), 'OSR': round(100 * r.get('oss_all', float('nan')), 1),
                         'NE': round(r.get('nes_all', float('nan')), 2), 'n': r.get('length'), 'eval_ts': r.get('timestamp'),
                         'last_loss': loss, 'step': step, 'n_result_rows': len(lines)})
        if not glob.glob(f'{run}/logs/*/result_h200.json'):
            rows.append({'group': grp, 'stage': int(stage), 'train': tl, 'eval': None, 'stamp': stamp, 'run': name, 'trained': trained, 'last_loss': loss, 'step': step})
json.dump(rows, open(sys.argv[1], 'w'), indent=1, ensure_ascii=False)
ok = [r for r in rows if r.get('SR') is not None]
print('rows', len(rows), 'with result', len(ok), '| n != 1839:', [(r['train'], r['eval'], r['n']) for r in ok if r['n'] != 1839][:10])
print('parse fail', [r['run'] for r in rows if r.get('parse') == 'fail'])
print('no result runs', [(r['group'], r['train'], r['stamp'], r['trained']) for r in rows if r.get('eval') is None and 'parse' not in r])
