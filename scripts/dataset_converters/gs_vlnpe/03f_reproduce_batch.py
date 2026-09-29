"""61씬에 `03 --mode reproduce`를 돌려 **"같은 루트 비율"**을 잰다.

무엇을 정하려고 재나
--------------------
`reproduce`는 GT의 start/goal을 그대로 쓰고 **경로만 우리 플래너가 새로 계획**한다. 우리 경로가
GT와 같은 루트를 타면 **GT 지시문이 그대로 유효**하므로 라벨링(vln-annotator)이 필요 없다.
다른 루트를 타면 "화장실 지나서"가 안 맞으므로 그 에피소드는 버리거나 다시 라벨링해야 한다.

그래서 증강 데이터를 만들기 전에 이 비율을 먼저 알아야 한다. 2씬 표본에서는
씬1 19/20 · 씬2 9/14로 씬마다 크게 달랐다 — 전수로 재야 판단이 선다.

판정 기준은 `03_sample_gt_paths.py:367`이 이미 갖고 있다:
`same_route = (Fréchet 거리 < 0.8 m)` — 문 너비보다 가까우면 같은 길로 본다.

단계와 의존
-----------
```
01_prepare_scene   -> logs/scene_meta/<scene>.json    (floor_z 등 씬 기하)
02_build_freemap   -> logs/esdf/<scene>_vlnpe.npz     (occupancy + ESDF)
03_sample_gt_paths -> logs/paths/<scene>_vlnpe.json   (경로 + GT 대조)   <- 여기서 지표를 읽는다
```
전부 CPU다 — Isaac Sim을 띄우지 않는다.

**01은 vln_n1 GT의 `camera_extrinsic`을 읽는다.** vln_pe parquet에는 그 컬럼이 없어서,
vln_n1 궤적이 없는 씬은 01을 돌릴 수 없다(실측: vln_pe 61씬 중 15씬). 그 씬들은
`--require_n1` 기본값에서 자동으로 건너뛰고 리포트에 목록으로 남는다.

커맨드
------
전수(vln_n1 GT가 있는 씬):
timeout --signal=KILL 21600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03f_reproduce_batch.py --dataset vln_pe --n_scenes 0 --log_dir logs/gs-vlnpe

몇 씬만 먼저:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03f_reproduce_batch.py --dataset vln_pe --scenes 17DRP5sb8fy,1LXtFkjw3qL --log_dir logs/gs-vlnpe
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCRIPT_NAME = '03f_reproduce_batch'
ISAAC_PY = '/workspace/isaaclab/_isaac_sim/python.sh'
S01, S02, S03 = (HERE / '01_prepare_scene.py', HERE / '02_build_freemap_esdf.py',
                 HERE / '03_sample_gt_paths.py')
# reproduce는 GT 개수로 잘리므로 큰 값을 준다 (0은 "0개"라는 뜻이다 — 실측으로 확인)
ALL_EPISODES = 999


def out_paths(work_dir: Path, scene: str, tag: str) -> dict:
    return {
        'scene_meta': work_dir / 'scene_meta' / f'{scene}.json',
        'esdf': work_dir / 'esdf' / f'{scene}{tag}.npz',
        'paths': work_dir / 'paths' / f'{scene}{tag}.json',
    }


def run(cmd: list, log_path: Path, timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a') as fh:
        fh.write('\n$ ' + ' '.join(cmd) + '\n')
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode


def process_scene(scene: str, args, work_dir: Path, tag: str, log_root: Path) -> dict:
    """01 -> 02 -> 03 을 필요한 것만 돌리고 03의 GT 대조 지표를 반환한다.

    각 단계의 성공은 **종료코드가 아니라 산출 파일 존재**로 본다 — 03은 게이트를 못 넘기면
    비영 종료코드를 내지만(같은 루트가 전부가 아니면 FAIL), 우리가 원하는 지표는 그때도 나온다.
    """
    P = out_paths(work_dir, scene, tag)
    log_path = log_root / f'{scene}.log'
    steps = []

    if not P['scene_meta'].is_file():
        run([ISAAC_PY, str(S01), '--scene', scene,
             '--out_dir', str(work_dir), '--log_dir', args.log_dir], log_path, args.step_timeout)
        steps.append('01')
        if not P['scene_meta'].is_file():
            return {'scene': scene, 'error': '01 실패 — scene_meta 없음', 'log': str(log_path),
                    'steps': steps}

    if not P['esdf'].is_file():
        run([ISAAC_PY, str(S02), '--dataset', args.dataset, '--scene', scene,
             '--r_b', str(args.r_b), '--h_nav_ratio', str(args.h_nav_ratio),
             '--out_dir', str(work_dir), '--scene_meta_dir', str(work_dir),
             '--log_dir', args.log_dir], log_path, args.step_timeout)
        steps.append('02')
        if not P['esdf'].is_file():
            return {'scene': scene, 'error': '02 실패 — esdf 없음', 'log': str(log_path),
                    'steps': steps}

    if args.skip_existing and P['paths'].is_file():
        steps.append('03(재사용)')
    else:
        cmd03 = [ISAAC_PY, str(S03), '--dataset', args.dataset, '--mode', 'reproduce',
                 '--scene', scene, '--num_episodes', str(ALL_EPISODES),
                 '--out_dir', str(work_dir), '--log_dir', args.log_dir]
        # 프로파일을 먼저 놓고, 개별 인자로 덮는다 (03의 apply_path_profile이 명시값을 존중한다)
        if args.path_profile:
            cmd03 += ['--path_profile', args.path_profile]
        cmd03 += ['--r_b', str(args.r_b), '--h_nav_ratio', str(args.h_nav_ratio)]
        if args.refine_radius is not None:
            cmd03 += ['--refine_radius', str(args.refine_radius)]
        run(cmd03 + (args.extra_03.split() if args.extra_03 else []), log_path, args.step_timeout)
        steps.append('03')
        if not P['paths'].is_file():
            return {'scene': scene, 'error': '03 실패 — paths json 없음', 'log': str(log_path),
                    'steps': steps}

    d = json.loads(P['paths'].read_text())
    st = d.get('stats', {})
    eps = d.get('episodes', [])
    fr = [e['gt_compare']['frechet_m'] for e in eps
          if e.get('gt_compare', {}).get('frechet_m') is not None]
    ch = [e['gt_compare']['chamfer_m'] for e in eps
          if e.get('gt_compare', {}).get('chamfer_m') is not None]
    n_ok = int(st.get('n_ok', len(fr)))
    n_same = int(st.get('n_same_route', 0))
    return {
        'scene': scene, 'steps': steps,
        'n_gt': int(st.get('n_total', len(eps))),
        'n_ok': n_ok, 'n_same_route': n_same,
        'same_route_frac': (n_same / n_ok) if n_ok else None,
        'frechet_median': float(np.median(fr)) if fr else None,
        'chamfer_median': float(np.median(ch)) if ch else None,
        'frechet': fr,
    }


def build_report(rows: list, skipped: list, args, out_dir: Path) -> Path:
    import viz_utils

    ok = [r for r in rows if 'error' not in r]
    bad = [r for r in rows if 'error' in r]
    tot_ok = sum(r['n_ok'] for r in ok)
    tot_same = sum(r['n_same_route'] for r in ok)
    tot_gt = sum(r['n_gt'] for r in ok)
    fracs = [r['same_route_frac'] for r in ok if r['same_route_frac'] is not None]
    all_fr = [x for r in ok for x in r['frechet']]

    def pill(frac):
        if frac is None:
            return '<span class="pill warn">—</span>'
        cls = 'good' if frac >= 0.9 else ('warn' if frac >= 0.7 else 'bad')
        return f'<span class="pill {cls}">{frac * 100:.0f}%</span>'

    rows_html = ''.join(
        f'<tr><td>{r["scene"]}</td><td>{r["n_gt"]}</td><td>{r["n_ok"]}</td>'
        f'<td>{r["n_same_route"]}</td><td>{pill(r["same_route_frac"])}</td>'
        f'<td>{r["frechet_median"]:.3f}</td><td>{r["chamfer_median"]:.3f}</td></tr>'
        for r in sorted(ok, key=lambda x: (x['same_route_frac'] is None, x['same_route_frac'] or 0)))
    table = ('<table><thead><tr><th>씬</th><th>GT 에피소드</th><th>계획 성공</th>'
             '<th>같은 루트</th><th>비율</th><th>Fréchet 중앙값(m)</th><th>chamfer 중앙값(m)</th>'
             '</tr></thead><tbody>' + rows_html + '</tbody></table>')

    summary = (
        f'<p><b>목적</b>: GT의 start/goal을 그대로 쓰고 경로만 우리 플래너로 다시 계획했을 때, '
        f'몇 %가 <b>GT와 같은 루트</b>를 타는지 잰다. 같은 루트면 GT 지시문을 그대로 쓸 수 있어 '
        f'라벨링이 필요 없다. 판정은 <code>Fréchet &lt; 0.8 m</code>(문 너비)다.</p>'
        f'<p><b>전체 {tot_same}/{tot_ok} 에피소드가 같은 루트 '
        f'({tot_same / tot_ok * 100:.1f}%)</b> · 씬 {len(ok)}개 · '
        f'계획 성공 {tot_ok}/{tot_gt} ({tot_ok / tot_gt * 100:.1f}%)</p>'
        f'<p>씬별 비율 중앙값 {np.median(fracs) * 100:.1f}% · 최악 씬 {min(fracs) * 100:.0f}% · '
        f'최선 씬 {max(fracs) * 100:.0f}% · 90% 이상인 씬 '
        f'{sum(1 for f in fracs if f >= 0.9)}/{len(fracs)}</p>'
        f'<p>Fréchet 전체 중앙값 {np.median(all_fr):.3f} m · p90 {np.percentile(all_fr, 90):.3f} m</p>'
        + table)

    body = ''
    if bad:
        body += ('<h2>실패한 씬</h2><ul>' + ''.join(
            f'<li><b>{r["scene"]}</b> — {r["error"]} (단계 {"→".join(r["steps"])}, 로그 '
            f'<code>{r["log"]}</code>)</li>' for r in bad) + '</ul>')
    if skipped:
        body += (f'<h2>건너뛴 씬 {len(skipped)}개 — vln_n1 GT 없음</h2>'
                 f'<p><code>01_prepare_scene.py</code>가 vln_n1 parquet의 '
                 f'<code>observation.camera_extrinsic</code>을 읽어 <code>floor_z</code>를 구한다. '
                 f'vln_pe parquet에는 그 컬럼이 없어서 vln_n1 궤적이 없는 씬은 01을 돌릴 수 없다.</p>'
                 f'<p>{", ".join(skipped)}</p>')
    return viz_utils.save_gallery(out_dir, 'report.html',
                                  f'{SCRIPT_NAME} — reproduce 같은 루트 비율 ({len(ok)}씬)',
                                  summary, body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_pe')
    ap.add_argument('--scenes', default=None, help='콤마로 지정. 생략 시 --n_scenes')
    ap.add_argument('--n_scenes', type=int, default=0, help='0이면 전수')
    ap.add_argument('--seed', type=int, default=20260909)
    ap.add_argument('--r_b', type=float, default=0.20)
    ap.add_argument('--h_nav_ratio', type=float, default=0.12)
    ap.add_argument('--path_profile', default='reproduce_v1',
                    help='경로 계획 프로파일 (path_profiles.py). 개별 인자를 주면 그게 이긴다')
    ap.add_argument('--refine_radius', type=float, default=None,
                    help='생략 시 프로파일 값')
    ap.add_argument('--require_n1', action='store_true', default=True,
                    help='vln_n1 GT가 없는 씬은 건너뛴다(01이 그 데이터를 읽으므로 기본 켬)')
    ap.add_argument('--no_require_n1', dest='require_n1', action='store_false')
    ap.add_argument('--work_dir', default='scripts/dataset_converters/gs_vlnpe/logs',
                    help='scene_meta/esdf/paths 가 쌓이는 곳 (기존 2씬 결과와 같은 위치)')
    ap.add_argument('--extra_03', default=None,
                    help='03에 그대로 넘길 추가 인자 (예: "--astar_cell_m 0.15 --smooth_step 0.02"). '
                         '**--r_b / --h_nav_ratio 는 여기로 주지 말 것** — 그 값은 02가 만든 '
                         'ESDF에 박혀 있어서, 기존 esdf를 재사용하면 옛 지도로 계획하게 된다.')
    ap.add_argument('--step_timeout', type=int, default=3600)
    ap.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                    help='이미 있는 paths json도 다시 계획한다')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    args = ap.parse_args()

    import dataset_utils

    tag = dataset_utils.dataset_tag(args.dataset)
    gt_root = Path(dataset_utils.default_data_root(args.dataset))
    n1_root = Path(dataset_utils.default_data_root('vln_n1'))
    work_dir = Path(args.work_dir)

    all_scenes = sorted(p.name for p in gt_root.glob('*') if (p / 'data' / 'chunk-000').is_dir())
    scenes = ([s for s in args.scenes.split(',') if s] if args.scenes else all_scenes)
    if not args.scenes and args.n_scenes > 0:
        import random

        scenes = sorted(random.Random(args.seed).sample(scenes, min(args.n_scenes, len(scenes))))

    skipped = []
    if args.require_n1:
        keep = []
        for s in scenes:
            if (n1_root / s / 'data' / 'chunk-000').is_dir():
                keep.append(s)
            else:
                skipped.append(s)
        scenes = keep

    print(f'[{SCRIPT_NAME}] dataset={args.dataset} 대상 {len(scenes)}씬 '
          f'(vln_n1 GT 없어 건너뜀 {len(skipped)}씬) · r_b={args.r_b} '
          f'refine={args.refine_radius}', flush=True)
    if skipped:
        print(f'[{SCRIPT_NAME}] 건너뜀: {", ".join(skipped)}', flush=True)

    log_root = Path(args.log_dir) / SCRIPT_NAME / 'logs'
    rows, t0 = [], time.time()
    for i, scene in enumerate(scenes, 1):
        ts = time.time()
        r = process_scene(scene, args, work_dir, tag, log_root)
        rows.append(r)
        if 'error' in r:
            print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  {r["error"]}  '
                  f'({time.time() - ts:.0f}s)', flush=True)
        else:
            frac = r['same_route_frac']
            print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  '
                  f'같은루트 {r["n_same_route"]}/{r["n_ok"]} '
                  f'({frac * 100:.0f}%)  GT {r["n_gt"]}ep  '
                  f'Fréchet {r["frechet_median"]:.3f}m  ({time.time() - ts:.0f}s)', flush=True)

    ok = [r for r in rows if 'error' not in r]
    if not ok:
        print(f'[{SCRIPT_NAME}] 성공한 씬이 없다 — {log_root} 의 로그를 볼 것', file=sys.stderr)
        return 2

    out_dir = Path(args.log_dir) / SCRIPT_NAME
    report = build_report(rows, skipped, args, out_dir)
    tot_ok = sum(r['n_ok'] for r in ok)
    tot_same = sum(r['n_same_route'] for r in ok)
    tot_gt = sum(r['n_gt'] for r in ok)
    fracs = [r['same_route_frac'] for r in ok if r['same_route_frac'] is not None]
    (out_dir / 'summary.json').write_text(json.dumps(
        {'scenes': rows, 'skipped_no_vln_n1': skipped,
         'total': {'gt_episodes': tot_gt, 'planned_ok': tot_ok, 'same_route': tot_same,
                   'same_route_frac': tot_same / tot_ok if tot_ok else None},
         'args': vars(args)}, indent=2, ensure_ascii=False, default=str))

    print(f'\n[{SCRIPT_NAME}] 씬 {len(ok)}/{len(scenes)} 성공 · {(time.time() - t0) / 60:.1f}분', flush=True)
    print(f'[{SCRIPT_NAME}] 계획 성공 {tot_ok}/{tot_gt} 에피소드 '
          f'({tot_ok / tot_gt * 100:.1f}%)', flush=True)
    print(f'[{SCRIPT_NAME}] **같은 루트 {tot_same}/{tot_ok} ({tot_same / tot_ok * 100:.1f}%)**', flush=True)
    print(f'[{SCRIPT_NAME}] 씬별 비율: 중앙값 {np.median(fracs) * 100:.1f}% · '
          f'최악 {min(fracs) * 100:.0f}% · 90%+ 씬 {sum(1 for f in fracs if f >= 0.9)}/{len(fracs)}',
          flush=True)
    print(f'[{SCRIPT_NAME}] report -> {report}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
