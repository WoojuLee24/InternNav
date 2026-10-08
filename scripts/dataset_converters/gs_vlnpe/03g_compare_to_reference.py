"""우리 계획 경로를 **계획 GT**(`reference_path`)와 비교한다 — `03`과 같은 기준으로.

왜 필요한가
-----------
`03`의 `gt_compare`는 우리 경로를 **실행 궤적**(parquet `camera_position`)과 비교한다. 그런데
실측해 보니 실행 궤적은 계획 경로를 로봇이 전진 0.25 m / 회전 15° 단위로 재현한 것이어서
**계획 경로보다 24% 길고 최대 1.15 m 벗어난다.** 우리는 경로를 *계획*하므로, 실행 궤적과
비교하면 우리가 만들지도 않은 로봇 실행 노이즈를 우리 오차로 계산하게 된다.

이 스크립트는 **계획 ↔ 계획**을 비교한다. 판정 기준은 `03`과 같게 둔다
(`same_route` = 이산 Fréchet < 0.8 m = 문 너비) — 그래야 기존 78.8%와 직접 견줄 수 있다.
`03e_compare_reference_path.py`는 같은 자료를 쓰지만 기준이 다르다(0.5 m 커버리지 비율)라
숫자를 바로 비교할 수 없어서 따로 만들었다.

기준선도 같이 낸다: **GT 실행 ↔ GT 계획**. 이게 "GT 자신의 내부 일관성"이고, 우리 점수가
그것보다 좋은지 나쁜지가 실질적인 판단 기준이다.

에피소드 매칭
-------------
`raw_data`의 `episode_id`는 parquet 인덱스와 다르다. `03e`와 같은 방식으로
**지시문 텍스트**로 후보를 찾고, 그중 `start_position`이 가장 가까운 것을 고른다.
좌표계도 변환해야 한다 — `reference_path`는 Habitat(y가 높이), parquet은 Isaac(z가 높이).

커맨드
------
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03g_compare_to_reference.py --paths_dir logs/gs-vlnpe/profile_check/v2/paths --label reproduce_v1 --log_dir logs/gs-vlnpe

두 설정을 나란히:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03g_compare_to_reference.py --paths_dir logs/gs-vlnpe/refine_sweep/r0.20/paths --label legacy --log_dir logs/gs-vlnpe

Isaac Sim을 띄우지 않는다 — 디스크만 읽는다.
"""

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from esdf_utils import chamfer_distance, discrete_frechet_distance, resample_by_arclength  # noqa: E402

SCRIPT_NAME = '03g_compare_to_reference'
RAW_SPLITS = ('train', 'val_seen', 'val_unseen')
# `03_sample_gt_paths.py:133` DOOR_WIDTH_M 과 같은 값 — 판정 기준을 일치시킨다
DOOR_WIDTH_M = 0.8
RESAMPLE_N = 200


def habitat_to_zup(p) -> np.ndarray:
    """Habitat(y=높이) -> Isaac(z=높이). `03e_compare_reference_path.py:40`과 같은 변환."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def load_raw(raw_root: Path) -> dict:
    """{씬: {지시문: [에피소드, ...]}} — 지시문으로 매칭하기 위한 색인."""
    out = {}
    for split in RAW_SPLITS:
        gz = raw_root / split / f'{split}.json.gz'
        if not gz.is_file():
            continue
        for e in json.load(gzip.open(gz))['episodes']:
            scene = e['scene_id'].split('/')[1]
            key = e['instruction']['instruction_text'].strip()
            out.setdefault(scene, {}).setdefault(key, []).append(e)
    return out


def pair(a_xy, b_xy) -> dict:
    a = resample_by_arclength(np.asarray(a_xy, dtype=np.float64), n=RESAMPLE_N)
    b = resample_by_arclength(np.asarray(b_xy, dtype=np.float64), n=RESAMPLE_N)
    length = lambda x: float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())  # noqa: E731
    fr = float(discrete_frechet_distance(a, b))
    return {'chamfer_m': float(chamfer_distance(a, b)), 'frechet_m': fr,
            'same_route': bool(fr < DOOR_WIDTH_M),
            'len_ratio': length(a) / max(length(b), 1e-9)}


def scene_rows(scene: str, paths_json: Path, gt_root: Path, raw: dict) -> list:
    """그 씬의 에피소드별 {우리↔계획, 실행↔계획} 비교."""
    import pyarrow.parquet as pq

    by_instr = raw.get(scene)
    if not by_instr:
        return []
    meta = gt_root / scene / 'meta' / 'episodes.jsonl'
    if not meta.is_file():
        return []
    tasks_by_ep = {}
    for line in meta.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            tasks_by_ep[d['episode_index']] = d['tasks']

    ours = {e['episode_id']: e for e in json.loads(paths_json.read_text()).get('episodes', [])}
    rows = []
    for ep, our in sorted(ours.items()):
        pq_path = gt_root / scene / 'data' / 'chunk-000' / f'episode_{ep:06d}.parquet'
        if not pq_path.is_file():
            continue
        cands = None
        for t in tasks_by_ep.get(ep, []):
            cands = by_instr.get(t.strip())
            if cands:
                break
        if not cands:
            continue
        exec_xy = np.asarray(pq.read_table(
            pq_path, columns=['observation.camera_position'])['observation.camera_position'].to_pylist())[:, :2]
        # 같은 지시문이 여러 raw 에피소드에 있을 수 있어 start가 가장 가까운 것을 고른다
        starts = np.stack([habitat_to_zup(np.array(c['start_position']))[:2] for c in cands])
        ref_xy = habitat_to_zup(np.array(cands[int(np.argmin(np.linalg.norm(starts - exec_xy[0], axis=1)))]
                                         ['reference_path']))[:, :2]
        our_xy = np.asarray(our['trajectory'], dtype=np.float64)[:, :2]
        if min(len(ref_xy), len(our_xy), len(exec_xy)) < 2:
            continue

        ck = our.get('check_trajectory') or (our.get('planned') or {}).get('check_trajectory') or {}
        rows.append({
            'scene': scene, 'episode': ep,
            'hard_ok': bool(ck.get('hard_ok', True)),
            'ours_vs_plan': pair(our_xy, ref_xy),
            'exec_vs_plan': pair(exec_xy, ref_xy),
            'ours_vs_exec': (our.get('gt_compare') or {}),
        })
    return rows


def summarize(rows: list, key: str, clean_only: bool) -> dict:
    use = [r for r in rows if (r['hard_ok'] or not clean_only)]
    if key == 'ours_vs_exec':
        vals = [(r[key].get('chamfer_m'), r[key].get('frechet_m'), r[key].get('same_route'),
                 r[key].get('len_ratio')) for r in use if r[key].get('frechet_m') is not None]
    else:
        vals = [(r[key]['chamfer_m'], r[key]['frechet_m'], r[key]['same_route'],
                 r[key]['len_ratio']) for r in use]
    if not vals:
        return {}
    ch, fr, sr, lr = (np.array([v[i] for v in vals], dtype=float) for i in range(4))
    return {'n': len(vals), 'chamfer_median': float(np.median(ch)),
            'frechet_median': float(np.median(fr)), 'frechet_p90': float(np.percentile(fr, 90)),
            'same_route': int(sr.sum()), 'same_route_frac': float(sr.mean()),
            'len_ratio_median': float(np.median(lr))}


def build_report(rows: list, summ: dict, args, out_dir: Path) -> Path:
    import viz_utils

    labels = {'ours_vs_plan': f'우리 계획 ↔ 계획 GT ({args.label})',
              'ours_vs_exec': f'우리 계획 ↔ GT 실행 ({args.label}, 03의 기존 기준)',
              'exec_vs_plan': 'GT 실행 ↔ 계획 GT (GT 자신의 내부 일관성)'}
    head = ['비교', 'ep', 'chamfer 중앙', 'Fréchet 중앙', 'Fréchet p90', '같은 루트', '길이비']
    body = ''
    for k in ('ours_vs_plan', 'ours_vs_exec', 'exec_vs_plan'):
        s = summ.get(k) or {}
        if not s:
            continue
        body += (f'<tr><td>{labels[k]}</td><td>{s["n"]}</td><td>{s["chamfer_median"]:.3f}</td>'
                 f'<td>{s["frechet_median"]:.3f}</td><td>{s["frechet_p90"]:.3f}</td>'
                 f'<td>{s["same_route"]} ({s["same_route_frac"] * 100:.1f}%)</td>'
                 f'<td>{s["len_ratio_median"]:.3f}</td></tr>')
    table = ('<table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr></thead><tbody>'
             + body + '</tbody></table>')

    op, oe, ep_ = summ.get('ours_vs_plan', {}), summ.get('ours_vs_exec', {}), summ.get('exec_vs_plan', {})
    verdict = ''
    if op and ep_:
        verdict = (f'<p><b>판정</b>: 계획 GT 기준으로 우리는 '
                   f'<b>{op["same_route_frac"] * 100:.1f}%</b>, GT 자신의 실행 궤적은 '
                   f'<b>{ep_["same_route_frac"] * 100:.1f}%</b>다. ')
        verdict += ('우리가 <b>더 좋다</b> — 남은 차이는 로봇 실행 노이즈이지 플래너 오차가 아니다.</p>'
                    if op['same_route_frac'] >= ep_['same_route_frac']
                    else 'GT 실행 쪽이 계획에 더 가깝다.</p>')
    if op and oe:
        verdict += (f'<p>비교 대상을 실행 궤적에서 계획 GT로 바꾸면 같은 루트 비율이 '
                    f'<b>{oe["same_route_frac"] * 100:.1f}% → {op["same_route_frac"] * 100:.1f}%</b>'
                    f'로 바뀐다 (판정 기준은 둘 다 Fréchet &lt; {DOOR_WIDTH_M} m).</p>')

    summary = (f'<p><b>목적</b>: 우리 계획 경로를 <b>계획 GT</b>(<code>reference_path</code>, R2R '
               f'navmesh 웨이포인트)와 비교한다. 기존 <code>03</code>은 <b>실행 궤적</b>(로봇이 '
               f'전진 0.25 m / 회전 15° 단위로 재현한 것)과 비교했는데, 그건 계획보다 24% 길고 '
               f'최대 1.15 m 벗어나므로 우리가 만들지 않은 노이즈를 우리 오차로 센다.</p>'
               f'{verdict}{table}')

    # 씬별 표 (계획 GT 기준, 비율 낮은 순)
    per = {}
    for r in rows:
        if not r['hard_ok']:
            continue
        per.setdefault(r['scene'], []).append(r['ours_vs_plan']['same_route'])
    srows = ''.join(
        f'<tr><td>{sc}</td><td>{len(v)}</td><td>{sum(v)}</td>'
        f'<td>{sum(v) / len(v) * 100:.0f}%</td></tr>'
        for sc, v in sorted(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1])))
    detail = ('<h2>씬별 (계획 GT 기준)</h2><table><thead><tr><th>씬</th><th>ep</th>'
              '<th>같은 루트</th><th>비율</th></tr></thead><tbody>' + srows + '</tbody></table>')
    return viz_utils.save_gallery(out_dir, 'report.html',
                                  f'{SCRIPT_NAME} — {args.label} vs 계획 GT', summary, detail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_dir', default='scripts/dataset_converters/gs_vlnpe/logs/paths',
                    help='03이 만든 `<씬>_vlnpe.json`들이 있는 폴더')
    ap.add_argument('--label', default='reproduce_v1', help='리포트에 쓸 설정 이름')
    ap.add_argument('--gt_root', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r')
    ap.add_argument('--clean_only', action='store_true', default=True,
                    help='하드 충돌 에피소드는 제외 (기본 켬)')
    ap.add_argument('--include_collisions', dest='clean_only', action='store_false')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    args = ap.parse_args()

    raw = load_raw(Path(args.raw_root))
    print(f'[{SCRIPT_NAME}] raw_data 씬 {len(raw)}개', flush=True)
    files = sorted(glob.glob(str(Path(args.paths_dir) / '*_vlnpe.json')))
    if not files:
        print(f'[{SCRIPT_NAME}] paths json이 없다: {args.paths_dir}', file=sys.stderr)
        return 2

    rows = []
    for f in files:
        scene = json.loads(Path(f).read_text())['scene_id']
        r = scene_rows(scene, Path(f), Path(args.gt_root), raw)
        rows += r
        if r:
            ok = [x for x in r if x['hard_ok']]
            sr = sum(x['ours_vs_plan']['same_route'] for x in ok)
            print(f'[{SCRIPT_NAME}] {scene:16s} ep {len(ok):3d}  계획GT 기준 같은루트 '
                  f'{sr}/{len(ok)} ({sr / max(len(ok), 1) * 100:3.0f}%)', flush=True)
        else:
            print(f'[{SCRIPT_NAME}] {scene:16s} 매칭 0건 (raw_data에 없거나 지시문 불일치)', flush=True)

    summ = {k: summarize(rows, k, args.clean_only)
            for k in ('ours_vs_plan', 'ours_vs_exec', 'exec_vs_plan')}
    out_dir = Path(args.log_dir) / SCRIPT_NAME / args.label
    report = build_report(rows, summ, args, out_dir)
    (out_dir / 'summary.json').write_text(json.dumps(
        {'label': args.label, 'paths_dir': args.paths_dir, 'summary': summ,
         'door_width_m': DOOR_WIDTH_M, 'rows': rows}, indent=2, ensure_ascii=False, default=str))

    print()
    for k, title in (('ours_vs_plan', '우리 계획 ↔ 계획 GT     '),
                     ('ours_vs_exec', '우리 계획 ↔ GT 실행     '),
                     ('exec_vs_plan', 'GT 실행  ↔ 계획 GT      ')):
        s = summ.get(k) or {}
        if s:
            print(f'[{SCRIPT_NAME}] {title} ep {s["n"]:4d} · chamfer {s["chamfer_median"]:.3f} · '
                  f'Fréchet {s["frechet_median"]:.3f} (p90 {s["frechet_p90"]:.3f}) · '
                  f'같은루트 {s["same_route"]}/{s["n"]} ({s["same_route_frac"] * 100:.1f}%) · '
                  f'길이비 {s["len_ratio_median"]:.3f}', flush=True)
    print(f'[{SCRIPT_NAME}] report -> {report}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
