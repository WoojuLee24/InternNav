"""s9가 남긴 씬별 json을 모아 **전 씬에 공통으로 쓸 톤맵 설정 하나**를 고른다.

왜 필요한가
-----------
`s9`는 씬마다 "그 씬의 최적"을 보고한다. 하지만 파이프라인은 `--film_iso` 하나를 **모든 씬에
같이** 쓴다. 그래서 "씬별 최적의 평균"이 아니라 **"어떤 한 설정이 전 씬에서 가장 좋은가"**를
따로 계산해야 한다. 둘은 다른 값이고, 그 차이가 "씬별로 맞추면 얼마나 더 얻는지"다.

`s8`이 씬마다 설정 전부의 SSIM을 json에 남기므로 GPU 없이 계산된다.

커맨드
------
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s10_global_pick.py --log_dir logs/gs-vlnpe/usd_study_tex_v2 --dataset vln_pe

두 데이터셋을 한 번에:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s10_global_pick.py --log_dir logs/gs-vlnpe/usd_study_tex_v2 --dataset both
"""

import argparse
import json
import statistics as st
from pathlib import Path

SCRIPT_NAME = 's10'
TM = {0: 'Clamp', 1: 'Linear', 2: 'Reinhard', 3: 'ReinhardMod',
      4: 'HejlHableAlu', 5: 'HableUc2', 6: 'Aces', 7: 'Iray'}


def load_scenes(log_dir: Path, dataset: str) -> dict:
    """{scene: {(op, iso, crush): ssim_median}} + 현재 설정 SSIM."""
    per_scene, current = {}, {}
    for jf in sorted(log_dir.glob(f's8_tonemap_adopt/*_{dataset}/s8_tonemap.json')):
        d = json.loads(jf.read_text())
        scene = jf.parent.name[: -len(f'_{dataset}')]
        table = {}
        for r in d['results']:
            if r.get('ssim_median') is None:
                continue
            table[(int(r['op']), float(r['iso']), float(r['crush']))] = float(r['ssim_median'])
        if not table:
            continue
        per_scene[scene] = table
        current[scene] = float(d['current']['ssim_median'])
    return per_scene, current


def rank_global(per_scene: dict) -> list:
    """모든 씬에서 측정된 설정만 후보로 두고, 씬 전체 SSIM 중앙값으로 순위."""
    common = None
    for table in per_scene.values():
        keys = set(table)
        common = keys if common is None else (common & keys)
    common = sorted(common or [])
    out = []
    for cfg in common:
        vals = [t[cfg] for t in per_scene.values()]
        out.append({'cfg': cfg, 'median': st.median(vals), 'mean': sum(vals) / len(vals),
                    'min': min(vals), 'n': len(vals)})
    out.sort(key=lambda x: -x['median'])
    return out


def label(cfg) -> str:
    op, iso, crush = cfg
    base = f'op{op} {TM.get(op, "?")} · iso {iso:g}'
    return base + (f' · crush {crush:g}' if op == 7 else '')


def report(log_dir: Path, dataset: str) -> None:
    per_scene, current = load_scenes(log_dir, dataset)
    if not per_scene:
        print(f'[{SCRIPT_NAME}] {dataset}: json이 없다 ({log_dir})')
        return

    ranked = rank_global(per_scene)
    cur_med = st.median(list(current.values()))
    oracle = st.median([max(t.values()) for t in per_scene.values()])

    print(f'\n===== {dataset} · 씬 {len(per_scene)}개 · 공통 설정 후보 {len(ranked)}개 =====')
    print(f'현재 설정 (op6 · iso 70 · crush 0.5)   SSIM 중앙값 {cur_med:.4f}')
    print('\n전 씬 공통 설정 상위 8')
    print(f'{"설정":44s} {"중앙값":>8s} {"현재대비":>9s} {"평균":>8s} {"최악씬":>8s}')
    for r in ranked[:8]:
        print(f'{label(r["cfg"]):44s} {r["median"]:8.4f} {r["median"]-cur_med:+9.4f} '
              f'{r["mean"]:8.4f} {r["min"]:8.4f}')

    best = ranked[0]
    print(f'\n권장(공통 하나)  {label(best["cfg"])}  ->  {best["median"]:.4f} ({best["median"]-cur_med:+.4f})')
    print(f'씬별로 맞추면    {oracle:.4f} ({oracle-cur_med:+.4f})  '
          f'-> 씬별 튜닝의 추가 이득 {oracle-best["median"]:+.4f}')

    # 그 설정이 손해를 보는 씬
    worse = sorted(((s, per_scene[s][best['cfg']] - current[s]) for s in per_scene), key=lambda x: x[1])
    bad = [(s, d) for s, d in worse if d < -0.005]
    print(f'\n권장 설정으로 0.005 이상 퇴보하는 씬: {len(bad)}개 / {len(per_scene)}')
    for s, d in bad[:8]:
        print(f'   {s:16s} {d:+.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log_dir', default='logs/gs-vlnpe/usd_study_tex_v2')
    ap.add_argument('--dataset', choices=['vln_pe', 'vln_n1', 'both'], default='both')
    args = ap.parse_args()
    datasets = ['vln_pe', 'vln_n1'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets:
        report(Path(args.log_dir), ds)


if __name__ == '__main__':
    main()
