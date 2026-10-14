"""출발·도착 지점에서 돌아본 시점 렌더 (frames.select 의 views 용).

trajectory 마다: raw start 에 놓고(at=start) 또는 parquet action 을 끝까지 재생한 뒤(at=goal; P1-R 로 릴리스 pose 와 0.1 mm 일치 확인)
turn_deg 만큼 제자리 회전(+ 왼쪽 / - 오른쪽)한 시점을 rig 별로 렌더한다.
입력은 sources.items 와 같다 (train = 릴리스 action, val = render_split 산출물 action).
저장: <cache_root>/views/<split>/<scene>/<trajectory_id>/<at>_<turn_deg>.<rig>.jpg (+ meta_<at>.json)
jpg 는 render_split 과 같은 규약 (릴리스 양자화 테이블, 4:2:0).

    /usr/bin/python scripts/dataset_converters/instruction_labeling/render/extra_views.py --split val_unseen --subset dev --select keyframes-startsurround-v1 --gpu 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'label'))
from habitat_rig_sim import RigSim  # noqa: E402
from release_io import load_cfg, scene_glb  # noqa: E402
from render_split import release_qtables, save_jpg  # noqa: E402

RENDER_CFG = HERE.parent / 'configs' / 'render.yaml'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', required=True)
    ap.add_argument('--subset', default='')
    ap.add_argument('--select', required=True, help='configs/frames.yaml 의 select 키 (views 목록을 거기서 읽는다)')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    import annotate
    from sources import episode_track, items, view_path
    views = yaml.safe_load(open(HERE.parent / 'configs' / 'frames.yaml'))['select'][args.select]['views'] or []
    cfg = load_cfg(RENDER_CFG)
    turn = float(cfg['agent']['turn_deg'])
    qt = release_qtables(cfg)
    its = annotate.select_subset(items(args.split), args.subset)
    by_traj = {}
    for it in its:
        by_traj.setdefault((it['scene'], it['ep']['trajectory_id']), it)
    rigs = list(cfg['camera']['rigs'])

    def todo_views(it):
        return [v for v in views if not all(view_path(it, v['at'], v['turn_deg'], r).is_file() for r in rigs)]

    done = 0
    for scene in sorted({s for s, _ in by_traj}):
        todo = [it for (s, _), it in by_traj.items() if s == scene and todo_views(it)]
        if not todo:
            continue
        rs = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
        for it in todo:
            tr = episode_track(it, rigs[0])
            for at in ('start', 'goal'):
                vs = [v for v in todo_views(it) if v['at'] == at]
                if not vs:
                    continue
                for v in vs:
                    k = int(round(abs(v['turn_deg']) / turn))
                    assert abs(k * turn - abs(v['turn_deg'])) < 1e-6, f'turn_deg 는 {turn}° 의 배수여야 한다'
                    rs.set_start(it['ep']['start_position'], it['ep']['start_rotation'])
                    if at == 'goal':
                        for a in tr['action'][1:]:
                            rs.step(int(a))
                        err = float(np.linalg.norm(rs.local_pose_cv(rigs[0])[:3, 3] - tr['pose'][-1][:3, 3]))
                        assert err < 0.01, f'{scene} traj {it["ep"]["trajectory_id"]}: 재생 끝 pose 오차 {err:.3f} m'
                    for _ in range(k):
                        rs.step(2 if v['turn_deg'] > 0 else 3)
                    for r, (rgb, _) in rs.observe().items():
                        p = view_path(it, at, v['turn_deg'], r)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        save_jpg(p, rgb, cfg, qt)
            done += 1
        rs.close()
        print(f'{scene}: {len(todo)} traj', flush=True)
    print(f'rendered views for {done} trajectories ({args.select})')


if __name__ == '__main__':
    main()
