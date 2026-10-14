"""P1-E: action 이 없는 split(val_unseen)을 릴리스 train 과 같은 vln_ce LeRobot 포맷으로 렌더한다.

흐름
  1. raw R2R json 을 trajectory 로 묶는다 (같은 trajectory 의 episode 는 경로가 같다).
  2. trajectory 마다 1회: action 생성(`action_gen`, config 의 action_gen) -> sim 재생 -> rig 별 RGB/depth/pose 저장
     -> 캐시 `<cache_root>/<split>/<scene>/<trajectory_id>/` (재실행 시 완료된 trajectory 는 건너뜀)
  3. scene 마다 export: episode(지시문) 마다 캐시 이미지를 복사(train 처럼 바이트 동일 사본),
     pixel goal 계산(`vlnce_pixel_goal.build_episode_goals`, 관문 아님), parquet+meta 기록(`vlnce_writer`).
     렌더하지 않은 rig 는 writer 가 센티널로 채워 21컬럼을 유지한다.
  4. `info.json` 의 splits 를 split 이름으로 고치고, `<out_root>/<split>/_index/<scene>.jsonl` 에
     (scene, episode_index, raw episode_id, trajectory_id) 매핑을 남긴다.

이미지 규약 (릴리스와 동일): RGB jpg = 릴리스 jpg 의 양자화 테이블 + 4:2:0, depth png = uint16 floor(m*1000), 10 m clip.

    /usr/bin/python scripts/dataset_converters/instruction_labeling/render/render_split.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --split val_unseen --gpu 0
"""

import argparse
import gzip
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_gen import generate_actions  # noqa: E402
from habitat_rig_sim import RigSim  # noqa: E402
from release_io import REPO, load_cfg, release_image_path, scene_glb  # noqa: E402

sys.path.insert(0, str(REPO / 'scripts/dataset_converters/gs_vlnpe'))
from vlnce_pixel_goal import build_episode_goals  # noqa: E402
from vlnce_writer import finalize_info_json, load_schema, save_episode_vlnce, scene_paths, stream_dir  # noqa: E402


def release_qtables(cfg):
    ref = cfg['image'].get('jpeg_qtables_ref')
    if not ref:
        return None
    return Image.open(REPO / ref).quantization


def save_jpg(path: Path, rgb: np.ndarray, cfg, qtables):
    im = Image.fromarray(rgb)
    sub = int(cfg['image']['jpeg_subsampling'])
    if qtables is not None:
        im.save(path, 'JPEG', qtables=qtables, subsampling=sub)
    else:
        im.save(path, 'JPEG', quality=int(cfg['image']['jpeg_quality']), subsampling=sub)


def save_depth(path: Path, depth_m: np.ndarray, max_m: float):
    mm = np.floor(np.clip(depth_m, 0.0, max_m) * 1000.0).astype(np.uint16)
    cv2.imwrite(str(path), mm)


def load_raw_split(cfg, split: str) -> dict:
    """scene -> trajectory_id -> [episodes] (episode_id 순)."""
    fp = cfg['paths']['raw_root'] / split / f'{split}.json.gz'
    d = json.load(gzip.open(fp))
    out = defaultdict(lambda: defaultdict(list))
    for e in d['episodes']:
        out[Path(e['scene_id']).stem][e['trajectory_id']].append(e)
    for s in out.values():
        for t in s.values():
            t.sort(key=lambda e: e['episode_id'])
    return out


def render_trajectory(rs: RigSim, ep: dict, cfg, tdir: Path, qtables) -> dict:
    """trajectory 1개를 렌더해 캐시에 쓴다. 완료 표시는 meta.json (마지막에 쓴다)."""
    ag = cfg['action_gen']
    g = generate_actions(rs, ep, ag['method'], float(ag['goal_radius']), int(ag['max_steps']))
    actions = g['actions']
    rigs = list(cfg['camera']['rigs'])
    dmax = float(cfg['camera']['depth_max_m'])
    tmp = tdir.with_name(tdir.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    for r in rigs:
        (tmp / f'rgb.{r}').mkdir(parents=True)
        (tmp / f'depth.{r}').mkdir(parents=True)
    rs.set_start(ep['start_position'], ep['start_rotation'])
    poses = {r: [] for r in rigs}
    positions = []
    for f, a in enumerate(actions):
        if f > 0:
            rs.step(a)
        obs = rs.observe()
        positions.append(rs.agent_position().tolist())
        for r in rigs:
            save_jpg(tmp / f'rgb.{r}' / f'{f}.jpg', obs[r][0], cfg, qtables)
            save_depth(tmp / f'depth.{r}' / f'{f}.png', obs[r][1], dmax)
            poses[r].append(rs.local_pose_cv(r))
    for r in rigs:
        np.save(tmp / f'pose.{r}.npy', np.stack(poses[r]).astype(np.float32))
    meta = {'trajectory_id': ep['trajectory_id'], 'scene': Path(ep['scene_id']).stem, 'frames': len(actions),
            'actions': actions, 'positions': positions, 'follower_errors': g['follower_errors'],
            'truncated': g['truncated'], 'action_gen': ag,
            'end_to_goal_m': float(np.linalg.norm(np.array(positions[-1]) - np.array(ep['goals'][0]['position']))),
            'end_to_last_ref_m': float(np.linalg.norm(np.array(positions[-1]) - np.array(ep['reference_path'][-1])))}
    (tmp / 'meta.json').write_text(json.dumps(meta))
    if tdir.exists():
        shutil.rmtree(tdir)
    tmp.rename(tdir)
    return meta


def export_scene(cfg, split: str, scene: str, trajs: dict, cache: Path, schema, index_f) -> dict:
    out = cfg['paths']['out_root'] / split
    P = scene_paths(out, scene)
    if P['base'].exists():
        shutil.rmtree(P['base'])  # meta 가 append 라서 scene 단위로 새로 쓴다
    rigs = list(cfg['camera']['rigs'])
    eps = sorted((e for t in trajs.values() for e in t), key=lambda e: e['episode_id'])
    offset, n_frames, goal_frac = 0, 0, []
    tmeta = {}
    for ei, ep in enumerate(eps):
        tdir = cache / str(ep['trajectory_id'])
        m = tmeta.setdefault(ep['trajectory_id'], json.loads((tdir / 'meta.json').read_text()))
        T = m['frames']
        per_rig = {}
        for r in rigs:
            for kind, ext in (('rgb', 'jpg'), ('depth', 'png')):
                dst = stream_dir(P['videos_dir'], kind, r)
                dst.mkdir(parents=True, exist_ok=True)
                for f in range(T):
                    shutil.copyfile(tdir / f'{kind}.{r}' / f'{f}.{ext}', dst / f'episode_{ei:06d}_{f}.{ext}')
            pose = np.load(tdir / f'pose.{r}.npy').astype(np.float64)
            depth_dir = tdir / f'depth.{r}'
            pg = build_episode_goals(pose, np.asarray(m['actions']), r,
                                     depth_imgs=lambda i, d=depth_dir: cv2.imread(str(d / f'{i}.png'), -1) / 1000.0)
            goal_frac.append(pg['frac_with_goal'])
            per_rig[r] = {'pose': pose, 'goal': pg['goal'], 'rel': pg['relative_goal_frame_id']}
        text = ep['instruction']['instruction_text']
        r_ = save_episode_vlnce(out, scene, ei, m['actions'], per_rig, text, schema=schema,
                                global_index_offset=offset)
        offset = r_['next_index_offset']
        n_frames += T
        index_f.write(json.dumps({'scene': scene, 'episode_index': ei, 'episode_id': ep['episode_id'],
                                  'trajectory_id': ep['trajectory_id'], 'frames': T}) + '\n')
    finalize_info_json(out, scene, schema, len(eps), n_frames)
    info_p = P['meta_dir'] / 'info.json'
    info = json.loads(info_p.read_text())
    info['splits'] = {split: f'0:{len(eps)}'}
    info_p.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return {'episodes': len(eps), 'frames': n_frames, 'goal_frac_mean': float(np.mean(goal_frac))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--scenes', default='all', help="'all' 또는 쉼표 구분")
    ap.add_argument('--max_traj', type=int, default=0, help='scene 당 trajectory 상한 (디버그용, 0=전부)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--skip_export', action='store_true', help='렌더 캐시만 만든다')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.split == 'train':
        raise SystemExit('train 은 릴리스가 있다. 이 스크립트는 action 이 없는 split 용이다.')
    qtables = release_qtables(cfg)
    schema = load_schema(REPO / cfg['image']['schema_ref'])
    data = load_raw_split(cfg, args.split)
    scenes = sorted(data) if args.scenes == 'all' else args.scenes.split(',')
    cache_root = cfg['paths']['cache_root'] / args.split
    out_split = cfg['paths']['out_root'] / args.split
    out_split.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = {}
    (out_split / '_index').mkdir(exist_ok=True)  # scene 별 파일 -> 병렬 실행 안전
    for si, scene in enumerate(scenes):
        trajs = data[scene]
        if args.max_traj:
            trajs = dict(list(trajs.items())[:args.max_traj])
        cache = cache_root / scene
        todo = [t for t in trajs if not (cache / str(t) / 'meta.json').is_file()]
        if todo:
            rs = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
            for k, tid in enumerate(todo):
                m = render_trajectory(rs, trajs[tid][0], cfg, cache / str(tid), qtables)
                print(f'[{scene}] traj {k + 1}/{len(todo)} id={tid} frames={m["frames"]} '
                      f'end->goal {m["end_to_goal_m"]:.2f} m err={m["follower_errors"]} trunc={m["truncated"]}',
                      flush=True)
            rs.close()
        if not args.skip_export:
            with open(out_split / '_index' / f'{scene}.jsonl', 'w') as index_f:
                report[scene] = export_scene(cfg, args.split, scene, trajs, cache, schema, index_f)
            (out_split / '_index' / f'{scene}.report.json').write_text(json.dumps(report[scene]))
            print(f'[{si + 1}/{len(scenes)}] {scene} export {report[scene]} ({time.time() - t0:.0f}s)', flush=True)
    tag = 'all' if args.scenes == 'all' else f'{scenes[0]}+{len(scenes) - 1}'
    (out_split / '_index' / f'_run_{tag}.json').write_text(json.dumps(
        {'args': vars(args), 'elapsed_s': round(time.time() - t0, 1), 'action_gen': cfg['action_gen'],
         'scenes': report}, indent=2))


if __name__ == '__main__':
    main()
