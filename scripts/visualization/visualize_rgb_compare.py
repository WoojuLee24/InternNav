"""
r2r 와 r2r_h1_replay(또는 r2r_h1) 의 observation.rgb 를 side-by-side PNG 로 비교한다.

두 데이터셋은 같은 LeRobot 포맷을 공유한다:
  <root>/<scan>/videos/chunk-000/observation.images.rgb/episode_XXXXXX.npy  (T,256,256,3) uint8
  <root>/<scan>/meta/tasks.jsonl  {"task_index": N, "task": "instruction", ...}

Usage:
    python scripts/eval/visualize_rgb_compare.py \
        --r2r_dir    data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r \
        --replay_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_debug/r2r_h1 \
        --out_dir    data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_debug/compare \
        [--scan SCAN_ID] [--r2r_ep 0] [--replay_ep 0] [--n_frames 8]

    --match_instruction  플래그를 추가하면 instruction text 로 episode 자동 매칭
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

def _rgb_dir(root, scan):
    return Path(root) / scan / 'videos' / 'chunk-000' / 'observation.images.rgb'


def _load_rgb(root, scan, ep_idx):
    """(T,256,256,3) uint8 npy 를 로드한다."""
    path = _rgb_dir(root, scan) / f'episode_{ep_idx:06d}.npy'
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def _load_tasks(root, scan):
    """tasks.jsonl → {task_index: task_text}"""
    path = Path(root) / scan / 'meta' / 'tasks.jsonl'
    out = {}
    if path.exists():
        for line in open(path):
            t = json.loads(line)
            out[t['task_index']] = t.get('task', '')
    return out


def _common_scans(r2r_dir, replay_dir):
    """두 디렉터리에 모두 존재하는 scan 목록(정렬)."""
    a = {d for d in os.listdir(r2r_dir)    if (Path(r2r_dir)    / d).is_dir()}
    b = {d for d in os.listdir(replay_dir) if (Path(replay_dir) / d).is_dir()}
    return sorted(a & b)


def _find_matching_ep(r2r_dir, replay_dir, scan, r2r_ep):
    """r2r episode N 의 instruction 과 동일한 replay episode index 를 반환한다."""
    r2r_tasks    = _load_tasks(r2r_dir,    scan)
    replay_tasks = _load_tasks(replay_dir, scan)

    target = r2r_tasks.get(r2r_ep, '').strip()
    if not target:
        return None

    replay_inv = {v.strip(): k for k, v in replay_tasks.items()}
    return replay_inv.get(target)


def _available_episodes(root, scan):
    """rgb npy 파일로부터 존재하는 episode index 목록을 반환한다."""
    d = _rgb_dir(root, scan)
    if not d.exists():
        return []
    return sorted(int(p.stem.split('_')[1]) for p in d.glob('episode_*.npy'))


# ─────────────────────────────────────────────────────────────────
# image composition
# ─────────────────────────────────────────────────────────────────

LABEL_H   = 28   # 상단 라벨 높이 (px)
FONT_SIZE = 16
PAD       = 6    # 이미지 간 여백


def _get_font(size=FONT_SIZE):
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _labeled_image(arr, label, font):
    """numpy (H,W,3) → PIL Image with top label bar."""
    h, w = arr.shape[:2]
    canvas = Image.new('RGB', (w, h + LABEL_H), (30, 30, 30))
    canvas.paste(Image.fromarray(arr), (0, LABEL_H))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, LABEL_H], fill=(50, 50, 50))
    draw.text((4, 4), label, fill=(230, 230, 230), font=font)
    return canvas


def make_comparison(r2r_frame, replay_frame, r2r_label, replay_label, font):
    """두 프레임을 좌우로 이어붙인 PIL Image 를 반환한다."""
    img_r2r    = _labeled_image(r2r_frame,    r2r_label,    font)
    img_replay = _labeled_image(replay_frame, replay_label, font)

    h = max(img_r2r.height, img_replay.height)
    w = img_r2r.width + PAD + img_replay.width
    out = Image.new('RGB', (w, h), (15, 15, 15))
    out.paste(img_r2r,    (0,                        0))
    out.paste(img_replay, (img_r2r.width + PAD, 0))
    return out


def make_strip(comparisons):
    """여러 comparison 이미지를 세로로 이어붙인 strip PNG 를 반환한다."""
    w = max(img.width  for img in comparisons)
    h = sum(img.height for img in comparisons) + PAD * (len(comparisons) - 1)
    strip = Image.new('RGB', (w, h), (15, 15, 15))
    y = 0
    for img in comparisons:
        strip.paste(img, (0, y))
        y += img.height + PAD
    return strip


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r2r_dir',    default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r')
    parser.add_argument('--replay_dir', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_debug/r2r_h1')
    parser.add_argument('--out_dir',    default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_debug/compare')
    parser.add_argument('--scan',       default=None,  help='비교할 scan ID (미지정 시 자동 선택)')
    parser.add_argument('--r2r_ep',     type=int, default=0, help='r2r episode index')
    parser.add_argument('--replay_ep',  type=int, default=None, help='replay episode index (미지정 시 r2r_ep 와 동일 또는 instruction 매칭)')
    parser.add_argument('--n_frames',   type=int, default=8, help='비교할 프레임 수')
    parser.add_argument('--match_instruction', action='store_true',
                        help='instruction text 로 replay episode 를 자동 매칭')
    parser.add_argument('--save_individual', action='store_true',
                        help='프레임별 PNG 도 개별 저장 (기본: strip 만)')
    args = parser.parse_args()

    font = _get_font()

    # ── scan 결정 ──
    if args.scan:
        scan = args.scan
    else:
        common = _common_scans(args.r2r_dir, args.replay_dir)
        if not common:
            # scan 이 겹치지 않아도 개별 탐색
            r2r_scans    = sorted(d for d in os.listdir(args.r2r_dir)    if (Path(args.r2r_dir)    / d).is_dir())
            replay_scans = sorted(d for d in os.listdir(args.replay_dir) if (Path(args.replay_dir) / d).is_dir())
            scan = r2r_scans[0] if r2r_scans else None
            replay_scan = replay_scans[0] if replay_scans else None
            if scan is None or replay_scan is None:
                print('[ERROR] r2r_dir 또는 replay_dir 에 scan 디렉터리가 없습니다.')
                return
            print(f'[WARN] 공통 scan 없음 → r2r:{scan}, replay:{replay_scan} 로 비교')
        else:
            scan        = common[0]
            replay_scan = scan
            print(f'공통 scan 자동 선택: {scan}')
    if not hasattr(args, 'replay_scan_override'):
        replay_scan = scan

    # ── replay episode 결정 ──
    replay_ep = args.replay_ep
    if replay_ep is None:
        if args.match_instruction:
            matched = _find_matching_ep(args.r2r_dir, args.replay_dir, scan, args.r2r_ep)
            if matched is not None:
                replay_ep = matched
                print(f'instruction 매칭 → replay ep={replay_ep}')
            else:
                replay_ep = args.r2r_ep
                print(f'[WARN] instruction 매칭 실패 → replay ep={replay_ep} 사용')
        else:
            replay_ep = args.r2r_ep

    # ── rgb 로드 ──
    try:
        r2r_rgb    = _load_rgb(args.r2r_dir,    scan,        args.r2r_ep)
        replay_rgb = _load_rgb(args.replay_dir, replay_scan, replay_ep)
    except FileNotFoundError as e:
        print(f'[ERROR] 파일 없음: {e}')
        # 사용 가능한 episode 목록 안내
        r2r_eps    = _available_episodes(args.r2r_dir,    scan)
        replay_eps = _available_episodes(args.replay_dir, replay_scan)
        print(f'  r2r({scan}) 사용 가능한 ep: {r2r_eps[:10]}')
        print(f'  replay({replay_scan}) 사용 가능한 ep: {replay_eps[:10]}')
        return

    T_r2r    = len(r2r_rgb)
    T_replay = len(replay_rgb)
    print(f'r2r    ep{args.r2r_ep} : {T_r2r} frames  {r2r_rgb.shape}')
    print(f'replay ep{replay_ep}   : {T_replay} frames {replay_rgb.shape}')

    # ── instruction 텍스트 ──
    r2r_tasks    = _load_tasks(args.r2r_dir,    scan)
    replay_tasks = _load_tasks(args.replay_dir, replay_scan)
    instr = (r2r_tasks.get(args.r2r_ep) or replay_tasks.get(replay_ep) or '')[:80]
    print(f'instruction: {instr}')

    # ── frame 샘플링 ──
    n = args.n_frames
    r2r_idxs    = [int(i * (T_r2r    - 1) / max(n - 1, 1)) for i in range(n)]
    replay_idxs = [int(i * (T_replay - 1) / max(n - 1, 1)) for i in range(n)]

    # ── 비교 이미지 생성 ──
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons = []
    for fi, (ri, pi) in enumerate(zip(r2r_idxs, replay_idxs)):
        r2r_label    = f'r2r    | ep{args.r2r_ep} frame{ri:03d}/{T_r2r}'
        replay_label = f'replay | ep{replay_ep} frame{pi:03d}/{T_replay}'
        comp = make_comparison(r2r_rgb[ri], replay_rgb[pi], r2r_label, replay_label, font)
        comparisons.append(comp)

        if args.save_individual:
            fname = out_dir / f'ep{args.r2r_ep:03d}_frame{fi:03d}.png'
            comp.save(fname)
            print(f'  saved: {fname}')

    # ── strip 저장 ──
    strip = make_strip(comparisons)
    # 하단에 instruction 텍스트 추가
    info_h = LABEL_H + 4
    final  = Image.new('RGB', (strip.width, strip.height + info_h), (15, 15, 15))
    final.paste(strip, (0, 0))
    draw = ImageDraw.Draw(final)
    draw.text((PAD, strip.height + 4), f'scan={scan}  {instr}', fill=(180, 180, 100), font=font)

    strip_path = out_dir / f'ep{args.r2r_ep:03d}_strip.png'
    final.save(strip_path)
    print(f'\nstrip saved → {strip_path}  ({final.width}x{final.height}px)')


if __name__ == '__main__':
    main()
