"""noeun 66 · WAIT 로 쓸 대기 프레임 목록 — 65 자동 판정(사람 보임) + 사람 검토(충돌 가능) -> 로더 `wait_mode=visible` 의 `wait_keep` 파일.

최종 기준 (2026-10-29 사용자): **사람과 충돌 가능** = WAIT. 사람이 보여도 부딪힐 가능성이 없으면 제외, 일부만 보여도 부딪힐 가능성이 있으면 WAIT.
65 의 "보임" 은 1 차 후보이고, 검토에서 충돌 가능 여부로 고친다.

검토 = 대기 프레임 검토 페이지(https://claude.ai/artifact/3piT2wsqNsAQdTkmGZUQFb)의 "결과 복사" json
  {"changed": [{"key": "<split>/<episode>/<frame>", "auto": bool, "human": bool}, ...]} — 자동 판정과 다르게 고친 것만 들어 있다.
출력: {"keep": [["hunav", episode, frame], ...] (train 만, 학습용), "val_seen": [[...]] (평가 참고), "source": {...}}
scene 이름 "hunav" = 64_build_train.py 의 hunav_v1 scene 접두사 (형제 hunav_s0~s2 가 같은 프레임을 공유).

사용 (저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/66_wait_keep.py --review <검토 결과 json>
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--visibility', default=str(HERE / 'wait' / 'wait_visibility_v1.json'))
    ap.add_argument('--review', default='', help='검토 페이지 결과 json (빈값 = 자동 판정 그대로)')
    ap.add_argument('--out', default=str(HERE / 'wait' / 'wait_keep_v1.json'))
    return ap.parse_args()


def main():
    a = parse()
    vis = json.load(open(a.visibility))['frames']
    final = {f"{r['split']}/{r['episode']}/{r['frame']}": r['visible'] for r in vis}
    changed = json.load(open(a.review))['changed'] if a.review else []
    for c in changed:
        assert c['key'] in final and final[c['key']] == c['auto'], f'검토 결과가 자동 판정 파일과 맞지 않다: {c}'
        final[c['key']] = c['human']
    keep = {'train': [], 'val_seen': []}
    for k, v in sorted(final.items()):
        split, ep, fr = k.split('/')
        if v:
            keep[split].append(['hunav', int(ep), int(fr)])
    out = {'keep': keep['train'], 'val_seen': keep['val_seen'],
           'source': {'criterion': '사람과 충돌 가능 (검토로 확정)', 'visibility': Path(a.visibility).name, 'review': Path(a.review).name if a.review else None,
                      'changed': len(changed), 'train_wait_frames': sum(k.startswith('train/') for k in final),
                      'val_wait_frames': sum(k.startswith('val_seen/') for k in final)}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"train keep {len(keep['train'])}/{out['source']['train_wait_frames']} · val keep {len(keep['val_seen'])}/{out['source']['val_wait_frames']} "
          f"(검토로 고친 것 {len(changed)}) -> {a.out}")


if __name__ == '__main__':
    main()
