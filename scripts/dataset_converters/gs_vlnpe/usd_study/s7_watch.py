"""S7 짝꿍 — `s7_live_view.py`가 흘려보내는 최신 프레임을 창에 띄운다.

**이 파일은 Isaac을 import하지 않는다.** 그게 존재 이유다 — `cv2.imshow`를 Isaac과 같은
프로세스에서 부르면 한 프레임 그린 뒤 세그폴트로 죽는다(실측). 그래서 렌더와 표시를
프로세스로 나눴다.

## 실행 (터미널 2에서, 한 줄)

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s7_watch.py --dir logs/gs-vlnpe/usd_study/s7_live_view

## 조작

    ESC / q   끝내기
    s         지금 화면 저장
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='s7_live_view가 흘려보내는 프레임을 창에 띄운다')
    ap.add_argument('--dir', default='logs/gs-vlnpe/usd_study/s7_live_view',
                    help='s7_live_view.py의 --out_dir 과 같은 폴더')
    ap.add_argument('--fps', type=float, default=20.0, help='화면 갱신 속도')
    ap.add_argument('--timeout_s', type=float, default=0,
                    help='0이면 무한 대기. 프레임이 이 시간 동안 안 갱신되면 끝낸다')
    args = ap.parse_args()

    d = Path(args.dir)
    img_path, state_path = d / 'live_latest.jpg', d / 'live_state.json'
    win = 's7 watch'
    delay = max(1, int(1000.0 / max(args.fps, 0.1)))
    saved, last_mtime, last_change = 0, None, time.time()

    # **창을 먼저 만든다.** `setWindowTitle`을 `imshow`보다 먼저 부르면 내용 없는 창이 생겨서
    # 작은 검은 사각형만 보인다(실측). WINDOW_GUI_NORMAL은 Qt 툴바를 없앤다.
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    def placeholder(lines):
        """아직 프레임이 없을 때 안내를 그림으로 보여준다 — 빈 창보다 낫다."""
        img = np.full((160, 900, 3), 24, dtype=np.uint8)
        for n, line in enumerate(lines):
            cv2.putText(img, line, (16, 38 + 34 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (210, 210, 210), 1, cv2.LINE_AA)
        return img

    print(f'[watch] 보는 파일: {img_path}')
    print('[watch] ESC 또는 q 로 끝냅니다.')
    while True:
        img = cv2.imread(str(img_path)) if img_path.is_file() else None
        if img is None:
            cv2.imshow(win, placeholder([
                'waiting for frames...',
                f'watching: {img_path}',
                'run s7_live_view.py in another terminal',
            ]))
        else:
            mtime = img_path.stat().st_mtime
            if mtime != last_mtime:
                last_mtime, last_change = mtime, time.time()
            cv2.imshow(win, img)                      # 반드시 imshow 먼저
            if state_path.is_file():
                try:
                    st = json.loads(state_path.read_text(encoding='utf-8'))
                    cv2.setWindowTitle(win, f"s7 watch  frame {st['frame']}/{st['total']}  "
                                            f"depth {st['depth_valid']:.3f}  {Path(st['usd']).name}")
                except Exception:  # noqa: BLE001 - 쓰는 중이면 잠깐 깨질 수 있다
                    pass

        key = cv2.waitKey(delay) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('s') and img is not None:
            out = d / f'watch_snapshot_{saved:03d}.jpg'
            cv2.imwrite(str(out), img)
            print(f'[watch] 저장: {out}')
            saved += 1
        if args.timeout_s and (time.time() - last_change) > args.timeout_s:
            print(f'[watch] {args.timeout_s}s 동안 갱신이 없어 끝냅니다')
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
