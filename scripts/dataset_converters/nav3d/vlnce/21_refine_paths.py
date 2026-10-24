"""vln_ce 21 · 경로 — 공식 navmesh 위 find_path 경로 다듬기 (wp · sg).

연결 파일: 코드는 `common/refine_paths.py` 에 있다. 이 파일은 거기에 아래 ARGS 를 앞에 붙여 실행만 한다
(뒤에 같은 인자를 주면 한 번만 받는 인자는 뒤 값이 이기고, 여러 번 받는 인자(--vlnce 등)는 더해진다).

사용 (저장소 루트):
    python3 scripts/dataset_converters/nav3d/vlnce/21_refine_paths.py --scene 17DRP5sb8fy  (씬마다) · --aggregate
"""

import runpy
import sys
from pathlib import Path

TARGET = 'common/refine_paths.py'
ARGS = ['--dataset', 'vln_ce']

if __name__ == '__main__':
    sys.argv = [sys.argv[0], *ARGS, *sys.argv[1:]]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / TARGET), run_name='__main__')
