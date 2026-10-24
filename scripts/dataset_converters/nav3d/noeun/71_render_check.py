"""노은역 71 · 검사 — RBQ v6 렌더 검사 M1~M8 (두 포맷).

연결 파일: 코드는 `checks/noeun_rbq_render_check.py` 에 있다. 이 파일은 거기에 아래 ARGS 를 앞에 붙여 실행만 한다
(뒤에 같은 인자를 주면 한 번만 받는 인자는 뒤 값이 이기고, 여러 번 받는 인자(--vlnce 등)는 더해진다).

사용 (저장소 루트):
    python3 scripts/dataset_converters/nav3d/noeun/71_render_check.py --max_episodes 5
"""

import runpy
import sys
from pathlib import Path

TARGET = 'checks/noeun_rbq_render_check.py'
ARGS = [
    '--walk_json',
    'scripts/dataset_converters/nav3d/logs/noeun/walk/walk_noeun_v6.json',
    '--paths_json',
    'scripts/dataset_converters/nav3d/logs/noeun/paths/noeun_station_paths_meshfloor_straight_v6.json',
    '--vlnce_root',
    'data/InternData-N1-v0.5-mini/vln_ce_noeun_nav3d_rbq_v6/traj_data/r2r/noeunstation',
    '--vlnpe_root',
    'data/InternData-N1-v0.5-mini/vln_pe_noeun_nav3d_rbq_v6/traj_data/r2r/noeunstation',
    '--out_dir',
    'scripts/dataset_converters/nav3d/logs/checks/noeun_rbq_render_check_v6',
]

if __name__ == '__main__':
    sys.argv = [sys.argv[0], *ARGS, *sys.argv[1:]]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / TARGET), run_name='__main__')
