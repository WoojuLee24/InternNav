"""nav3d 파이프라인 실행기 — 데이터셋(vlnce · vlnpe · noeun)마다 단계를 순서대로 돌린다. 단계 · 단계 묶음 · 구간만 돌릴 수도 있다.

각 단계는 데이터셋 폴더의 번호 스크립트(또는 common/ 연결 파일)를 **입력·출력 경로를 다 적어서** 부를 뿐이다.
그래서 스크립트를 따로 돌려도 똑같다 (`--dry_run` 이 그 명령을 보여 준다).

번호 앞자리 = 단계 (세 데이터셋 공통): 1 준비 · 2 경로 만들기 · 3 경로 보정 · 4 걷기 · 5 렌더(데이터 저장) · 6 라벨 · 7 검사.

| 데이터셋 | 무엇 | 최종 산출 |
|---|---|---|
| vlnce | MP3D nav3d 경로 → habitat 렌더 → b1 라벨 | `vln_ce_nav3d/traj_data/r2r/nav3d_{wp,sg}` |
| vlnpe | MP3D nav3d sg 경로 → (A) 경로 위 카메라 Isaac 렌더 · (B) H1 물리 걷기 | `vln_pe_nav3d_sg` · `vln_pe_nav3d_h1` |
| noeun | 노은역 navmesh 경로 → RBQ 물리 걷기 + 경로 보정 → Isaac 렌더 (v6) | `vln_{ce,pe}_noeun_nav3d_rbq_v6` |

컨테이너: 단계마다 `habitat` (/usr/bin/python + habitat_sim) 또는 `isaac` (/workspace/isaaclab/_isaac_sim/python.sh).
지금 컨테이너에서 못 도는 단계에 닿으면 멈추고, 다른 컨테이너에서 이어 돌릴 명령(`--from <단계>`)을 알려 준다.
출력이 이미 있으면 멈춘다 (`--overwrite` 로 덮어쓰기). 지금 logs/·data/ 의 결과를 실수로 덮지 않게.

사용 (저장소 루트, 아무 python3):
    python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --list
    python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --dry_run
    python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --stage 3,4 --overwrite
    python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --only straighten_stairs --overwrite
    python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --scenes 17DRP5sb8fy,1LXtFkjw3qL --max_episodes 3
"""

import argparse
import contextlib
import functools
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HAB = '/usr/bin/python'
ISAAC = '/workspace/isaaclab/_isaac_sim/python.sh'
N = 'scripts/dataset_converters/nav3d'
L = f'{N}/logs'
D = 'data/InternData-N1-v0.5-mini'
PATHS = f'{L}/noeun/paths'      # 노은역 paths json (04 → 보정 → v6)
WALK = f'{L}/noeun/walk'        # 노은역 걷기 입력(eps_*)·결과(walk_*)
DATA_ROOTS = {'mini': 'data/InternData-N1-v0.5-mini', 'full': 'data/InternData-N1'}   # utils/data_roots.py 와 같은 값
LOCAL_VLM_MODEL = 'qwen3vl8b-local'      # common/vlm_server.py 가 띄우는 모델 (configs/model/qwen3vl8b-local.yaml)
LABEL_MODELS = 'scripts/dataset_converters/instruction_labeling/configs/model'
STAGES = {1: '준비', 2: '경로 만들기', 3: '경로 보정', 4: '걷기', 5: '렌더', 6: '라벨', 7: '검사'}
# 노은역 RBQ 걷기 설정 (walk_noeun_v3 · v5trig json 의 args 에 남은 값): 전진 중 방향 보정 1.5 · 제자리 회전 문턱 30° · 높이 스캔 필터 −0.2
RBQ_WALK = ['--steer_gain', '1.5', '--turn_threshold_deg', '30', '--scan_obstacle_depth', '-0.2']


@dataclass
class Step:
    stage: int                                    # 번호 앞자리 (STAGES)
    name: str                                     # 단계 이름 (--only · --from · --to)
    env: str                                      # 'habitat' | 'isaac'
    what: str                                     # 한 줄 설명
    cmds: list                                    # 명령들 (인자 리스트). '{scene}' 이 들어간 명령은 --scenes 마다 병렬
    inputs: list = field(default_factory=list)    # 돌리기 전에 있어야 하는 파일 (이 실행 밖에서 오는 것)
    outputs: list = field(default_factory=list)   # 이미 있으면 --overwrite 없이 멈춘다. '{scene}' 은 씬마다


def data_args(args) -> dict:
    """--data 에 따른 데이터 루트 · logs 위치 · 스크립트에 넘길 경로 인자.
    mini 는 스크립트 기본값 그대로 (경로 인자를 넘기지 않는다 — 검증된 이전 동작), full 은 전부 명시하고 logs 도 logs/full/ 로 나눈다."""
    root = DATA_ROOTS[args.data]
    LD = L if args.data == 'mini' else f'{L}/full'
    full = args.data == 'full'
    return {
        'root': root, 'LD': LD,
        'gt': ['--data_root', f'{root}/vln_ce/traj_data/r2r', '--raw_root', f'{root}/vln_ce/raw_data/r2r',
               '--scene_root', f'{root}/scene_data/mp3d_ce/mp3d', '--out_dir', f'{LD}/common/compare_vlnce_gt'] if full else [],
        'refine': ['--data_root', f'{root}/vln_ce/traj_data/r2r', '--raw_root', f'{root}/vln_ce/raw_data/r2r',
                   '--scene_root', f'{root}/scene_data/mp3d_ce/mp3d', '--out_dir', f'{LD}/common/refine_paths'] if full else [],
        'nav': ['--nav02_dir', f'{LD}/common/compare_vlnce_gt/scenes', '--nav03_dir', f'{LD}/common/refine_paths/scenes'] if full else [],
    }


def mp3d_paths_steps(ds: str, args) -> list:
    """vlnce · vlnpe 가 같이 쓰는 앞 단계 (공식 MP3D navmesh 위 경로). 연결 파일은 데이터셋 폴더에."""
    da, jobs = data_args(args), args.jobs
    n = args.max_episodes
    return [
        Step(1, 'compare_gt', 'habitat', f'릴리스 GT 를 같은 지도·도구로 다시 계획 (층 이동 표시를 뒤 단계가 읽는다, 씬마다 동시 {jobs})',
             [[HAB, f'{N}/{ds}/11_compare_gt.py', '--scene', '{scene}', '--recompute', *da['gt']],
              [HAB, f'{N}/{ds}/11_compare_gt.py', '--aggregate', *da['gt']]],
             outputs=[f'{da["LD"]}/common/compare_vlnce_gt/scenes/{{scene}}.json']),
        Step(2, 'refine_paths', 'habitat', f'navmesh find_path 경로 다듬기 (wp = GT 경유점, sg = 시작·끝만; 평지 경로 '
             f'{"씬마다 " + str(n) + " 개까지" if n else "전부"} + 층 이동 전부, 씬마다 동시 {jobs})',
             [[HAB, f'{N}/{ds}/21_refine_paths.py', '--scene', '{scene}', '--max_flat', str(n), *da['refine']],
              [HAB, f'{N}/{ds}/21_refine_paths.py', '--aggregate', *da['refine']]],
             outputs=[f'{da["LD"]}/common/refine_paths/scenes/{{scene}}_traj.npz']),
    ]


def pipelines(args) -> dict:
    n = args.max_episodes
    lim = lambda flag: [flag, str(n)] if n else []  # noqa: E731 — 일부 에피소드만 (검증용)
    ep = ['--episodes_json', args.episodes_json] if args.episodes_json else []      # 빈값 = 학습 데이터와 같은 에피소드 전부
    sel_in = [args.episodes_json] if args.episodes_json else []
    da = data_args(args)
    root, LD = da['root'], da['LD']
    nav3d_out = f'{root}/vln_ce_nav3d/traj_data/r2r'
    full_render = ['--data_root', root, '--out_root', nav3d_out, '--out_dir', f'{LD}/vlnce/51_render', *da['nav']] if args.data == 'full' else []
    full_label = ['--data_root', root, '--nav3d_root', nav3d_out, '--out_root', f'{root}/instruction_labels_nav3d',
                  '--nav03_dir', f'{LD}/common/refine_paths/scenes'] if args.data == 'full' else []
    scenes = ['--scenes', args.scenes] if args.scenes else []
    vlnce = mp3d_paths_steps('vlnce', args) + [
        Step(5, 'render', 'habitat', 'wp·sg 경로 → GreedyGeodesicFollower → habitat 렌더 → 릴리스 포맷 (지시문마다 에피소드)',
             [[HAB, f'{N}/vlnce/51_render.py', '--instructions', 'all', *ep, *([] if ep else scenes), *full_render]],
             inputs=sel_in, outputs=[nav3d_out]),
        Step(6, 'label', 'habitat', f'instruction 라벨 (method {args.label_method}, model {args.label_model}) — VLM 서버 필요',
             [[HAB, f'{N}/vlnce/61_label.py', '--method', args.label_method, '--model', args.label_model, *ep, *lim('--limit'), *full_label]],
             inputs=sel_in),
        Step(7, 'loader_check', 'habitat', '학습 로더로 nav3d wp · sg 읽기 (릴리스 한 씬 대조)',
             [[HAB, f'{N}/vlnce/71_loader_check.py',
               '--vlnce', f'nav3d wp ({args.data})={nav3d_out}/nav3d_wp=125,0,30', '--vlnce', f'nav3d sg ({args.data})={nav3d_out}/nav3d_sg=125,0,30',
               '--out', f'{LD}/checks/loader_check/summary_vlnce_nav3d.json']],
             outputs=[f'{LD}/checks/loader_check/summary_vlnce_nav3d.json']),
    ]
    sub_scenes = ['--scenes', args.scenes, '--stairs_scenes', ''] if args.scenes else []
    vlnpe = mp3d_paths_steps('vlnpe', args) + [
        Step(5, 'h1_pairs', 'habitat', '(A) vln_pe H1 궤적 ↔ vln_ce ep 짝 (21 경로가 있어야 한다)',
             [[HAB, f'{N}/vlnpe/51_h1_pairs.py']], outputs=[f'{L}/vlnpe/51_h1_pairs/summary.json']),
        Step(5, 'camera_values', 'habitat', '(A) vln_pe GT 에서 카메라 높이·앞 거리·pitch 재기',
             [[HAB, f'{N}/vlnpe/52_camera_values.py', '--stage', 'prepare', *sub_scenes]],
             outputs=[f'{L}/vlnpe/52_camera_values/prepare.json']),
        Step(5, 'camera_prepare', 'habitat', '(A) sg 경로 → mesh 바닥 → 씬별 렌더 명령 (render_cmds.sh)',
             [[HAB, f'{N}/vlnpe/53_camera.py', '--stage', 'prepare', *ep]],
             inputs=[args.episodes_json], outputs=[f'{L}/vlnpe/53_camera/render_cmds.sh']),
        Step(5, 'camera_render', 'isaac', '(A) 씬별 Isaac 렌더 → vln_pe 포맷 (render_cmds.sh = 54_render_camera.py 씬마다)',
             [['bash', f'{L}/vlnpe/53_camera/render_cmds.sh']], outputs=[f'{D}/vln_pe_nav3d_sg/traj_data']),
        Step(5, 'walk_prepare', 'habitat', '(B) 같은 ep 로 raw split 두 개 (nav3d_gt · nav3d_sg)',
             [[HAB, f'{N}/vlnpe/55_walk_h1.py', '--stage', 'prepare', *ep]],
             inputs=[args.episodes_json], outputs=[f'{D}/vln_pe_nav3d_raw/r2r']),
        Step(5, 'walk_h1', 'isaac', '(B) H1 물리 걷기 → vln_pe 포맷 (gt · sg)',
             [[ISAAC, f'{N}/vlnpe/55_walk_h1.py', '--stage', 'walk', '--split', 'nav3d_gt', *lim('--max_episodes')],
              [ISAAC, f'{N}/vlnpe/55_walk_h1.py', '--stage', 'walk', '--split', 'nav3d_sg', *lim('--max_episodes')]],
             outputs=[f'{D}/vln_pe_nav3d_h1/traj_data/nav3d_gt', f'{D}/vln_pe_nav3d_h1/traj_data/nav3d_sg']),
        Step(7, 'loader_check', 'habitat', '(A) 학습 로더로 읽기 (camera_check 가 결과를 읽는다)',
             [[HAB, f'{N}/vlnpe/71_loader_check.py']], outputs=[f'{L}/checks/loader_check/summary_06q.json']),
        Step(7, 'camera_check', 'habitat', '(A) 판정 (카메라 높이 · 경로 · 로더)',
             [[HAB, f'{N}/vlnpe/53_camera.py', '--stage', 'check', *ep]], outputs=[f'{L}/vlnpe/53_camera/summary.json']),
        Step(7, 'walk_compare', 'habitat', '(B) 성공률 · 도착 오차 · sg vs gt',
             [[HAB, f'{N}/vlnpe/55_walk_h1.py', '--stage', 'compare', *ep]], outputs=[f'{L}/vlnpe/55_walk_h1/summary.json']),
    ]
    P0, P_MF, P_ST = (f'{PATHS}/noeun_station_paths{s}.json' for s in ('', '_meshfloor', '_meshfloor_straight'))
    P_V4, P_V5, P_V6 = (f'{PATHS}/noeun_station_paths_meshfloor_straight_{v}.json' for v in ('v4', 'v5', 'v6'))
    CAND, CAND_MF, CAND_ST = (f'{PATHS}/noeun_station_paths_cand{s}.json' for s in ('', '_meshfloor', '_meshfloor_straight'))
    CE6, PE6 = f'{D}/vln_ce_noeun_nav3d_rbq_v6', f'{D}/vln_pe_noeun_nav3d_rbq_v6'
    # 42_walk_rbq 는 <log>/walk_<tag>.json 을 쓰고, 넘겨받은 ep 목록도 <log>/eps_<tag>.json 으로 다시 쓴다
    # (tag 이름이 입력과 같으면 같은 내용으로 덮는다 — noeun_cand · noeun_v6, 이전 실행과 같음).
    walk = lambda eps, tag, extra=(): [  # noqa: E731
        'python3', f'{N}/noeun/42_walk_rbq.py', '--episodes_json', eps, '--tag', tag, '--workers', str(args.walk_workers),
        '--', *RBQ_WALK, *extra]
    flat = ['--n_multifloor', str(n), '--flat', f'0:{n // 2},1:{n - n // 2}'] if n else []
    noeun = [
        Step(1, 'check_assets', 'habitat', '충돌 mesh ↔ 3DGS 정합 · 바닥 구멍 (holes.csv = select_paths U4 입력)',
             [[HAB, f'{N}/noeun/11_check_assets.py', '--scene', 'noeun_station']],
             outputs=[f'{L}/noeun/11_check_assets/noeun_station/holes.csv']),
        Step(1, 'build_navmesh', 'habitat', 'mesh → navmesh · 층 · 계단',
             [[HAB, f'{N}/noeun/12_build_navmesh.py', '--scene', 'noeun_station']],
             outputs=[f'{L}/noeun/12_build_navmesh/noeun_station/noeun_station.navmesh']),
        Step(2, 'refine_paths', 'habitat', '층 사이 경로 다듬기',
             [[HAB, f'{N}/noeun/21_refine_paths.py', *lim('--n_pairs')]],
             outputs=[f'{L}/common/refine_paths/scenes/noeun_station_traj.npz']),
        Step(2, 'select_paths', 'habitat', '층 이동 + 평지 고르기 (사용 가능 판정 U1~U5) → paths json',
             [[HAB, f'{N}/noeun/22_select_paths.py', *flat]], outputs=[P0]),
        Step(3, 'floor_to_mesh', 'habitat', '점별 바닥 z: navmesh → 장면 mesh',
             [[HAB, f'{N}/noeun/31_floor_to_mesh.py', '--paths_json', P0, '--out', P_MF]], outputs=[P_MF]),
        Step(3, 'straighten_narrow', 'habitat', '좁은 통로(개찰구) 안 경로를 곧게',
             [[HAB, f'{N}/noeun/32_straighten_narrow.py', '--paths_json', P_MF, '--out', P_ST]], outputs=[P_ST]),
        Step(4, 'make_walk_inputs', 'habitat', 'RBQ 걷기 입력 (0.5 m 경유점 + 출발 yaw)',
             [[HAB, f'{N}/noeun/41_make_walk_inputs.py', '--paths_json', P_ST, '--out', f'{WALK}/eps_noeun_all_v2.json']],
             outputs=[f'{WALK}/eps_noeun_all_v2.json']),
        Step(4, 'walk_first', 'isaac', 'RBQ 첫 걷기 → 넘어진 ep 를 고른다 (v3)',
             [walk(f'{WALK}/eps_noeun_all_v2.json', 'noeun_v3')], outputs=[f'{WALK}/walk_noeun_v3.json']),
        Step(4, 'replace_candidates', 'habitat', '넘어진 ep 마다 같은 종류의 새 경로 후보 3 개 (U1~U6)',
             [[HAB, f'{N}/noeun/43_replace_fallen.py', 'gen', '--paths_json', P_ST, '--walk_json', f'{WALK}/walk_noeun_v3.json',
               '--cand_json', CAND]], outputs=[CAND]),
        Step(4, 'candidates_fix_paths', 'habitat', '후보에 31 · 32 를 그대로',
             [[HAB, f'{N}/noeun/31_floor_to_mesh.py', '--paths_json', CAND, '--out', CAND_MF],
              [HAB, f'{N}/noeun/32_straighten_narrow.py', '--paths_json', CAND_MF, '--out', CAND_ST]], outputs=[CAND_MF, CAND_ST]),
        Step(4, 'candidates_walk_inputs', 'habitat', '후보 → 걷기 입력',
             [[HAB, f'{N}/noeun/43_replace_fallen.py', 'eps', '--cand_straight_json', CAND_ST,
               '--eps_json', f'{WALK}/eps_noeun_all_v2.json', '--cand_eps_json', f'{WALK}/eps_noeun_cand.json']],
             outputs=[f'{WALK}/eps_noeun_cand.json']),
        Step(4, 'walk_candidates', 'isaac', '후보 걷기',
             [walk(f'{WALK}/eps_noeun_cand.json', 'noeun_cand')], outputs=[f'{WALK}/walk_noeun_cand.json']),
        Step(4, 'replace_merge', 'habitat', '성공한 첫 후보로 갈아 끼움 → v4 (paths · eps · walk)',
             [[HAB, f'{N}/noeun/43_replace_fallen.py', 'merge', '--paths_json', P_ST, '--eps_json', f'{WALK}/eps_noeun_all_v2.json',
               '--walk_json', f'{WALK}/walk_noeun_v3.json', '--cand_walk', f'{WALK}/walk_noeun_cand.json',
               '--cand_eps_json', f'{WALK}/eps_noeun_cand.json', '--cand_straight_json', CAND_ST, '--out_tag', 'v4']],
             outputs=[P_V4, f'{WALK}/eps_noeun_v4.json', f'{WALK}/walk_noeun_v4.json']),
        Step(4, 'shift_starts', 'habitat', '출발 자리(U6) 실패 ep 의 출발점을 경로 따라 앞으로 → v5',
             [[HAB, f'{N}/noeun/44_shift_starts.py', '--paths_json', P_V4, '--eps_json', f'{WALK}/eps_noeun_v4.json', '--out_tag', 'v5']],
             outputs=[P_V5, f'{WALK}/eps_noeun_v5.json']),
        Step(4, 'straighten_stairs', 'habitat', f'계단 구간을 곧게 (ep {args.stairs_episode_ids}: 계단 위 제자리 회전으로 넘어짐) → v6',
             [[HAB, f'{N}/noeun/45_straighten_stairs.py', '--paths_json', P_V5, '--eps_json', f'{WALK}/eps_noeun_v5.json',
               '--episode_ids', args.stairs_episode_ids, '--out_tag', 'v6']],
             outputs=[P_V6, f'{WALK}/eps_noeun_v6.json']),
        Step(4, 'walk', 'isaac', 'RBQ 최종 걷기 (프레임 = 0.25 m / 15° 마다, --record trigger)',
             [walk(f'{WALK}/eps_noeun_v6.json', 'noeun_v6', ['--record', 'trigger'])], outputs=[f'{WALK}/walk_noeun_v6.json']),
        Step(5, 'render', 'isaac', '걸은 자세로 D455 렌더 → vln_ce · vln_pe 포맷',
             [[ISAAC, f'{N}/noeun/51_render_rbq.py', '--walk_json', f'{WALK}/walk_noeun_v6.json', '--format', fmt,
               '--out_root', root, '--log_dir', f'{L}/noeun/51_render_rbq/v6', *lim('--max_episodes')]
              for fmt, root in (('vlnce', CE6), ('vlnpe', PE6))],
             outputs=[CE6, PE6]),
        Step(7, 'render_check', 'habitat', '렌더 검사 M1~M8 (두 포맷)',
             [[HAB, f'{N}/noeun/71_render_check.py', *lim('--max_episodes')]],
             outputs=[f'{L}/checks/noeun_rbq_render_check_v6/summary.json']),
        Step(7, 'loader_check', 'habitat', '학습 로더로 읽기 (릴리스 한 씬 대조)',
             [[HAB, f'{N}/noeun/72_loader_check.py']], outputs=[f'{L}/checks/loader_check/summary_rbq_v6.json']),
    ]
    return {'vlnce': vlnce, 'vlnpe': vlnpe, 'noeun': noeun}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target', choices=['vlnce', 'vlnpe', 'noeun'])
    ap.add_argument('--data', default='mini', choices=list(DATA_ROOTS),
                    help='mini = data/InternData-N1-v0.5-mini (기본) · full = data/InternData-N1. 지금은 vlnce 만 (vlnpe 는 나중에, noeun 은 무관)')
    ap.add_argument('--list', action='store_true', help='단계 목록만')
    ap.add_argument('--dry_run', action='store_true', help='명령만 출력 (실행 안 함)')
    ap.add_argument('--stage', default='', help='이 단계 묶음만 (번호 앞자리, 쉼표). 1 준비 2 경로 3 보정 4 걷기 5 렌더 6 라벨 7 검사')
    ap.add_argument('--only', default='', help='이 단계들만 (이름, 쉼표)')
    ap.add_argument('--from', dest='start', default='', help='이 단계부터')
    ap.add_argument('--to', dest='stop', default='', help='이 단계까지')
    ap.add_argument('--overwrite', action='store_true', help='출력이 이미 있어도 돌린다')
    ap.add_argument('--scenes', default='', help='씬마다 도는 단계(compare_gt · refine_paths)의 씬 (쉼표). 비우면 vln_ce 릴리스 61 씬 전부')
    ap.add_argument('--jobs', type=int, default=4, help='씬마다 도는 단계의 동시 프로세스 수 (이전 기록과 같은 4)')
    ap.add_argument('--episodes_json', default='',
                    help='{씬: [릴리스 episode_index]} 로 일부만. 빈값 = 학습 데이터와 같은 에피소드 전부 (vlnce). '
                         'vlnpe 는 아직 이 목록이 필요하다 (기본 logs/common/episodes.json — 나중에 바꾼다)')
    ap.add_argument('--max_episodes', type=int, default=0,
                    help='테스트용 개수 제한. 0 = 전부. vlnce: 씬마다 평지 경로 수(층 이동은 전부) · 라벨 수, noeun: 경로 수 · 렌더 ep, vlnpe: H1 걷기 ep')
    ap.add_argument('--stairs_episode_ids', default='86', help='noeun straighten_stairs 가 펼 ep (v5 걷기를 보고 사람이 고른 값)')
    ap.add_argument('--walk_workers', type=int, default=5, help='noeun RBQ 걷기 동시 프로세스 (24 코어에 5)')
    ap.add_argument('--label_method', default='b1', help='instruction_labeling configs/methods/<이름>.yaml')
    ap.add_argument('--label_model', default='qwen3vl8b-local', help='configs/model/<이름>.yaml (기본 = 이 PC 의 로컬 서버)')
    ap.add_argument('--vlm_server', default='auto', choices=['auto', 'external'],
                    help='auto = 라벨 단계에서 VLM 서버가 안 떠 있으면 띄우고 끝나면 내린다 (qwen3vl8b-local 만) / external = 직접 띄운 서버만 쓴다')
    ap.add_argument('--vlm_pythonpath', default='/ws/src/InternNav/.venv_label/site', help='VLM 서버용 transformers 4.57 설치 경로')
    ap.add_argument('--vlm_wait_s', type=int, default=900, help='서버 준비를 기다릴 최대 시간 (모델 로드 ≈ 4 분)')
    return ap.parse_args()


def select(steps, args) -> list:
    names = [s.name for s in steps]
    wanted = [x for x in (args.only.split(',') if args.only else []) + [args.start, args.stop] if x]
    unknown = [x for x in wanted if x not in names]
    if unknown:
        sys.exit(f'모르는 단계 {unknown}. 있는 단계: {names}')
    if args.only:
        return [s for s in steps if s.name in args.only.split(',')]
    i0 = names.index(args.start) if args.start else 0
    i1 = names.index(args.stop) if args.stop else len(steps) - 1
    todo = steps[i0:i1 + 1]
    if args.stage:
        keep = {int(x) for x in args.stage.split(',')}
        todo = [s for s in todo if s.stage in keep]
    return todo


@functools.lru_cache(maxsize=None)
def env_ok(env: str) -> bool:
    if env == 'isaac':
        return Path(ISAAC).exists()
    elif env == 'habitat':
        return Path(HAB).exists() and subprocess.run([HAB, '-c', 'import habitat_sim'], capture_output=True).returncode == 0
    else:
        assert False, f'unreachable env={env!r}'


@functools.lru_cache(maxsize=None)
def scene_list(scenes_arg: str) -> tuple:
    if scenes_arg:
        return tuple(scenes_arg.split(','))
    root = Path(f'{D}/vln_ce/traj_data/r2r')
    if not root.is_dir():
        sys.exit(f'씬 목록을 못 읽음: {root} 없음 (--scenes 로 주기)')
    return tuple(sorted(p.name for p in root.iterdir() if p.is_dir()))


def expand(items, args) -> list:
    """'{scene}' 이 든 경로·명령은 씬마다 하나씩."""
    out = []
    for x in items:
        has = any('{scene}' in t for t in x) if isinstance(x, list) else '{scene}' in x
        if not has:
            out.append(x)
        elif isinstance(x, list):
            out += [[t.replace('{scene}', s) for t in x] for s in scene_list(args.scenes)]
        else:
            out += [x.replace('{scene}', s) for s in scene_list(args.scenes)]
    return out


def label_server_error(model: str):
    """서버형 model 이면 base_url 이 응답하는지 본다. 문제면 이유 문자열."""
    cfg = Path(LABEL_MODELS) / f'{model}.yaml'
    if not cfg.exists():
        return f'model 설정 없음: {cfg}'
    m = re.search(r'^base_url:\s*(\S+)', cfg.read_text(), re.M)
    if not m or 'managed:' in cfg.read_text():
        return None
    try:
        urllib.request.urlopen(m.group(1).rstrip('/') + '/models', timeout=3)
        return None
    except Exception as e:  # noqa: BLE001 — 연결 실패 이유를 그대로 보여 준다
        return (f'VLM 서버가 응답하지 않음 ({m.group(1)}: {e}). 다른 터미널에서 먼저 띄운다:\n'
                f'  PYTHONPATH=/ws/src/InternNav/.venv_label/site {HAB} {N}/common/vlm_server.py')


def vlm_server_cmd(args) -> list:
    m = re.search(r':(\d+)', label_base_url(args.label_model) or '')
    return [HAB, f'{N}/common/vlm_server.py', '--port', m.group(1) if m else '8200']


def label_base_url(model: str):
    cfg = Path(LABEL_MODELS) / f'{model}.yaml'
    m = re.search(r'^base_url:\s*(\S+)', cfg.read_text(), re.M) if cfg.exists() else None
    return m.group(1) if m else None


@contextlib.contextmanager
def vlm_server(args):
    """라벨 단계 동안 VLM 서버. 이미 떠 있으면 그대로 쓰고 내리지 않는다 (다른 곳에서 쓰는 서버일 수 있다).
    안 떠 있고 model 이 이 PC 로컬 서버(LOCAL_VLM_MODEL)면 띄우고, 준비될 때까지 기다리고, 끝나면(실패해도) 내린다.
    yield 값 = 문제 설명 (없으면 None)."""
    err = label_server_error(args.label_model)
    if err is None or args.vlm_server == 'external' or args.label_model != LOCAL_VLM_MODEL:
        yield err
        return
    log = Path(f'{L}/run_pipeline/{args.target}/vlm_server.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in (args.vlm_pythonpath, os.environ.get('PYTHONPATH', '')) if x))
    cmd = vlm_server_cmd(args)
    print(f'  VLM 서버가 안 떠 있어 띄운다 (끝나면 내린다): PYTHONPATH={args.vlm_pythonpath} {" ".join(cmd)} · 로그 {log}', flush=True)
    with open(log, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        t0, ready = time.time(), False
        while time.time() - t0 < args.vlm_wait_s and proc.poll() is None:
            if label_server_error(args.label_model) is None:
                ready = True
                break
            time.sleep(10)
        if ready:
            print(f'  VLM 서버 준비 ({time.time() - t0:.0f} s)', flush=True)
            yield None
        else:
            why = f'종료됨 (exit {proc.returncode})' if proc.poll() is not None else f'{args.vlm_wait_s} s 안에 응답 없음'
            yield f'VLM 서버를 띄웠지만 {why} — 로그 {log}'
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        print('  VLM 서버 내림', flush=True)


def run_one(cmd, log: Path = None) -> int:
    if log is None:
        return subprocess.run(cmd).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def run_step(step: Step, args) -> None:
    for c in step.cmds:
        if '{scene}' in ' '.join(c):
            cmds = expand([c], args)
            logs = [Path(f'{L}/run_pipeline/{args.target}/{step.name}/{s}.log') for s in scene_list(args.scenes)]
            print(f'  씬 {len(cmds)} 개 · 동시 {args.jobs} · 로그 {logs[0].parent}/<씬>.log', flush=True)
            with ThreadPoolExecutor(args.jobs) as ex:
                rcs = list(ex.map(run_one, cmds, logs))
            bad = [lg.stem for lg, rc in zip(logs, rcs) if rc != 0]
            if bad:
                sys.exit(f'[{step.name}] 실패한 씬 {len(bad)} 개: {bad[:10]} (로그 {logs[0].parent})')
        else:
            print('  $ ' + ' '.join(c), flush=True)
            rc = run_one(c)
            if rc != 0:
                sys.exit(f'[{step.name}] 실패 (exit {rc}): {" ".join(c)}')


def scripts_of(step: Step) -> str:
    return ' · '.join(dict.fromkeys(Path(t).name for c in step.cmds for t in c if t.endswith(('.py', '.sh')) and t not in (HAB, ISAAC)))


def main():
    args = parse()
    if args.data != 'mini' and args.target != 'vlnce':
        sys.exit(f'--data {args.data} 는 지금 vlnce 만 된다 (vlnpe 는 나중에, noeun 은 mini · full 과 무관)')
    if args.target == 'vlnpe' and not args.episodes_json:
        args.episodes_json = f'{L}/common/episodes.json'      # vlnpe 는 아직 목록이 필요하다 (나중에 vlnce 처럼 바꾼다)
    steps = pipelines(args)[args.target]
    if args.list:
        print(f'{args.target} 단계 ({len(steps)}):')
        for i, s in enumerate(steps, 1):
            print(f'  {i:2d}. [{s.stage} {STAGES[s.stage]:5s}] {s.name:22s} [{s.env:7s}] {scripts_of(s):40s} {s.what}')
        return
    todo = select(steps, args)
    for s in todo:
        print(f'\n[{args.target}] {s.stage} {STAGES[s.stage]} · {s.name} [{s.env}] — {s.what}', flush=True)
        if args.dry_run:
            for c in s.cmds:
                print('  $ ' + ' '.join(c))
            for p in s.inputs:
                print(f'  입력 {"있음" if Path(p).exists() else "없음"}: {p}')
            if s.outputs:
                print(f'  출력 (이미 있으면 --overwrite 필요): {", ".join(s.outputs)}')
            if s.name == 'label' and args.label_model == LOCAL_VLM_MODEL and args.vlm_server == 'auto':
                print(f'  (VLM 서버가 안 떠 있으면 먼저 띄우고 끝나면 내림) PYTHONPATH={args.vlm_pythonpath} {" ".join(vlm_server_cmd(args))}')
            continue
        if not env_ok(s.env):
            sys.exit(f'[{s.name}] 는 {s.env} 컨테이너에서 돈다 (지금 컨테이너에 없음). 거기서 이어서:\n'
                     f'  python3 {N}/run_pipeline.py {" ".join(_resume_args(args, s.name))}')
        missing = [p for p in s.inputs if not Path(p).exists()]
        if missing:
            sys.exit(f'[{s.name}] 입력 없음: {missing}')
        exist = [p for p in expand(s.outputs, args) if Path(p).exists()]
        if exist and not args.overwrite:
            sys.exit(f'[{s.name}] 출력이 이미 있다 ({len(exist)} 개, 예: {exist[:3]}). 덮어쓰려면 --overwrite')
        if s.name == 'label':
            with vlm_server(args) as err:
                if err:
                    sys.exit(f'[{s.name}] {err}')
                run_step(s, args)
            continue
        run_step(s, args)
    print(f'\n[{args.target}] {"명령만 출력 (실행 안 함)" if args.dry_run else "끝"}: {", ".join(s.name for s in todo)}')


def _resume_args(args, step_name: str) -> list:
    """다른 컨테이너에서 이어 돌릴 때의 인자 — 지금 준 인자에서 --from 만 바꾼다."""
    out = [args.target, '--from', step_name] + (['--data', args.data] if args.data != 'mini' else [])
    for flag, val in (('--to', args.stop), ('--stage', args.stage), ('--scenes', args.scenes)):
        if val:
            out += [flag, val]
    for flag, val, default in (('--max_episodes', args.max_episodes, 0), ('--jobs', args.jobs, 4),
                               ('--walk_workers', args.walk_workers, 5), ('--stairs_episode_ids', args.stairs_episode_ids, '86'),
                               ('--episodes_json', args.episodes_json, f'{L}/common/episodes.json' if args.target == 'vlnpe' else '')):
        if val != default:
            out += [flag, str(val)]
    return out + (['--overwrite'] if args.overwrite else [])


if __name__ == '__main__':
    main()
