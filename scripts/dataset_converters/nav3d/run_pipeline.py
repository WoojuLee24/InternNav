"""nav3d 파이프라인 실행기 — 대상(vlnce · vlnpe · noeun)마다 단계를 순서대로 돌린다. 한 단계만, 구간만 돌릴 수도 있다.

각 단계는 기존 번호 스크립트를 **입력·출력 경로를 모두 적어서** 부를 뿐이다 (스크립트 코드·기본값은 그대로).
그래서 단계마다 그 스크립트를 따로 돌려도 똑같다 (`--dry_run` 이 그 명령을 보여 준다).

| 대상 | 무엇 | 최종 산출 |
|---|---|---|
| vlnce | MP3D 61 씬 nav3d 경로 → habitat 렌더 → b1 라벨 | `vln_ce_nav3d/traj_data/r2r/nav3d_{wp,sg}` |
| vlnpe | MP3D nav3d sg 경로 → (a) 경로 위 카메라 Isaac 렌더 · (b) H1 물리 걷기 | `vln_pe_nav3d_sg` · `vln_pe_nav3d_h1` |
| noeun | 노은역 navmesh 경로 → RBQ 물리 걷기 + 경로 보정 → Isaac 렌더 (v6) | `vln_{ce,pe}_noeun_nav3d_rbq_v6` |

컨테이너: 단계마다 `habitat` (/usr/bin/python + habitat_sim) 또는 `isaac` (/workspace/isaaclab/_isaac_sim/python.sh).
지금 컨테이너에서 못 도는 단계에 닿으면 멈추고, 다른 컨테이너에서 이어 돌릴 명령(`--from <단계>`)을 알려 준다.
출력이 이미 있으면 멈춘다 (`--overwrite` 로 덮어쓰기). 지금 logs/·data/ 의 결과를 실수로 덮지 않게.

사용 (저장소 루트, 아무 python3):
    python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --list
    python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --dry_run
    python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --only straighten_stairs --overwrite
    python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --from floor_to_mesh --to straighten_stairs --overwrite
    python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --scenes 17DRP5sb8fy,1LXtFkjw3qL --jobs 4
"""

import argparse
import functools
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HAB = '/usr/bin/python'
ISAAC = '/workspace/isaaclab/_isaac_sim/python.sh'
N = 'scripts/dataset_converters/nav3d'
C = f'{N}/checks'
L = f'{N}/logs'
D = 'data/InternData-N1-v0.5-mini'
D04 = f'{L}/04_export_paths/noeun_station'
D06K = f'{L}/06k_rbq_walk'
EPISODES = f'{L}/05_vlnce_gt_render/episodes.json'   # (씬, 릴리스 episode_index) 고른 목록 — 만든 스크립트가 없다 (README)
LABEL_MODELS = 'scripts/dataset_converters/instruction_labeling/configs/model'
# 06k RBQ 걷기 설정 (walk_noeun_v3 · v5trig json 의 args 에 남은 값): 전진 중 방향 보정 1.5 · 제자리 회전 문턱 30° · 높이 스캔 필터 −0.2
WALK = ['--steer_gain', '1.5', '--turn_threshold_deg', '30', '--scan_obstacle_depth', '-0.2']


@dataclass
class Step:
    name: str                                     # 단계 이름 (--only · --from · --to)
    env: str                                      # 'habitat' | 'isaac'
    what: str                                     # 한 줄 설명
    cmds: list                                    # 명령들 (인자 리스트). '{scene}' 이 들어간 명령은 --scenes 마다 병렬
    inputs: list = field(default_factory=list)    # 돌리기 전에 있어야 하는 파일 (이 실행 밖에서 오는 것)
    outputs: list = field(default_factory=list)   # 이미 있으면 --overwrite 없이 멈춘다. '{scene}' 은 씬마다


def mp3d_paths_steps(jobs_note: str) -> list:
    """vlnce · vlnpe 가 같이 쓰는 앞 단계 (공식 MP3D navmesh 위 경로)."""
    return [
        Step('gt_compare', 'habitat', f'02 vln_ce GT 와 같은 지도·도구로 다시 계획 (05b · 06q 가 층 이동 표시를 읽는다) {jobs_note}',
             [[HAB, f'{N}/02_vlnce_gt_compare.py', '--scene', '{scene}', '--recompute'],
              [HAB, f'{N}/02_vlnce_gt_compare.py', '--aggregate']],
             outputs=[f'{L}/02_vlnce_gt_compare/scenes/{{scene}}.json']),
        Step('refine_paths', 'habitat', f'03 navmesh find_path 경로 다듬기 (wp = GT 경유점, sg = 시작·끝만) {jobs_note}',
             [[HAB, f'{N}/03_refine_paths.py', '--dataset', 'vln_ce', '--scene', '{scene}'],
              [HAB, f'{N}/03_refine_paths.py', '--aggregate']],
             outputs=[f'{L}/03_refine_paths/scenes/{{scene}}_traj.npz']),
    ]


def pipelines(args) -> dict:
    ctrl_ce = '릴리스 vln_ce (대조)=CONTROL_CE=125,0,30'
    ctrl_pe = '릴리스 vln_pe (대조)=CONTROL_PE'
    per_scene = f'(씬마다, 동시 {args.jobs})'
    vlnce = mp3d_paths_steps(per_scene) + [
        Step('render_habitat', 'habitat', '05b wp·sg 경로 → GreedyGeodesicFollower → habitat 렌더 → 릴리스 포맷',
             [[HAB, f'{N}/05b_vlnce_render_paths.py']],
             inputs=[EPISODES], outputs=[f'{D}/vln_ce_nav3d/traj_data/r2r']),
        Step('label', 'habitat', f'07 instruction 라벨 (method {args.label_method}, model {args.label_model}) — VLM 서버 필요',
             [[HAB, f'{N}/07_label_paths.py', '--sources', 'nav3d_wp,nav3d_sg', '--method', args.label_method, '--model', args.label_model]],
             inputs=[EPISODES]),
        Step('loader_check', 'habitat', '08 학습 로더로 읽기 (릴리스 한 씬 대조)',
             [[HAB, f'{C}/08_loader_check.py', '--vlnce', ctrl_ce,
               '--vlnce', f'nav3d wp (MP3D)={D}/vln_ce_nav3d/traj_data/r2r/nav3d_wp=125,0,30',
               '--vlnce', f'nav3d sg (MP3D)={D}/vln_ce_nav3d/traj_data/r2r/nav3d_sg=125,0,30',
               '--vlnpe', ctrl_pe, '--out', f'{L}/08_loader_check/summary_vlnce_nav3d.json']],
             outputs=[f'{L}/08_loader_check/summary_vlnce_nav3d.json']),
    ]
    vlnpe = mp3d_paths_steps(per_scene) + [
        # (a) 경로 위에 카메라를 바로 놓는 방식
        Step('h1_pairs', 'habitat', '06h vln_pe H1 궤적 ↔ vln_ce ep 짝 (06q 입력)',
             [[HAB, f'{N}/06h_vlnpe_h1_paths.py']], outputs=[f'{L}/06h_vlnpe_h1/summary.json']),
        Step('camera_values', 'habitat', '06v vln_pe GT 에서 카메라 높이·앞 거리·pitch 재기 (06q 입력)',
             [[HAB, f'{N}/06v_vlnpe_check.py', '--stage', 'prepare']], outputs=[f'{L}/06v_vlnpe/prepare.json']),
        Step('camera_prepare', 'habitat', '06q sg 경로 → mesh 바닥 → 씬별 렌더 명령 (render_cmds.sh)',
             [[HAB, f'{N}/06q_vlnpe_mp3d_check.py', '--stage', 'prepare']],
             inputs=[EPISODES], outputs=[f'{L}/06q_vlnpe_mp3d/render_cmds.sh']),
        Step('camera_render', 'isaac', '06p 씬별 Isaac 렌더 → vln_pe 포맷 (render_cmds.sh 실행)',
             [['bash', f'{L}/06q_vlnpe_mp3d/render_cmds.sh']], outputs=[f'{D}/vln_pe_nav3d_sg/traj_data']),
        Step('camera_loader_check', 'habitat', '08 학습 로더로 읽기 (06q check 가 결과를 읽는다)',
             [[HAB, f'{C}/08_loader_check.py', '--vlnce', ctrl_ce, '--vlnpe', f'MP3D sg vln_pe={D}/vln_pe_nav3d_sg/traj_data',
               '--out', f'{L}/08_loader_check/summary_06q.json']],
             outputs=[f'{L}/08_loader_check/summary_06q.json']),
        Step('camera_check', 'habitat', '06q 판정 (카메라 높이 · 경로 · 로더)',
             [[HAB, f'{N}/06q_vlnpe_mp3d_check.py', '--stage', 'check']], outputs=[f'{L}/06q_vlnpe_mp3d/summary.json']),
        # (b) H1 이 물리로 걷는 방식
        Step('walk_prepare', 'habitat', '06r 같은 ep 로 raw split 두 개 (nav3d_gt · nav3d_sg)',
             [[HAB, f'{N}/06r_h1_walk_vlnpe.py', '--stage', 'prepare']],
             inputs=[EPISODES], outputs=[f'{D}/vln_pe_nav3d_raw/r2r']),
        Step('walk_h1', 'isaac', '06r H1 물리 걷기 → vln_pe 포맷 (gt · sg)',
             [[ISAAC, f'{N}/06r_h1_walk_vlnpe.py', '--stage', 'walk', '--split', 'nav3d_gt'],
              [ISAAC, f'{N}/06r_h1_walk_vlnpe.py', '--stage', 'walk', '--split', 'nav3d_sg']],
             outputs=[f'{D}/vln_pe_nav3d_h1/traj_data/nav3d_gt', f'{D}/vln_pe_nav3d_h1/traj_data/nav3d_sg']),
        Step('walk_compare', 'habitat', '06r 성공률 · 도착 오차 · sg vs gt',
             [[HAB, f'{N}/06r_h1_walk_vlnpe.py', '--stage', 'compare']], outputs=[f'{L}/06r_h1_walk/summary.json']),
    ]
    P0, P_MF, P_ST = (f'{D04}/noeun_station_paths{s}.json' for s in ('', '_meshfloor', '_meshfloor_straight'))
    P_V4, P_V5, P_V6 = (f'{D04}/noeun_station_paths_meshfloor_straight_{v}.json' for v in ('v4', 'v5', 'v6'))
    CAND, CAND_MF, CAND_ST = (f'{D04}/noeun_station_paths_cand{s}.json' for s in ('', '_meshfloor', '_meshfloor_straight'))
    CE6, PE6 = f'{D}/vln_ce_noeun_nav3d_rbq_v6', f'{D}/vln_pe_noeun_nav3d_rbq_v6'
    # 06k_parallel 은 <log>/walk_<tag>.json 을 쓰고, 넘겨받은 ep 목록도 <log>/eps_<tag>.json 으로 다시 쓴다
    # (tag 이름이 입력과 같으면 같은 내용으로 덮는다 — noeun_cand · noeun_v6, 이전 실행과 같음).
    walk = lambda eps, tag, extra=():['python3', f'{N}/06k_parallel.py', '--episodes_json', eps, '--tag', tag, '--', *WALK, *extra]  # noqa: E731
    noeun = [
        Step('mesh_gs_check', 'habitat', '00 충돌 mesh ↔ 3DGS 정합 · 바닥 구멍 (holes.csv = 04 U4 입력)',
             [[HAB, f'{N}/00_mesh_gs_check.py', '--scene', 'noeun_station']], outputs=[f'{L}/00_mesh_gs/noeun_station/holes.csv']),
        Step('build_navmesh', 'habitat', '01 mesh → navmesh · 층 · 계단',
             [[HAB, f'{N}/01_build_navmesh.py', '--scene', 'noeun_station']],
             outputs=[f'{L}/01_build_navmesh/noeun_station/noeun_station.navmesh']),
        Step('refine_paths', 'habitat', '03 층 사이 경로 100 개 다듬기',
             [[HAB, f'{N}/03_refine_paths.py', '--scene', 'noeun_station']],
             outputs=[f'{L}/03_refine_paths/scenes/noeun_station_traj.npz']),
        Step('export_paths', 'habitat', '04 층 이동 100 + 평지 100 고르기 (U1~U5) → paths json',
             [[HAB, f'{N}/04_export_paths.py']], outputs=[P0]),
        Step('floor_to_mesh', 'habitat', '04b 점별 바닥 z: navmesh → 장면 mesh',
             [[HAB, f'{N}/04b_paths_mesh_floor.py', '--paths_json', P0, '--out', P_MF]], outputs=[P_MF]),
        Step('straighten_narrow', 'habitat', '04c 좁은 통로(개찰구) 안 경로를 곧게',
             [[HAB, f'{N}/04c_straighten_narrow.py', '--paths_json', P_MF, '--out', P_ST]], outputs=[P_ST]),
        Step('make_walk_inputs', 'habitat', '06k 입력 (0.5 m 경유점 + 출발 yaw)',
             [[HAB, f'{N}/06k_make_walk_inputs.py', '--paths_json', P_ST, '--out', f'{D06K}/eps_noeun_all_v2.json']],
             outputs=[f'{D06K}/eps_noeun_all_v2.json']),
        Step('walk_first', 'isaac', '06k RBQ 200 ep 첫 걷기 → 넘어진 ep 를 고른다 (v3)',
             [walk(f'{D06K}/eps_noeun_all_v2.json', 'noeun_v3')], outputs=[f'{D06K}/walk_noeun_v3.json']),
        Step('replace_candidates', 'habitat', '04d gen 넘어진 ep 마다 같은 종류의 새 경로 후보 3 개 (U1~U6)',
             [[HAB, f'{N}/04d_replace_episodes.py', 'gen', '--paths_json', P_ST, '--walk_json', f'{D06K}/walk_noeun_v3.json',
               '--cand_json', CAND]], outputs=[CAND]),
        Step('candidates_fix_paths', 'habitat', '후보에 04b · 04c 를 그대로',
             [[HAB, f'{N}/04b_paths_mesh_floor.py', '--paths_json', CAND, '--out', CAND_MF],
              [HAB, f'{N}/04c_straighten_narrow.py', '--paths_json', CAND_MF, '--out', CAND_ST]], outputs=[CAND_MF, CAND_ST]),
        Step('candidates_walk_inputs', 'habitat', '04d eps 후보 → 06k 입력',
             [[HAB, f'{N}/04d_replace_episodes.py', 'eps', '--cand_straight_json', CAND_ST,
               '--eps_json', f'{D06K}/eps_noeun_all_v2.json', '--cand_eps_json', f'{D06K}/eps_noeun_cand.json']],
             outputs=[f'{D06K}/eps_noeun_cand.json']),
        Step('walk_candidates', 'isaac', '06k 후보 걷기',
             [walk(f'{D06K}/eps_noeun_cand.json', 'noeun_cand')], outputs=[f'{D06K}/walk_noeun_cand.json']),
        Step('replace_merge', 'habitat', '04d merge 성공한 첫 후보로 갈아 끼움 → v4 (paths · eps · walk)',
             [[HAB, f'{N}/04d_replace_episodes.py', 'merge', '--paths_json', P_ST, '--eps_json', f'{D06K}/eps_noeun_all_v2.json',
               '--walk_json', f'{D06K}/walk_noeun_v3.json', '--cand_walk', f'{D06K}/walk_noeun_cand.json',
               '--cand_eps_json', f'{D06K}/eps_noeun_cand.json', '--cand_straight_json', CAND_ST, '--out_tag', 'v4']],
             outputs=[P_V4, f'{D06K}/eps_noeun_v4.json', f'{D06K}/walk_noeun_v4.json']),
        Step('shift_unsafe_starts', 'habitat', '04f 출발 자리(U6) 실패 ep 의 출발점을 경로 따라 앞으로 → v5',
             [[HAB, f'{N}/04f_fix_start.py', '--paths_json', P_V4, '--eps_json', f'{D06K}/eps_noeun_v4.json', '--out_tag', 'v5']],
             outputs=[P_V5, f'{D06K}/eps_noeun_v5.json']),
        Step('straighten_stairs', 'habitat', '04g 계단 구간을 곧게 (ep 86: 계단 위 제자리 회전으로 넘어짐) → v6',
             [[HAB, f'{N}/04g_straighten_stairs.py', '--paths_json', P_V5, '--eps_json', f'{D06K}/eps_noeun_v5.json',
               '--episode_ids', '86', '--out_tag', 'v6']],
             outputs=[P_V6, f'{D06K}/eps_noeun_v6.json']),
        Step('walk', 'isaac', '06k RBQ 200 ep 최종 걷기 (프레임 = 0.25 m / 15° 마다, --record trigger)',
             [walk(f'{D06K}/eps_noeun_v6.json', 'noeun_v6', ['--record', 'trigger'])], outputs=[f'{D06K}/walk_noeun_v6.json']),
        Step('render', 'isaac', '06m 걸은 자세로 D455 렌더 → vln_ce · vln_pe 포맷',
             [[ISAAC, f'{N}/06m_render_rbq.py', '--walk_json', f'{D06K}/walk_noeun_v6.json', '--format', 'vlnce',
               '--out_root', CE6, '--log_dir', f'{L}/06m_render_rbq/v6'],
              [ISAAC, f'{N}/06m_render_rbq.py', '--walk_json', f'{D06K}/walk_noeun_v6.json', '--format', 'vlnpe',
               '--out_root', PE6, '--log_dir', f'{L}/06m_render_rbq/v6']],
             outputs=[CE6, PE6]),
        Step('check_render', 'habitat', '06n 렌더 검사 M1~M8 (두 포맷)',
             [[HAB, f'{C}/06n_check_rbq_render.py', '--walk_json', f'{D06K}/walk_noeun_v6.json', '--paths_json', P_V6,
               '--vlnce_root', f'{CE6}/traj_data/r2r/noeunstation', '--vlnpe_root', f'{PE6}/traj_data/r2r/noeunstation',
               '--out_dir', f'{L}/06n_check_rbq_render_v6']],
             outputs=[f'{L}/06n_check_rbq_render_v6/summary.json']),
        Step('loader_check', 'habitat', '08 학습 로더로 읽기 (릴리스 한 씬 대조)',
             [[HAB, f'{C}/08_loader_check.py', '--vlnce', ctrl_ce, '--vlnce', f'노은역 RBQ v6={CE6}/traj_data/r2r=60,15,15',
               '--vlnpe', ctrl_pe, '--vlnpe', f'노은역 RBQ v6={PE6}/traj_data', '--out', f'{L}/08_loader_check/summary_rbq_v6.json']],
             outputs=[f'{L}/08_loader_check/summary_rbq_v6.json']),
    ]
    return {'vlnce': vlnce, 'vlnpe': vlnpe, 'noeun': noeun}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target', choices=['vlnce', 'vlnpe', 'noeun'])
    ap.add_argument('--list', action='store_true', help='단계 목록만')
    ap.add_argument('--dry_run', action='store_true', help='명령만 출력 (실행 안 함)')
    ap.add_argument('--only', default='', help='이 단계들만 (쉼표)')
    ap.add_argument('--from', dest='start', default='', help='이 단계부터')
    ap.add_argument('--to', dest='stop', default='', help='이 단계까지')
    ap.add_argument('--overwrite', action='store_true', help='출력이 이미 있어도 돌린다')
    ap.add_argument('--scenes', default='', help='씬마다 도는 단계(vlnce · vlnpe 02 · 03)의 씬 (쉼표). 비우면 vln_ce 릴리스 61 씬 전부')
    ap.add_argument('--jobs', type=int, default=4, help='씬마다 도는 단계의 동시 프로세스 수 (README 기록과 같은 4)')
    ap.add_argument('--label_method', default='b1', help='instruction_labeling configs/methods/<이름>.yaml')
    ap.add_argument('--label_model', default='qwen3vl8b-local', help='configs/model/<이름>.yaml (07 기본 = 이 PC 의 로컬 서버)')
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
    return steps[i0:i1 + 1]


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
                f'  PYTHONPATH=/ws/src/InternNav/.venv_label/site {HAB} {N}/07_vlm_server.py')


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


def main():
    args = parse()
    steps = pipelines(args)[args.target]
    if args.list:
        print(f'{args.target} 단계 ({len(steps)}):')
        for i, s in enumerate(steps, 1):
            print(f'  {i:2d}. {s.name:24s} [{s.env:7s}] {s.what}')
        return
    todo = select(steps, args)
    for s in todo:
        print(f'\n[{args.target}] {s.name} [{s.env}] — {s.what}', flush=True)
        if args.dry_run:
            for c in s.cmds:
                print('  $ ' + ' '.join(c))
            for p in s.inputs:
                print(f'  입력 {"있음" if Path(p).exists() else "없음"}: {p}')
            if s.outputs:
                print(f'  출력 (이미 있으면 --overwrite 필요): {", ".join(s.outputs)}')
            continue
        if not env_ok(s.env):
            sys.exit(f'[{s.name}] 는 {s.env} 컨테이너에서 돈다 (지금 컨테이너에 없음). 거기서 이어서:\n'
                     f'  python3 {N}/run_pipeline.py {args.target} --from {s.name}'
                     + (f' --to {args.stop}' if args.stop else '') + (' --overwrite' if args.overwrite else ''))
        missing = [p for p in s.inputs if not Path(p).exists()]
        if missing:
            sys.exit(f'[{s.name}] 입력 없음: {missing}')
        exist = [p for p in expand(s.outputs, args) if Path(p).exists()]
        if exist and not args.overwrite:
            sys.exit(f'[{s.name}] 출력이 이미 있다 ({len(exist)} 개, 예: {exist[:3]}). 덮어쓰려면 --overwrite')
        if s.name == 'label':
            err = label_server_error(args.label_model)
            if err:
                sys.exit(f'[{s.name}] {err}')
        run_step(s, args)
    print(f'\n[{args.target}] {"명령만 출력 (실행 안 함)" if args.dry_run else "끝"}: {", ".join(s.name for s in todo)}')


if __name__ == '__main__':
    main()
