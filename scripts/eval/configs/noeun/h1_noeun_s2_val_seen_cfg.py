"""노은역 val_seen VLN-PE(Isaac) 평가 — S2 단독 checkpoint + H1 flash(바닥 높이 따라가기).

평가 정보 정본: scripts/eval/configs/noeun/noeun_val_seen_v01.yaml (카메라 rig 60cm_15deg = 바닥 위 0.6 m · 아래 15 deg · hfov 68.7, 성공 3 m, 렌더 ambient_only).
값은 환경 변수로 바꾼다 (기본값은 괄호):
  NOEUN_MODEL      S2 checkpoint 경로 (필수)
  NOEUN_DATA       raw_data/r2r 폴더 (data/data/vln_pe_noeun_nav3d_rbq_v6/raw_data/r2r) — 그 아래 val_seen/val_seen.json.gz
  NOEUN_TASK       결과 이름 = logs/<NOEUN_TASK>/ (noeun_s2_val_seen)
  NOEUN_MAX_EPS    앞에서부터 N 개만 (0 = 전부)
  NOEUN_PORT       agent server 포트 (8031). use_agent_server 일 때 start_server.py 도 같은 config 로 띄운다
  NOEUN_AGENT_SERVER  1 = 모델을 agent server(다른 GPU) 에서 (1)
  NOEUN_HEADLESS   1 = 창 없음 (1)
  NOEUN_AGENT      internvla_n1_s2 (HF bf16) | internvla_n1_s2_gguf (llama.cpp 경량화 — NOEUN_MODEL 은 쓰지 않는다)
  NOEUN_FOLLOWER   pixel goal 추종 straight (직선) | navmesh (habitat 과 같은 navmesh 최단 경로 — noeun_navmesh_server.py 필요)
  NOEUN_NAVMESH_URL · NOEUN_BASE_HEIGHT  navmesh 서버 (http://127.0.0.1:8040) · base z - 바닥 (1.05 = H1, sim RBQ10 은 0.55)
  NOEUN_MAX_FOLLOW  pixel goal 하나를 따라가는 최대 step (8 = habitat). 줄이면 S2 를 더 자주 묻는다
  NOEUN_STOP_CONFIRM  STOP 이 이 횟수만큼 연달아 나와야 멈춘다 (1 = habitat)
  NOEUN_GGUF_DIR   GGUF 폴더 (vln-deploy/quantization_kit/out/gguf_noeun_v01_b1wait) — text-Q8_0-embf16.gguf + mmproj-f16.gguf

    # 1) 모델 서버 (GPU 1)
    CUDA_VISIBLE_DEVICES=1 NOEUN_MODEL=<ckpt> /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/start_server.py --config scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py --port 8031
    # 2) Isaac 평가 (GPU 0)
    CUDA_VISIBLE_DEVICES=0 NOEUN_MODEL=<ckpt> /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/eval.py --config scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py
"""
import os

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    MetricCfg,
    SceneCfg,
    TaskCfg,
)

ROOT = '/ws/src/InternNav'
TASK = os.environ.get('NOEUN_TASK', 'noeun_s2_val_seen')
GGUF_DIR = os.environ.get('NOEUN_GGUF_DIR', f'{ROOT}/vln-deploy/quantization_kit/out/gguf_noeun_v01_b1wait')

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=int(os.environ.get('NOEUN_PORT', '8031')),
        model_name=os.environ.get('NOEUN_AGENT', 'internvla_n1_s2'),
        ckpt_path='',
        model_settings={
            'env_num': 1,
            'sim_num': 1,
            'model_path': os.environ.get('NOEUN_MODEL', ''),
            'device': 'cuda:0',
            'width': 640,
            'height': 480,
            'hfov': 68.7,
            'resize_w': 384,
            'resize_h': 384,
            'num_history': 8,
            'max_new_tokens': 128,
            'lookdown_same_view': True,
            'log_dir': f'{ROOT}/logs/{TASK}/agent',
            'save_images': os.environ.get('NOEUN_SAVE_IMAGES', '0') == '1',
            'gguf_text': f"{GGUF_DIR}/text-Q8_0-embf16.gguf",
            'gguf_mmproj': f"{GGUF_DIR}/mmproj-f16.gguf",
            'runner_bin': f'{ROOT}/vln-deploy/quantization_kit/llama.cpp/build_x86/bin/llama-s2-runner',
            'follower': os.environ.get('NOEUN_FOLLOWER', 'straight'),  # navmesh = habitat 처럼 navmesh 최단 경로 (noeun_navmesh_server.py)
            'navmesh_url': os.environ.get('NOEUN_NAVMESH_URL', 'http://127.0.0.1:8040'),
            'navmesh_base_height': float(os.environ.get('NOEUN_BASE_HEIGHT', '1.05')),  # base z - 바닥: H1 1.05 · RBQ10(sim) 0.55
            'max_follow_steps': int(os.environ.get('NOEUN_MAX_FOLLOW', '8')),  # pixel goal 추종 최대 step (habitat 8)
            'stop_confirm': int(os.environ.get('NOEUN_STOP_CONFIRM', '1')),  # STOP 연속 횟수 (1 = habitat)
        },
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': os.environ.get('NOEUN_HEADLESS', '1') == '1',
            'max_episodes': int(os.environ.get('NOEUN_MAX_EPS', '0')),
        },
    ),
    task=TaskCfg(
        task_name=TASK,
        task_settings={
            'env_num': 1,
            'use_distributed': False,
            'proc_num': 1,
            'max_step': 600,  # habitat max_steps_per_episode 500 (↓ 는 step 으로 안 셈) + ↓ 여유
            'type': 'NoeunVLNEvalTask',
            'light_mode': 'ambient_only',
            'rtx_ambient': 10.0,
            'film_iso': 70.0,
            'camera_hfov_deg': 68.7,
            'render_subframes': int(os.environ.get('NOEUN_SUBFRAMES', '2')),
            'success_dz': 1.5,
            'camera_floor_height': 0.60,  # vln_ce rig 60cm_15deg (camera_profiles._vlnce_rig): 바닥 위 0.6 m
            'camera_pitch_deg': 15.0,
            'hide_robot': True,  # 카메라가 H1 다리 사이(0.6 m)라 다리가 찍힌다
            'robot_physics': os.environ.get('NOEUN_ROBOT_PHYSICS', 'off'),  # flash: 위치는 컨트롤러가 놓고 막힘은 raycast. 켜 두면 개찰구에서 PhysX 가 멈춘다
        },
        scene=SceneCfg(
            scene_type='usdz',
            scene_asset_path=f'{ROOT}/data/GS_USDZ/noeun_station_collision.usdz',
            scene_scale=(1, 1, 1),
            scene_settings={},
            scene_data_dir='',
        ),
        robot_name='h1',
        robot_flash=True,
        flash_collision='stop',
        flash_controller_type='VlnMoveByFlashFloorController',
        flash_controller_settings={'base_height': 1.05},
        robot_platform_size=0.3,
        skip_aux_cameras=True,  # 위 flash floor 컨트롤러는 topdown depth 를 안 쓴다
        robot_usd_path=f'{ROOT}/data/InternData-N1-v0.5-mini/Embodiments/vln-pe/h1/h1_internvla.usd',
        camera_resolution=[640, 480],
        camera_prim_path='torso_link/h1_1_25_down_30',  # 자세는 관측마다 task 가 바닥 + 0.6 m, 15 deg 로 다시 놓는다
        one_step_stand_still=True,
        metric=MetricCfg(metric_setting={'metric_config': {'type': 'NoeunVLNPEMetric'}}),
    ),
    dataset=EvalDatasetCfg(
        dataset_type='mp3d',  # 노은역 raw json 은 habitat 좌표 (nav3d/common/label_paths.py ours_to_hab)
        dataset_settings={
            'base_data_dir': os.environ.get('NOEUN_DATA', f'{ROOT}/data/data/vln_pe_noeun_nav3d_rbq_v6/raw_data/r2r'),
            'split_data_types': ['val_seen'],
            'filter_stairs': False,  # 층 이동 에피소드도 평가 (flash 가 바닥 높이를 따라간다)
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': True,
        'vis_output': False,
        'show_rgb': False,
        'use_agent_server': os.environ.get('NOEUN_AGENT_SERVER', '1') == '1',
    },
)
