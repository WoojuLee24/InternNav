"""
h1_internvla_n1_async_cfg_orig.py 기반.
camera_translation / camera_orientation 파라미터로 카메라 포즈를 변경한다.

USD 파일은 그대로 사용하며, VLNCameraCfg의 translation/orientation이
ICamera.create()에 전달되어 prim의 local transform을 override한다.

카메라 방향 설정:
  - CAMERA_PITCH_DEG: 양수=하향, 음수=상향 (0이면 수평 전방)
  - CAMERA_YAW_DEG:   양수=좌향, 음수=우향 (0이면 정면)
"""
import math

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)


def _euler_to_camera_orientation(pitch_deg: float, yaw_deg: float):
    """Convert pitch/yaw Euler angles to Isaac Sim quaternion (w,x,y,z).

    torso_link frame: X=forward, Y=left, Z=up.
    Camera looks along its local -Z axis.
    pitch_deg > 0 = downward, yaw_deg > 0 = leftward.
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    # forward direction the camera should look at (world-space in torso_link frame)
    fx = math.cos(p) * math.cos(y)
    fy = math.cos(p) * math.sin(y)
    fz = -math.sin(p)

    # right = forward × world_up(0,0,1)
    rx, ry, rz = fy, -fx, 0.0
    r_len = math.sqrt(rx * rx + ry * ry)
    if r_len < 1e-6:
        rx, ry, rz, r_len = 1.0, 0.0, 0.0, 1.0
    rx, ry, rz = rx / r_len, ry / r_len, rz / r_len

    # camera up = camera_forward_neg × right  →  (-f) × right
    ucx = -fy * rz + fz * ry
    ucy = -fz * rx + fx * rz
    ucz = -fx * ry + fy * rx

    # rotation matrix columns: [right, cam_up, -forward]
    M = [[rx, ucx, -fx], [ry, ucy, -fy], [rz, ucz, -fz]]
    trace = M[0][0] + M[1][1] + M[2][2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (M[2][1] - M[1][2]) * s
        y_ = (M[0][2] - M[2][0]) * s
        z = (M[1][0] - M[0][1]) * s
    elif M[0][0] > M[1][1] and M[0][0] > M[2][2]:
        s = 2.0 * math.sqrt(1.0 + M[0][0] - M[1][1] - M[2][2])
        w = (M[2][1] - M[1][2]) / s
        x = 0.25 * s
        y_ = (M[0][1] + M[1][0]) / s
        z = (M[0][2] + M[2][0]) / s
    elif M[1][1] > M[2][2]:
        s = 2.0 * math.sqrt(1.0 + M[1][1] - M[0][0] - M[2][2])
        w = (M[0][2] - M[2][0]) / s
        x = (M[0][1] + M[1][0]) / s
        y_ = 0.25 * s
        z = (M[1][2] + M[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + M[2][2] - M[0][0] - M[1][1])
        w = (M[1][0] - M[0][1]) / s
        x = (M[0][2] + M[2][0]) / s
        y_ = (M[1][2] + M[2][1]) / s
        z = 0.25 * s

    return (w, x, y_, z)


# ── 카메라 파라미터 설정 ──────────────────────────────────────────────
# translation: torso_link 기준 local 좌표 offset (x, y, z) [m]
# 전방 x, 좌측 y, 상방 z 양수. (0,0,0) = torso_link과 동일 위치
# CAMERA_PITCH_DEG: 양수=하향, 음수=상향 (0=수평 전방, 30=기본 USD와 동일)
# CAMERA_YAW_DEG:   양수=좌향, 음수=우향
CAMERA_TRANSLATION = (0.2, 0.0, 0.2)
CAMERA_PITCH_DEG = 30.0 # 30.0: 0.6123724356957947, 0.3535533905932736, -0.3535533905932738, -0.6123724356957944)
CAMERA_YAW_DEG = 0.0
CAMERA_ORIENTATION = _euler_to_camera_orientation(CAMERA_PITCH_DEG, CAMERA_YAW_DEG) 
# ──────────────────────────────────────────────────────────────────────

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8023,
        model_name='internvla_n1',
        ckpt_path='',
        model_settings={
            'env_num': 1,
            'sim_num': 1,
            'model_path': '/ws/src/InternNav/checkpoints/InternVLA-N1-w-NavDP',
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            'resize_w': 384,
            'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
            'infer_mode': 'partial_async',
            'vis_debug': False,
            'vis_debug_path': './logs/test_n1/vis_debug',
        },
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='test_n1',
        task_settings={
            'env_num': 1,
            'use_distributed': False,
            'proc_num': 1,
            'max_step': 1000,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            scene_data_dir='/ws/src/InternNav/data/InternData-N1-v0.5-mini/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_flash=True,
        robot_platform_size=None,
        flash_collision=None,
        robot_usd_path='/ws/src/InternNav/data/InternData-N1-v0.5-mini/Embodiments/vln-pe/h1/h1_internvla.usd',
        camera_resolution=[640, 480],
        camera_prim_path='torso_link/h1_1_25_down_30',
        one_step_stand_still=True,
        camera_translation=CAMERA_TRANSLATION,
        camera_orientation=CAMERA_ORIENTATION,
    ),
    dataset=EvalDatasetCfg(
        dataset_type='mp3d',
        dataset_settings={
            'base_data_dir': '/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r',
            'split_data_types': ['val_unseen'],
            'filter_stairs': True,
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': True,
        'vis_output': True,
        'show_rgb': False,
        'use_agent_server': False,
    },
)
