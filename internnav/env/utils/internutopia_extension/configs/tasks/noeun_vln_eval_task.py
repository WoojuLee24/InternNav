from typing import Optional

from .vln_eval_task import VLNEvalTaskCfg


class NoeunVLNEvalTaskCfg(VLNEvalTaskCfg):
    """VLNEvalTask + 노은역 usdz (NuRec GS rgb, collision mesh depth) 렌더 조건. 값의 정본 = 데이터 렌더러
    (`scripts/dataset_converters/gs_vlnpe/apply_real/04_render_obs_isaac.py`, `nav3d/noeun/51_render_rbq.py`).

    light_mode: 'ambient_only' = 조명 prim 없이 RTX ambient 만 (데이터와 같음) | 'default' = VLNEvalTask 의 distant + disk
    camera_hfov_deg: None 이면 USD 카메라 그대로. 값이 있으면 focal length 를 바꿔 이 hfov 로 맞춘다 (60cm_15deg rig = 68.7)
    render_subframes: 관측 1 장마다 RTX subframe 수 (GS 볼륨이 수렴하도록; VLNEvalTask 는 2)
    success_dz: 성공 판정에 쓰는 목표와의 높이 차 상한 [m] (같은 층 판정). None 이면 기존 xy 판정만
    camera_floor_height: None 이면 카메라는 USD 장착 그대로. 값이 있으면 관측마다 카메라를 로봇 xy · yaw 에,
        바닥(raycast) + 이 높이, 아래 camera_pitch_deg 로 놓는다 (vln_ce rig 정의: 바닥 위 고정 높이 + 고정 pitch)
    camera_base_height: 바닥 raycast 가 실패할 때 바닥 = base z - 이 값
    hide_robot: True 면 로봇 geometry 를 렌더에서 숨긴다 (물리 충돌은 그대로)
    robot_physics: 'on' = 로봇 충돌 · 중력 그대로 (물리 이동) | 'off' = 로봇 충돌 · 중력을 끈다 (flash 전용: 위치는
        컨트롤러가 매 step 직접 놓고 막힘은 raycast 로 본다. 켜 두면 좁은 개찰구에서 몸이 씬 mesh 와 겹쳐 PhysX 가 멈춘다)
    """

    type: Optional[str] = 'NoeunVLNEvalTask'
    light_mode: Optional[str] = 'ambient_only'
    rtx_ambient: Optional[float] = 10.0
    film_iso: Optional[float] = 70.0
    camera_hfov_deg: Optional[float] = None
    camera_clip: Optional[tuple] = (0.05, 12.0)
    render_subframes: Optional[int] = 2
    success_dz: Optional[float] = None
    camera_floor_height: Optional[float] = None
    camera_pitch_deg: Optional[float] = 15.0
    camera_base_height: Optional[float] = 1.05
    hide_robot: Optional[bool] = False
    robot_physics: Optional[str] = 'on'
