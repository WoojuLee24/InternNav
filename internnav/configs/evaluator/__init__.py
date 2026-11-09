from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from internnav.configs.agent import AgentCfg


class EnvCfg(BaseModel):
    env_type: str
    env_settings: Dict[str, Any]


class SensorCfg(BaseModel):
    sensor_settings: Dict[str, Any] = None


class ControllerCfg(BaseModel):
    controller_settings: Dict[str, Any] = None


class RobotCfg(BaseModel):
    robot_settings: Dict[str, Any] = None
    sensors: List[SensorCfg] = None
    controllers: List[ControllerCfg] = None


class SceneCfg(BaseModel):
    scene_type: Optional[str] = None
    scene_asset_path: str = None
    scene_scale: Optional[Any] = None
    scene_settings: Optional[Dict[str, Any]] = None
    scene_data_dir: Optional[str] = None


class MetricCfg(BaseModel):
    save_dir: str = None
    metric_setting: Dict[str, Any] = None


class TaskCfg(BaseModel):
    task_name: Optional[str] = None
    task_settings: Dict[str, Any] = None
    scene: SceneCfg = None
    robot_name: Optional[str] = None
    robot: Optional[RobotCfg] = None
    robot_flash: Optional[bool] = None
    flash_collision: Optional[str] = None  # None: no detection, 'stop': stop on collision, 'reset': episode failure
    robot_platform_size: Optional[float] = None  # robot platform radius in meters; None = auto from AABB
    robot_usd_path: Optional[str] = None
    camera_resolution: Optional[List[int]] = None
    metric: Optional[MetricCfg] = None
    camera_prim_path: Optional[str] = None
    one_step_stand_still: Optional[bool] = None
    camera_translation: Optional[tuple] = None
    camera_orientation: Optional[tuple] = None
    flash_controller_type: Optional[str] = None  # None: robot_flash/flash_collision 로 고른 기본 flash 컨트롤러
    flash_controller_settings: Optional[Dict[str, Any]] = None  # flash_controller_type 컨트롤러의 추가 설정
    skip_aux_cameras: Optional[bool] = None  # True: topdown_camera_500 을 붙이지 않음 (렌더 시간 절약; flash_collision 기본 컨트롤러는 이 카메라가 필요)


class EvalDatasetCfg(BaseModel):
    dataset_type: Optional[str] = None
    dataset_settings: Dict[str, Any]


class EvalCfg(BaseModel):
    eval_type: Optional[str] = None
    eval_settings: Optional[Dict[str, Any]] = {}
    agent: Optional[AgentCfg] = None
    env: EnvCfg = None
    task: TaskCfg = None
    dataset: EvalDatasetCfg = None


__all__ = [
    'AgentCfg',
    'EnvCfg',
    'SensorCfg',
    'ControllerCfg',
    'RobotCfg',
    'SceneCfg',
    'MetricCfg',
    'EvalDatasetCfg',
    'EvalCfg',
]
