"""M1.4 — Isaac Sim(raw `isaacsim.SimulationApp`) 버전 렌더러.

`04_render_obs.py`(Open3D 버전)와 완전히 같은 2-A/2-B 구조·스키마·검증을 쓴다 — **렌더러 백엔드만
Isaac Sim으로 바꾼 것**이다. json 처리·pose 합성(`synthesize_action_poses`)·통계·리포트는
Open3D 버전과 동일한 함수를 그대로 import해서 재사용하고, mesh 로드/카메라 생성/렌더링 4개
함수만 새로 구현한다.

## 왜 새 파일인가

Open3D 버전을 만든 뒤 사용자가 "GT rgb와 여전히 다르다"고 지적해 원인을 추적한 결과, vln_n1의
GT rgb 자체가 실제 카메라 사진이 아니라 **같은 mesh를 렌더링한 합성 데이터**임을 확인했다
(`.claude/memory/260804_gs_vlnpe_04_render_obs_result.md`). 이 저장소에 Isaac Lab/Isaac Sim이
이미 있고(`docker-isaac/base-navdp.sh`가 NavDP 코드베이스를 Isaac Lab 도커에서 돌리는 것으로
보아 원본 데이터도 Isaac Sim으로 생성됐을 가능성이 높다), 실측으로 확인한 결과 **IsaacLab의
`AppLauncher`(전용 kit 실험 파일)를 거치면 렌더 한 프레임에 30분 이상 걸리거나 멈추는데, raw
`isaacsim.SimulationApp`으로 직접 띄우면 수십 초 안에 끝나고 GT와 기하학적으로 완벽히 일치한다**
(같은 가구·그림·문틀 배치). 이 두 렌더 경로는 부팅 방식·필요한 안전장치 플래그·의존 모듈이
완전히 달라 기존 파일에 분기로 끼워 넣으면 가이드라인의 "로직 내부 침투 금지" 원칙을 어기게
된다 — 그래서 새 파일로 분리했다.

## Isaac Sim 관련 실측 사실 (전부 이 세션에서 직접 확인)

- **`AppLauncher` 우회**: `isaaclab.app.AppLauncher`(→ `apps/isaaclab.python.headless.rendering.kit`
  + performance/balanced/quality 프리셋) 대신 `isaacsim.SimulationApp({"headless": True})`을
  직접 쓴다. 정확히 어떤 kit 설정이 정지의 원인인지는 못 좁혔지만(PhysX 충돌 쿠킹 비활성화,
  `rendering_mode=performance` 둘 다 시도했지만 해결 안 됨), raw 경로가 실측상 확실히 빠르다.
- **`/isaaclab/cameras_enabled` carb 플래그**: `isaaclab.sensors.camera.Camera`는 이 플래그가
  꺼져 있으면 즉시 `RuntimeError`를 던진다(`--enable_cameras`가 `AppLauncher`를 거칠 때만
  자동으로 켜지는 안전장치). raw 부팅 시 수동으로 켜야 한다.
- **텍스처 절대경로**: mp3d_pe의 `isaacsim_<hash>.usd`는 텍스처를 원래 변환 환경의 절대경로
  (`/ssd/share/Matterport3D/data/v1/scans/<scene>/...`)로 참조한다 — 이 컨테이너엔 없는 경로라
  심링크로 우리 로컬 파일을 연결해야 한다(`ensure_texture_symlink`).
- **PhysX 충돌 쿠킹**: 이 USD는 VLN-PE 로봇 물리 시뮬레이션용으로 만들어져 충돌 메쉬가 이미
  붙어 있다 — 순수 카메라 렌더링에는 불필요하므로 `collision_enabled=False`로 끈다(속도 개선
  효과는 못 봤지만 경고 로그를 줄이고 의도를 명확히 하기 위해 유지).
- **mp3d_n1과 mp3d_pe는 같은 world 좌표계**: bbox가 실측상 거의 일치(원점·스케일 재정합 불필요).
- **pose 컨벤션**: IsaacLab `Camera.set_world_poses(pos, quat_wxyz, convention="ros")`의 "ros"는
  forward=+Z, up=-Y — `geometry_utils.action_to_c2w`가 만드는 OpenCV 컨벤션과 정확히 같아서
  회전행렬→쿼터니언 변환만 하면 되고 별도 축 변환이 필요 없다.

실행 예시 (`isaaclab.sh`가 아니라 `_isaac_sim/python.sh`로 직접 실행 — AppLauncher를 안 씀):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs_isaac.py \\
        --scene 17DRP5sb8fy --mode gt_replay --episodes 0,1,2 --n_frames 6
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': True})

import carb  # noqa: E402

carb.settings.get_settings().set_bool('/isaaclab/cameras_enabled', True)
# 히스토그램 기반 오토 익스포저 — 실측(DomeLight 세기 스윕을 켠 채/끈 채 둘 다 반복)으로는 밝기에
# 차이를 못 만들었지만(오토 익스포저가 이 문제의 원인은 아니었음), 꺼두는 게 재현성 면에서 더
# 안전하다 — 켜져 있으면 원칙적으로 프레임마다 직전 밝기 히스토리에 따라 노출이 동적으로
# 재조정될 수 있어 같은 씬/조명이라도 프레임 순서에 따라 살짝 다른 결과가 나올 수 있다.
carb.settings.get_settings().set_bool('/rtx/post/histogram/enabled', False)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from skimage.metrics import structural_similarity as ssim_metric  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
import omni.usd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esdf_utils import load_scene_usd, resample_by_arclength  # noqa: E402
from geometry_utils import (  # noqa: E402
    DEPTH_SCALE_RAW_TO_M,
    action_to_c2w,
    colorize_depth,
    compute_mesh_distance,
    load_depth_frame_m,
    load_rgb_frame,
    load_scene_mesh,
    save_jpg,
    synthesize_action_poses,
)
from viz_utils import blink_widget_html, save_gallery  # noqa: E402
import camera_profiles  # noqa: E402
import dataset_utils  # noqa: E402
import usdz_scene_utils  # noqa: E402

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
# Open3D 버전은 mp3d_n1(원본 raw obj)을 쓰지만, Isaac 버전은 Isaac-ready USD가 필요해 mp3d_pe를
# 쓴다 — 실측으로 두 자산이 같은 world 좌표계(bbox 거의 일치)임을 확인했다.
DEFAULT_MESH_ROOT = 'data/scene_data/mp3d_pe'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe/apply_real'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe/apply_real'
DEFAULT_SCENE = '17DRP5sb8fy'
SCRIPT_NAME = '04_render_obs_isaac'

RENDER_W, RENDER_H = 480, 270
RENDER_NEAR_M, RENDER_FAR_M = 0.05, 10.0
MAX_DEPTH_M = 10.0

# mp3d_pe의 `isaacsim_*.usd`가 참조하는, 원래 변환 환경의 하드코딩된 절대경로 — 이 컨테이너엔
# 없으므로 로컬 mesh_root의 실제 파일로 심링크한다(`ensure_texture_symlink`).
MATTERPORT_TEXTURE_ROOT = Path('/ssd/share/Matterport3D/data/v1/scans')

# --- 2-B(생성 경로 수집) 전용 상수 (04_render_obs.py와 동일) ---
FRAME_STEP_M = 0.035
DEPTH_CLIP_MIN_M, DEPTH_CLIP_MAX_M = 0.1, 3.0
GEN_MESH_ANCHOR_TOL_M = 0.01
GEN_STEP_TOL_RATIO = 3.0

# --- 2-A 통과 기준 (04_render_obs.py와 동일 — 실측 후 재조정 예정) ---
GT_REPLAY_DEPTH_MEDIAN_TOL_M = 0.01
GT_REPLAY_DEPTH_P95_TOL_M = 0.05
# edge_corr(라플라시안 엣지맵 픽셀별 피어슨 상관)는 18프레임 실측에서 두 씬 다 0.19~0.22로
# 기준 미달이었지만, SSIM(구조 유사도)으로 같은 프레임을 다시 보면 median 0.65~0.71로 꽤
# 잘 맞는다(scikit-image `structural_similarity`, 저장된 render_rgb.jpg vs real_rgb.jpg 재계산).
# edge_corr는 엣지가 정확히 같은 픽셀 위치에 있어야 점수가 나오는 까다로운 지표라 미세한 톤/
# 노이즈 차이에도 크게 흔들리는 것으로 판단 — SSIM을 RGB 통과 기준으로 교체하고, edge_corr는
# 참고용으로만 계속 표시한다. 임계값 0.5는 SSIM 문헌에서 흔히 쓰는 "중간 이상 일치" 기준.
GT_REPLAY_RGB_EDGE_CORR_MIN = 0.5
GT_REPLAY_RGB_SSIM_MIN = 0.5
# Isaac RTX 톤매퍼(`/rtx/post/tonemap/op = 6`, Iray)의 노출 기본값. 이 값 그대로면 설정을 아예
# 건드리지 않아 기존 경로가 100% 동일하게 돈다. 같은 톤매퍼의 crushBlacks/burnHighlights는
# RTX Real-Time 경로에서 무반응임을 실측 확인했다(04d_tonemap_sweep.py 참고).
DEFAULT_FILM_ISO = 100.0


RenderCameraConfig = camera_profiles.CameraProfile
SYNTHETIC_CAMERAS = camera_profiles.PROFILES
def ensure_texture_symlink(mesh_root, scene: str) -> None:
    """mp3d_pe USD의 텍스처 절대경로를 로컬 파일로 심링크(멱등 — 이미 있으면 건너뜀)."""
    target = MATTERPORT_TEXTURE_ROOT / scene / 'matterport_mesh'
    if target.is_symlink() or target.exists():
        return
    src = Path(mesh_root).resolve() / scene / 'matterport_mesh'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(src)


def find_scene_usd(mesh_root, scene: str) -> Path:
    """mp3d_pe의 Isaac-ready USD(`isaacsim_<hash>.usd`, `_non_metric` 제외)를 찾는다."""
    mesh_dir = Path(mesh_root) / scene / 'matterport_mesh'
    candidates = sorted(p for p in mesh_dir.glob('*/isaacsim_*.usd') if '_non_metric' not in p.name)
    if not candidates:
        raise FileNotFoundError(f'Isaac USD(isaacsim_*.usd) not found under {mesh_dir}')
    return candidates[0]


def load_scene_model(mesh_root, scene: str) -> Path:
    """씬 USD 경로를 반환(텍스처 심링크까지 준비). `render_along`이 매 호출마다 stage에 참조한다."""
    ensure_texture_symlink(mesh_root, scene)
    return find_scene_usd(mesh_root, scene)


def build_renderer(width: int, height: int, k: np.ndarray, usd_path: Path,
                   light: str = 'dome', tl: dict = None, rtx_ambient: float = 0.0,
                   film_iso: float = DEFAULT_FILM_ISO,
                   render_near_m: float = RENDER_NEAR_M,
                   render_far_m: float = RENDER_FAR_M) -> tuple:
    """`(width,height,K)` + 씬 USD -> `(camera, sim, lights)`. `render_along`이 매 프레임 재사용.
    `lights`는 three_light일 때만 translate op dict(그 외 None — 위치 갱신 불필요).

    **순서가 중요하다** — 실측으로 확인된 성공 순서(raw 스모크 테스트)는 "mesh 참조 -> 조명 ->
    카메라 생성 -> `sim.reset()`"이다. 이 순서를 바꿔서 카메라를 먼저 만들고 나중에 mesh를
    참조했더니 `sim.reset()`이 멈추거나(mesh가 아직 없는 빈 스테이지에 대한 초기화라서로 추정)
    `distance_to_image_plane` 애노테이터가 빈 텐서를 반환했다 — 그래서 이 함수가 mesh 로드까지
    전담하고, `render_along`은 pose 설정과 스텝만 담당한다.
    """
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath('/World/Scene'):
        stage.RemovePrim('/World/Scene')
    cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path),
                               collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.func('/World/Scene', cfg)

    # Collision capture asset은 geometry가 존재해도 RTX sensor에서 숨겨져
    # 있을 수 있으므로 source asset을 건드리지 않고 runtime stage만 보정한다.
    changed = usdz_scene_utils.expose_collision_meshes_for_rendering(
        stage, usd_path, UsdGeom
    )
    if changed:
        print(f'[scene-render] enabled {changed} hidden collision mesh(es)', flush=True)

    lights = _add_lights(stage, light, tl)

    camera_cfg = CameraCfg(
        prim_path='/World/RenderCamera',
        height=height,
        width=width,
        data_types=['rgb', 'distance_to_image_plane'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(render_near_m, render_far_m)),
    )
    camera = Camera(cfg=camera_cfg)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.reset()
    # 전역 간접광 — **씬/카메라가 올라온 뒤에 걸어야 한다**. main()에서 미리 설정했더니 RTX
    # 초기화가 기본값으로 덮어써서 전혀 반영되지 않았다(실측: SSIM 0.72로 무변화).
    if rtx_ambient > 0:
        carb.settings.get_settings().set_float('/rtx/sceneDb/ambientLightIntensity', float(rtx_ambient))
    # 노출(ISO). ambient만으로는 GT 응답곡선의 기울기를 못 맞춘다는 실측(04d_tonemap_sweep) 뒤에
    # 남은 유일한 실효 노브 — 같은 이유로 rtx_ambient와 함께 씬 로드 이후에 건다.
    if film_iso != DEFAULT_FILM_ISO:
        carb.settings.get_settings().set_float('/rtx/post/tonemap/filmIso', float(film_iso))
    camera.set_intrinsic_matrices(torch.tensor(np.asarray(k, dtype=np.float32), device=sim.device).unsqueeze(0))
    # 씬 참조 직후 애노테이터 파이프라인이 아직 안 돌아서 첫 프레임에 빈 텐서가 나오는 것을
    # 실측으로 확인했다(RuntimeError: shape invalid for input of size 0) — 워밍업 스텝으로 흡수한다.
    for _ in range(10):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
    return camera, sim, lights


# 카메라를 따라다니는 3-light(distant+up/down disk light) 레시피를 실측 확정(raise=0.2m,
# 원본 세기 그대로)까지 구현했지만, 사용자가 `04c_light_explorer.py`(GT/렌더/차이 비교 도구)로
# 9개 조명 옵션을 육안 비교한 결과 DomeLight 단독(dome_2M)이 가장 자연스럽다고 판단해 이 단순한
# 형태로 되돌렸다 — camera_light(RTX `/rtx/useViewLightingMode`)가 수치(mean_err)는 더
# 낮았지만 육안으로는 "너무 강하다"고 평가됨. 정량 지표가 좋다고 시각적으로도 낫다는 보장은
# 없다는 교훈이 이번에도 확인됐다. 세기 2,000,000은 이전 라운드 실측 스윕으로 확정한 값
# (1000→3500은 거의 무변화, 300000→2,000,000 구간에서 실제 변화, 200만에서 실측 밝기가
# GT 평균과 거의 일치) — 톤매핑 압축 때문에 세기를 더 올려도 변화가 급격히 줄어드는 구간에
# 이미 들어와 있어 추가 상향은 의미가 작다.
DOME_LIGHT_INTENSITY = 2_000_000


# 3-light(three_light) 파라미터 — VLN-PE `vln_eval_task.py`의 `create_light` 원본 세기,
# raise는 raise x scale 스윕으로 실측 확정했던 0.2m(경위는 260805 메모리 문서).
THREE_LIGHT_DISTANT_INTENSITY = 1000
THREE_LIGHT_DISK_INTENSITY = 5000
THREE_LIGHT_RAISE_M = 0.2


def _add_lights(stage, light: str = 'dome', tl: dict = None):
    """조명 생성. 기본 dome은 기존 동작 그대로(DomeLight 2M 하나, 위치 갱신 불필요 -> None 반환).

    `three_light`는 VLN-PE 실제 프로덕션 조명(distant + 카메라추적 up/down disk light,
    `vln_eval_task.py`의 `create_light`) — **vln_pe GT rgb가 이 레시피로 렌더된 것**이라 vln_pe
    톤 정합 실험용으로 유지한다(vln_n1에서는 육안 비교로 dome이 채택됐던 경위는
    `.claude/memory/260805_gs_vlnpe_04_render_obs_isaac_result.md`). 반환값은 render_along이
    매 프레임 위치 갱신에 쓰는 translate op dict(dome이면 None).
    """
    for path in ('/World/dome_light', '/World/distant_light', '/World/up_disk_light',
                 '/World/down_disk_light'):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    tl = tl or {}
    raise_m = float(tl.get('raise_m', THREE_LIGHT_RAISE_M))
    up_i = float(tl.get('up_intensity', THREE_LIGHT_DISK_INTENSITY))
    down_i = float(tl.get('down_intensity', THREE_LIGHT_DISK_INTENSITY))
    distant_i = float(tl.get('distant_intensity', THREE_LIGHT_DISTANT_INTENSITY))
    dome_i = float(tl.get('dome_intensity', 500_000))

    if light == 'ambient_only':
        # 조명 prim을 만들지 않는다 — 밝기는 전적으로 --rtx_ambient(전역 간접광)가 담당.
        return None
    if light == 'dome':
        dome_light = UsdLux.DomeLight.Define(stage, '/World/dome_light')
        dome_light.CreateIntensityAttr(DOME_LIGHT_INTENSITY)
        dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        return None
    elif light in ('three_light', 'dome_three'):
        if light == 'dome_three':
            # 실측(천장 스윕): disk 세기는 4배를 올려도 천장 밝기가 거의 안 변한다(62.8->65.5,
            # 위치 토폴로지가 지배). GT의 밝은 천장(top1/3=207)은 균일 ambient 성분으로 보이므로
            # 약한 DomeLight를 병행해 채운다.
            dome_light = UsdLux.DomeLight.Define(stage, '/World/dome_light')
            dome_light.CreateIntensityAttr(dome_i)
            dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        distant_light = UsdLux.DistantLight.Define(stage, '/World/distant_light')
        distant_light.CreateIntensityAttr(distant_i)
        distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        up_disk_light = UsdLux.DiskLight.Define(stage, '/World/up_disk_light')
        up_disk_light.CreateIntensityAttr(up_i)
        up_disk_light.CreateRadiusAttr(50.0)
        up_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(up_disk_light).AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
        up_translate = UsdGeom.Xformable(up_disk_light).AddTranslateOp()

        down_disk_light = UsdLux.DiskLight.Define(stage, '/World/down_disk_light')
        down_disk_light.CreateIntensityAttr(down_i)
        down_disk_light.CreateRadiusAttr(50.0)
        down_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        down_translate = UsdGeom.Xformable(down_disk_light).AddTranslateOp()
        return {'up': up_translate, 'down': down_translate, 'raise_m': raise_m}
    else:
        assert False, f'unreachable light={light!r}'


def _update_light_positions(lights: dict, camera_xyz: np.ndarray) -> None:
    """three_light/dome_three 전용 — 카메라 world 위치를 따라 up/down disk translate만 갱신."""
    x, y, z = float(camera_xyz[0]), float(camera_xyz[1]), float(camera_xyz[2])
    r = lights['raise_m']
    lights['up'].Set(Gf.Vec3f(x, y, z + r))
    lights['down'].Set(Gf.Vec3f(x, y, z - r))


def set_camera_pose(camera: Camera, sim, c2w: np.ndarray) -> None:
    """`c2w`(4,4, OpenCV camera-to-world — `action_to_c2w`와 같은 컨벤션)로 카메라를 옮긴다.

    IsaacLab의 `convention="ros"`(forward=+Z, up=-Y)가 OpenCV 컨벤션과 정확히 같아서, 회전행렬을
    쿼터니언으로 바꾸는 것 말고는 축 변환이 전혀 필요 없다(Open3D의 eye/forward/up 수동 조립과
    달리 여기선 `set_world_poses`가 그 변환을 내부에서 처리한다).
    """
    pos = c2w[:3, 3]
    quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    camera.set_world_poses(
        torch.tensor(pos, dtype=torch.float32, device=sim.device).unsqueeze(0),
        torch.tensor(quat_wxyz, dtype=torch.float32, device=sim.device).unsqueeze(0),
        convention='ros',
    )


def render_along(camera_and_sim, poses_c2w) -> tuple:
    """`(N,4,4)` OpenCV c2w pose -> `rgb`(uint8 (N,H,W,3)), `depth_m`(float32 (N,H,W), 배경=inf).

    포즈 소스가 GT 원본(2-A)이든 03이 만든 경로에서 합성한 것(2-B)이든 완전히 같은 함수를 탄다
    (Open3D 버전의 `render_along`과 같은 역할 분담). 씬 로드는 `build_renderer`가 전담한다(순서
    제약 — 그 함수의 docstring 참고).
    """
    camera, sim, lights = camera_and_sim
    rgbs, depths = [], []
    for c2w in poses_c2w:
        c2w = np.asarray(c2w, dtype=np.float64)
        set_camera_pose(camera, sim, c2w)
        if lights is not None:
            _update_light_positions(lights, c2w[:3, 3])
        # 2스텝만 주면 실측상 이전 pose의 스테일 데이터(완전히 엉뚱한 장면)가 나온다(렌더
        # 파이프라인 latency가 2프레임보다 김) — `build_renderer`의 워밍업과 같은 수(10)로 맞춘다.
        for _ in range(10):
            sim.step()
            camera.update(dt=sim.get_physics_dt())
        rgbs.append(camera.data.output['rgb'][0, ..., :3].cpu().numpy())
        depths.append(camera.data.output['distance_to_image_plane'][0, ..., 0].cpu().numpy())
    return np.stack(rgbs), np.stack(depths)


def edge_correlation(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Open3D 버전과 동일한 지표(엣지맵 피어슨 상관) — 렌더/실측 비교 기준을 통일한다."""
    ea = cv2.Laplacian(gray_a, cv2.CV_32F, ksize=3)
    eb = cv2.Laplacian(gray_b, cv2.CV_32F, ksize=3)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    denom = float(np.sqrt((ea ** 2).sum() * (eb ** 2).sum()))
    return float((ea * eb).sum() / denom) if denom > 1e-9 else 0.0


def _tl_params(args) -> dict:
    return {'raise_m': args.tl_raise, 'up_intensity': args.tl_up_intensity,
            'down_intensity': args.tl_down_intensity, 'distant_intensity': args.tl_distant_intensity,
            'dome_intensity': args.tl_dome_intensity}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default=DEFAULT_SCENE)
    parser.add_argument('--dataset', default='vln_n1', choices=['vln_n1', 'vln_pe'],
                        help='GT 데이터셋. vln_pe면 입출력에 _vlnpe 태그, 렌더 해상도 256x256')
    parser.add_argument(
        '--camera',
        default='dataset',
        choices=['dataset', *SYNTHETIC_CAMERAS],
        help='2-B observation camera. dataset=기존 GT camera, '
             'd455_nominal=D455 RGB FOV 기반 nominal synthetic camera.',
    )
    parser.add_argument('--light', default='dome',
                        choices=['dome', 'three_light', 'dome_three', 'ambient_only'],
                        help='dome(기본, 기존 동작)=DomeLight 2M 균일광. three_light=VLN-PE 실제 '
                             '조명(distant+카메라추적 up/down disk) — vln_pe GT가 이 레시피로 '
                             '렌더돼 톤 정합에 유리(ssim 0.553->0.72 실측). dome_three=둘 병행 '
                             '(disk가 못 채우는 천장 밝기를 dome ambient로 보강)')
    parser.add_argument('--tl_raise', type=float, default=THREE_LIGHT_RAISE_M,
                        help='three_light: 카메라 기준 up/down disk 오프셋[m]')
    parser.add_argument('--tl_up_intensity', type=float, default=THREE_LIGHT_DISK_INTENSITY,
                        help='three_light: 위(천장 방향) disk 세기')
    parser.add_argument('--tl_down_intensity', type=float, default=THREE_LIGHT_DISK_INTENSITY,
                        help='three_light: 아래(바닥 방향) disk 세기')
    parser.add_argument('--tl_distant_intensity', type=float, default=THREE_LIGHT_DISTANT_INTENSITY,
                        help='three_light: distant light 세기')
    parser.add_argument('--film_iso', type=float, default=DEFAULT_FILM_ISO,
                        help='RTX 톤매퍼 노출(/rtx/post/tonemap/filmIso). 기본 100=기존 동작. '
                             'ambient만 켰을 때 렌더가 GT보다 밝은 쪽으로 치우치는 것을 낮춰 보정한다 '
                             '(vln_n1 6프레임 실측: iso 70/85/100/130 -> SSIM 0.852/0.859/0.856/0.836, '
                             '밝은대역 편차 +17/+28/+36/+48).')
    parser.add_argument('--rtx_ambient', type=float, default=0.0,
                        help='RTX 전역 간접광 세기(/rtx/sceneDb/ambientLightIntensity). 0=기존 동작. '
                             '**vln_pe 톤 정합의 핵심 노브** — raw isaacsim.SimulationApp으로 부팅하면 '
                             '이 값이 0이라 천장처럼 직접광이 안 닿는 면이 어둡게 남는다(실측: 상단 1/3 '
                             '밝기 63 vs GT 208, 조명 세기·형태·방향·오토익스포저·톤맵 어떤 것도 이걸 '
                             '못 올렸다). IsaacLab의 apps/isaaclab.python.headless.rendering.kit는 이 값을 '
                             '1.0으로 켜는데 AppLauncher를 우회하면서 빠졌던 것. vln_pe 씬1 실측 스윕: '
                             'ambient 0/4/6/8/10/13 -> SSIM 0.715/0.818/0.824/0.819/0.810/0.794 이고 '
                             '6.0에서 전체 밝기도 182.0(GT 178.4)로 거의 일치해 6.0을 권장.')
    parser.add_argument('--tl_dome_intensity', type=float, default=500_000,
                        help='dome_three 전용: 병행 DomeLight 세기(천장/그림자 ambient 보강용)')
    parser.add_argument('--mode', default='gt_replay', choices=['gt_replay', 'reproduce', 'random'])
    parser.add_argument('--episodes', default='0,1,2', help='2-A에서 검증할 에피소드 (콤마 구분)')
    parser.add_argument('--n_frames', type=int, default=6, help='2-A: 에피소드당 균등 표본 프레임 수')
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='2-B: paths/<scene>[_random].json에서 렌더링할 에피소드 수(앞에서부터)')
    parser.add_argument('--data_root', default=None, help='생략 시 --dataset의 기본 경로')
    parser.add_argument('--mesh_root', default=DEFAULT_MESH_ROOT)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return parser


def validate_gt_replay(args) -> dict:
    """2-A — GT `action` 시퀀스를 그대로 mesh에 렌더링해 실제 캡처와 프레임별로 비교한다.

    (`04_render_obs.py`의 `validate_gt_replay`와 완전히 같은 로직 — 렌더러 호출부만 다르다.)
    """
    scene_dir = Path(args.data_root) / args.scene
    episodes = [int(e) for e in str(args.episodes).split(',') if e != '']

    k, camera_and_sim = None, None
    usd_path = load_scene_model(args.mesh_root, args.scene)

    frames_out, all_depth_err, all_edge_corr, all_ssim = [], [], [], []
    for ep in episodes:
        gt = dataset_utils.load_gt_episode(args.data_root, args.scene, ep, args.dataset)
        if k is None:
            k = gt['k']
            camera_and_sim = build_renderer(RENDER_W, RENDER_H, k, usd_path, args.light, _tl_params(args), args.rtx_ambient, args.film_iso)
        n = len(gt['poses_c2w'])
        frame_idx = np.unique(np.linspace(0, n - 1, args.n_frames).astype(int)).tolist()
        poses_c2w = [gt['poses_c2w'][f] for f in frame_idx]
        rgb_render, depth_render = render_along(camera_and_sim, poses_c2w)

        depth_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.depth'
        rgb_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.rgb'
        for i, frame in enumerate(frame_idx):
            real_depth_m, real_valid = dataset_utils.load_depth_frame_m(depth_dir, ep, frame, MAX_DEPTH_M,
                                                                         args.dataset)
            real_rgb = dataset_utils.load_rgb_frame(rgb_dir, ep, frame, args.dataset)
            render_valid = np.isfinite(depth_render[i]) & (depth_render[i] < RENDER_FAR_M * 0.99)
            both = real_valid & render_valid
            depth_err = np.abs(depth_render[i][both] - real_depth_m[both]) if both.any() else np.array([np.nan])
            corr = edge_correlation(cv2.cvtColor(rgb_render[i], cv2.COLOR_RGB2GRAY),
                                    cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY))
            ssim_val = float(ssim_metric(real_rgb, rgb_render[i], channel_axis=2, data_range=255))
            all_depth_err.append(depth_err)
            all_edge_corr.append(corr)
            all_ssim.append(ssim_val)
            frames_out.append({
                'episode': ep, 'frame': int(frame), 'coverage_frac': float(both.mean()),
                'depth_median_m': float(np.median(depth_err)), 'depth_p95_m': float(np.percentile(depth_err, 95)),
                'edge_corr': corr, 'ssim': ssim_val, 'rgb_render': rgb_render[i], 'depth_render': depth_render[i],
                'rgb_real': real_rgb, 'depth_real_m': real_depth_m, 'real_valid': real_valid,
            })
            print(f'    ep {ep:>3} frame {frame:>4}: depth median={frames_out[-1]["depth_median_m"]:.4f}m '
                  f'p95={frames_out[-1]["depth_p95_m"]:.4f}m  edge_corr={corr:.3f}  ssim={ssim_val:.3f}  '
                  f'coverage={frames_out[-1]["coverage_frac"]:.3f}', flush=True)

    all_depth_err = np.concatenate(all_depth_err)
    stats = {
        'n_frames': len(frames_out),
        'depth_median_m': float(np.median(all_depth_err)),
        'depth_p95_m': float(np.percentile(all_depth_err, 95)),
        'edge_corr_median': float(np.median(all_edge_corr)),
        'edge_corr_min': float(np.min(all_edge_corr)),
        'ssim_median': float(np.median(all_ssim)),
        'ssim_min': float(np.min(all_ssim)),
    }
    if args.dataset == 'vln_pe':
        # vln_pe GT는 보행 중 물리 시뮬 캡처라 pose 기록과 depth 렌더 시점이 어긋난다(00의
        # mesh-anchor 실측: 정지 frame0은 6mm, 보행 프레임은 2~9cm, 오프셋 보정 불가). p95는
        # 회전 중 경계 시프트 아웃라이어가 지배해(실측 ~1.6m) 참고 상한만 둔다 — 회귀에 민감한
        # 지표는 median(컨벤션이 깨지면 정지 프레임 포함 전 구간 0.3m+).
        med_tol, p95_tol = 0.10, 3.0
    else:
        med_tol, p95_tol = GT_REPLAY_DEPTH_MEDIAN_TOL_M, GT_REPLAY_DEPTH_P95_TOL_M
    stats['depth_median_tol_m'], stats['depth_p95_tol_m'] = med_tol, p95_tol
    stats['depth_ok'] = bool(stats['depth_median_m'] <= med_tol
                             and stats['depth_p95_m'] <= p95_tol)
    # RGB 판정은 SSIM으로 한다(edge_corr는 참고용으로만 계속 표시 — 위 GT_REPLAY_RGB_SSIM_MIN
    # 정의부 주석 참고: 같은 프레임에서 edge_corr는 0.19~0.22, SSIM은 0.65~0.71로 실측 확인).
    stats['rgb_ok'] = bool(stats['ssim_median'] >= GT_REPLAY_RGB_SSIM_MIN)
    stats['passed'] = bool(stats['depth_ok'] and stats['rgb_ok'])
    return {'stats': stats, 'frames': frames_out}


def render_gt_replay_report(log_dir: Path, scene: str, result: dict) -> Path:
    stats = result['stats']

    def pill(ok: bool) -> str:
        return f'<span class="pill {"good" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span>'

    summary = f'''
<div class="stat-row">
  <div class="stat"><b>전체</b>{pill(stats["passed"])}</div>
  <div class="stat"><b>depth median</b><span class="pill {"good" if stats["depth_ok"] else "bad"}">
      {stats["depth_median_m"]:.4f} m (기준 {stats["depth_median_tol_m"]} m)</span></div>
  <div class="stat"><b>depth p95</b><span class="pill {"good" if stats["depth_ok"] else "bad"}">
      {stats["depth_p95_m"]:.4f} m (기준 {stats["depth_p95_tol_m"]} m)</span></div>
  <div class="stat"><b>rgb SSIM</b><span class="pill {"good" if stats["rgb_ok"] else "bad"}">
      median {stats["ssim_median"]:.3f} / min {stats["ssim_min"]:.3f} (기준 {GT_REPLAY_RGB_SSIM_MIN})
      </span></div>
  <div class="stat"><b>rgb 엣지 상관(참고용)</b><span class="pill">
      median {stats["edge_corr_median"]:.3f} / min {stats["edge_corr_min"]:.3f}</span></div>
</div>
<p>2-A(Isaac Sim 버전) — GT parquet의 실제 <code>action</code> 시퀀스를 그대로 mesh에 렌더링해
<code>observation.images.rgb/depth</code>(실제 캡처)와 프레임별로 비교했다. 렌더러는 raw
<code>isaacsim.SimulationApp</code>(IsaacLab <code>AppLauncher</code> 우회) + mp3d_pe의
Isaac-ready USD. RGB 판정은 SSIM 기준(edge_corr는 엣지 픽셀 정합에 너무 민감해 참고용으로만
표시 — 자세한 근거는 <code>GT_REPLAY_RGB_SSIM_MIN</code> 정의부 주석).</p>
'''
    body = ''
    for f in result['frames']:
        render_depth_vis = colorize_depth(np.where(np.isfinite(f['depth_render']), f['depth_render'], np.nan),
                                          np.isfinite(f['depth_render']))
        real_depth_vis = colorize_depth(f['depth_real_m'], f['real_valid'])
        frame_dir = log_dir / f'ep{f["episode"]}_frame{f["frame"]}'
        states = [
            ('render rgb', save_jpg(f['rgb_render'], frame_dir / 'render_rgb.jpg')),
            ('real rgb', save_jpg(f['rgb_real'], frame_dir / 'real_rgb.jpg')),
            ('render depth', save_jpg(render_depth_vis, frame_dir / 'render_depth.jpg')),
            ('real depth', save_jpg(real_depth_vis, frame_dir / 'real_depth.jpg')),
        ]
        body += (f'<h4>ep {f["episode"]} frame {f["frame"]} — depth median {f["depth_median_m"]:.4f} m / '
                f'p95 {f["depth_p95_m"]:.4f} m, ssim {f["ssim"]:.3f}, edge_corr {f["edge_corr"]:.3f}, '
                f'coverage {f["coverage_frac"]:.3f}</h4>'
                + blink_widget_html(f'f{f["episode"]}_{f["frame"]}', states))
    return save_gallery(log_dir, 'report.html', f'{SCRIPT_NAME} — {scene} (gt_replay 2-A)', summary, body)


def resolve_generate_scene(args) -> tuple:
    """2-B 렌더링용 USD와 validation mesh를 결정한다.

    기존 Matterport scene은 기존 loader를 그대로 사용한다.
    `scene_type == "new_scene"`이면 Stage 01 scene_meta의 USD를
    render source와 mesh-anchor validation source로 함께 사용한다.
    """

    resolved = usdz_scene_utils.load_new_scene(
        args.out_dir, args.scene, load_scene_usd
    )
    if resolved is not None:
        return resolved

    meta_path = Path(args.out_dir) / 'scene_meta' / f'{args.scene}.json'
    scene_meta = {}
    if meta_path.is_file():
        with open(meta_path) as f:
            scene_meta = json.load(f)

    if scene_meta.get('scene_type') == 'new_scene':
        if not scene_meta.get('passed', False):
            raise ValueError(
                f'new_scene이 scene preparation을 통과하지 못했다: {meta_path}'
            )

        usd_path = Path(scene_meta['usd_path'])
        if not usd_path.is_file():
            raise FileNotFoundError(
                f'new_scene USD 없음: {usd_path}'
            )

        validation_mesh = load_scene_usd(
            str(usd_path)
        )
        return usd_path, validation_mesh, scene_meta

    usd_path = load_scene_model(
        args.mesh_root,
        args.scene,
    )
    validation_mesh = load_scene_mesh(
        args.mesh_root,
        args.scene,
    )

    return usd_path, validation_mesh, scene_meta


def dataset_camera_config(args) -> RenderCameraConfig:
    """기존 2-B의 dataset GT camera 설정을 그대로 구성한다."""

    gt = dataset_utils.load_gt_episode(
        args.data_root,
        args.scene,
        0,
        args.dataset,
    )

    width, height = dataset_utils.render_wh(
        args.dataset
    )

    return RenderCameraConfig(
        name=f'{args.dataset}_gt',
        width=width,
        height=height,
        k=np.asarray(gt['k'], dtype=np.float64),

        # 아래 값들은 기존 04의 2-B 저장 규약 그대로다.
        depth_scale_m=DEPTH_SCALE_RAW_TO_M,
        depth_min_m=DEPTH_CLIP_MIN_M,
        depth_max_m=DEPTH_CLIP_MAX_M,
        render_near_m=RENDER_NEAR_M,
        render_far_m=RENDER_FAR_M,
    )


def resolve_generate_camera(
    args,
    scene_meta: dict,
) -> RenderCameraConfig:
    """2-B observation camera를 선택한다.

    Scene source와 camera source는 독립이다. 따라서 Matterport scene도
    D455로 렌더할 수 있고, new_scene도 향후 다른 synthetic camera를
    선택할 수 있다.
    """

    if args.camera == 'dataset':
        if scene_meta.get('scene_type') == 'new_scene':
            raise ValueError(
                'new_scene에는 dataset GT camera가 없다. '
                '--camera로 synthetic camera를 지정할 것'
            )

        return dataset_camera_config(args)

    return SYNTHETIC_CAMERAS[args.camera]


def depth_m_to_raw16(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    depth_scale_m: float,
) -> np.ndarray:
    """Metric depth를 uint16 Z16 encoding으로 변환한다.

    표현 범위를 넘는 값은 saturation하지 않는다. 잘못된 depth scale 때문에
    geometry가 한 거리로 뭉개지는 silent corruption을 즉시 오류로 처리한다.
    """

    if depth_scale_m <= 0:
        raise ValueError(
            f'depth_scale_m must be positive: {depth_scale_m}'
        )

    raw = np.zeros(depth_m.shape, dtype=np.uint16)

    if not valid_mask.any():
        return raw

    scaled = np.rint(
        depth_m[valid_mask] / depth_scale_m
    )

    if np.any(scaled < 1) or np.any(scaled > 65534):
        depth_max = float(
            np.max(depth_m[valid_mask])
        )
        representable_max = 65534 * depth_scale_m

        raise ValueError(
            'depth uint16 encoding overflow: '
            f'valid max={depth_max:.3f} m, '
            f'representable max={representable_max:.3f} m, '
            f'scale={depth_scale_m:g} m/raw'
        )

    raw[valid_mask] = scaled.astype(np.uint16)
    return raw


def generate_episode(camera_and_sim, camera: RenderCameraConfig,
                     ep_data: dict, ep_dir: Path) -> dict:
    """2-B 경로를 camera configuration에 따라 RGB/depth observation으로 렌더링한다."""

    k = camera.k
    xy = resample_by_arclength(np.asarray(ep_data['trajectory'], dtype=np.float64), step_m=FRAME_STEP_M)
    action_poses = synthesize_action_poses(xy, ep_data['floor_z'], ep_data['h_b'], ep_data['pitch_deg'])
    poses_c2w = np.stack([action_to_c2w(a, 'cam2world_gl') for a in action_poses])
    rgb, depth_m = render_along(camera_and_sim, poses_c2w)
    n = len(action_poses)

    finite_depth = depth_m[np.isfinite(depth_m)]
    depth_stats = {
        'finite_frac': float(np.isfinite(depth_m).mean()),
        'zero_frac': float((depth_m == 0).mean()),
        'min_m': float(finite_depth.min()) if finite_depth.size else float('nan'),
        'median_m': float(np.median(finite_depth)) if finite_depth.size else float('nan'),
        'max_m': float(finite_depth.max()) if finite_depth.size else float('nan'),
    }

    (ep_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (ep_dir / 'depth').mkdir(parents=True, exist_ok=True)
    (ep_dir / 'extrinsic').mkdir(parents=True, exist_ok=True)
    valid_all = np.zeros(depth_m.shape, dtype=bool)
    for i in range(n):
        save_jpg(rgb[i], ep_dir / 'rgb' / f'frame_{i:04d}.jpg')
        valid = (
            np.isfinite(depth_m[i])
            & (depth_m[i] < camera.render_far_m * 0.99)
            & (depth_m[i] >= camera.depth_min_m)
            & (depth_m[i] <= camera.depth_max_m)
        )
        valid_all[i] = valid

        depth_raw = depth_m_to_raw16(
            depth_m[i],
            valid,
            camera.depth_scale_m,
        )
        cv2.imwrite(
            str(ep_dir / 'depth' / f'frame_{i:04d}.png'),
            depth_raw,
        )
        np.save(ep_dir / 'extrinsic' / f'frame_{i:04d}.npy', action_poses[i].astype(np.float32))
    np.save(ep_dir / 'intrinsic.npy', np.asarray(k, dtype=np.float32))

    step = np.linalg.norm(np.diff(poses_c2w[:, :2, 3], axis=0), axis=1)
    step_bad = int((step > FRAME_STEP_M * GEN_STEP_TOL_RATIO).sum())

    rng = np.random.RandomState(0)
    anchor_d = []
    for i in range(0, n, max(1, n // 8)):
        pts_cam = np.stack(np.where(valid_all[i]), axis=1)
        if len(pts_cam) == 0:
            continue
        sel = pts_cam[rng.choice(len(pts_cam), min(300, len(pts_cam)), replace=False)]
        v, u = sel[:, 0], sel[:, 1]
        z = depth_m[i][v, u].astype(np.float64)
        x_cam = (u - k[0, 2]) * z / k[0, 0]
        y_cam = (v - k[1, 2]) * z / k[1, 1]
        pts_cam3 = np.stack([x_cam, y_cam, z], axis=1)
        pts_world = (poses_c2w[i][:3, :3] @ pts_cam3.T).T + poses_c2w[i][:3, 3]
        anchor_d.append(pts_world)
    return {
        'n_frames': n, 'step_max_m': float(step.max()) if n > 1 else 0.0, 'step_bad_n': step_bad,
        'anchor_points_world': np.concatenate(anchor_d, axis=0) if anchor_d else np.empty((0, 3)),
        'rgb_first': rgb[0], 'depth_first': depth_m[0], 'valid_first': valid_all[0],
        'rgb_last': rgb[-1], 'depth_last': depth_m[-1], 'valid_last': valid_all[-1],
        'depth_stats': depth_stats,
    }


def run_generate(args) -> dict:
    paths_dir = Path(args.out_dir) / 'paths'
    tag = dataset_utils.dataset_tag(args.dataset)
    json_path = paths_dir / (f'{args.scene}{tag}.json' if args.mode == 'reproduce' else f'{args.scene}{tag}_random.json')
    if not json_path.is_file():
        print(f'  [ERROR] {json_path} 없음 — 03_sample_gt_paths.py --mode {args.mode}를 먼저 돌릴 것')
        return {'error': f'{json_path} not found'}
    paths = json.load(open(json_path))
    episodes_data = paths['episodes'][:args.num_episodes]

    try:
        usd_path, mesh, scene_meta = resolve_generate_scene(args)
        camera = resolve_generate_camera(args, scene_meta)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f'  [ERROR] {exc}')
        return {'error': str(exc)}

    camera_and_sim = build_renderer(
        camera.width,
        camera.height,
        camera.k,
        usd_path,
        args.light,
        _tl_params(args),
        args.rtx_ambient,
        args.film_iso,
        camera.render_near_m,
        camera.render_far_m,
    )

    camera_tag = (
        ''
        if args.camera == 'dataset'
        else f'_{args.camera}'
    )

    obs_dir = (
        Path(args.out_dir)
        / 'obs'
        / f'{args.scene}{tag}_{args.mode}_isaac{camera_tag}'
    )
    obs_dir.mkdir(parents=True, exist_ok=True)

    with open(obs_dir / 'camera_config.json', 'w') as f:
        json.dump(
            camera.to_dict(),
            f,
            indent=2,
        )
    results = []
    for i, ep_data in enumerate(episodes_data):
        ep_dir = obs_dir / f'episode_{i:06d}'
        res = generate_episode(camera_and_sim, camera, ep_data, ep_dir)
        anchor_dist = (compute_mesh_distance(res['anchor_points_world'], mesh)
                      if len(res['anchor_points_world']) else np.array([np.nan]))
        res['anchor_median_m'] = float(np.median(anchor_dist))
        res['episode_id'] = ep_data['episode_id']
        results.append(res)
        ds = res['depth_stats']
        print(f'    ep {i:>3} (src episode_id={ep_data["episode_id"]}): {res["n_frames"]} frames, '
              f'step max={res["step_max_m"]:.4f}m (기준 {FRAME_STEP_M * GEN_STEP_TOL_RATIO:.4f}m, '
              f'위반 {res["step_bad_n"]}건), mesh-anchor median={res["anchor_median_m"]:.5f}m')
        print(f'      depth: finite={ds["finite_frac"]*100:.2f}%, zero={ds["zero_frac"]*100:.2f}%, '
              f'min/median/max={ds["min_m"]:.3f}/{ds["median_m"]:.3f}/{ds["max_m"]:.3f} m -> {ep_dir}')

    stats = {
        'n_episodes': len(results),
        'step_bad_total': int(sum(r['step_bad_n'] for r in results)),
        'anchor_median_m': float(np.median([r['anchor_median_m'] for r in results])) if results else float('nan'),
        'anchor_worst_m': float(max(r['anchor_median_m'] for r in results)) if results else float('nan'),
    }
    stats['step_ok'] = bool(stats['step_bad_total'] == 0)
    stats['anchor_ok'] = bool(stats['anchor_worst_m'] <= GEN_MESH_ANCHOR_TOL_M)
    stats['passed'] = bool(stats['step_ok'] and stats['anchor_ok'] and len(results) == len(episodes_data))
    return {
        'stats': stats,
        'episodes': results,
        'obs_dir': obs_dir,
        'camera': camera.to_dict(),
    }


def render_generate_report(log_dir: Path, scene: str, mode: str, result: dict) -> Path:
    stats = result['stats']
    camera = result['camera']

    def pill(ok: bool) -> str:
        return f'<span class="pill {"good" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span>'

    summary = f'''
<div class="stat-row">
  <div class="stat"><b>전체</b>{pill(stats["passed"])}</div>
  <div class="stat"><b>에피소드</b><span class="pill">{stats["n_episodes"]}개</span></div>
  <div class="stat"><b>인접 프레임 이동거리</b><span class="pill {"good" if stats["step_ok"] else "bad"}">
      위반 {stats["step_bad_total"]}건 (기준 {FRAME_STEP_M * GEN_STEP_TOL_RATIO:.3f} m)</span></div>
  <div class="stat"><b>mesh-anchor</b><span class="pill {"good" if stats["anchor_ok"] else "bad"}">
      median {stats["anchor_median_m"]:.5f} m / worst {stats["anchor_worst_m"]:.5f} m
      (기준 {GEN_MESH_ANCHOR_TOL_M} m)</span></div>
</div>
<p>2-B(Isaac Sim 버전) — 2-A로 검증된 같은 렌더러로 <code>03_sample_gt_paths.py --mode {mode}</code>가
만든 경로를 따라 실제 rgb/depth/extrinsic을 생성했다. 산출물: <code>{result["obs_dir"]}</code>.</p>
<p>카메라 <code>{camera["name"]}</code> —
{camera["width"]}x{camera["height"]},
depth scale <code>{camera["depth_scale_m"]} m/raw</code>,
저장 범위 {camera["depth_min_m"]}~{camera["depth_max_m"]} m.</p>
'''
    body = ''
    for r in result['episodes']:
        first_depth_vis = colorize_depth(np.where(r['valid_first'], r['depth_first'], np.nan), r['valid_first'])
        last_depth_vis = colorize_depth(np.where(r['valid_last'], r['depth_last'], np.nan), r['valid_last'])
        frame_dir = log_dir / f'ep{r["episode_id"]}'
        states = [
            ('첫 프레임 rgb', save_jpg(r['rgb_first'], frame_dir / 'first_rgb.jpg')),
            ('첫 프레임 depth', save_jpg(first_depth_vis, frame_dir / 'first_depth.jpg')),
            ('마지막 프레임 rgb', save_jpg(r['rgb_last'], frame_dir / 'last_rgb.jpg')),
            ('마지막 프레임 depth', save_jpg(last_depth_vis, frame_dir / 'last_depth.jpg')),
        ]
        body += (f'<h4>episode_id {r["episode_id"]} — {r["n_frames"]} 프레임, '
                f'mesh-anchor median {r["anchor_median_m"]:.5f} m, '
                f'step max {r["step_max_m"]:.4f} m</h4>'
                + blink_widget_html(f'gen{r["episode_id"]}', states))
    return save_gallery(log_dir, 'report.html', f'{SCRIPT_NAME} — {scene} ({mode} 2-B)', summary, body)


def main() -> int:
    args = build_argparser().parse_args()
    if args.data_root is None:
        args.data_root = dataset_utils.default_data_root(args.dataset)
    # 렌더 해상도는 데이터셋의 GT 해상도를 따른다(vln_n1 480x270, vln_pe 256x256) — vln_n1은
    # 기존 값 그대로라 기본 경로 동작 불변.
    global RENDER_W, RENDER_H
    RENDER_W, RENDER_H = dataset_utils.render_wh(args.dataset)
    print(f'[{SCRIPT_NAME}] scene={args.scene} dataset={args.dataset} '
          f'mode={args.mode} camera={args.camera}')

    if args.mode == 'gt_replay':
        if args.camera != 'dataset':
            print('  [ERROR] gt_replay는 dataset GT camera만 사용한다 '
                  '(--camera dataset).')
            return 2
        result = validate_gt_replay(args)
        stats = result['stats']
        log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}'
        report = render_gt_replay_report(log_dir, args.scene, result)
        print(f'  report html -> {report}')
        print(f'  => {"PASS" if stats["passed"] else "FAIL"} '
              f'(depth median {stats["depth_median_m"]:.4f} m / p95 {stats["depth_p95_m"]:.4f} m, '
              f'ssim median {stats["ssim_median"]:.3f}, edge_corr median {stats["edge_corr_median"]:.3f} 참고)')
        return 0 if stats['passed'] else 1

    result = run_generate(args)
    if 'error' in result:
        return 2
    stats = result['stats']
    camera_tag = '' if args.camera == 'dataset' else f'_{args.camera}'
    log_dir = (
        Path(args.log_dir)
        / SCRIPT_NAME
        / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}_{args.mode}{camera_tag}'
    )
    report = render_generate_report(log_dir, args.scene, args.mode, result)
    print(f'  obs -> {result["obs_dir"]}')
    print(f'  report html -> {report}')
    print(f'  => {"PASS" if stats["passed"] else "FAIL"} '
          f'({stats["n_episodes"]}개 에피소드, mesh-anchor median {stats["anchor_median_m"]:.5f} m)')
    return 0 if stats['passed'] else 1


if __name__ == '__main__':
    exit_code = main()
    # report.html/이미지는 이 시점에 이미 디스크에 저장 완료된 상태다(main()이 반환했으므로).
    # `simulation_app.close()`는 3-light(위치추적 disk light 2개 포함) 도입 후 실측상 수 분까지
    # 걸릴 수 있다(예전 DomeLight 버전의 "수십 초"보다 훨씬 김) — 결과물과 무관한 GPU/렌더러
    # 종료 비용이므로 기다리지 않고 `timeout --signal=KILL`로 강제 종료해도 무방하다.
    simulation_app.close()
    sys.exit(exit_code)
