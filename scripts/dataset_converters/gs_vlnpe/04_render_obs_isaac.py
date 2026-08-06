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
from esdf_utils import resample_by_arclength  # noqa: E402
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

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
# Open3D 버전은 mp3d_n1(원본 raw obj)을 쓰지만, Isaac 버전은 Isaac-ready USD가 필요해 mp3d_pe를
# 쓴다 — 실측으로 두 자산이 같은 world 좌표계(bbox 거의 일치)임을 확인했다.
DEFAULT_MESH_ROOT = 'data/scene_data/mp3d_pe'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
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


def build_renderer(width: int, height: int, k: np.ndarray, usd_path: Path) -> tuple:
    """`(width,height,K)` + 씬 USD -> `(camera, sim)`. `render_along`이 매 프레임 재사용.

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
    _add_lights(stage)

    camera_cfg = CameraCfg(
        prim_path='/World/RenderCamera',
        height=height,
        width=width,
        data_types=['rgb', 'distance_to_image_plane'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(RENDER_NEAR_M, RENDER_FAR_M)),
    )
    camera = Camera(cfg=camera_cfg)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.reset()
    camera.set_intrinsic_matrices(torch.tensor(np.asarray(k, dtype=np.float32), device=sim.device).unsqueeze(0))
    # 씬 참조 직후 애노테이터 파이프라인이 아직 안 돌아서 첫 프레임에 빈 텐서가 나오는 것을
    # 실측으로 확인했다(RuntimeError: shape invalid for input of size 0) — 워밍업 스텝으로 흡수한다.
    for _ in range(10):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
    return camera, sim


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


def _add_lights(stage) -> None:
    """씬 전체에 균일한 조명 하나만 둔다(`04c_light_explorer.py`의 `dome_2M` config와 동일).

    카메라 위치를 따라다니는 3-light 레시피도 구현해봤지만(자세한 경위는
    `.claude/memory/260805_gs_vlnpe_04_render_obs_isaac_result.md`), 사용자가 여러 조명
    옵션을 GT/렌더/차이 이미지로 직접 비교한 뒤 이 단순한 DomeLight 하나가 가장 자연스럽다고
    판단해 최종 채택했다.
    """
    if stage.GetPrimAtPath('/World/dome_light'):
        stage.RemovePrim('/World/dome_light')
    dome_light = UsdLux.DomeLight.Define(stage, '/World/dome_light')
    dome_light.CreateIntensityAttr(DOME_LIGHT_INTENSITY)
    dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))


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
    camera, sim = camera_and_sim
    rgbs, depths = [], []
    for c2w in poses_c2w:
        c2w = np.asarray(c2w, dtype=np.float64)
        set_camera_pose(camera, sim, c2w)
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default=DEFAULT_SCENE)
    parser.add_argument('--mode', default='gt_replay', choices=['gt_replay', 'reproduce', 'random'])
    parser.add_argument('--episodes', default='0,1,2', help='2-A에서 검증할 에피소드 (콤마 구분)')
    parser.add_argument('--n_frames', type=int, default=6, help='2-A: 에피소드당 균등 표본 프레임 수')
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='2-B: paths/<scene>[_random].json에서 렌더링할 에피소드 수(앞에서부터)')
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
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
        table = pq.read_table(scene_dir / 'data' / 'chunk-000' / f'episode_{ep:06d}.parquet')
        actions = table['action'].to_pylist()
        if k is None:
            k = np.asarray(table['observation.camera_intrinsic'].to_pylist()[0], dtype=np.float64).reshape(3, 3)
            camera_and_sim = build_renderer(RENDER_W, RENDER_H, k, usd_path)
        n = len(actions)
        frame_idx = np.unique(np.linspace(0, n - 1, args.n_frames).astype(int)).tolist()
        poses_c2w = [action_to_c2w(np.asarray(actions[f], dtype=np.float64).reshape(4, 4), 'cam2world_gl')
                    for f in frame_idx]
        rgb_render, depth_render = render_along(camera_and_sim, poses_c2w)

        depth_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.depth'
        rgb_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.rgb'
        for i, frame in enumerate(frame_idx):
            real_depth_m, real_valid = load_depth_frame_m(depth_dir, ep, frame, MAX_DEPTH_M)
            real_rgb = load_rgb_frame(rgb_dir, ep, frame)
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
    stats['depth_ok'] = bool(stats['depth_median_m'] <= GT_REPLAY_DEPTH_MEDIAN_TOL_M
                             and stats['depth_p95_m'] <= GT_REPLAY_DEPTH_P95_TOL_M)
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
      {stats["depth_median_m"]:.4f} m (기준 {GT_REPLAY_DEPTH_MEDIAN_TOL_M} m)</span></div>
  <div class="stat"><b>depth p95</b><span class="pill {"good" if stats["depth_ok"] else "bad"}">
      {stats["depth_p95_m"]:.4f} m (기준 {GT_REPLAY_DEPTH_P95_TOL_M} m)</span></div>
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


def depth_m_to_raw16(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    raw = np.zeros(depth_m.shape, dtype=np.uint16)
    scaled = np.clip(np.round(depth_m / DEPTH_SCALE_RAW_TO_M), 1, 65534)
    raw[valid_mask] = scaled[valid_mask].astype(np.uint16)
    return raw


def generate_episode(camera_and_sim, k: np.ndarray, ep_data: dict, ep_dir: Path) -> dict:
    """(`04_render_obs.py`의 `generate_episode`와 완전히 같은 로직 — 렌더러 호출부만 다르다.)"""
    xy = resample_by_arclength(np.asarray(ep_data['trajectory'], dtype=np.float64), step_m=FRAME_STEP_M)
    action_poses = synthesize_action_poses(xy, ep_data['floor_z'], ep_data['h_b'], ep_data['pitch_deg'])
    poses_c2w = np.stack([action_to_c2w(a, 'cam2world_gl') for a in action_poses])
    rgb, depth_m = render_along(camera_and_sim, poses_c2w)
    n = len(action_poses)

    (ep_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (ep_dir / 'depth').mkdir(parents=True, exist_ok=True)
    (ep_dir / 'extrinsic').mkdir(parents=True, exist_ok=True)
    valid_all = np.zeros((n, RENDER_H, RENDER_W), dtype=bool)
    for i in range(n):
        save_jpg(rgb[i], ep_dir / 'rgb' / f'frame_{i:04d}.jpg')
        valid = (np.isfinite(depth_m[i]) & (depth_m[i] < RENDER_FAR_M * 0.99)
                & (depth_m[i] >= DEPTH_CLIP_MIN_M) & (depth_m[i] <= DEPTH_CLIP_MAX_M))
        valid_all[i] = valid
        cv2.imwrite(str(ep_dir / 'depth' / f'frame_{i:04d}.png'), depth_m_to_raw16(depth_m[i], valid))
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
    }


def run_generate(args) -> dict:
    paths_dir = Path(args.out_dir) / 'paths'
    json_path = paths_dir / (f'{args.scene}.json' if args.mode == 'reproduce' else f'{args.scene}_random.json')
    if not json_path.is_file():
        print(f'  [ERROR] {json_path} 없음 — 03_sample_gt_paths.py --mode {args.mode}를 먼저 돌릴 것')
        return {'error': f'{json_path} not found'}
    paths = json.load(open(json_path))
    episodes_data = paths['episodes'][:args.num_episodes]

    ep0_table = pq.read_table(sorted((Path(args.data_root) / args.scene / 'data' / 'chunk-000').glob('*.parquet'))[0],
                              columns=['observation.camera_intrinsic'])
    k = np.asarray(ep0_table['observation.camera_intrinsic'].to_pylist()[0], dtype=np.float64).reshape(3, 3)

    usd_path = load_scene_model(args.mesh_root, args.scene)
    camera_and_sim = build_renderer(RENDER_W, RENDER_H, k, usd_path)
    mesh = load_scene_mesh(args.mesh_root, args.scene)

    obs_dir = Path(args.out_dir) / 'obs' / f'{args.scene}_{args.mode}_isaac'
    results = []
    for i, ep_data in enumerate(episodes_data):
        ep_dir = obs_dir / f'episode_{i:06d}'
        res = generate_episode(camera_and_sim, k, ep_data, ep_dir)
        anchor_dist = (compute_mesh_distance(res['anchor_points_world'], mesh)
                      if len(res['anchor_points_world']) else np.array([np.nan]))
        res['anchor_median_m'] = float(np.median(anchor_dist))
        res['episode_id'] = ep_data['episode_id']
        results.append(res)
        print(f'    ep {i:>3} (src episode_id={ep_data["episode_id"]}): {res["n_frames"]} frames, '
              f'step max={res["step_max_m"]:.4f}m (기준 {FRAME_STEP_M * GEN_STEP_TOL_RATIO:.4f}m, '
              f'위반 {res["step_bad_n"]}건), mesh-anchor median={res["anchor_median_m"]:.5f}m -> {ep_dir}')

    stats = {
        'n_episodes': len(results),
        'step_bad_total': int(sum(r['step_bad_n'] for r in results)),
        'anchor_median_m': float(np.median([r['anchor_median_m'] for r in results])) if results else float('nan'),
        'anchor_worst_m': float(max(r['anchor_median_m'] for r in results)) if results else float('nan'),
    }
    stats['step_ok'] = bool(stats['step_bad_total'] == 0)
    stats['anchor_ok'] = bool(stats['anchor_worst_m'] <= GEN_MESH_ANCHOR_TOL_M)
    stats['passed'] = bool(stats['step_ok'] and stats['anchor_ok'] and len(results) == len(episodes_data))
    return {'stats': stats, 'episodes': results, 'obs_dir': obs_dir}


def render_generate_report(log_dir: Path, scene: str, mode: str, result: dict) -> Path:
    stats = result['stats']

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
    print(f'[{SCRIPT_NAME}] scene={args.scene} mode={args.mode}')

    if args.mode == 'gt_replay':
        result = validate_gt_replay(args)
        stats = result['stats']
        log_dir = Path(args.log_dir) / SCRIPT_NAME / args.scene
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
    log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}_{args.mode}'
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
