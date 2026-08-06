"""M1.4 부속 — Isaac Sim light 옵션 비교 탐색기.

여러 라운드의 조명 튜닝(DomeLight 세기 스윕 -> 3-light 카메라추적 raise/scale 스윕)에도
`04_render_obs_isaac.py`의 RGB 엣지 상관계수가 기준(0.5) 미달로 남아, 사용자가 "수치만 보고
추측하는" 반복 튜닝 대신 **GT/렌더/차이 이미지를 눈으로 비교**할 수 있는 도구를 요청했다.
이 스크립트는 `04_render_obs_isaac.py`(공식 gt_replay 검증)는 건드리지 않고, mesh 로드·pose
변환·시각화 유틸을 그대로 import해서 재사용하는 **조명 비교 전용 신규 스크립트**다(가이드라인
"새 기능은 새 파일" 원칙).

## 비교 대상 (LIGHT_CONFIGS)

- `dome_2M`: DomeLight 단독(이전 확정값, intensity=2,000,000)
- `isaaclab_default_dome3000`: IsaacLab 튜토리얼 관용 기본값(`DomeLightCfg(intensity=3000)`)
- `gui_default_distant3000`: Isaac Sim GUI가 새 스테이지에 자동으로 넣는 "sunlight" 템플릿
  기본값(`DistantLight(intensity=3000, angle=1.0)`) — `omni.kit.stage_templates`
  (`isaacsim.exp.full.kit`에만 포함, 우리 headless 베이스 익스피리언스엔 없음)에서 실측 확인.
- `three_light_raiseX`: VLN-PE 원본 레시피(distant+up/down disk light)를 카메라 추적으로
  구현한 버전 — raise(카메라 기준 위/아래 오프셋) 0.1/0.2/0.3/0.5 스윕. 0.2가
  `04_render_obs_isaac.py`의 현재 확정값.
- `camera_light`: Isaac Sim 뷰포트의 "Camera Light"(헤드램프) 모드. GUI 프리뷰 전용이 아니라
  진짜 RTX 렌더 설정 `/rtx/useViewLightingMode`(bool) — `omni.usd.schema.render_settings.rtx`의
  `OmniRtxDebugSettingsAPI`로 실제 렌더 패스에 반영되므로 headless에서도 `camera.data.
  output['rgb']`에 그대로 적용된다(카메라 위치에서 카메라가 보는 방향으로 비추는 라이트).
  스테이지의 다른 라이트를 무시하는 경향이 있다는 문서 힌트가 있어, 단독(`camera_light_only`)과
  기존 3-light(raise0.2) 위에 얹은 병행(`camera_light_plus_three_light`) 둘 다 비교한다.

## 실행

```
timeout --signal=KILL 900 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04c_light_explorer.py --scene 17DRP5sb8fy
```
(`04_render_obs_isaac.py`와 동일하게 raw `isaacsim.SimulationApp` 사용 — `AppLauncher` 우회,
`simulation_app.close()`가 수 분 걸릴 수 있어 `timeout --signal=KILL`로 감싼다. report.html은
`main()` 반환 시점에 이미 저장 완료.)
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': True})

import carb  # noqa: E402

carb.settings.get_settings().set_bool('/isaaclab/cameras_enabled', True)
carb.settings.get_settings().set_bool('/rtx/post/histogram/enabled', False)

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
import omni.usd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_utils import action_to_c2w, load_rgb_frame, save_jpg  # noqa: E402
from viz_utils import blink_widget_html, save_gallery  # noqa: E402

# `04_render_obs_isaac.py`를 통째로 import하지 않는다 — 그 파일도 최상위에서
# `SimulationApp({'headless': True})`를 만들기 때문에(이미 이 파일 위에서 하나 띄웠다), import시
# 두 번째 SimulationApp을 만들려다 충돌/행이 난다. 대신 필요한 상수·헬퍼 3개만 그대로 복제한다
# (전부 몇 줄짜리라 가이드라인상 신규 파일 중복 허용 범위).
DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_MESH_ROOT = 'data/scene_data/mp3d_pe'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
DEFAULT_SCENE = '17DRP5sb8fy'
RENDER_W, RENDER_H = 480, 270
RENDER_NEAR_M, RENDER_FAR_M = 0.05, 10.0
MATTERPORT_TEXTURE_ROOT = Path('/ssd/share/Matterport3D/data/v1/scans')
SCRIPT_NAME = '04c_light_explorer'


def ensure_texture_symlink(mesh_root, scene: str) -> None:
    """`04_render_obs_isaac.py`의 동명 함수와 동일 — mp3d_pe USD의 텍스처 절대경로를 로컬로 심링크."""
    target = MATTERPORT_TEXTURE_ROOT / scene / 'matterport_mesh'
    if target.is_symlink() or target.exists():
        return
    src = Path(mesh_root).resolve() / scene / 'matterport_mesh'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(src)


def find_scene_usd(mesh_root, scene: str) -> Path:
    mesh_dir = Path(mesh_root) / scene / 'matterport_mesh'
    candidates = sorted(p for p in mesh_dir.glob('*/isaacsim_*.usd') if '_non_metric' not in p.name)
    if not candidates:
        raise FileNotFoundError(f'Isaac USD(isaacsim_*.usd) not found under {mesh_dir}')
    return candidates[0]


def load_scene_model(mesh_root, scene: str) -> Path:
    ensure_texture_symlink(mesh_root, scene)
    return find_scene_usd(mesh_root, scene)

# 서로 다른 방을 지나는 7프레임 — raise x scale 스윕과 동일(연속성 유지, 국소 편차 비교 목적).
TARGET_FRAMES = [(0, 0), (0, 47), (0, 95), (0, 142), (0, 190), (0, 238), (1, 14)]

# 3-light(distant+up/down disk light) 원본 세기 — VLN-PE `vln_eval_task.py`의 `create_light`
# 값 그대로(재조정 무의미함을 이전 라운드 스윕으로 확인).
THREE_LIGHT_DISTANT_INTENSITY = 1000
THREE_LIGHT_DISK_INTENSITY = 5000

# 하드코딩 방지 — 비교할 조합을 전부 여기 한 곳에 모은다. 각 원소는
# {'label': str, 'kind': 'dome'|'distant'|'three_light'|'camera_light', ...kind별 파라미터}.
LIGHT_CONFIGS = [
    {'label': 'dome_2M', 'kind': 'dome', 'intensity': 2_000_000},
    {'label': 'isaaclab_default_dome3000', 'kind': 'dome', 'intensity': 3000},
    {'label': 'gui_default_distant3000', 'kind': 'distant', 'intensity': 3000, 'angle': 1.0},
    {'label': 'three_light_raise0.1', 'kind': 'three_light', 'raise_m': 0.1},
    {'label': 'three_light_raise0.2', 'kind': 'three_light', 'raise_m': 0.2},
    {'label': 'three_light_raise0.3', 'kind': 'three_light', 'raise_m': 0.3},
    {'label': 'three_light_raise0.5', 'kind': 'three_light', 'raise_m': 0.5},
    {'label': 'camera_light_only', 'kind': 'camera_light', 'with_three_light_raise_m': None},
    {'label': 'camera_light_plus_three_light', 'kind': 'camera_light', 'with_three_light_raise_m': 0.2},
]

_LIGHT_PATHS = ('/World/dome_light', '/World/distant_light', '/World/up_disk_light', '/World/down_disk_light',
               '/World/headlamp_light')

# OpenCV(forward=+Z, up=-Y) -> USD/UsdLux(forward=-Z, up=+Y) 회전 변환. `geometry_utils.
# CAM_CV_TO_GL`(diag(1,-1,-1,1))과 정확히 같은 변환(이미 mesh 표면거리 0.00003m로 실측 검증됨) —
# 카메라뿐 아니라 raw로 방향을 설정하는 조명 prim에도 그대로 적용된다(둘 다 "local -Z가 정면,
# local +Y가 위"라는 같은 USD 컨벤션을 따르므로).
_CAM_CV_TO_GL_ROT = np.diag([1.0, -1.0, -1.0])


def _clear_lights(stage) -> None:
    for path in _LIGHT_PATHS:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    carb.settings.get_settings().set_bool('/rtx/useViewLightingMode', False)


def _spawn_dome(stage, intensity: float) -> None:
    dome = UsdLux.DomeLight.Define(stage, '/World/dome_light')
    dome.CreateIntensityAttr(float(intensity))
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))


def _spawn_distant(stage, intensity: float, angle: float) -> None:
    distant = UsdLux.DistantLight.Define(stage, '/World/distant_light')
    distant.CreateIntensityAttr(float(intensity))
    distant.CreateAngleAttr(float(angle))
    distant.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))


def _spawn_three_light(stage) -> dict:
    """`04_render_obs_isaac.py`의 `_add_lights`와 동일한 3-light 레시피(위치는 매 프레임 갱신)."""
    distant_light = UsdLux.DistantLight.Define(stage, '/World/distant_light')
    distant_light.CreateIntensityAttr(THREE_LIGHT_DISTANT_INTENSITY)
    distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    up_disk_light = UsdLux.DiskLight.Define(stage, '/World/up_disk_light')
    up_disk_light.CreateIntensityAttr(THREE_LIGHT_DISK_INTENSITY)
    up_disk_light.CreateRadiusAttr(50.0)
    up_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    UsdGeom.Xformable(up_disk_light).AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
    up_translate = UsdGeom.Xformable(up_disk_light).AddTranslateOp()

    down_disk_light = UsdLux.DiskLight.Define(stage, '/World/down_disk_light')
    down_disk_light.CreateIntensityAttr(THREE_LIGHT_DISK_INTENSITY)
    down_disk_light.CreateRadiusAttr(50.0)
    down_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    down_translate = UsdGeom.Xformable(down_disk_light).AddTranslateOp()

    return {'up': up_translate, 'down': down_translate}


def _update_three_light_positions(lights: dict, camera_xyz: np.ndarray, raise_m: float) -> None:
    x, y, z = float(camera_xyz[0]), float(camera_xyz[1]), float(camera_xyz[2])
    lights['up'].Set(Gf.Vec3f(x, y, z + raise_m))
    lights['down'].Set(Gf.Vec3f(x, y, z - raise_m))


def _spawn_headlamp(stage, radius: float = 0.15) -> dict:
    """카메라를 따라다니는 작은 DiskLight 하나 — 세기/높이(raise)/아래로 기울이는 각도(pitch)를
    직접 조절 가능(RTX 내장 `/rtx/useViewLightingMode`는 이 세 값을 전혀 못 건드린다)."""
    light = UsdLux.DiskLight.Define(stage, '/World/headlamp_light')
    light.CreateRadiusAttr(radius)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    intensity_attr = light.CreateIntensityAttr(1.0)
    # 순서 중요 — `_spawn_three_light`(회전 op를 translate op보다 먼저 추가, 실측으로 이미
    # 검증된 패턴)와 같은 순서로 맞춘다. 반대로 하면(translate를 먼저 추가) USD가 로컬 좌표를
    # "회전 후 이동"이 아니라 "이동 후 회전"으로 합성해, translate로 넣은 카메라 world 위치
    # 자체가 orient 행렬에 의해 다시 회전당해 엉뚱한 곳으로 어긋난다(실측: pitch를 ±30도
    # 바꿔도 렌더 결과가 거의 똑같았던 원인 — 위치 오차가 pitch 변화보다 훨씬 지배적이었다).
    orient_op = UsdGeom.Xformable(light).AddOrientOp()
    translate_op = UsdGeom.Xformable(light).AddTranslateOp()
    return {'intensity_attr': intensity_attr, 'translate': translate_op, 'orient': orient_op}


def _update_headlamp(headlamp: dict, c2w: np.ndarray, raise_m: float, pitch_deg: float) -> None:
    """카메라 위치보다 `raise_m`만큼 world +Z로 띄우고, 카메라가 보는 방향을 `pitch_deg`만큼
    (양수=위, 음수=아래) 추가로 기울인 방향을 비춘다.

    **부호는 유도만 하고 아직 실측 검증 전이다** — `_CAM_CV_TO_GL_ROT`로 카메라 forward(+Z, OpenCV)를
    조명의 기본 정면(-Z, USD)에 맞춘 뒤, 조명 로컬 X축 기준 `pitch_deg` 회전을 그 뒤에 곱한다.
    로컬 +Y가 "위"에 대응함을 대수적으로 확인했으므로 양의 pitch는 위로 향해야 하지만, 실제
    렌더 결과(바닥 vs 천장 밝기)로 최종 확인할 것 — 이 세션 내내 반복된 원칙("좌표 부호는 실측
    으로 검증").
    """
    pos = c2w[:3, 3] + np.array([0.0, 0.0, raise_m])
    headlamp['translate'].Set(Gf.Vec3f(*[float(v) for v in pos]))
    base_rot = c2w[:3, :3] @ _CAM_CV_TO_GL_ROT
    pitch_rot = Rotation.from_euler('x', pitch_deg, degrees=True).as_matrix()
    final_rot = base_rot @ pitch_rot
    quat_xyzw = Rotation.from_matrix(final_rot).as_quat()
    headlamp['orient'].Set(Gf.Quatf(float(quat_xyzw[3]),
                                    Gf.Vec3f(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))))


def apply_light_config(stage, config: dict) -> dict:
    """`config`에 맞는 light prim을 새로 만들고, 매 프레임 위치 갱신에 필요한 상태를 반환한다.

    반환값 `state`는 `kind`에 따라 다르다:
    - 'three_light' 계열: {'kind': 'three_light', 'lights': {...}, 'raise_m': float}
    - 그 외: {'kind': config['kind']} (위치 갱신 불필요)
    """
    _clear_lights(stage)
    kind = config['kind']
    if kind == 'dome':
        _spawn_dome(stage, config['intensity'])
        return {'kind': 'dome'}
    elif kind == 'distant':
        _spawn_distant(stage, config['intensity'], config['angle'])
        return {'kind': 'distant'}
    elif kind == 'three_light':
        lights = _spawn_three_light(stage)
        return {'kind': 'three_light', 'lights': lights, 'raise_m': config['raise_m']}
    elif kind == 'camera_light':
        carb.settings.get_settings().set_bool('/rtx/useViewLightingMode', True)
        raise_m = config['with_three_light_raise_m']
        if raise_m is None:
            return {'kind': 'camera_light'}
        lights = _spawn_three_light(stage)
        return {'kind': 'camera_light', 'lights': lights, 'raise_m': raise_m}
    elif kind == 'headlamp':
        headlamp = _spawn_headlamp(stage)
        headlamp['intensity_attr'].Set(float(config['intensity']))
        return {'kind': 'headlamp', 'headlamp': headlamp, 'raise_m': config['raise_m'],
                'pitch_deg': config['pitch_deg']}
    else:
        assert False, f'unreachable kind={kind!r}'


def update_light_state_for_pose(state: dict, c2w: np.ndarray) -> None:
    if state['kind'] == 'three_light' or (state['kind'] == 'camera_light' and 'lights' in state):
        _update_three_light_positions(state['lights'], c2w[:3, 3], state['raise_m'])
    elif state['kind'] == 'headlamp':
        _update_headlamp(state['headlamp'], c2w, state['raise_m'], state['pitch_deg'])


def colorize_diff(render_rgb: np.ndarray, real_rgb: np.ndarray, max_diff: float = 255.0) -> np.ndarray:
    """render/real 간 픽셀별 절대차이(채널 평균)를 컬러맵으로 시각화.

    `max_diff`를 config 전체에서 고정해야 밝기가 조합마다 다른 컬러맵으로 왜곡되지 않고
    공정하게 비교된다(예: 항상 0~255 전체 범위로 정규화).
    """
    diff = np.abs(render_rgb.astype(np.float32) - real_rgb.astype(np.float32)).mean(axis=-1)
    norm = np.clip(diff / max_diff, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    colormap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
    color_bgr = cv2.applyColorMap(u8, colormap)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def make_composite(real_rgb: np.ndarray, render_rgb: np.ndarray, diff_rgb: np.ndarray, label: str) -> np.ndarray:
    """[GT | Render | Diff] 가로 합성 + 상단 라벨 텍스트."""
    pad = 28
    h, w = real_rgb.shape[:2]
    panels = [real_rgb, render_rgb, diff_rgb]
    titles = ['GT', f'Render ({label})', 'Diff']
    canvas = np.zeros((h + pad, w * 3, 3), dtype=np.uint8)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        canvas[pad:, i * w:(i + 1) * w] = panel
        cv2.putText(canvas, title, (i * w + 6, pad - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default=DEFAULT_SCENE)
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--mesh_root', default=DEFAULT_MESH_ROOT)
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    print(f'[{SCRIPT_NAME}] scene={args.scene}')

    scene_dir = Path(args.data_root) / args.scene
    usd_path = load_scene_model(args.mesh_root, args.scene)

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath('/World/Scene'):
        stage.RemovePrim('/World/Scene')
    cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path),
                               collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.func('/World/Scene', cfg)

    tables = {}
    real_rgbs, c2ws = {}, {}
    k = None
    for ep, frame in TARGET_FRAMES:
        if ep not in tables:
            tables[ep] = pq.read_table(scene_dir / 'data/chunk-000' / f'episode_{ep:06d}.parquet')
        table = tables[ep]
        if k is None:
            k = np.asarray(table['observation.camera_intrinsic'].to_pylist()[0], dtype=np.float64).reshape(3, 3)
        actions = table['action'].to_pylist()
        c2ws[(ep, frame)] = action_to_c2w(np.asarray(actions[frame], dtype=np.float64).reshape(4, 4), 'cam2world_gl')
        real_rgbs[(ep, frame)] = load_rgb_frame(scene_dir / 'videos/chunk-000/observation.images.rgb', ep, frame)

    camera_cfg = CameraCfg(
        prim_path='/World/RenderCamera', height=RENDER_H, width=RENDER_W,
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(RENDER_NEAR_M, RENDER_FAR_M)),
    )
    camera = Camera(cfg=camera_cfg)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.reset()
    camera.set_intrinsic_matrices(torch.tensor(np.asarray(k, dtype=np.float32), device=sim.device).unsqueeze(0))
    for _ in range(10):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    log_dir = Path(args.log_dir) / SCRIPT_NAME / args.scene
    # frame -> [(config_label, composite_jpg_path), ...] — blink_widget_html의 states로 그대로 쓴다.
    frame_states = {t: [] for t in TARGET_FRAMES}

    for config in LIGHT_CONFIGS:
        label = config['label']
        state = apply_light_config(stage, config)
        print(f'  config={label}', flush=True)
        for t in TARGET_FRAMES:
            ep, frame = t
            c2w = c2ws[t]
            pos = c2w[:3, 3]
            quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            camera.set_world_poses(
                torch.tensor(pos, dtype=torch.float32, device=sim.device).unsqueeze(0),
                torch.tensor(quat_wxyz, dtype=torch.float32, device=sim.device).unsqueeze(0),
                convention='ros',
            )
            update_light_state_for_pose(state, c2w)
            for _ in range(10):
                sim.step()
                camera.update(dt=sim.get_physics_dt())
            render_rgb = camera.data.output['rgb'][0, ..., :3].cpu().numpy()
            real_rgb = real_rgbs[t]
            diff_rgb = colorize_diff(render_rgb, real_rgb)
            composite = make_composite(real_rgb, render_rgb, diff_rgb, label)
            mean_err = float(np.abs(render_rgb.astype(np.float32) - real_rgb.astype(np.float32)).mean())
            composite_path = save_jpg(composite, log_dir / f'ep{ep}_frame{frame}' / f'{label}.jpg')
            frame_states[t].append((f'{label} (mean_err={mean_err:.1f})', composite_path))
            print(f'    ep{ep} frame{frame}: mean_err={mean_err:.1f}', flush=True)

    body = ''
    for t in TARGET_FRAMES:
        ep, frame = t
        body += f'<h4>ep {ep} frame {frame}</h4>' + blink_widget_html(f'f{ep}_{frame}', frame_states[t])
    summary = (f'<p>{len(LIGHT_CONFIGS)}개 light 설정 x {len(TARGET_FRAMES)}개 프레임(서로 다른 방 포함) '
              f'— 화살표(&lsaquo; &rsaquo;)로 설정을 전환하며 GT/렌더/차이(diff, 색이 진할수록 GT와 큰 차이)를 '
              f'비교한다. mean_err는 픽셀당 RGB 채널 평균 절대차(0~255, 작을수록 GT에 가까움).</p>')
    report = save_gallery(log_dir, 'report.html', f'{SCRIPT_NAME} — {args.scene} (light 옵션 비교)', summary, body)
    print(f'  report html -> {report}')
    return 0


if __name__ == '__main__':
    exit_code = main()
    # report.html/이미지는 이 시점에 이미 저장 완료 — `simulation_app.close()`는 조명 종류가
    # 많을수록(특히 disk light) 수 분 걸릴 수 있어(04_render_obs_isaac.py에서 실측 확인) 기다릴
    # 필요 없이 `timeout --signal=KILL`로 강제 종료해도 결과물 손실이 없다.
    simulation_app.close()
    sys.exit(exit_code)
