"""S7 — Isaac Sim을 **GUI로 띄우고 RGB/depth를 실시간으로 본다**.

S1~S6은 전부 headless(화면 없이)로 돌려 결과를 파일로만 봤다. 이 스크립트는 창을 띄운다.

- **Isaac Sim 창**: 씬을 3D로 직접 돌려볼 수 있다 (마우스로 이동/회전)
- **별도 창 하나**: 카메라 센서의 **RGB와 depth를 나란히** 실시간 표시

원본 USD를 열 수도 있고, 내가 만든 **덧칠 레이어를 열 수도 있다** — 같은 카메라 경로로
번갈아 띄워보면 덧칠이 뭘 바꾸는지 눈으로 확인된다.

## 왜 이 파일이 따로 있나 (기존 렌더러를 재사용 못 하는 이유)

`04_render_obs_isaac.py`는 **import하는 순간** `SimulationApp({'headless': True})`을 부른다.
GUI로 먼저 띄운 뒤 그 모듈을 import하면 **프로세스가 그대로 죽는다**(실측: 예외도 아니고
즉시 종료). 그래서 씬 로드·조명·카메라 생성을 여기서 직접 한다 —
다만 **순서는 그 파일의 `build_renderer`와 똑같이** 맞췄다(mesh 참조 -> 조명 -> 카메라 ->
`sim.reset()`). 그 순서를 바꾸면 멈추거나 빈 텐서가 나온다고 그 파일에 기록돼 있다.

## 창은 **두 개의 터미널**로 띄운다

`cv2.imshow`를 Isaac과 같은 프로세스에서 부르면 **한 프레임 그리고 죽는다**(실측, headless로도 같음).
Qt 충돌이라 피할 방법이 없어서 역할을 나눴다:

    터미널 1 : s7_live_view.py   렌더해서 최신 프레임을 파일로 흘려보낸다
    터미널 2 : s7_watch.py       그 파일을 읽어 창에 띄운다 (Isaac을 안 쓴다)

## 실행 (한 줄)

노은역 원본:

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s7_live_view.py --usd data/GS_USDZ/Subway/noeun_station_collision.usdz --poses json --paths_json scripts/dataset_converters/gs_vlnpe/apply_real/paths/noeun_station_mid_random.json --camera d455_nominal

같은 씬을 **덧칠 레이어로** (충돌 메시가 보이게 된 상태):

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s7_live_view.py --usd logs/gs-vlnpe/usd_study/s6_use_layers/b_noeun/override_visibility.usda --poses json --paths_json scripts/dataset_converters/gs_vlnpe/apply_real/paths/noeun_station_mid_random.json --camera d455_nominal

vln_pe 씬:

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s7_live_view.py --usd data/scene_data/mp3d_pe/17DRP5sb8fy/matterport_mesh/bed1a77d92d64f5cbbaaae4feed64ec1/isaacsim_bed1a77d92d64f5cbbaaae4feed64ec1.usd --poses gt --scene 17DRP5sb8fy --dataset vln_pe --rtx_ambient 10.0

## 창 조작 — **끌 때는 ESC 또는 q 를 쓸 것**

Isaac 창을 X로 닫으면 Kit이 자기 방식으로 종료하면서 세그폴트를 낸다(결과에는 영향 없지만
화면이 실패처럼 보인다). ESC/q 로 끄면 그 경로를 안 탄다. Ctrl+C 도 막아 두었다.

    ESC / q   끝내기  <- 이걸로 끄세요
    SPACE     멈춤 / 재생
    n / p     (멈춘 상태에서) 다음 / 이전 프레임
    s         지금 화면을 파일로 저장

## 주의

**GUI 모드에서는 `print`가 터미널에 안 나온다**(Kit이 stdout을 가져간다, 실측).
그래서 진행 상황은 창 위에 글자로 얹고, 로그 파일에도 남긴다.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

GS_VLNPE = Path(__file__).resolve().parent.parent


def parse_args():
    """**`SimulationApp`보다 먼저** 인자를 읽어야 headless 여부를 정할 수 있다."""
    ap = argparse.ArgumentParser(description='S7 — Isaac Sim GUI로 RGB/depth 실시간 보기')
    ap.add_argument('--usd', required=True, help='열 USD/USDZ 또는 내가 만든 덧칠 레이어(.usda)')
    ap.add_argument('--poses', choices=['gt', 'json'], default='json', help='카메라 경로를 어디서 가져올지')
    ap.add_argument('--paths_json', default=str(GS_VLNPE / 'apply_real' / 'paths' / 'noeun_station_mid_random.json'),
                    help='[--poses json] 카메라 경로 json')
    ap.add_argument('--camera', default='d455_nominal', help='[--poses json] 카메라 프로파일')
    ap.add_argument('--scene', default='17DRP5sb8fy', help='[--poses gt] 씬 이름')
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_pe', help='[--poses gt]')
    ap.add_argument('--data_root', default=None, help='[--poses gt] 궤적 폴더')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--light', default='ambient_only',
                    choices=['ambient_only', 'dome', 'three_light', 'dome_three'],
                    help='조명 레시피. 실행 중 control.json 으로도 바꿀 수 있다')
    ap.add_argument('--rtx_ambient', type=float, default=10.0, help='전역 간접광 세기')
    ap.add_argument('--film_iso', type=float, default=70.0, help='노출')
    ap.add_argument('--fps', type=float, default=6.0, help='자동 재생 속도')
    ap.add_argument('--steps_per_frame', type=int, default=10,
                    help='프레임당 sim.step 횟수 — 줄이면 빨라지지만 이전 pose가 남을 수 있다')
    ap.add_argument('--headless', action='store_true', help='Isaac 창 없이 (그래도 RGB/depth 창은 뜬다)')
    ap.add_argument('--save_only', action='store_true', help='프레임을 번호별로 저장하고 끝낸다 (동작 확인용)')
    ap.add_argument('--display', choices=['stream', 'window'], default='stream',
                    help='stream=최신 프레임을 파일로 흘려보낸다(권장, s7_watch.py로 본다) / '
                         'window=이 프로세스에서 직접 창을 띄운다 — **Isaac과 Qt가 충돌해 죽는다**(실측)')
    ap.add_argument('--out_dir', default='logs/gs-vlnpe/usd_study/s7_live_view')
    ap.add_argument('--max_frames', type=int, default=0, help='0이면 무한 반복')
    ap.add_argument('--start_frame', type=int, default=0, help='몇 번째 pose에서 시작할지')
    ap.add_argument('--hold', action='store_true',
                    help='카메라를 움직이지 않고 --start_frame 한 장면에 고정 (자세히 볼 때)')
    return ap.parse_args()


ARGS = parse_args()

# ---------------------------------------------------------------------------
# 여기부터 Isaac — **반드시 다른 무엇보다 먼저 부팅한다**
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({'headless': bool(ARGS.headless)})

import carb  # noqa: E402

# `isaaclab.sensors.camera.Camera`는 이 플래그가 꺼져 있으면 즉시 RuntimeError를 낸다.
# `AppLauncher`를 거칠 때만 자동으로 켜지므로 여기서 직접 켠다(04_render_obs_isaac.py와 동일).
carb.settings.get_settings().set_bool('/isaaclab/cameras_enabled', True)
carb.settings.get_settings().set_bool('/rtx/post/histogram/enabled', False)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from pxr import UsdGeom  # noqa: E402

sys.path.insert(0, str(GS_VLNPE))
sys.path.insert(0, str(GS_VLNPE / 'apply_real'))

# --- 조명 레시피 상수 — `04_render_obs_isaac.py`의 값을 그대로 옮긴 것 ---
# 그 모듈을 import하면 headless로 또 부팅해 프로세스가 죽어서 재사용할 수 없다(파일 상단 설명).
# 값을 바꿀 일이 생기면 **두 곳을 같이** 고쳐야 한다.
DOME_LIGHT_INTENSITY = 2_000_000
THREE_LIGHT_DISTANT_INTENSITY = 1000
THREE_LIGHT_DISK_INTENSITY = 5000
THREE_LIGHT_RAISE_M = 0.2
DOME_THREE_DOME_INTENSITY = 500_000
LIGHT_CHOICES = ('ambient_only', 'dome', 'three_light', 'dome_three')

# 재질(OmniPBR) 입력 이름 — S5에서 실측으로 확인한 것.
# `diffuse_color_constant`는 diffuse_texture가 있으면 무시된다(S5 실측) → 여기서는 참고용.
MAT_ALBEDO = 'diffuse_color_constant'
MAT_ROUGH = 'reflection_roughness_constant'
MAT_TEX = 'diffuse_texture'
MAT_TEX_SCALE = 'texture_scale'        # float2 — 타일링 배율
MAT_TEX_ROT = 'texture_rotate'         # float  — 회전(도)
MAT_TEX_TRANS = 'texture_translate'    # float2 — 이동


OUT_DIR = Path(ARGS.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / 'live_view.log'


def log(msg: str):
    """GUI 모드에서는 print가 터미널에 안 나오므로 파일에도 남긴다."""
    print(msg, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def load_poses_and_camera():
    """카메라 경로와 렌즈 설정을 준비한다. 반환: (poses_c2w, k, W, H, near, far)"""
    if ARGS.poses == 'gt':
        import dataset_utils

        root = ARGS.data_root or dataset_utils.default_data_root(ARGS.dataset)
        gt = dataset_utils.load_gt_episode(root, ARGS.scene, ARGS.episode, ARGS.dataset)
        W, H = dataset_utils.render_wh(ARGS.dataset)
        return list(gt['poses_c2w']), np.asarray(gt['k'], dtype=np.float32), W, H, 0.05, 10.0

    import camera_profiles
    import geometry_utils

    paths = json.loads(Path(ARGS.paths_json).read_text(encoding='utf-8'))
    ep = paths['episodes'][ARGS.episode]
    traj = np.asarray(ep['trajectory'], dtype=np.float64)[:, :2]
    action_poses = geometry_utils.synthesize_action_poses(traj, float(ep['floor_z']), float(ep['h_b']),
                                                          float(ep['pitch_deg']))
    # **이 변환을 빠뜨리면 그림이 180° 뒤집힌다.** `synthesize_action_poses`는 GT parquet의
    # `action` 포맷을 만들고, 렌더러는 OpenCV c2w를 받는다. 두 규약은 X축 기준 180° 다르다
    # (실측: right 내적 +1.0, down/forward 내적 -1.0). 파이프라인
    # (`apply_real/04_render_obs_isaac.py:683`)도 정확히 이 두 단계를 거친다.
    poses = [geometry_utils.action_to_c2w(a, 'cam2world_gl') for a in action_poses]
    prof = camera_profiles.get(ARGS.camera)
    return list(poses), np.asarray(prof.k, dtype=np.float32), prof.width, prof.height, \
        prof.render_near_m, prof.render_far_m


def build_scene(usd_path: Path, k, width, height, near, far):
    """씬 로드 -> 조명 -> 카메라 -> reset. **순서는 04_render_obs_isaac.build_renderer와 동일.**"""
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath('/World/Scene'):
        stage.RemovePrim('/World/Scene')

    cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path),
                               collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.func('/World/Scene', cfg)

    # 원본 usdz를 직접 열었을 때만, 숨은 충돌 메시를 런타임에서 보이게 한다(원본은 안 건드린다).
    # 덧칠 레이어(.usda)를 열었다면 이 함수는 이름 검사에서 빠져나가므로 아무 일도 안 한다.
    import usdz_scene_utils

    changed = usdz_scene_utils.expose_collision_meshes_for_rendering(stage, str(usd_path), UsdGeom)
    log(f'[s7] 런타임 visibility 뒤집기: {changed}개 (덧칠 레이어를 열었다면 0이 정상)')

    # 조명은 카메라 **전에** 만든다 — `build_renderer`의 성공 순서(mesh -> 조명 -> 카메라 -> reset).
    lights = rebuild_lights(stage, ARGS.light)
    log(f'[s7] 조명 {ARGS.light} 생성 (three_light 계열이면 카메라를 따라감: {lights is not None})')

    camera = Camera(cfg=CameraCfg(
        prim_path='/World/RenderCamera', height=height, width=width,
        data_types=['rgb', 'distance_to_image_plane'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(near, far))))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.reset()

    # 전역 간접광과 노출은 **씬이 올라온 뒤에** 걸어야 먹는다(04_render_obs_isaac.py의 실측 주석).
    if ARGS.rtx_ambient > 0:
        carb.settings.get_settings().set_float('/rtx/sceneDb/ambientLightIntensity', float(ARGS.rtx_ambient))
    carb.settings.get_settings().set_float('/rtx/post/tonemap/filmIso', float(ARGS.film_iso))

    camera.set_intrinsic_matrices(torch.tensor(k, dtype=torch.float32, device=sim.device).unsqueeze(0))
    for _ in range(10):
        sim.step()
    shaders = collect_shaders(stage)
    log(f'[s7] shader {len(shaders)}개 (재질 실시간 조절 대상)')
    return camera, sim, stage, lights, shaders


def rebuild_lights(stage, light: str, dome_i=None, distant_i=None, disk_i=None, raise_m=None):
    """조명 prim을 다시 만든다. 반환: three_light 계열이면 위치 갱신용 translate op dict.

    **`04_render_obs_isaac._add_lights`와 같은 레시피**다(세기·반경·회전 전부 동일).
    실시간으로 조명을 바꾸려면 기존 prim을 지우고 다시 만들어야 하므로 여기서 매번 호출한다.
    """
    from pxr import Gf, UsdLux

    for path in ('/World/dome_light', '/World/distant_light', '/World/up_disk_light',
                 '/World/down_disk_light'):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)

    dome_i = DOME_THREE_DOME_INTENSITY if dome_i is None else float(dome_i)
    distant_i = THREE_LIGHT_DISTANT_INTENSITY if distant_i is None else float(distant_i)
    disk_i = THREE_LIGHT_DISK_INTENSITY if disk_i is None else float(disk_i)
    raise_m = THREE_LIGHT_RAISE_M if raise_m is None else float(raise_m)

    if light == 'ambient_only':
        return None                      # 조명 prim 없음 — 밝기는 전적으로 rtx_ambient가 담당
    if light == 'dome':
        d = UsdLux.DomeLight.Define(stage, '/World/dome_light')
        d.CreateIntensityAttr(DOME_LIGHT_INTENSITY)
        d.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        return None
    if light in ('three_light', 'dome_three'):
        if light == 'dome_three':
            d = UsdLux.DomeLight.Define(stage, '/World/dome_light')
            d.CreateIntensityAttr(dome_i)
            d.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        dl = UsdLux.DistantLight.Define(stage, '/World/distant_light')
        dl.CreateIntensityAttr(distant_i)
        dl.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        up = UsdLux.DiskLight.Define(stage, '/World/up_disk_light')
        up.CreateIntensityAttr(disk_i)
        up.CreateRadiusAttr(50.0)
        up.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(up).AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
        up_t = UsdGeom.Xformable(up).AddTranslateOp()
        dn = UsdLux.DiskLight.Define(stage, '/World/down_disk_light')
        dn.CreateIntensityAttr(disk_i)
        dn.CreateRadiusAttr(50.0)
        dn.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        dn_t = UsdGeom.Xformable(dn).AddTranslateOp()
        return {'up': up_t, 'down': dn_t, 'raise_m': raise_m}
    assert False, f'unreachable light={light!r}'


def move_lights(lights, camera_xyz):
    """three_light 계열의 up/down disk를 카메라 위치로 따라가게 한다(레시피 동일)."""
    from pxr import Gf

    if not lights:
        return
    x, y, z = (float(v) for v in camera_xyz[:3])
    r = lights['raise_m']
    lights['up'].Set(Gf.Vec3f(x, y, z + r))
    lights['down'].Set(Gf.Vec3f(x, y, z - r))


def collect_shaders(stage):
    """스폰된 씬 안의 Shader prim 목록. 재질을 실시간으로 바꾸려면 매번 이걸 순회한다."""
    return [p for p in stage.Traverse() if p.GetTypeName() == 'Shader']


def valid_input(sh, name):
    """`sh.GetInput(name)`은 없는 입력에도 **`None`이 아니라 무효 객체**를 준다.

    그래서 `if inp is not None:` 로 검사하면 통과해 버리고, 그 뒤 `GetAttr().GetName()`에서
    `RuntimeError: Accessed invalid attribute '' on null prim`으로 죽는다(실측).
    pxr 게터 전반이 이렇다 — **`is not None`이 아니라 유효성으로 검사해야 한다.**
    """
    inp = sh.GetInput(name)
    try:
        return inp if inp and inp.GetAttr().IsValid() else None
    except Exception:  # noqa: BLE001 - 무효 객체는 접근 자체가 예외를 던질 수 있다
        return None


def snapshot_material(shaders) -> dict:
    """시작 시 재질 원래 값을 전부 기억한다 — 제어 파일에 `null`을 주면 되돌리기 위해.

    **`null`은 "그냥 안 건드림"이 아니라 "원래대로"여야 한다.** 처음엔 값이 있을 때만 적용하고
    `null`은 무시했는데, 그러면 한 번 준 `texture_scale`이 계속 남아 원본으로 복귀가 안 됐다
    (실측: 복귀 후에도 원본과 평균차 33). 저작돼 있지 않던 입력은 `None`으로 기억해 두고,
    복귀 시 그 입력을 지운다.
    """
    from pxr import UsdShade

    snap = {}
    for prim in shaders:
        sh = UsdShade.Shader(prim)
        rec = {}
        for name in (MAT_TEX, MAT_ALBEDO, MAT_ROUGH, MAT_TEX_SCALE, MAT_TEX_ROT, MAT_TEX_TRANS):
            inp = valid_input(sh, name)
            rec[name] = None if inp is None else inp.Get()
        tex_inp = valid_input(sh, MAT_TEX)
        rec['colorSpace'] = tex_inp.GetAttr().GetColorSpace() if tex_inp is not None else None
        snap[str(prim.GetPath())] = rec
    return snap


def _as_vec2(v):
    """숫자 하나면 (v, v), 리스트면 그대로 2개로."""
    if isinstance(v, (int, float)):
        return (float(v), float(v))
    return (float(v[0]), float(v[1]))


def apply_material(shaders, state: dict, snap: dict) -> dict:
    """런타임 스테이지의 재질/텍스처를 바꾼다. **원본 USD는 안 건드린다**(스테이지만).

    각 키는 값을 주면 적용하고 **`null`을 주면 시작 시점 값으로 되돌린다.**

    S5에서 확인한 사실:
      - 이 씬은 MDL `OmniPBR`이고 `diffuse_texture`가 붙어 있으면 `diffuse_color_constant`
        (albedo)는 **무시된다**
      - `colorSpace`는 크게 먹는다(`raw`로 바꾸면 SSIM 0.869 -> 0.728)
      - `reflection_roughness_constant`는 저작돼 있지 않아 새로 만들어야 한다
    """
    from pxr import Sdf, UsdShade

    n = {k: 0 for k in ('texture', 'colorspace', 'albedo', 'roughness',
                        'tex_scale', 'tex_rotate', 'tex_translate')}
    tex = state.get('texture')
    if tex is not None and not Path(str(tex)).is_file():
        return dict(n, error=f'texture 파일 없음: {tex}')
    tex_abs = str(Path(str(tex)).resolve()) if tex is not None else None

    # (제어키, USD 입력이름, 타입, 변환) — 값 적용과 복귀를 같은 표로 처리한다
    plan = [('albedo', MAT_ALBEDO, Sdf.ValueTypeNames.Color3f, lambda v: (float(v),) * 3, 'albedo'),
            ('roughness', MAT_ROUGH, Sdf.ValueTypeNames.Float, float, 'roughness'),
            ('texture_scale', MAT_TEX_SCALE, Sdf.ValueTypeNames.Float2, _as_vec2, 'tex_scale'),
            ('texture_rotate', MAT_TEX_ROT, Sdf.ValueTypeNames.Float, float, 'tex_rotate'),
            ('texture_translate', MAT_TEX_TRANS, Sdf.ValueTypeNames.Float2, _as_vec2, 'tex_translate')]

    import time as _t

    t0 = _t.time()
    for idx, prim in enumerate(shaders):
        if idx % 8 == 0:
            log(f'[s7]   재질 적용 중 {idx}/{len(shaders)} ({_t.time() - t0:.1f}s)')
        sh = UsdShade.Shader(prim)
        rec = snap.get(str(prim.GetPath()), {})

        for key, name, vtype, conv, counter in plan:
            if key not in state:
                continue
            val = state[key]
            if val is not None:
                inp = valid_input(sh, name) or sh.CreateInput(name, vtype)
                inp.Set(conv(val))
                n[counter] += 1
            else:
                orig = rec.get(name)
                inp = valid_input(sh, name)
                if orig is not None:
                    inp = inp or sh.CreateInput(name, vtype)
                    inp.Set(orig)
                    n[counter] += 1
                elif inp is not None:
                    # 원래 저작돼 있지 않던 입력 — 지워서 MDL 기본값으로 되돌린다
                    prim.RemoveProperty(inp.GetAttr().GetName())
                    n[counter] += 1

        tex_inp = valid_input(sh, MAT_TEX)
        if tex_inp is not None:
            if 'texture' in state:
                if tex_abs is not None:
                    tex_inp.Set(Sdf.AssetPath(tex_abs))
                    n['texture'] += 1
                elif rec.get(MAT_TEX) is not None:
                    tex_inp.Set(rec[MAT_TEX])
                    n['texture'] += 1
            if 'colorspace' in state:
                cs = state['colorspace']
                tex_inp.GetAttr().SetColorSpace(str(cs) if cs is not None else (rec.get('colorSpace') or ''))
                n['colorspace'] += 1
    log(f'[s7]   재질 적용 완료 ({_t.time() - t0:.1f}s)')
    return n


def make_checker(path: Path, size=512, cells=8) -> Path:
    """UV 매핑을 눈으로 보기 위한 체커보드 텍스처를 만들어 둔다(실습 편의).

    실제 텍스처로 바꿔보기 전에, **텍스처가 어디에 어떻게 붙는지**를 먼저 보는 데 쓴다.
    """
    import cv2
    import numpy as np

    if path.is_file():
        return path
    step = size // cells
    img = np.zeros((size, size, 3), np.uint8)
    for r in range(cells):
        for c in range(cells):
            if (r + c) % 2 == 0:
                img[r * step:(r + 1) * step, c * step:(c + 1) * step] = (235, 235, 235)
            else:
                img[r * step:(r + 1) * step, c * step:(c + 1) * step] = (40, 90, 200)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def read_control(path: Path, state: dict) -> tuple:
    """제어 파일을 읽어 state를 갱신한다. 반환: (바뀐 키 목록, 에러문자열|None)

    **왜 파일인가**: 창은 별도 프로세스(`s7_watch.py`)가 띄우므로 키 입력을 이 프로세스로
    보낼 수 없다. 그래서 작은 json 파일을 매 프레임 읽는다 — 터미널에서 `echo`로 고치면
    다음 프레임에 바로 반영된다.
    """
    if not path.is_file():
        return [], None
    try:
        new = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001 - 편집 중이면 깨진 json을 읽을 수 있다
        return [], f'{type(exc).__name__}: {exc}'
    changed = [k for k, v in new.items() if state.get(k) != v]
    state.update(new)
    return changed, None


def set_pose(camera, sim, c2w):
    """OpenCV c2w -> Isaac 카메라 pose.

    **회전 변환은 `04_render_obs_isaac.set_camera_pose`와 같은 방식(scipy)을 쓴다.**
    처음엔 행렬->쿼터니언을 손으로 구현했는데, 검증된 코드가 있는데 직접 쓸 이유가 없다
    (부호/순서를 한 군데만 틀려도 그림이 뒤집히고, 그걸 눈으로 구분하기 어렵다).

    IsaacLab의 `convention='ros'`(forward=+Z, up=-Y)가 OpenCV와 같은 축이라 축 변환은 없다.
    """
    from scipy.spatial.transform import Rotation

    c2w = np.asarray(c2w, dtype=np.float64)
    quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    camera.set_world_poses(
        torch.tensor(c2w[:3, 3], dtype=torch.float32, device=sim.device).unsqueeze(0),
        torch.tensor(quat_wxyz, dtype=torch.float32, device=sim.device).unsqueeze(0),
        convention='ros')


def depth_to_color(d, far_m):
    """깊이 -> 볼 수 있는 그림. **깊이가 없는 픽셀은 검정**(그게 이 실습의 관전 포인트다)."""
    valid = np.isfinite(d) & (d < far_m * 0.99)
    norm = np.zeros(d.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        norm = (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    col = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    col[~valid] = 0
    return col, float(valid.mean())


def compose(rgb, depth_col, text_lines):
    """RGB | depth 를 나란히 붙이고 위에 글자를 얹는다."""
    panel = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_col])
    bar = np.zeros((26 * len(text_lines) + 8, panel.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(text_lines):
        cv2.putText(bar, line, (10, 20 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def install_clean_exit():
    """Ctrl+C 로 끌 때도 Isaac 정리 단계를 건너뛰게 한다.

    Kit이 종료를 주도하면 `atexit` -> carb 플러그인 정리에서 **세그폴트가 난다**(실측).
    일이 다 끝난 뒤라 결과에는 영향이 없지만 화면이 실패처럼 보인다. `os._exit()`은
    그 경로를 통째로 건너뛴다.

    **Isaac 창을 X로 닫는 경우는 못 막는다** — Kit이 먼저 죽어서 파이썬으로 제어가 안 온다.
    그래서 끌 때는 RGB/depth 창에서 ESC 또는 q 를 쓰는 것을 권한다.
    """
    import signal

    def bye(signum, frame):
        log(f'[s7] 신호 {signum} 받고 종료')
        sys.stdout.flush()
        os._exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, bye)
        except Exception:  # noqa: BLE001 - 일부 환경에서 핸들러 등록이 막혀 있을 수 있다
            pass


def main():
    install_clean_exit()
    usd_path = Path(ARGS.usd)
    if not usd_path.is_file():
        log(f'[s7] 파일이 없다: {usd_path}')
        sys.stdout.flush()
        os._exit(2)
    log(f'[s7] 여는 파일: {usd_path}  ({usd_path.stat().st_size:,} B)')

    poses, k, W, H, near, far = load_poses_and_camera()
    log(f'[s7] pose {len(poses)}개 · 해상도 {W}x{H} · near/far {near}/{far} m')

    camera, sim, stage, lights, shaders = build_scene(usd_path, k, W, H, near, far)

    # 제어 파일 초기 상태를 써 둔다 — 이 파일을 고치면 다음 프레임에 반영된다.
    mat_snap = snapshot_material(shaders)
    checker = make_checker(OUT_DIR / 'assets' / 'checker.png')
    log(f'[s7] 재질 원래값 {len(mat_snap)}개 기억 · 체커보드 준비: {checker}')
    state = {'light': ARGS.light, 'ambient': ARGS.rtx_ambient, 'iso': ARGS.film_iso,
             'albedo': None, 'roughness': None, 'colorspace': None,
             'texture': None, 'texture_scale': None, 'texture_rotate': None,
             'texture_translate': None,
             'frame': int(ARGS.start_frame), 'hold': bool(ARGS.hold), 'fps': ARGS.fps}
    ctl_path = OUT_DIR / 'control.json'
    ctl_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'[s7] 제어 파일: {ctl_path}  (이 파일을 고치면 실시간 반영)')
    log('[s7] 씬 준비 완료 — 창을 보세요 (ESC/q 끝내기, SPACE 멈춤, n/p 프레임 이동, s 저장)')

    win = f'RGB | depth  —  {usd_path.name}'
    paused, i, saved, shown = False, int(ARGS.start_frame), 0, 0
    delay = max(1, int(1000.0 / max(ARGS.fps, 0.1)))

    log(f'[s7] 루프 시작 (is_running={simulation_app.is_running()}, save_only={ARGS.save_only})')
    ctl_err, applied = None, {}
    while simulation_app.is_running():
        changed, ctl_err = read_control(ctl_path, state)
        if changed:
            log(f'[s7] 제어 변경: {changed} -> ' + json.dumps({k: state[k] for k in changed}, ensure_ascii=False))
            if 'light' in changed:
                if state['light'] not in LIGHT_CHOICES:
                    log(f"[s7]   무시 — light 는 {LIGHT_CHOICES} 중 하나여야 한다")
                    state['light'] = ARGS.light
                else:
                    lights = rebuild_lights(stage, state['light'])
            if {'ambient', 'iso'} & set(changed):
                carb.settings.get_settings().set_float('/rtx/sceneDb/ambientLightIntensity',
                                                       float(state['ambient']))
                carb.settings.get_settings().set_float('/rtx/post/tonemap/filmIso', float(state['iso']))
            if {'albedo', 'roughness', 'colorspace', 'texture',
                'texture_scale', 'texture_rotate', 'texture_translate'} & set(changed):
                applied = apply_material(shaders, state, mat_snap)
                log(f'[s7]   재질/텍스처 적용: {applied}')
            if 'frame' in changed:
                i = int(state['frame'])
            for _ in range(ARGS.steps_per_frame):
                sim.step()

        i = int(state['frame']) if state.get('hold') else i
        set_pose(camera, sim, poses[i % len(poses)])
        move_lights(lights, poses[i % len(poses)][:3, 3])
        for _ in range(ARGS.steps_per_frame):
            sim.step()
            camera.update(dt=sim.get_physics_dt())
        rgb = camera.data.output['rgb'][0, ..., :3].cpu().numpy()
        depth = camera.data.output['distance_to_image_plane'][0, ..., 0].cpu().numpy()
        depth_col, valid = depth_to_color(depth, far)

        mat = ' '.join(f'{k}={Path(str(state[k])).name if k == "texture" else state[k]}'
                       for k in ('texture', 'texture_scale', 'texture_rotate', 'colorspace',
                                 'albedo', 'roughness')
                       if state.get(k) is not None) or 'material=default'
        frame = compose(rgb, depth_col, [
            f'frame {i % len(poses)}/{len(poses)}  {"[HOLD]" if state.get("hold") else ""}'
            f'{"[PAUSED]" if paused else ""}',
            f'light={state["light"]}  ambient={state["ambient"]}  iso={state["iso"]}',
            f'{mat}',
            f'depth {valid:.3f}  brightness {rgb.mean():.1f}   {usd_path.name}',
        ] + ([f'control.json ERROR: {ctl_err}'] if ctl_err else []))
        shown += 1
        if shown == 1 or shown % 10 == 0:
            log(f'[s7] 루프 {shown}회 · frame {i % len(poses)} · depth {valid:.3f}')

        if ARGS.save_only:
            cv2.imwrite(str(OUT_DIR / f'live_{i % len(poses):04d}.jpg'), frame)
            if ARGS.max_frames and shown >= ARGS.max_frames:
                log(f'[s7] {shown} 프레임 저장하고 종료: {OUT_DIR}')
                break
            i += 1
            continue

        if ARGS.display == 'stream':
            # **왜 이 프로세스에서 창을 안 띄우나**: `cv2.imshow`를 Isaac과 같은 프로세스에서
            # 부르면 **한 프레임 그린 뒤 세그폴트로 죽는다**(실측 — headless로 띄워도 같다).
            # opencv의 Qt5 백엔드와 Kit이 들고 있는 Qt/GL이 충돌하는 것으로 본다.
            # 그래서 최신 프레임만 파일로 흘려보내고, 창은 **별도 프로세스**(`s7_watch.py`)가 띄운다.
            # 덮어쓰다 반쯤 쓰인 파일을 읽지 않도록 임시파일에 쓴 뒤 rename 한다.
            tmp = OUT_DIR / '.live_tmp.jpg'
            cv2.imwrite(str(tmp), frame)
            os.replace(tmp, OUT_DIR / 'live_latest.jpg')
            (OUT_DIR / 'live_state.json').write_text(json.dumps(
                {'frame': i % len(poses), 'total': len(poses), 'depth_valid': valid,
                 'brightness': float(rgb.mean()), 'usd': str(usd_path), 'shown': shown},
                ensure_ascii=False), encoding='utf-8')
            if ARGS.max_frames and shown >= ARGS.max_frames:
                log(f'[s7] {shown} 프레임 흘려보내고 종료')
                break
            time.sleep(max(0.0, 1.0 / max(ARGS.fps, 0.1)))
            if not state.get('hold'):
                i += 1
                state['frame'] = i % len(poses)
            continue

        cv2.imshow(win, frame)
        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            i += 1
        elif key == ord('p'):
            i -= 1
        elif key == ord('s'):
            p = OUT_DIR / f'snapshot_{saved:03d}.jpg'
            cv2.imwrite(str(p), frame)
            log(f'[s7] 저장: {p}')
            saved += 1
        elif not paused:
            i += 1

    log(f'[s7] 루프 종료 (총 {shown}회, is_running={simulation_app.is_running()})')
    if not ARGS.save_only and ARGS.display == 'window':
        cv2.destroyAllWindows()
    log('[s7] 종료')
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    # main()이 어떤 이유로든 빠져나오면 여기서 확실히 끊는다 — 그냥 return 하면
    # 인터프리터 정리(atexit -> carb)로 넘어가 세그폴트가 난다(실측).
    # **단 예외는 반드시 남긴다** — 처음엔 bare finally 로 감싸서 오류가 조용히 사라졌다.
    try:
        main()
    except BaseException:
        import traceback

        tb = traceback.format_exc()
        try:
            log('[s7] 예외로 종료:\n' + tb)
        except Exception:  # noqa: BLE001 - 로그 자체가 실패해도 stderr엔 남긴다
            pass
        sys.stderr.write(tb)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
