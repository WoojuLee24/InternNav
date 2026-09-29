"""세 자산(vln_ce / vln_pe / 노은역 GS)에서 조명 모드별 렌더를 **같은 시점으로** 나란히 저장.

왜 필요한가
-----------
`04c_light_explorer.py`(Matterport)와 `apply_real/04c_light_explorer.py`(노은역)는 각각 한 자산
안에서만 조명을 비교한다. 그런데 "조명이 로봇을 따라가면 더 현실감 있나"라는 질문의 답은
**자산 종류에 따라 다르다** — baked 텍스처냐, Gaussian splat이냐. 그래서 세 자산을 같은 조명
모드 목록으로 훑어 **한 장의 그림으로 비교**할 수 있게 한다.

세 컨텍스트
-----------
`vln_ce`  Habitat이 조명 없이(`NO_LIGHT_KEY`) 렌더한 GT 프레임 + **같은 pose**를 Isaac에서 각 조명
          모드로 렌더한 것. habitat_sim이 이 컨테이너에 없어 GT는 저장된 이미지를 그대로 쓴다
          (VLN-CE의 조명 모드는 하나뿐이라 스윕할 것도 없다).
`vln_pe`  Isaac 프로덕션 3-light로 캡처된 GT + 같은 pose의 각 모드 렌더.
`noeun`   NuRec Gaussian-splat usdz. **GT rgb 없음** — 모드 간 비교만.

`--film_iso`는 전 컨텍스트 100(기본값) 고정이다. 조명만 변수로 두기 위해서다 — 프로덕션 권장값
(70)과 다르므로 밝기 절대값을 프로덕션 산출물과 직접 비교하면 안 된다.

실행 (컨텍스트마다 해상도가 달라 `Camera`를 다시 만들 수 없으므로 **프로세스를 분리한다**)
```
for C in vln_ce vln_pe noeun; do timeout --signal=KILL 3000 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04f_light_gallery.py --context $C; done
```
`simulation_app.close()`가 수 분 걸릴 수 있으나 산출물은 `main()` 반환 시점에 저장 완료다 —
**성공 여부는 report.html 파일 존재로 판단**할 것.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': True})

import carb  # noqa: E402

carb.settings.get_settings().set_bool('/isaaclab/cameras_enabled', True)
carb.settings.get_settings().set_bool('/rtx/post/histogram/enabled', False)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
import omni.usd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camera_profiles  # noqa: E402
import dataset_utils  # noqa: E402
import usdz_scene_utils  # noqa: E402
from geometry_utils import action_to_c2w, save_jpg, synthesize_action_poses  # noqa: E402
from viz_utils import blink_widget_html, save_gallery  # noqa: E402

SCRIPT_NAME = '04f_light_gallery'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
MESH_ROOT = 'data/scene_data/mp3d_pe'
MATTERPORT_TEXTURE_ROOT = Path('/ssd/share/Matterport3D/data/v1/scans')
DATA_ROOT = 'data/InternData-N1-v0.5-mini'
APPLY_REAL_DIR = HERE / 'apply_real'

DOME_LIGHT_INTENSITY = 2_000_000
THREE_LIGHT_DISTANT_INTENSITY = 1000
THREE_LIGHT_DISK_INTENSITY = 5000
_LIGHT_PATHS = ('/World/dome_light', '/World/distant_light',
                '/World/up_disk_light', '/World/down_disk_light')

# 세 컨텍스트 공통 목록. `follow`는 "광원이 카메라를 따라오는가" — 리포트에서 이 축으로 묶는다.
LIGHT_MODES = [
    {'label': 'no_light',             'kind': 'none',        'ambient': 0.0,  'view_light': False, 'follow': False},
    {'label': 'ambient_only_10',      'kind': 'none',        'ambient': 10.0, 'view_light': False, 'follow': False},
    {'label': 'dome_2M',              'kind': 'dome',        'ambient': 0.0,  'view_light': False, 'follow': False},
    {'label': 'three_light_raise0.2', 'kind': 'three_light', 'ambient': 0.0,  'view_light': False, 'follow': True, 'raise_m': 0.2},
    {'label': 'three_light_raise0.5', 'kind': 'three_light', 'ambient': 0.0,  'view_light': False, 'follow': True, 'raise_m': 0.5},
    {'label': 'camera_light',         'kind': 'none',        'ambient': 0.0,  'view_light': True,  'follow': True},
]

# vln_ce는 parquet에 intrinsic이 없다 — 학습 dataloader가 쓰는 하드코딩 값과 같은 것을 쓴다
# (`internvla_n1_unified_provider.py:56-59`). 640x480, hfov 2*atan(320/388.19) = 78.8도.
VLNCE_K = np.array([[388.19, 0.0, 319.5], [0.0, 388.19, 239.5], [0.0, 0.0, 1.0]])
VLNCE_SIZE = (640, 480)
VLNCE_RIG = '125cm_0deg'
# vln_pe는 USD 실측값(90도 hfov 256x256) — dataset_utils 모듈 docstring 참고.
VLNPE_K = np.array([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]])
VLNPE_SIZE = (256, 256)


# ---------------------------------------------------------------- 조명
def _clear_lights(stage) -> None:
    for path in _LIGHT_PATHS:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)


def apply_light_mode(stage, mode: dict) -> dict:
    """`04_render_obs_isaac.py:_add_lights`와 같은 구성을 만들고 RTX 설정을 건다.

    ambient / view_light는 **모드마다 명시적으로 다시 건다** — 켜기만 하고 끄지 않으면 뒤 모드로
    샌다(본가 04c가 그 구조였다).
    """
    _clear_lights(stage)
    s = carb.settings.get_settings()
    s.set_bool('/rtx/useViewLightingMode', bool(mode['view_light']))
    s.set_float('/rtx/sceneDb/ambientLightIntensity', float(mode['ambient']))
    kind = mode['kind']
    if kind == 'none':
        return {'kind': 'none'}
    elif kind == 'dome':
        dome = UsdLux.DomeLight.Define(stage, '/World/dome_light')
        dome.CreateIntensityAttr(DOME_LIGHT_INTENSITY)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        return {'kind': 'dome'}
    elif kind == 'three_light':
        distant = UsdLux.DistantLight.Define(stage, '/World/distant_light')
        distant.CreateIntensityAttr(THREE_LIGHT_DISTANT_INTENSITY)
        distant.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        up = UsdLux.DiskLight.Define(stage, '/World/up_disk_light')
        up.CreateIntensityAttr(THREE_LIGHT_DISK_INTENSITY)
        up.CreateRadiusAttr(50.0)
        up.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(up).AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
        up_t = UsdGeom.Xformable(up).AddTranslateOp()
        down = UsdLux.DiskLight.Define(stage, '/World/down_disk_light')
        down.CreateIntensityAttr(THREE_LIGHT_DISK_INTENSITY)
        down.CreateRadiusAttr(50.0)
        down.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        down_t = UsdGeom.Xformable(down).AddTranslateOp()
        return {'kind': 'three_light', 'up': up_t, 'down': down_t, 'raise_m': mode['raise_m']}
    else:
        assert False, f'unreachable kind={kind!r}'


def update_light_for_pose(state: dict, c2w: np.ndarray) -> None:
    if state['kind'] != 'three_light':
        return
    x, y, z = c2w[:3, 3]
    r = state['raise_m']
    state['up'].Set(Gf.Vec3f(float(x), float(y), float(z + r)))
    state['down'].Set(Gf.Vec3f(float(x), float(y), float(z - r)))


# ---------------------------------------------------------------- 씬/시점
def mp3d_usd(scene: str) -> Path:
    target = MATTERPORT_TEXTURE_ROOT / scene / 'matterport_mesh'
    if not (target.is_symlink() or target.exists()):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(MESH_ROOT).resolve() / scene / 'matterport_mesh')
    cands = sorted(p for p in (Path(MESH_ROOT) / scene / 'matterport_mesh').glob('*/isaacsim_*.usd')
                   if '_non_metric' not in p.name)
    assert cands, f'Isaac USD 없음: {scene}'
    return cands[0]


def views_vln_ce(scene: str, episodes: list, n_frames: int) -> tuple:
    root = Path(DATA_ROOT) / 'vln_ce/traj_data/r2r' / scene
    rgb_dir = root / 'videos/chunk-000' / f'observation.images.rgb.{VLNCE_RIG}'
    views = []
    for ep in episodes:
        df = pd.read_parquet(root / 'data/chunk-000' / f'episode_{ep:06d}.parquet')
        poses = np.stack([np.stack(x) for x in df[f'pose.{VLNCE_RIG}'].values])
        # 제자리 회전 프레임(이동 0)이 38%라 균등 표본이 같은 지점에 몰릴 수 있다 — 이동이 있는
        # 프레임 사이에서 고르게 뽑는다.
        moved = np.r_[True, np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) > 1e-4]
        cand = np.flatnonzero(moved)
        for i in cand[np.linspace(0, len(cand) - 1, n_frames).round().astype(int)]:
            gt = rgb_dir / f'episode_{ep:06d}_{i}.jpg'
            views.append({'key': f'ep{ep}_f{i:04d}', 'c2w': poses[i].astype(np.float64),
                          'gt': gt if gt.is_file() else None})
    return views, mp3d_usd(scene), VLNCE_K, VLNCE_SIZE, (0.05, 10.0)


def views_vln_pe(scene: str, episodes: list, n_frames: int) -> tuple:
    root = Path(DATA_ROOT) / 'vln_pe/traj_data/r2r'
    rgb_dir = root / scene / 'videos/chunk-000/observation.images.rgb'
    views = []
    for ep in episodes:
        gt = dataset_utils.load_gt_episode(str(root), scene, ep, dataset='vln_pe')
        stack = np.load(rgb_dir / f'episode_{ep:06d}.npy')
        n = min(len(gt['poses_c2w']), len(stack))
        for i in np.linspace(0, n - 1, n_frames).round().astype(int):
            views.append({'key': f'ep{ep}_f{i:04d}', 'c2w': gt['poses_c2w'][i],
                          'gt_array': stack[i]})
    return views, mp3d_usd(scene), VLNPE_K, VLNPE_SIZE, (0.05, 10.0)


def views_noeun(scene: str, episodes: list, n_frames: int) -> tuple:
    meta = json.loads((APPLY_REAL_DIR / 'scene_meta' / f'{scene}.json').read_text())
    paths = json.loads((APPLY_REAL_DIR / 'paths' / f'{scene}_random.json').read_text())
    by_id = {e['episode_id']: e for e in paths['episodes']}
    prof = camera_profiles.get('d455_nominal')
    views = []
    for ep in episodes:
        e = by_id[ep]
        poses = synthesize_action_poses(np.asarray(e['trajectory'], dtype=np.float64),
                                        e['floor_z'], e['h_b'], e['pitch_deg'])
        for i in np.linspace(0, len(poses) - 1, n_frames).round().astype(int):
            views.append({'key': f'ep{ep}_f{i:04d}', 'c2w': action_to_c2w(poses[i], 'cam2world_gl')})
    return (views, Path(meta['usd_path']), prof.k, (prof.width, prof.height),
            (prof.render_near_m, prof.render_far_m))


CONTEXTS = {
    'vln_ce': {'scene': '17DRP5sb8fy', 'episodes': [0, 3], 'views': views_vln_ce,
               'gt_label': 'GT — Habitat 무조명(no_lights)'},
    'vln_pe': {'scene': '17DRP5sb8fy', 'episodes': [0, 3], 'views': views_vln_pe,
               'gt_label': 'GT — Isaac 프로덕션 3-light(에피소드 시작점 고정)'},
    'noeun':  {'scene': 'noeun_station_mid', 'episodes': [0, 5], 'views': views_noeun,
               'gt_label': None},
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--context', required=True, choices=list(CONTEXTS))
    p.add_argument('--scene', default=None, help='생략 시 컨텍스트 기본 씬')
    p.add_argument('--episodes', default=None, help='콤마 구분. 생략 시 컨텍스트 기본값')
    p.add_argument('--n_frames', type=int, default=2, help='에피소드당 균등 표본 시점 수')
    p.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    ctx = CONTEXTS[args.context]
    scene = args.scene or ctx['scene']
    episodes = [int(x) for x in args.episodes.split(',')] if args.episodes else ctx['episodes']
    views, usd_path, k, (w, h), (near, far) = ctx['views'](scene, episodes, args.n_frames)
    print(f'[{SCRIPT_NAME}] context={args.context} scene={scene} views={len(views)} '
          f'modes={len(LIGHT_MODES)} size={w}x{h} usd={Path(usd_path).name}', flush=True)

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath('/World/Scene'):
        stage.RemovePrim('/World/Scene')
    cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path),
                               collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.func('/World/Scene', cfg)
    changed = usdz_scene_utils.expose_collision_meshes_for_rendering(stage, usd_path, UsdGeom)
    if changed:
        print(f'[scene-render] enabled {changed} hidden collision mesh(es)', flush=True)

    camera = Camera(cfg=CameraCfg(
        prim_path='/World/RenderCamera', height=h, width=w, data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(near, far))))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.reset()
    camera.set_intrinsic_matrices(
        torch.tensor(np.asarray(k, dtype=np.float32), device=sim.device).unsqueeze(0))
    for _ in range(10):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    def render_at(c2w: np.ndarray) -> np.ndarray:
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        camera.set_world_poses(
            torch.tensor(c2w[:3, 3], dtype=torch.float32, device=sim.device).unsqueeze(0),
            torch.tensor(np.array([q[3], q[0], q[1], q[2]]), dtype=torch.float32,
                         device=sim.device).unsqueeze(0),
            convention='ros')
        for _ in range(10):
            sim.step()
            camera.update(dt=sim.get_physics_dt())
        return camera.data.output['rgb'][0, ..., :3].cpu().numpy()

    log_dir = Path(args.log_dir) / SCRIPT_NAME / args.context
    stats = {}
    states = {v['key']: [] for v in views}

    # GT 먼저 (있는 컨텍스트만) — 저장 위치를 모드 이미지와 같게 두어 리포트에서 함께 넘긴다.
    for v in views:
        if v.get('gt') is not None:
            from PIL import Image
            arr = np.asarray(Image.open(v['gt']).convert('RGB'))
            p = save_jpg(arr, log_dir / v['key'] / 'gt.jpg')
            states[v['key']].append((ctx['gt_label'], p))
        elif v.get('gt_array') is not None:
            p = save_jpg(np.asarray(v['gt_array']), log_dir / v['key'] / 'gt.jpg')
            states[v['key']].append((ctx['gt_label'], p))

    for mode in LIGHT_MODES:
        label = mode['label']
        state = apply_light_mode(stage, mode)
        lums = []
        for v in views:
            update_light_for_pose(state, v['c2w'])
            rgb = render_at(v['c2w'])
            p = save_jpg(rgb, log_dir / v['key'] / f'{label}.jpg')
            g = rgb.astype(np.float32).mean(axis=-1)
            lums.append({'lum': float(g.mean()), 'std': float(g.std()),
                         'sat': float((rgb >= 250).mean())})
            states[v['key']].append((f'{label}{" · 추종" if mode["follow"] else ""}', p))
        stats[label] = {kk: float(np.mean([x[kk] for x in lums])) for kk in lums[0]}
        stats[label]['follow'] = bool(mode['follow'])
        print(f'  {label:22s} lum={stats[label]["lum"]:6.2f} std={stats[label]["std"]:6.2f} '
              f'sat={stats[label]["sat"]*100:5.2f}%', flush=True)

    if any(v.get('gt') is not None or v.get('gt_array') is not None for v in views):
        gts = []
        for v in views:
            p = log_dir / v['key'] / 'gt.jpg'
            if p.is_file():
                from PIL import Image
                a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32).mean(axis=-1)
                gts.append({'lum': float(a.mean()), 'std': float(a.std())})
        if gts:
            stats['__gt__'] = {kk: float(np.mean([x[kk] for x in gts])) for kk in gts[0]}
            print(f'  {"GT":22s} lum={stats["__gt__"]["lum"]:6.2f} std={stats["__gt__"]["std"]:6.2f}',
                  flush=True)

    rows = ''.join(
        f'<tr><td>{m["label"]}</td><td>{"추종" if m["follow"] else "-"}</td>'
        f'<td>{stats[m["label"]]["lum"]:.2f}</td><td>{stats[m["label"]]["std"]:.2f}</td>'
        f'<td>{stats[m["label"]]["sat"]*100:.3f}%</td></tr>' for m in LIGHT_MODES)
    gt_row = (f'<tr><td><b>GT</b></td><td>-</td><td><b>{stats["__gt__"]["lum"]:.2f}</b></td>'
              f'<td>{stats["__gt__"]["std"]:.2f}</td><td>-</td></tr>') if '__gt__' in stats else ''
    summary = (f'<p>컨텍스트 <b>{args.context}</b> · 씬 <b>{scene}</b> · 시점 {len(views)}개 x '
               f'조명 {len(LIGHT_MODES)}모드. film_iso는 전부 100 고정(조명만 변수).</p>'
               f'<table border="1" cellpadding="6" style="border-collapse:collapse">'
               f'<tr><th>mode</th><th>카메라 추종</th><th>평균 휘도</th><th>휘도 std</th>'
               f'<th>포화 &ge;250</th></tr>{gt_row}{rows}</table>')
    body = ''.join(f'<h4>{v["key"]}</h4>' + blink_widget_html(v['key'], states[v['key']]) for v in views)
    report = save_gallery(log_dir, 'report.html',
                          f'{SCRIPT_NAME} — {args.context} / {scene} (조명 모드별 렌더)', summary, body)
    (log_dir / 'stats.json').write_text(json.dumps(
        {'context': args.context, 'scene': scene, 'size': [w, h],
         'views': [v['key'] for v in views], 'stats': stats}, indent=2))
    print(f'  report html -> {report}', flush=True)
    return 0


if __name__ == '__main__':
    code = main()
    simulation_app.close()
    sys.exit(code)
