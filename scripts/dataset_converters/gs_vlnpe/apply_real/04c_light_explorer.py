"""노은역(NuRec Gaussian-splat usdz) 조명 옵션 비교 — GT가 없는 씬용.

본가 `gs_vlnpe/04c_light_explorer.py`는 Matterport GT rgb와 렌더를 비교해 `mean_err`로 순위를
매긴다. 노은역에는 **GT rgb가 없다**(03이 만든 random 경로를 렌더한 것이 전부) — 그래서 이
스크립트는 "정답과의 거리"가 아니라 **설정끼리의 상대 비교**만 한다:

- 기준(reference)은 기본이 현재 노은역 기본값 `dome_2M`이고 `--reference`로 바꾼다.
- 각 설정은 같은 프레임을 렌더해 `[기준 | 이 설정 | 차이]` 합성 이미지로 저장한다.
- 수치는 프레임 평균 휘도/표준편차/포화·흑색 클립 비율과 **기준 대비 최대 절대차**다.

## 이 스크립트가 답하려는 질문

**NuRec Gaussian-splat 볼륨에 USD 조명이 영향을 주는가?** GS는 radiance가 이미 구워져 있어
조명에 반응하지 않을 수 있다. 반면 `usdz_scene_utils.expose_collision_meshes_for_rendering`이
충돌 mesh를 runtime stage에서 visible로 바꾸므로, 그 mesh는 PBR 표면이라 조명에 반응한다.
두 성분이 한 화면에 섞여 있으므로 **기준 대비 최대 절대차가 0이면 조명이 아무 일도 안 한 것**이고,
0이 아니면 어느 성분이 반응했는지 diff 이미지 위치로 판별할 수 있다.

`04_render_obs_isaac.py`(공식 렌더)는 건드리지 않는다 — 조명 비교 전용 신규 파일이다.

## 실행

```
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/04c_light_explorer.py --scene noeun_station_mid
```
`simulation_app.close()`가 수 분 걸릴 수 있으나 report.html은 `main()` 반환 시점에 저장 완료다
— **성공 여부는 report.html 파일 존재로 판단**할 것.
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
import omni.usd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_profiles  # noqa: E402
import usdz_scene_utils  # noqa: E402
from geometry_utils import action_to_c2w, save_jpg, synthesize_action_poses  # noqa: E402
from viz_utils import blink_widget_html, save_gallery  # noqa: E402

SCRIPT_NAME = '04c_light_explorer'
DEFAULT_SCENE = 'noeun_station_mid'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe/apply_real'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe/apply_real'
DEFAULT_CAMERA = 'd455_nominal'

# 본가 04c와 같은 세기 상수 (`04_render_obs_isaac.py`의 확정값).
DOME_LIGHT_INTENSITY = 2_000_000
THREE_LIGHT_DISTANT_INTENSITY = 1000
THREE_LIGHT_DISK_INTENSITY = 5000

_LIGHT_PATHS = ('/World/dome_light', '/World/distant_light',
                '/World/up_disk_light', '/World/down_disk_light')

# `ambient`는 RTX `/rtx/sceneDb/ambientLightIntensity`, `view_light`는
# `/rtx/useViewLightingMode`(헤드램프). 둘 다 **설정마다 명시적으로 다시 건다** — 본가 04c는
# view_light를 켜기만 하고 끄지 않아 뒤 설정에 새는 구조였다.
LIGHT_CONFIGS = [
    {'label': 'no_light',            'kind': 'none',        'ambient': 0.0,  'view_light': False},
    {'label': 'ambient_only_10',     'kind': 'none',        'ambient': 10.0, 'view_light': False},
    {'label': 'dome_2M',             'kind': 'dome',        'ambient': 0.0,  'view_light': False},
    {'label': 'three_light_raise0.2', 'kind': 'three_light', 'ambient': 0.0, 'view_light': False, 'raise_m': 0.2},
    {'label': 'three_light_raise0.5', 'kind': 'three_light', 'ambient': 0.0, 'view_light': False, 'raise_m': 0.5},
    {'label': 'camera_light',        'kind': 'none',        'ambient': 0.0,  'view_light': True},
]
REFERENCE_LABEL = 'dome_2M'


def _clear_lights(stage) -> None:
    for path in _LIGHT_PATHS:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)


def _spawn_dome(stage, intensity: float) -> None:
    dome = UsdLux.DomeLight.Define(stage, '/World/dome_light')
    dome.CreateIntensityAttr(intensity)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))


def _spawn_three_light(stage) -> dict:
    """`04_render_obs_isaac.py:_add_lights`의 three_light와 같은 구성(회전 op를 translate보다 먼저)."""
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
    return {'up': up_t, 'down': down_t}


def apply_light_config(stage, config: dict) -> dict:
    """light prim을 새로 만들고 RTX 설정을 건다. 반환값은 프레임별 위치 갱신에 필요한 상태."""
    _clear_lights(stage)
    settings = carb.settings.get_settings()
    settings.set_bool('/rtx/useViewLightingMode', bool(config['view_light']))
    settings.set_float('/rtx/sceneDb/ambientLightIntensity', float(config['ambient']))

    kind = config['kind']
    if kind == 'none':
        return {'kind': 'none'}
    elif kind == 'dome':
        _spawn_dome(stage, DOME_LIGHT_INTENSITY)
        return {'kind': 'dome'}
    elif kind == 'three_light':
        return {'kind': 'three_light', 'lights': _spawn_three_light(stage), 'raise_m': config['raise_m']}
    else:
        assert False, f'unreachable kind={kind!r}'


def update_light_state_for_pose(state: dict, c2w: np.ndarray) -> None:
    if state['kind'] != 'three_light':
        return
    x, y, z = c2w[:3, 3]
    r = state['raise_m']
    state['lights']['up'].Set(Gf.Vec3f(float(x), float(y), float(z + r)))
    state['lights']['down'].Set(Gf.Vec3f(float(x), float(y), float(z - r)))


def colorize_diff(a: np.ndarray, b: np.ndarray, max_diff: float = 255.0) -> np.ndarray:
    """두 렌더의 픽셀별 절대차(채널 평균)를 고정 스케일 컬러맵으로. 스케일을 고정해야 설정 간
    비교가 공정하다(설정마다 정규화하면 차이가 클수록 오히려 연해 보인다)."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=-1)
    u8 = (np.clip(diff / max_diff, 0, 1) * 255).astype(np.uint8)
    colormap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
    return cv2.cvtColor(cv2.applyColorMap(u8, colormap), cv2.COLOR_BGR2RGB)


def make_composite(ref_rgb, this_rgb, diff_rgb, label: str, ref_label: str) -> np.ndarray:
    pad = 28
    h, w = this_rgb.shape[:2]
    canvas = np.zeros((h + pad, w * 3, 3), dtype=np.uint8)
    for i, (panel, title) in enumerate(zip([ref_rgb, this_rgb, diff_rgb],
                                           [f'ref ({ref_label})', f'this ({label})', 'diff x1'])):
        canvas[pad:, i * w:(i + 1) * w] = panel
        cv2.putText(canvas, title, (i * w + 6, pad - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def frame_stats(rgb: np.ndarray) -> dict:
    g = rgb.astype(np.float32).mean(axis=-1)
    return {'lum': float(g.mean()), 'std': float(g.std()),
            'sat': float((rgb >= 250).mean()), 'black': float((g <= 5).mean())}


def pick_frames(episodes: list, n_ep: int, n_frames: int) -> list:
    """앞에서부터 `n_ep`개 에피소드, 각 궤적에서 균등 간격으로 `n_frames`개 pose를 뽑는다."""
    out = []
    for ep in episodes[:n_ep]:
        xy = np.asarray(ep['trajectory'], dtype=np.float64)
        poses = synthesize_action_poses(xy, ep['floor_z'], ep['h_b'], ep['pitch_deg'])
        idx = np.linspace(0, len(poses) - 1, n_frames).round().astype(int)
        for i in idx:
            out.append((ep['episode_id'], int(i), action_to_c2w(poses[i], 'cam2world_gl')))
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', default=DEFAULT_SCENE)
    p.add_argument('--camera', default=DEFAULT_CAMERA, choices=list(camera_profiles.PROFILES))
    p.add_argument('--num_episodes', type=int, default=2, help='비교에 쓸 에피소드 수(앞에서부터)')
    p.add_argument('--n_frames', type=int, default=4, help='에피소드당 균등 표본 프레임 수')
    p.add_argument('--reference', default=REFERENCE_LABEL,
                   choices=[c['label'] for c in LIGHT_CONFIGS],
                   help='차이를 재는 기준 설정. 기본은 현재 노은역 기본값. '
                        '"추종 조명이 조명 없음과 다른가"를 보려면 no_light을 준다.')
    p.add_argument('--self_check', action='store_true',
                   help='기준 설정을 같은 라벨 하나 더 붙여 두 번 렌더한다. 조명 차이를 해석하려면 '
                        '먼저 이 값(렌더 자체의 재현 불가능한 잡음)을 알아야 한다 — NuRec GS 경로가 '
                        '결정론적이면 0이 나오고, 그렇지 않으면 그 값이 잡음 바닥이다.')
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    p.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    meta = json.loads((out_dir / 'scene_meta' / f'{args.scene}.json').read_text())
    usd_path = Path(meta['usd_path'])
    assert usd_path.is_file(), f'씬 USD 없음: {usd_path}'
    paths = json.loads((out_dir / 'paths' / f'{args.scene}_random.json').read_text())
    frames = pick_frames(paths['episodes'], args.num_episodes, args.n_frames)
    prof = camera_profiles.get(args.camera)
    reference = args.reference
    configs = list(LIGHT_CONFIGS)
    if args.self_check:
        # 렌더 파이프라인에 프레임 패리티(짝/홀) 의존이 보여서 재현 2회를 붙인다 — 한 번만
        # 붙이면 기준과 같은 패리티에만 떨어져 잡음 바닥을 과소평가할 수 있다.
        ref = next(c for c in configs if c['label'] == reference)
        configs.append({**ref, 'label': f'{reference}__repeat1'})
        configs.append({**ref, 'label': f'{reference}__repeat2'})
    print(f'[{SCRIPT_NAME}] scene={args.scene} usd={usd_path.name} '
          f'frames={len(frames)} configs={len(configs)}', flush=True)

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath('/World/Scene'):
        stage.RemovePrim('/World/Scene')
    cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path),
                               collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.func('/World/Scene', cfg)
    changed = usdz_scene_utils.expose_collision_meshes_for_rendering(stage, usd_path, UsdGeom)
    print(f'[scene-render] enabled {changed} hidden collision mesh(es)', flush=True)

    camera_cfg = CameraCfg(
        prim_path='/World/RenderCamera', height=prof.height, width=prof.width,
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(prof.render_near_m, prof.render_far_m)),
    )
    camera = Camera(cfg=camera_cfg)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.reset()
    camera.set_intrinsic_matrices(
        torch.tensor(np.asarray(prof.k, dtype=np.float32), device=sim.device).unsqueeze(0))
    for _ in range(10):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    def render_at(c2w: np.ndarray) -> np.ndarray:
        pos = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        camera.set_world_poses(
            torch.tensor(pos, dtype=torch.float32, device=sim.device).unsqueeze(0),
            torch.tensor(np.array([q[3], q[0], q[1], q[2]]), dtype=torch.float32,
                         device=sim.device).unsqueeze(0),
            convention='ros')
        for _ in range(10):
            sim.step()
            camera.update(dt=sim.get_physics_dt())
        return camera.data.output['rgb'][0, ..., :3].cpu().numpy()

    log_dir = Path(args.log_dir) / SCRIPT_NAME / args.scene
    renders = {}
    stats = {}
    for config in configs:
        label = config['label']
        state = apply_light_config(stage, config)
        per_frame = []
        for ep, fi, c2w in frames:
            update_light_state_for_pose(state, c2w)
            rgb = render_at(c2w)
            renders[(label, ep, fi)] = rgb
            per_frame.append(frame_stats(rgb))
        stats[label] = {k: float(np.mean([f[k] for f in per_frame])) for k in per_frame[0]}
        print(f'  {label:22s} lum={stats[label]["lum"]:6.2f} std={stats[label]["std"]:6.2f} '
              f'sat={stats[label]["sat"]*100:5.2f}% black={stats[label]["black"]*100:5.2f}%', flush=True)

    # 기준 대비 차이 + 합성 이미지
    frame_states = {(ep, fi): [] for ep, fi, _ in frames}
    for config in configs:
        label = config['label']
        maxdiffs, meandiffs = [], []
        for ep, fi, _ in frames:
            ref = renders[(reference, ep, fi)]
            cur = renders[(label, ep, fi)]
            d = np.abs(cur.astype(np.float32) - ref.astype(np.float32))
            maxdiffs.append(float(d.max()))
            meandiffs.append(float(d.mean()))
            comp = make_composite(ref, cur, colorize_diff(cur, ref), label, reference)
            p = save_jpg(comp, log_dir / f'ep{ep}_frame{fi:04d}' / f'{label}.jpg')
            frame_states[(ep, fi)].append((f'{label} (vs ref: mean {meandiffs[-1]:.2f}, max {maxdiffs[-1]:.0f})', p))
        stats[label]['ref_mean_diff'] = float(np.mean(meandiffs))
        stats[label]['ref_max_diff'] = float(np.max(maxdiffs))
        print(f'  {label:22s} vs {reference}: mean {stats[label]["ref_mean_diff"]:.3f} '
              f'max {stats[label]["ref_max_diff"]:.0f}', flush=True)

    rows = ''.join(
        f'<tr><td>{c["label"]}</td><td>{stats[c["label"]]["lum"]:.2f}</td>'
        f'<td>{stats[c["label"]]["std"]:.2f}</td>'
        f'<td>{stats[c["label"]]["sat"]*100:.3f}%</td><td>{stats[c["label"]]["black"]*100:.3f}%</td>'
        f'<td>{stats[c["label"]]["ref_mean_diff"]:.3f}</td>'
        f'<td>{stats[c["label"]]["ref_max_diff"]:.0f}</td></tr>'
        for c in configs)
    summary = (
        f'<p>씬 <b>{args.scene}</b> (NuRec Gaussian-splat usdz, <b>GT rgb 없음</b>) — '
        f'{len(configs)}개 조명 설정 x {len(frames)}개 프레임. 기준은 현재 기본값 '
        f'<code>{reference}</code>이며, 아래 수치는 모두 <b>정답과의 거리가 아니라 기준과의 차이</b>다. '
        f'<code>ref_max_diff</code>가 0이면 그 설정이 기준과 <b>픽셀 단위로 동일</b> = 조명이 화면에 '
        f'아무 영향도 주지 않았다는 뜻이다.</p>'
        f'<table border="1" cellpadding="6" style="border-collapse:collapse">'
        f'<tr><th>config</th><th>평균 휘도</th><th>휘도 std</th><th>포화(&ge;250)</th>'
        f'<th>흑색(&le;5)</th><th>기준 대비 평균차</th><th>기준 대비 최대차</th></tr>{rows}</table>')
    body = ''
    for ep, fi, _ in frames:
        body += f'<h4>ep {ep} frame {fi}</h4>' + blink_widget_html(f'f{ep}_{fi}', frame_states[(ep, fi)])
    report = save_gallery(log_dir, 'report.html',
                          f'{SCRIPT_NAME} — {args.scene} (조명 옵션 비교, GT 없음)', summary, body)
    print(f'  report html -> {report}', flush=True)
    (log_dir / 'stats.json').write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == '__main__':
    code = main()
    simulation_app.close()
    sys.exit(code)
