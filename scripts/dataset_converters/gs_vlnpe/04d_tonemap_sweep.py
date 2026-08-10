"""M1.4 부속 — Isaac RTX **톤매핑/노출** 스윕 (조명 세기가 아니라 응답곡선을 맞춘다).

## 왜 만들었나

`--light ambient_only --rtx_ambient 6.0`으로 vln_n1 SSIM 0.648 -> 0.876까지 올렸지만 GT와
차이가 남아서 원인을 분해했더니(18프레임 실측):

| GT 휘도대 | 렌더-GT 평균 |
|---|---|
| 0~60 (어두움)   | **-4.3** |
| 60~150 (중간)   | **+19.4** |
| 150~255 (밝음)  | **+37.1** |

어두운 곳은 더 어둡고 밝은 곳은 더 밝다 = **렌더의 응답곡선이 GT보다 가파르다**. 조명 세기를
줄이면 밝은 쪽은 맞지만 어두운 쪽이 더 틀어진다(단일 감마 보정도 SSIM 0.860 -> 0.839로 악화
확인). 즉 남은 차이는 광량이 아니라 **톤매퍼 설정**이다.

Isaac 기본 톤매퍼는 `/rtx/post/tonemap/op = 6` (Iray)이고, 실측한 기본값 중 정확히 이 증상에
대응하는 노브가 있다:

- `/rtx/post/tonemap/irayReinhard/crushBlacks` = 0.5 — 클수록 어두운 쪽을 검게 뭉갠다
- `/rtx/post/tonemap/irayReinhard/burnHighlights` = 0.7 — 클수록 밝은 쪽이 덜 눌린다(더 밝게 탐)
- `/rtx/post/tonemap/filmIso` = 100 — 전체 노출 게인

## 하는 일

Isaac을 **한 번만** 띄우고 (ambient, filmIso, crushBlacks, burnHighlights) 조합을 순회하며 같은
프레임을 반복 렌더한다(조합당 재기동 1~2분을 피하기 위함 — carb 설정은 렌더 사이에 바꿔도 즉시
반영된다). 조합마다 SSIM과 위 휘도대별 편차를 같이 찍어서, SSIM만 보고 밝기 오프셋을 놓치는
실수를 막는다.

`04_render_obs_isaac.py`(공식 검증)는 건드리지 않는 별도 탐색 스크립트다.

## 실행

```
timeout --signal=KILL 1800 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04d_tonemap_sweep.py --dataset vln_n1 --scene 17DRP5sb8fy
```
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
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from skimage.metrics import structural_similarity as compare_ssim  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
import omni.usd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_utils import default_data_root, load_gt_episode, load_rgb_frame, render_wh  # noqa: E402
from geometry_utils import save_jpg  # noqa: E402
from viz_utils import blink_widget_html, save_gallery  # noqa: E402

DEFAULT_MESH_ROOT = 'data/scene_data/mp3d_pe'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
DEFAULT_SCENE = '17DRP5sb8fy'
RENDER_NEAR_M, RENDER_FAR_M = 0.05, 10.0
MATTERPORT_TEXTURE_ROOT = Path('/ssd/share/Matterport3D/data/v1/scans')
SCRIPT_NAME = '04d_tonemap_sweep'

# 데이터셋별 프레임 표본 — 서로 다른 방/밝기를 섞는다(밝은 벽면·어두운 구석 둘 다 필요).
TARGET_FRAMES = {
    'vln_n1': [(0, 0), (0, 47), (0, 95), (0, 142), (0, 190), (0, 238)],
    'vln_pe': [(0, 0), (0, 10), (0, 20), (0, 30), (1, 5), (1, 15)],
}

# 데이터셋별 ambient 기준값 — 앞선 스윕에서 확정된 값(vln_n1 6.0 / vln_pe 10.0)을 중심으로 둔다.
BASE_AMBIENT = {'vln_n1': 6.0, 'vln_pe': 10.0}

TM_ISO = '/rtx/post/tonemap/filmIso'
TM_CRUSH = '/rtx/post/tonemap/irayReinhard/crushBlacks'
TM_BURN = '/rtx/post/tonemap/irayReinhard/burnHighlights'
RTX_AMBIENT = '/rtx/sceneDb/ambientLightIntensity'


def build_sweep_tonemap(base_ambient: float) -> list:
    """`--mode tonemap` — Iray 톤매퍼 노브 스윕.

    **실측 결론(vln_n1 6프레임): `crushBlacks`/`burnHighlights`는 RTX Real-Time 경로에서 전혀
    반영되지 않는다**(baseline과 SSIM·대역편차가 소수 셋째 자리까지 동일). 듣는 건 `filmIso`뿐인데
    전 대역을 같이 올려서 응답곡선 기울기는 못 고친다. 재현용으로 남겨 둔 스윕이다.
    """
    configs = [dict(label='baseline', ambient=base_ambient)]
    for crush in (0.0, 0.25, 0.75):
        configs.append(dict(label=f'crush{crush}', ambient=base_ambient, crush=crush))
    for burn in (0.1, 0.3, 1.0):
        configs.append(dict(label=f'burn{burn}', ambient=base_ambient, burn=burn))
    for iso in (70.0, 85.0, 130.0):
        configs.append(dict(label=f'iso{iso:.0f}', ambient=base_ambient, iso=iso))
    return configs


def build_sweep_light_mix(base_ambient: float) -> list:
    """`--mode light_mix` — **실제 광원 + ambient** 혼합 스윕.

    톤매퍼로 기울기를 못 고친다는 결과를 받고 세운 가설: ambient 단독이면 씬에 광원이 하나도 없어
    **GI 바운스가 아예 없다**. 평평한 ambient는 픽셀값을 (알베도 x 상수)로 만들어 알베도 대비를
    그대로 노출시키는 반면, 실제 광원이 있으면 밝은 벽에서 튄 빛이 어두운 구석을 들어올려
    (어두움 상승 / 밝음 하강) 기울기가 완만해진다 — 우리에게 필요한 방향과 정확히 일치한다.

    그래서 DomeLight 세기 x ambient 격자를 돈다. 두 값 모두 0이면 완전 암흑이므로 제외.
    """
    configs = []
    for dome in (0.0, 3e5, 6e5, 1e6, 2e6):
        for amb in (0.0, base_ambient * 0.5, base_ambient):
            if dome == 0.0 and amb == 0.0:
                continue
            configs.append(dict(label=f'dome{dome/1e6:.2f}M_amb{amb:.1f}', ambient=amb, dome=dome))
    return configs


def build_sweep_flatten(base_ambient: float) -> list:
    """`--mode flatten` — light_mix에서 찾은 "기울기는 맞는데 전체가 밝다" 조합을 노출로 내린다.

    light_mix 실측(vln_n1 6프레임)에서 `dome 0.3M + ambient 3.0`의 대역 편차가
    **+14.7 / +20.4 / +19.3**으로 거의 평평했다 — ambient 단독(-5.2 / +17.4 / +36.2, 폭 41)과 달리
    기울기 문제가 사라진 상태다. 남은 건 균일한 +18 오프셋뿐이고, 이건 `filmIso`(유일하게 반응하는
    톤맵 노브)로 내릴 수 있다. 그래서 (dome, ambient) 혼합비를 몇 개 두고 iso를 낮춰 훑는다.

    첫 줄의 `ambient_only_control`은 **대조군**이다 — 세션 중간에 조명을 바꾸며 재는 방식이
    믿을 만한지 확인하려면 이미 아는 값(ambient 6.0 = SSIM 0.856)이 재현되는지 봐야 한다.
    """
    configs = [dict(label='ambient_only_control', ambient=base_ambient)]
    for dome, amb in ((3e5, 3.0), (1.5e5, 3.0), (1.5e5, 4.5)):
        for iso in (70.0, 80.0, 90.0):
            configs.append(dict(label=f'dome{dome/1e6:.2f}M_amb{amb:.1f}_iso{iso:.0f}',
                                ambient=amb, dome=dome, iso=iso))
    return configs


def build_sweep_iso(base_ambient: float) -> list:
    """`--mode iso` — ambient 고정, `filmIso`만 정밀 스윕.

    앞선 세 스윕의 결론: 톤맵 하위 노브(crushBlacks/burnHighlights)는 무반응, DomeLight를 섞으면
    밝기는 맞지만 SSIM이 0.86 -> 0.75로 떨어진다(60스텝으로 올려도 동일 -> 수렴 문제 아님).
    실제로 쓸 수 있는 노브는 `filmIso` 하나뿐이라 이것만 촘촘히 훑어 데이터셋별 최적값을 정한다.
    """
    return [dict(label=f'iso{iso:.0f}', ambient=base_ambient, iso=iso)
            for iso in (65.0, 75.0, 80.0, 85.0, 90.0, 100.0, 110.0)]


SWEEP_MODES = {'tonemap': build_sweep_tonemap, 'light_mix': build_sweep_light_mix,
               'flatten': build_sweep_flatten, 'iso': build_sweep_iso}


def _spawn_dome(stage, intensity: float) -> None:
    from pxr import UsdLux

    path = '/World/dome_light'
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    if intensity <= 0:
        return
    light = UsdLux.DomeLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))


def apply_config(stage, config: dict) -> None:
    s = carb.settings.get_settings()
    s.set_float(RTX_AMBIENT, float(config.get('ambient', 0.0)))
    s.set_float(TM_ISO, float(config.get('iso', 100.0)))
    s.set_float(TM_CRUSH, float(config.get('crush', 0.5)))
    s.set_float(TM_BURN, float(config.get('burn', 0.7)))
    _spawn_dome(stage, float(config.get('dome', 0.0)))


def band_deltas(render_rgb: np.ndarray, real_rgb: np.ndarray) -> dict:
    """GT 휘도대별 (렌더-GT) 평균. 원인 분해에 쓴 지표와 동일하게 유지한다."""
    g = real_rgb.astype(np.float64).mean(2)
    r = render_rgb.astype(np.float64).mean(2)
    out = {}
    for lo, hi, name in ((0, 60, 'dark'), (60, 150, 'mid'), (150, 256, 'bright')):
        m = (g >= lo) & (g < hi)
        out[name] = float((r[m] - g[m]).mean()) if m.sum() > 500 else float('nan')
    return out


def ensure_texture_symlink(mesh_root, scene: str) -> None:
    """`04_render_obs_isaac.py`의 동명 함수와 동일 — mp3d_pe USD의 텍스처 절대경로를 로컬로 심링크."""
    target = MATTERPORT_TEXTURE_ROOT / scene
    if not target.exists():
        return
    link = Path(mesh_root) / scene / 'matterport_mesh'
    if link.exists() or link.is_symlink():
        return
    src = target / 'matterport_mesh'
    if src.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(src)


def find_scene_usd(mesh_root, scene: str) -> Path:
    candidates = sorted(Path(mesh_root).glob(f'{scene}/**/*.usd*'))
    if not candidates:
        raise FileNotFoundError(f'{mesh_root}/{scene} 아래 usd 없음')
    return candidates[0]


def colorize_diff(render_rgb: np.ndarray, real_rgb: np.ndarray) -> np.ndarray:
    import cv2

    d = np.abs(render_rgb.astype(np.float32) - real_rgb.astype(np.float32)).mean(2)
    return cv2.cvtColor(cv2.applyColorMap(np.clip(d * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)


def make_composite(real_rgb, render_rgb, diff_rgb, label: str) -> np.ndarray:
    import cv2

    def put(img, text):
        out = img.copy()
        cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return out

    return np.hstack([put(real_rgb, 'GT'), put(render_rgb, label), put(diff_rgb, '|diff|x3')])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='vln_n1', choices=['vln_n1', 'vln_pe'])
    parser.add_argument('--scene', default=DEFAULT_SCENE)
    parser.add_argument('--data_root', default=None, help='생략하면 --dataset 기본 경로')
    parser.add_argument('--mesh_root', default=DEFAULT_MESH_ROOT)
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    parser.add_argument('--ambient', type=float, default=None, help='생략하면 데이터셋 확정값')
    parser.add_argument('--supersample', type=int, default=1,
                        help='N배 해상도로 렌더한 뒤 INTER_AREA로 축소(SSAA). GT보다 렌더가 미세하게 '
                             '더 선명해서(라플라시안 분산 243 vs 231) 고주파가 어긋나는데, 오프라인 '
                             '블러(sigma 0.8)를 먹이면 SSIM이 0.860->0.876으로 오르는 것을 실측했다 — '
                             '그 이득을 후처리 블러가 아니라 렌더 단계에서 정직하게 얻는 방법.')
    parser.add_argument('--steps', type=int, default=10,
                        help='프레임당 sim.step 횟수. DomeLight처럼 샘플링이 필요한 광원은 10스텝에서 '
                             '평평한 벽면에 노이즈가 남아 SSIM이 부당하게 깎인다(실측: dome 조합이 '
                             '밝기는 완벽히 맞는데 SSIM만 0.86->0.75). 조명 비교 시 40 이상 권장.')
    parser.add_argument('--mode', default='light_mix', choices=sorted(SWEEP_MODES),
                        help='tonemap=Iray 노브 스윕(대부분 무반응 확인됨), light_mix=DomeLight x ambient 격자')
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    data_root = args.data_root or default_data_root(args.dataset)
    ambient = args.ambient if args.ambient is not None else BASE_AMBIENT[args.dataset]
    targets = TARGET_FRAMES[args.dataset]
    w, h = render_wh(args.dataset)
    print(f'[{SCRIPT_NAME}] dataset={args.dataset} scene={args.scene} mode={args.mode} '
          f'ambient={ambient} frames={len(targets)}')

    scene_dir = Path(data_root) / args.scene
    ensure_texture_symlink(args.mesh_root, args.scene)
    usd_path = find_scene_usd(args.mesh_root, args.scene)

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath('/World/Scene'):
        stage.RemovePrim('/World/Scene')
    cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path),
                               collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.func('/World/Scene', cfg)

    real_rgbs, c2ws, k = {}, {}, None
    rgb_dir = scene_dir / 'videos/chunk-000/observation.images.rgb'
    for ep, frame in targets:
        gt = load_gt_episode(data_root, args.scene, ep, args.dataset)
        k = gt['k'] if k is None else k
        c2ws[(ep, frame)] = gt['poses_c2w'][frame]
        real_rgbs[(ep, frame)] = load_rgb_frame(rgb_dir, ep, frame, args.dataset)

    camera_cfg = CameraCfg(
        prim_path='/World/RenderCamera', height=h * args.supersample, width=w * args.supersample,
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(RENDER_NEAR_M, RENDER_FAR_M)),
    )
    camera = Camera(cfg=camera_cfg)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.reset()
    k_render = np.asarray(k, dtype=np.float64).copy()
    k_render[:2, :] *= args.supersample  # fx,fy,cx,cy를 같이 키워야 화각이 그대로 유지된다
    camera.set_intrinsic_matrices(
        torch.tensor(k_render.astype(np.float32), device=sim.device).unsqueeze(0))
    for _ in range(10):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    suffix = f'_ss{args.supersample}' if args.supersample > 1 else ''
    log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}_{args.dataset}_{args.mode}{suffix}'
    frame_states = {t: [] for t in targets}
    rows = []

    for config in SWEEP_MODES[args.mode](ambient):
        label = config['label']
        apply_config(stage, config)
        ssims, bands = [], []
        for t in targets:
            ep, frame = t
            c2w = c2ws[t]
            quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            camera.set_world_poses(
                torch.tensor(c2w[:3, 3], dtype=torch.float32, device=sim.device).unsqueeze(0),
                torch.tensor(quat_wxyz, dtype=torch.float32, device=sim.device).unsqueeze(0),
                convention='ros',
            )
            for _ in range(args.steps):
                sim.step()
                camera.update(dt=sim.get_physics_dt())
            render_rgb = camera.data.output['rgb'][0, ..., :3].cpu().numpy()
            if args.supersample > 1:
                import cv2 as _cv2
                render_rgb = _cv2.resize(render_rgb, (w, h), interpolation=_cv2.INTER_AREA)
            real_rgb = real_rgbs[t]
            ssims.append(float(compare_ssim(real_rgb, render_rgb, channel_axis=2, data_range=255)))
            bands.append(band_deltas(render_rgb, real_rgb))
            composite = make_composite(real_rgb, render_rgb, colorize_diff(render_rgb, real_rgb), label)
            path = save_jpg(composite, log_dir / f'ep{ep}_frame{frame}' / f'{label}.jpg')
            frame_states[t].append((f'{label} (ssim={ssims[-1]:.3f})', path))
        row = dict(label=label, ambient=config.get('ambient', 0.0), film_iso=config.get('iso', 100.0),
                   dome=config.get('dome', 0.0), crush_blacks=config.get('crush', 0.5),
                   burn_highlights=config.get('burn', 0.7),
                   ssim=float(np.mean(ssims)), ssim_min=float(np.min(ssims)),
                   **{f'd_{n}': float(np.nanmean([b[n] for b in bands])) for n in ('dark', 'mid', 'bright')})
        rows.append(row)
        print(f"  {label:34s} ssim={row['ssim']:.3f} (min {row['ssim_min']:.3f})  "
              f"d_dark={row['d_dark']:+6.1f} d_mid={row['d_mid']:+6.1f} d_bright={row['d_bright']:+6.1f}",
              flush=True)

    rows_sorted = sorted(rows, key=lambda r: -r['ssim'])
    best = rows_sorted[0]
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'sweep.json').write_text(json.dumps(dict(dataset=args.dataset, scene=args.scene, rows=rows),
                                                   indent=2, ensure_ascii=False))

    table = ('<table><tr><th>설정</th><th>filmIso</th><th>crushBlacks</th><th>burnHighlights</th>'
             '<th>dome</th><th>ambient</th>'
             '<th>SSIM</th><th>SSIM min</th><th>어두움 편차</th><th>중간 편차</th><th>밝음 편차</th></tr>')
    for r in rows_sorted:
        mark = ' style="font-weight:700"' if r is best else ''
        table += (f"<tr{mark}><td>{r['label']}</td><td>{r['film_iso']:.0f}</td><td>{r['crush_blacks']:.2f}</td>"
                  f"<td>{r['burn_highlights']:.2f}</td><td>{r['dome']:.3g}</td><td>{r['ambient']:.1f}</td>"
                  f"<td>{r['ssim']:.3f}</td><td>{r['ssim_min']:.3f}</td>"
                  f"<td>{r['d_dark']:+.1f}</td><td>{r['d_mid']:+.1f}</td><td>{r['d_bright']:+.1f}</td></tr>")
    table += '</table>'

    body = table
    for t in targets:
        ep, frame = t
        body += f'<h4>ep {ep} frame {frame}</h4>' + blink_widget_html(f'f{ep}_{frame}', frame_states[t])
    summary = (f"<p>mode={args.mode}, 기준 ambient={ambient}. 편차 = (렌더-GT) 평균 밝기차를 "
               f"GT 휘도대(0~60 / 60~150 / 150~255)별로 나눈 값 — 0에 가까울수록 좋고, 부호가 "
               f"대(帶)마다 다르면 응답곡선 기울기가 틀린 것이다. 최적: <b>{best['label']}</b> "
               f"(SSIM {best['ssim']:.3f}).</p>")
    report = save_gallery(log_dir, 'report.html', f'{SCRIPT_NAME} — {args.scene} / {args.dataset}', summary, body)
    print(f"\n  BEST: {best['label']} ssim={best['ssim']:.3f} "
          f"d_dark={best['d_dark']:+.1f} d_mid={best['d_mid']:+.1f} d_bright={best['d_bright']:+.1f}")
    print(f'  report html -> {report}')
    return 0


if __name__ == '__main__':
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)
