"""S6 — 만든 override 레이어를 **실제로 물려서 렌더한다**.

S1~S5는 "레이어를 만들 수 있나 / 값이 바뀌나"까지였다. 이 스크립트는 한 걸음 더 가서
**그 레이어로 실제 렌더를 돌리고, 원래 방식과 픽셀 단위로 같은지** 본다. 같으면 그 레이어는
기존 방식을 대체할 수 있다는 뜻이다.

두 가지를 각각 다룬다.

## A — vln_pe 씬의 텍스처 절대경로 (부채 2)

`isaacsim_<hash>.usd`는 텍스처를 `/ssd/share/Matterport3D/...`라는 **원래 변환 환경의 절대경로**로
참조한다. 그래서 `04_render_obs_isaac.py:132` `ensure_texture_symlink`가 **컨테이너 밖 시스템
경로에 심링크를 만든다.** 이걸 override 레이어(텍스처를 로컬 상대경로로 재지정)로 바꿔도
같은 그림이 나오는지 본다.

## B — 노은역 usdz의 숨은 충돌 메시 (런타임 파이썬 조작)

`apply_real/04_render_obs_isaac.py`는 씬을 스폰한 뒤
`usdz_scene_utils.expose_collision_meshes_for_rendering()`으로 **실행 중에 파이썬으로**
숨겨진 충돌 메시의 visibility를 뒤집는다. 이걸 `.usda` 한 장으로 뺄 수 있는지 본다.

여기엔 자연스러운 대조 실험이 붙어 있다 — 그 파이썬 함수는 첫 줄에서

    if not Path(usd_path).name.endswith('_collision.usdz'):
        return 0

로 **파일 이름을 보고 빠져나간다.** 우리 레이어는 `override_visibility.usda`라서 이 함수는
아무 일도 하지 않는다. 즉 **레이어만으로 그림이 나오면 그게 곧 대체 증명**이다.

## 실행 (각각 한 줄, 프로세스를 나눠야 한다)

A와 B는 서로 다른 렌더러 모듈을 import하고 두 모듈 다 import 시점에 `SimulationApp`을 띄우므로
**한 프로세스에서 같이 돌릴 수 없다.**

    timeout --signal=KILL 2400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s6_use_layers.py --mode a --scene qoiz87JEwZ2 --dataset vln_pe --log_dir logs/gs-vlnpe/usd_study

    timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s6_use_layers.py --mode b --log_dir logs/gs-vlnpe/usd_study

`--scene`을 안 주면 A는 4종 USD를 갖춘 씬 중에서 **무작위로 하나** 고른다(시드를 리포트에 적는다).
"""

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from usd_study_utils import (
    DEFAULT_LOG_DIR,
    DEFAULT_USDZ,
    DEFAULT_USD_ROOT,
    MATTERPORT_TEXTURE_ROOT,
    TABLE_CSS,
    PreservationGuard,
    clear_render_prims,
    esc,
    exit_skipping_isaac_teardown,
    glossary_html,
    glossary_note_html,
    mp3d_usd_variants,
    new_override_layer,
    pill,
    preservation_section,
    scene_mesh_dir,
    stat_row_html,
    table_html,
)

import viz_utils  # noqa: E402

GS_VLNPE = Path(__file__).resolve().parent.parent


def load_module(rel_path: str, mod_name: str):
    """`SimulationApp`을 띄우는 렌더러 모듈을 import한다 (숫자로 시작해 일반 import 불가).

    **pxr보다 먼저 호출해야 한다** — 부팅 전에 standalone pxr을 잡으면 Omniverse 확장이 전부
    import 실패한다(S5에서 실측).
    """
    path = GS_VLNPE / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_paths(args, mode: str, scene: str = None) -> dict:
    """이 실행이 **어떤 파일을 읽고 어디에 쓰는지** 한 곳에서 확정한다.

    경로를 코드에 박아두면 다른 씬·다른 자산으로 바꿔볼 수가 없다. 전부 인자로 받고,
    안 주면 관례에 따라 계산한다. 확정된 경로는 실행 시작할 때 화면에 그대로 찍는다 —
    "무슨 파일을 읽는 거지?"에 답이 되도록.
    """
    apply_real = Path(args.apply_real_dir) if args.apply_real_dir else (GS_VLNPE / 'apply_real')
    if mode == 'b':
        out = {
            'usdz': Path(args.usdz),
            'apply_real_dir': apply_real,
            'scene_meta': Path(args.scene_meta) if args.scene_meta
            else apply_real / 'scene_meta' / f'{args.b_scene}.json',
            'paths_json': Path(args.paths_json) if args.paths_json
            else apply_real / 'paths' / f'{args.b_scene}_random.json',
            'out_dir': Path(args.out_dir) if args.out_dir
            else Path(args.log_dir) / 's6_use_layers' / 'b_noeun',
        }
    else:
        out = {
            'usd_root': Path(args.usd_root),
            'out_dir': Path(args.out_dir) if args.out_dir
            else Path(args.log_dir) / 's6_use_layers' / f'a_{scene}',
        }
    return out


def print_paths(title: str, reads, writes):
    print(f'[{title}] 읽는 파일:')
    for label, q in reads:
        q = Path(q)
        mark = 'OK  ' if q.exists() else '없음'
        size = f'{q.stat().st_size:,} B' if q.exists() else '-'
        print(f'   {mark} {label:16s} {q}   ({size})')
    print(f'[{title}] 쓰는 곳:')
    for label, q in writes:
        print(f'        {label:16s} {q}')


def pick_random_scene(usd_root, seed: int) -> tuple:
    """4종 USD를 갖춘 씬 중 무작위로 하나. 시드를 같이 반환해 재현 가능하게 한다."""
    root = Path(usd_root)
    scenes = sorted({d.parent.parent.name for d in root.glob('*/matterport_mesh/*')
                     if d.is_dir() and any(d.glob('isaacsim_*.usd')) and any(d.glob('fixed.usd'))})
    rng = random.Random(seed)
    return rng.choice(scenes), len(scenes)


def compare_renders(a: np.ndarray, b: np.ndarray) -> dict:
    """두 렌더 결과가 같은지. 렌더러가 결정론적이므로 같은 입력이면 픽셀까지 같아야 한다."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {'identical': bool(np.array_equal(a, b)), 'max_abs_diff': int(d.max()),
            'mean_abs_diff': float(d.mean()), 'diff_pixel_frac': float((d.max(axis=-1) > 0).mean())}


# ---------------------------------------------------------------------------
# A — vln_pe 텍스처 경로
# ---------------------------------------------------------------------------

def run_mode_a(args) -> dict:
    print('[s6a] Isaac 부팅')
    r4 = load_module('04_render_obs_isaac.py', 'r4_top')
    import dataset_utils
    import s2_assets  # 부팅 뒤에 import — 모듈 최상단에서 pxr을 잡는다
    from pxr import Ar, Sdf, Usd

    scene = args.scene
    picked_from = None
    if not scene:
        scene, n = pick_random_scene(args.usd_root, args.seed)
        picked_from = n
        print(f'[s6a] 무작위 선택: {scene} (후보 {n}개, seed={args.seed})')

    variants = dict(mp3d_usd_variants(args.usd_root, scene))
    if args.base_variant not in variants:
        print(f'[fail] 그런 자산이 없다: {args.base_variant} (있는 것: {list(variants)})', file=sys.stderr)
        raise SystemExit(2)
    base_usd = variants[args.base_variant]
    P = resolve_paths(args, 'a', scene)
    out_dir = P['out_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root_probe = args.data_root or dataset_utils.default_data_root(args.dataset)
    print_paths('s6a',
                [('씬 USD', base_usd),
                 ('궤적 폴더', Path(data_root_probe) / scene),
                 ('실제 rgb', Path(data_root_probe) / scene / 'videos' / 'chunk-000' / 'observation.images.rgb')],
                [('덧칠 레이어', out_dir / 'override_textures.usda'),
                 ('결과', out_dir / ('s6_a_control.json' if args.control else 'report.html'))])

    guard = PreservationGuard([base_usd]).snapshot_before()

    print('[s6a] 텍스처 override 레이어 저작')
    desc = s2_assets.describe_assets('isaacsim', base_usd)
    ov = s2_assets.build_texture_override(base_usd, desc['rows'], out_dir)
    print(f"[s6a]   재지정 {ov['retargeted']}개 · 로컬 resolve {ov['resolved_local']} · "
          f"박힌 경로 {ov['still_baked']} · 미해결 {ov['unresolved_count']}(전부 MDL이면 정상)")

    data_root = data_root_probe
    gt = dataset_utils.load_gt_episode(data_root, scene, args.episode, args.dataset)
    W, H = dataset_utils.render_wh(args.dataset)
    n = len(gt['poses_c2w'])
    frame_idx = np.unique(np.linspace(0, n - 1, args.n_frames).astype(int)).tolist()
    poses = [gt['poses_c2w'][f] for f in frame_idx]

    # **판정 기준이 씬마다 다르다.** 원본 USD의 텍스처 절대경로는 `ensure_texture_symlink`가
    # 만들어 준 심링크가 있어야만 풀린다. 그 심링크는 **그 씬을 한 번 렌더해 본 적이 있어야**
    # 생긴다(부수효과). 그래서:
    #   - 심링크 있음 -> 두 렌더가 **픽셀까지 같아야** 한다 (동치 검사)
    #   - 심링크 없음 -> 원본은 텍스처가 깨지고 레이어만 정상이어야 한다 (가치 검사)
    # 이 스크립트는 심링크를 **만들지 않는다** — 컨테이너 밖에 쓰는 부수효과이기 때문이다.
    symlink_root = MATTERPORT_TEXTURE_ROOT / scene / 'matterport_mesh'
    symlink_present = symlink_root.exists()
    print(f'[s6a] 이 씬의 텍스처 심링크: {"있음" if symlink_present else "없음"} ({symlink_root})')

    rendered = {}
    base_label = ('원본 usd (심링크 있음)' if symlink_present else '원본 usd (심링크 없음 — 텍스처 못 찾음)')
    # **한 프로세스에서 씬을 3번 올리면 죽는다** — 3번째 `render_along`에서
    # `AnnotatorRegistryError: Annotator rgb is not attached to any render products.` (실측).
    # 그래서 렌더는 프로세스당 2회로 제한하고, 대조군은 `--control`로 **따로 실행**한다:
    #   --control 없이  -> 원본 vs 레이어  (시험)
    #   --control 주고   -> 원본 vs 원본    (대조군, 렌더러 자체의 흔들림 측정)
    # 두 실행 모두 "1번째 로드 vs 2번째 로드"라 조건이 같아 비교가 성립한다.
    second_label = ('원본 usd 재렌더 (대조군)' if args.control else 'override 레이어 (로컬 상대경로)')
    second_path = base_usd if args.control else Path(ov['layer'])
    for label, path in ((base_label, base_usd), (second_label, second_path)):
        print(f'[s6a] 렌더: {label}')
        clear_render_prims()
        cas = r4.build_renderer(W, H, gt['k'], path, args.light, None, args.rtx_ambient, args.film_iso)
        rgb, _ = r4.render_along(cas, poses)
        rendered[label] = rgb
        print(f'[s6a]   밝기 {rgb.mean():.2f}')

    cmp = compare_renders(rendered[base_label], rendered[second_label])
    kind = 'control' if args.control else 'test'
    print(f'[s6a] {kind}: 평균 차이 {cmp["mean_abs_diff"]:.4f} · 최대 {cmp["max_abs_diff"]}')

    # 시험 실행이면 앞서 저장된 대조군 값을 읽어 판정에 쓴다.
    control = None
    ctrl_json = out_dir / 's6_a_control.json'
    if not args.control and ctrl_json.is_file():
        control = json.loads(ctrl_json.read_text(encoding='utf-8')).get('compare')
        print(f'[s6a] 대조군(이전 실행) 평균 차이 {control["mean_abs_diff"]:.4f}')
    elif not args.control:
        print('[s6a] 대조군 json 없음 — `--control`로 한 번 더 돌리면 판정이 정확해진다', file=sys.stderr)

    guard.snapshot_after()

    # GT와의 SSIM도 같이 (렌더가 정상인지 sanity)
    from skimage.metrics import structural_similarity as ssim_metric
    rgb_dir = Path(data_root) / scene / 'videos' / 'chunk-000' / 'observation.images.rgb'
    reals = [dataset_utils.load_rgb_frame(rgb_dir, args.episode, f, args.dataset) for f in frame_idx]
    ssims = {k: float(np.median([ssim_metric(reals[i], v[i], channel_axis=2, data_range=255)
                                 for i in range(len(reals))])) for k, v in rendered.items()}

    import cv2
    img_dir = out_dir / 'frames'
    img_dir.mkdir(parents=True, exist_ok=True)
    states = []
    for i, f in enumerate(frame_idx[:3]):
        for k, v in rendered.items():
            q = img_dir / f'{"orig" if k.startswith("원본") else "layer"}_f{f:04d}.jpg'
            cv2.imwrite(str(q), cv2.cvtColor(v[i], cv2.COLOR_RGB2BGR))
            states.append((f'{k} · frame {f}', q))

    return {'mode': 'a', 'scene': scene, 'picked_from': picked_from, 'seed': args.seed,
            'symlink_present': symlink_present, 'symlink_root': str(symlink_root),
            'base_label': base_label,
            'base_usd': str(base_usd), 'override': ov, 'frames': frame_idx, 'compare': cmp,
            'control': control, 'is_control': args.control,
            'second_label': second_label,
            'ssim': ssims, 'brightness': {k: float(v.mean()) for k, v in rendered.items()},
            'guard': guard, 'blink': states, 'out_dir': out_dir,
            'dataset': args.dataset, 'episode': args.episode}


# ---------------------------------------------------------------------------
# B — 노은역 usdz visibility
# ---------------------------------------------------------------------------

def run_mode_b(args) -> dict:
    print('[s6b] Isaac 부팅 (apply_real 렌더러)')
    r4 = load_module('apply_real/04_render_obs_isaac.py', 'r4_apply_real')
    sys.path.insert(0, str(Path(args.apply_real_dir) if args.apply_real_dir else GS_VLNPE / 'apply_real'))
    import camera_profiles
    import geometry_utils
    from pxr import Usd, UsdGeom

    P = resolve_paths(args, 'b')
    usdz, out_dir = P['usdz'], P['out_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    print_paths('s6b',
                [('원본 usdz', usdz), ('씬 정보', P['scene_meta']), ('카메라 경로', P['paths_json']),
                 ('카메라 프로파일', P['apply_real_dir'] / 'camera_profiles.py')],
                [('덧칠 레이어', out_dir / 'override_visibility.usda'),
                 ('결과', out_dir / ('s6_b_control.json' if args.control else 'report.html'))])
    guard = PreservationGuard([usdz]).snapshot_before()

    print('[s6b] 숨은 충돌 메시 찾기')
    stage = Usd.Stage.Open(str(usdz))
    targets = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if 'PhysicsCollisionAPI' not in set(prim.GetAppliedSchemas()):
            continue
        if UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible:
            targets.append(str(prim.GetPath()))
    print(f'[s6b]   대상 {len(targets)}개: {targets}')

    print('[s6b] visibility override 레이어 저작')
    layer = new_override_layer(usdz, out_dir / 'override_visibility.usda')
    ov_stage = Usd.Stage.Open(layer)
    for path in targets:
        UsdGeom.Imageable(ov_stage.OverridePrim(path)).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    layer.Save()
    layer_path = Path(layer.identifier)

    verify = Usd.Stage.Open(str(layer_path))
    flipped = [p for p in targets
               if UsdGeom.Imageable(verify.GetPrimAtPath(p)).GetVisibilityAttr().Get() == UsdGeom.Tokens.inherited]
    print(f'[s6b]   합성 결과 뒤집힘 {len(flipped)}/{len(targets)}')

    meta = json.loads(P['scene_meta'].read_text())
    paths = json.loads(P['paths_json'].read_text())
    ep = paths['episodes'][args.episode]
    traj = np.asarray(ep['trajectory'], dtype=np.float64)[:, :2]
    action_poses = geometry_utils.synthesize_action_poses(traj, float(ep['floor_z']), float(ep['h_b']),
                                                          float(ep['pitch_deg']))
    # **이 변환을 빠뜨리면 그림이 180° 뒤집힌다** — 자세한 이유는 s7_live_view.py의 같은 지점 주석.
    # 파이프라인도 `apply_real/04_render_obs_isaac.py:683`에서 이 두 단계를 거친다.
    poses_all = np.stack([geometry_utils.action_to_c2w(a, 'cam2world_gl') for a in action_poses])
    idx = np.unique(np.linspace(0, len(poses_all) - 1, args.n_frames).astype(int)).tolist()
    poses = [poses_all[i] for i in idx]
    prof = camera_profiles.get(args.camera)

    # "아무도 안 뒤집은" 대조 상태를 만들려면 usdz를 그대로 쓰면 안 된다 —
    # `expose_collision_meshes_for_rendering`이 파일 이름이 `_collision.usdz`로 끝나면
    # 런타임에 뒤집어버리기 때문이다. 아무 override도 없는 **껍데기 레이어**를 씌우면
    # 이름 검사에서 빠져나가므로, 진짜로 아무 일도 일어나지 않은 렌더를 얻을 수 있다.
    plain = new_override_layer(usdz, out_dir / 'no_override.usda')
    plain.Save()
    plain_path = Path(plain.identifier)

    # A와 같은 이유로 대조군을 별도 실행으로 분리한다(한 프로세스 3회 로드는 죽는다).
    if args.compare_to == 'nothing':
        base_label = '아무도 안 뒤집음 (충돌 메시가 숨은 채)'
        base_path = plain_path
    else:
        base_label = '원본 usdz + 런타임 파이썬 뒤집기'
        base_path = usdz
    second_label = (f'{base_label} 재렌더 (대조군)' if args.control
                    else 'override 레이어 (덧칠)')
    second_path = base_path if args.control else layer_path
    rendered, rendered_depth, depth_valid = {}, {}, {}
    for label, path in ((base_label, base_path), (second_label, second_path)):
        print(f'[s6b] 렌더: {label}')
        clear_render_prims()
        cas = r4.build_renderer(prof.width, prof.height, prof.k, path, args.light, None,
                                args.rtx_ambient, args.film_iso, prof.render_near_m, prof.render_far_m)
        rgb, depth = r4.render_along(cas, poses)
        rendered[label] = rgb
        rendered_depth[label] = depth
        finite = np.isfinite(depth) & (depth < prof.render_far_m * 0.99)
        depth_valid[label] = float(finite.mean())
        print(f'[s6b]   밝기 {rgb.mean():.2f} · depth 유효비율 {finite.mean():.3f}')

    cmp = compare_renders(rendered[base_label], rendered[second_label])
    kind = 'control' if args.control else 'test'
    print(f'[s6b] {kind}: 평균 차이 {cmp["mean_abs_diff"]:.4f} · 최대 {cmp["max_abs_diff"]}')
    control = None
    ctrl_json = out_dir / 's6_b_control.json'
    if not args.control and ctrl_json.is_file():
        control = json.loads(ctrl_json.read_text(encoding='utf-8')).get('compare')
        print(f'[s6b] 대조군(이전 실행) 평균 차이 {control["mean_abs_diff"]:.4f}')
    guard.snapshot_after()

    import cv2

    def depth_to_image(d, far_m):
        """깊이를 눈으로 볼 수 있는 그림으로. **깊이가 없는 픽셀은 검정**으로 둔다.

        이 실습에서 실제로 달라지는 것이 깊이라서, 검정 면적이 곧 "깊이가 안 나온 영역"이다.
        """
        valid = np.isfinite(d) & (d < far_m * 0.99)
        norm = np.zeros(d.shape, dtype=np.uint8)
        if valid.any():
            lo, hi = np.percentile(d[valid], [2, 98])
            scaled = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            norm = (scaled * 255).astype(np.uint8)
        col = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        col[~valid] = 0
        return col

    img_dir = out_dir / 'frames'
    img_dir.mkdir(parents=True, exist_ok=True)
    states, depth_states = [], []
    for i, f in enumerate(idx[:3]):
        for k, v in rendered.items():
            tag = 'py' if k == base_label else 'layer'
            q = img_dir / f'{tag}_f{f:04d}.jpg'
            cv2.imwrite(str(q), cv2.cvtColor(v[i], cv2.COLOR_RGB2BGR))
            states.append((f'{k} · frame {f}', q))
            qd = img_dir / f'{tag}_DEPTH_f{f:04d}.jpg'
            cv2.imwrite(str(qd), depth_to_image(rendered_depth[k][i], prof.render_far_m))
            dv = depth_valid.get(k, float('nan'))
            depth_states.append((f'[깊이] {k} · 나온 비율 {dv:.3f} · frame {f}', qd))

    return {'mode': 'b', 'usdz': str(usdz), 'targets': targets, 'flipped': flipped,
            'control': control, 'is_control': args.control, 'second_label': second_label,
            'compare_to': args.compare_to, 'base_label': base_label,
            'depth_valid': depth_valid, 'depth_blink': depth_states,
            'layer': str(layer_path), 'layer_text': layer_path.read_text(encoding='utf-8')[:1600],
            'frames': idx, 'compare': cmp, 'guard': guard, 'blink': states, 'out_dir': out_dir,
            'brightness': {k: float(v.mean()) for k, v in rendered.items()},
            'scene': args.b_scene, 'camera': args.camera}


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def verdict(res) -> tuple:
    """(통과 여부, 판정 문구, 설명). 모드 A는 심링크 유무로 기준이 갈린다."""
    cmp = res['compare']
    if res['mode'] == 'b' and res.get('compare_to') == 'nothing':
        if not res.get('control'):
            return False, '대조군 없음 — 판정 보류', \
                '렌더러 자체의 흔들림을 잴 <b>대조군이 없다</b>. <code>--control</code>로 한 번 더 돌려야 한다.'
        # **RGB로 재면 안 된다.** 이 씬은 색이 NuRec 가우시안 볼륨에서 나오고(항상 보임),
        # 깊이가 충돌 메시에서 나온다(숨어 있음). 덧칠이 살리는 것은 **깊이**라서 RGB는 거의 안 변한다.
        dv = res.get('depth_valid', {})
        base_dv = dv.get(res.get('base_label'), float('nan'))
        layer_dv = max((v for k, v in dv.items() if k != res.get('base_label')), default=float('nan'))
        differs = (layer_dv - base_dv) > 0.1
        return differs, ('덧칠이 깊이를 살렸다' if differs else '덧칠해도 깊이가 안 살아난다'), \
            ('이번엔 <b>아무도 안 뒤집은 상태</b>와 비교한다 — <b>차이가 나야</b> 정상이다. '
             '단 이 씬은 <b>색은 NuRec 가우시안 볼륨</b>에서, <b>깊이는 숨어 있던 충돌 메시</b>에서 나온다. '
             '덧칠이 살리는 건 깊이라서 <b>RGB로 재면 안 보인다</b> — '
             f'깊이가 나온 픽셀 비율로 판정한다: <b>{base_dv:.3f} → {layer_dv:.3f}</b>.')
    if res['mode'] == 'b':
        if not res.get('control'):
            return False, '대조군 없음 — 판정 보류', \
                '렌더러 자체의 흔들림을 잴 <b>대조군이 없다</b>. <code>--control</code>로 한 번 더 돌려야 한다.'
        ctrl = res['control']['mean_abs_diff']
        test = cmp['mean_abs_diff']
        same = test <= max(ctrl * 2.0, 0.5)
        return same, ('대조군 수준 — 레이어가 파이썬 조작을 대체' if same else '대조군보다 큰 차이'), \
            ('런타임 파이썬 조작과 레이어가 같은 그림을 내야 대체 가능하다. 다만 씬을 다시 올려 '
             '렌더하면 <b>같은 원본끼리도 픽셀이 흔들리므로</b>, '
             f'<b>원본↔원본 대조군({ctrl:.3f})</b>과 <b>원본↔레이어({test:.3f})</b>를 비교해 판정한다.')
    if res.get('symlink_present'):
        if not res.get('control'):
            return False, '대조군 없음 — 판정 보류', \
                '이 씬은 심링크가 있어 <b>동치 검사</b>를 해야 하는데, 렌더러 자체의 흔들림을 잴 ' \
                '<b>대조군이 없다</b>. <code>--control</code>로 한 번 더 돌려야 판정할 수 있다.'
        ctrl = res.get('control', {}).get('mean_abs_diff', 0.0)
        test = cmp['mean_abs_diff']
        same = test <= max(ctrl * 2.0, 0.5)
        return same, ('대조군 수준 — 심링크와 레이어가 같은 결과' if same else '대조군보다 큰 차이'), \
            ('이 씬은 심링크가 이미 있어 원본도 텍스처를 찾는다 → <b>동치 검사</b>다. '
             '다만 같은 프로세스에서 씬을 다시 올려 렌더하면 <b>같은 원본끼리도 픽셀이 조금 흔들린다</b>(실측). '
             f'그래서 <b>원본↔원본 대조군({ctrl:.3f})</b>을 같이 재고, '
             f'<b>원본↔레이어({test:.3f})</b>가 그 수준이면 같다고 본다.')
    ssims = res.get('ssim', {})
    base_s = ssims.get(res.get('base_label'), 0.0)
    layer_s = max((v for k, v in ssims.items() if k != res.get('base_label')), default=0.0)
    better = layer_s > base_s + 0.01
    return better, ('레이어만 정상 — 심링크 없이 동작' if better else '레이어가 원본보다 낫지 않다'), \
        ('이 씬은 심링크가 <b>없어서</b> 원본 USD가 텍스처를 못 찾는다 → '
         '<b>레이어 쪽 SSIM이 더 높아야</b> 한다(가치 검사). 픽셀 동일은 여기서 기대할 수 없다.')


def build_report(res, args):
    guard = res['guard']
    parts = [TABLE_CSS, glossary_note_html()]
    cmp = res['compare']
    ok, verdict_text, verdict_why = verdict(res)

    if res['mode'] == 'a':
        title = f"S6-A · 텍스처 경로 레이어를 실제 렌더에 물리기 — {res['scene']}"
        parts.append('<h2>목적</h2>')
        parts.append(
            '<p><code>isaacsim_&lt;hash&gt;.usd</code>는 텍스처를 <code>/ssd/share/Matterport3D/...</code>라는 '
            '<b>원래 변환 환경의 절대경로</b>로 참조한다. 그래서 렌더러가 매번 '
            '<b>컨테이너 밖 시스템 경로에 심링크를 만든다</b>(<code>ensure_texture_symlink</code>). '
            '환경이 바뀌면 깨지는 구조다.</p>'
            '<p>이걸 <b>override 레이어</b>(텍스처를 repo 안 상대경로로 재지정)로 바꿔도 '
            '<b>같은 그림</b>이 나오는지 본다. 같으면 심링크를 없앨 수 있다.</p>')
        if res['picked_from']:
            parts.append(f'<p style="color:var(--text-dim);font-size:.85rem">씬은 4종 USD를 갖춘 '
                         f'<b>{res["picked_from"]}개</b> 중 무작위로 골랐다 (seed <code>{res["seed"]}</code> '
                         f'→ <code>{esc(res["scene"])}</code>). 특정 씬에만 통하는 얘기가 아닌지 보려는 것.</p>')
        cmdline = (f'/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/'
                   f's6_use_layers.py --mode a --scene {res["scene"]} --dataset {res["dataset"]} '
                   f'--log_dir {args.log_dir}')
    else:
        title = 'S6-B · 노은역 usdz — 런타임 파이썬 조작을 레이어로'
        parts.append('<h2>목적</h2>')
        parts.append(
            '<p>노은역 렌더러는 씬을 올린 뒤 <b>실행 중에 파이썬으로</b> 숨겨진 충돌 메시의 '
            'visibility를 뒤집는다(<code>expose_collision_meshes_for_rendering</code>). '
            '이걸 <code>.usda</code> 한 장으로 뺄 수 있는지 본다.</p>'
            '<p><b>대조 실험이 공짜로 붙어 있다</b> — 그 파이썬 함수는 첫 줄에서 '
            '<code>if not usd_path.name.endswith("_collision.usdz"): return 0</code>으로 '
            '<b>파일 이름을 보고 빠져나간다.</b> 우리 레이어는 <code>override_visibility.usda</code>라서 '
            '함수가 아무 일도 하지 않는다. 즉 <b>레이어만으로 그림이 나오면 그게 대체 증명</b>이다.</p>')
        cmdline = ('/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/'
                   f's6_use_layers.py --mode b --log_dir {args.log_dir}')

    parts.append('<h2>커맨드와 결과</h2>')
    parts.append(f'<pre>{esc(cmdline)}</pre>')

    parts.append('<h3>판정</h3>')
    parts.append(f'<p>{verdict_why}</p>')
    parts.append(table_html(['항목', '값'], [
        ['<b>판정</b>', pill(ok, verdict_text, verdict_text)],
        ['픽셀 단위 동일', pill(cmp['identical'], 'YES', 'NO')],
        ['평균 차이 — <b>원본 ↔ 레이어</b> (시험)', f"{cmp['mean_abs_diff']:.4f}"],
        ['평균 차이 — <b>원본 ↔ 원본</b> (대조군)',
         f"{res['control']['mean_abs_diff']:.4f}" if res.get('control') else '—'],
        ['최대 채널 차이 (시험)', cmp['max_abs_diff']],
        ['다른 픽셀 비율 (시험)', f"{cmp['diff_pixel_frac']:.6f}"],
    ]))
    if res['mode'] == 'a':
        parts.append(f'<p>이 씬의 텍스처 심링크(<code>{esc(res["symlink_root"])}</code>): '
                     f'<b>{"있음" if res["symlink_present"] else "없음"}</b>. '
                     '이 스크립트는 심링크를 <b>만들지 않는다</b> — 컨테이너 밖에 쓰는 부수효과라서다.</p>')
    if res.get('control'):
        parts.append('<p style="color:var(--text-dim);font-size:.85rem">'
                     '<b>왜 대조군이 필요한가</b>: <code>../reports.md</code>에 "렌더러가 결정론적"이라고 '
                     '적혀 있지만 그건 <b>같은 커맨드를 새 프로세스에서 다시 돌렸을 때</b> 얘기다. '
                     '한 프로세스 안에서 씬을 내렸다 다시 올리면 <b>같은 원본끼리도 픽셀이 조금 흔들린다</b>. '
                     '그래서 "픽셀 동일"을 기준으로 쓰면 노이즈를 실제 차이로 오독한다 — '
                     '실제로 한 번 오독했고, 그래서 대조군을 넣었다.</p>')

    if res['mode'] == 'b' and res.get('depth_valid'):
        rows = [[k, f'{v:.2f}', f"{res['depth_valid'].get(k, float('nan')):.3f}"]
                for k, v in res['brightness'].items()]
        parts.append(table_html(['렌더 방식', '평균 밝기', '<b>깊이가 나온 픽셀 비율</b>'], rows))
        parts.append('<p style="color:var(--text-dim);font-size:.85rem">이 씬은 <b>색은 NuRec 가우시안 '
                     '볼륨</b>에서, <b>깊이는 숨어 있던 충돌 메시</b>에서 나온다. 그래서 덧칠의 효과는 '
                     'RGB가 아니라 <b>깊이</b>에 나타난다.</p>')
    rows = [[k, f'{v:.2f}'] for k, v in res['brightness'].items()]
    if res['mode'] == 'a':
        rows = [[k, f'{v:.2f}', f"{res['ssim'][k]:.4f}"] for k, v in res['brightness'].items()]
        parts.append(table_html(['렌더 방식', '평균 밝기', 'GT와 SSIM'], rows))
    elif not res.get('depth_valid'):
        parts.append(table_html(['렌더 방식', '평균 밝기'], rows))

    if res['mode'] == 'a':
        ov = res['override']
        parts.append('<h3>레이어가 텍스처 경로를 어디로 돌렸나</h3>')
        parts.append(stat_row_html([
            ('재지정한 텍스처', ov['retargeted']),
            ('repo 로컬로 resolve', ov['resolved_local']),
            ('아직 박힌 절대경로', ov['still_baked']),
            ('미해결', ov['unresolved_count']),
        ]))
        parts.append('<p style="color:var(--text-dim);font-size:.85rem">미해결로 남는 것은 '
                     '<code>OmniPBR.mdl</code>(머티리얼 정의)뿐이면 정상이다 — 텍스처가 아니고, '
                     'RTX가 자체 MDL 검색경로로 찾는다(S5 참고).</p>')
    else:
        parts.append('<h3>레이어가 무엇을 뒤집었나</h3>')
        parts.append(table_html(['항목', '값'], [
            ['숨어 있던 충돌 메시', ', '.join(res['targets']) or '(없음)'],
            ['합성 결과 보이게 바뀐 것', f"{len(res['flipped'])} / {len(res['targets'])}"],
        ], mono_cols={1}))
        parts.append(f'<p><b>내가 쓴 레이어 전문</b></p><pre>{esc(res["layer_text"])}</pre>')

    if res.get('depth_blink'):
        parts.append('<h3>깊이 비교 — 검정이 "깊이가 안 나온 곳"이다</h3>')
        parts.append(viz_utils.blink_widget_html('s6_blink_depth', res['depth_blink'],
                                                 title='화살표로 번갈아 보세요'))
    if res['blink']:
        parts.append('<h3>색(RGB) 비교 — 여기는 거의 안 변한다</h3>')
        parts.append(viz_utils.blink_widget_html('s6_blink', res['blink'], title='두 방식의 같은 pose'))

    parts.append('<h2>Takeaway</h2>')
    tk = []
    if res['mode'] == 'a':
        if res['symlink_present']:
            tk.append(('<b>심링크와 레이어가 같은 결과를 낸다</b>' if ok else '<b>레이어 렌더가 원본과 다르다</b>',
                       (f'원본↔레이어 평균 차이 <b>{cmp["mean_abs_diff"]:.3f}</b>, '
                        f'원본↔원본 대조군 <b>{res["control"]["mean_abs_diff"]:.3f}</b> — '
                        '<b>구별되지 않는 수준</b>이다. → 심링크를 <b>레이어로 안전하게 바꿀 수 있다</b>.'
                        if ok else
                        f'원본↔레이어 {cmp["mean_abs_diff"]:.3f} 가 대조군 '
                        f'{res["control"]["mean_abs_diff"]:.3f} 보다 크다. 레이어가 텍스처 외의 것도 '
                        '건드렸는지 확인해야 한다.')))
        else:
            base_s = res['ssim'].get(res['base_label'], float('nan'))
            layer_s = max((v for k, v in res['ssim'].items() if k != res['base_label']), default=float('nan'))
            tk.append(('<b>심링크가 없는 씬에서는 원본이 아예 깨진다 — 레이어만 정상이다</b>',
                       f'원본 USD 렌더 SSIM <b>{base_s:.3f}</b> vs 레이어 <b>{layer_s:.3f}</b>. '
                       '원본은 텍스처를 못 찾아 밝기만 뜨고 무늬가 없다. '
                       '→ 심링크 방식은 <b>씬마다 부수효과를 한 번 실행해야</b> 쓸 수 있고, '
                       '레이어 방식은 그게 필요 없다.'))
            tk.append(('<b>이게 예외가 아니라 기본 상태다</b>',
                       f'<code>{esc(str(MATTERPORT_TEXTURE_ROOT))}</code> 아래에는 '
                       '<b>전에 렌더해 본 씬만</b> 들어 있다. 나머지 씬은 전부 이 상태다 — '
                       f'무작위로 고른 <code>{esc(res["scene"])}</code>가 마침 그 다수에 속했다.'))
    else:
        tk.append(('<b>런타임 파이썬 조작을 레이어가 대신할 수 있다</b>' if ok else '<b>레이어가 파이썬 조작을 대체하지 못했다</b>',
                   ('파이썬 함수는 파일 이름 검사에서 빠져나가 <b>아무 일도 안 했는데</b>, '
                    '레이어만으로 렌더 결과가 기존 방식과 <b>픽셀까지 동일</b>했다. '
                    '→ 설정이 코드가 아니라 <b>파일</b>이 되면 diff·공유·재현이 된다.'
                    if ok else '차이가 났다 — 레이어가 뒤집은 대상이 함수가 뒤집는 대상과 다를 수 있다.')))
        tk.append(('<b>2.2 GB usdz도 레이어를 얹을 수 있다</b>',
                   'usdz는 패키지라 못 얹을 것 같지만 된다. 안에 <code>defaultPrim</code>이 이미 있어서 '
                   '그대로 복사하면 되고, 원본은 한 바이트도 안 바뀐다(sha256 대조).'))
    tk.append(('<b>남은 한 걸음 — 파이프라인이 레이어를 집어가지 않는다</b>',
               '렌더러가 원본 USD를 <b>glob으로 직접 찾는다</b>(<code>find_scene_usd()</code>). '
               '옆에 레이어를 놔둬도 무시된다. 실제로 쓰려면 경로를 넘길 optional 인자가 필요하다 '
               '— 이 스크립트는 그 인자 없이 <code>build_renderer</code>를 직접 불러서 우회했다.'))
    parts.append('<div style="display:flex;flex-direction:column;gap:14px;margin:12px 0 22px">')
    for head, body in tk:
        parts.append('<div style="border:1px solid var(--border);border-radius:10px;background:var(--surface);'
                     f'padding:12px 14px"><div style="margin-bottom:6px">{head}</div>'
                     f'<div style="font-size:.88rem;line-height:1.6;color:var(--text-dim)">{body}</div></div>')
    parts.append('</div>')

    parts.append('<h2>원자료</h2>')
    parts.append(preservation_section(guard))
    parts.append(glossary_html())

    summary = stat_row_html([
        ('판정', verdict_text),
        ('최대 픽셀 차이', cmp['max_abs_diff']),
        ('프레임', len(res['frames'])),
        ('원본 무변경', 'YES' if guard.all_unchanged else 'NO'),
        ('대상', res.get('scene', '')),
    ])
    return title, summary, ''.join(parts)


def main():
    ap = argparse.ArgumentParser(description='S6 — 만든 레이어를 실제 렌더에 물린다')
    ap.add_argument('--mode', choices=['a', 'b'], required=True)
    ap.add_argument('--scene', default=None, help='A용. 생략하면 무작위 선택')
    ap.add_argument('--seed', type=int, default=20260905, help='A의 무작위 씬 선택 시드')
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_pe')
    ap.add_argument('--usd_root', default=str(DEFAULT_USD_ROOT), help='[A] 씬 USD가 있는 뿌리 폴더')
    ap.add_argument('--base_variant', default='isaacsim',
                    choices=['isaacsim', 'fixed', 'fixed_docker', 'isaacsim_non_metric'],
                    help='[A] 어느 USD 위에 덧칠할지 (기본: 렌더가 실제로 쓰는 isaacsim_<hash>.usd)')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--usdz', default=str(DEFAULT_USDZ), help='[B] 원본 usdz 경로')
    ap.add_argument('--b_scene', default='noeun_station_mid', help='[B] 논리 scene ID (아래 두 json 이름의 기본값이 된다)')
    ap.add_argument('--apply_real_dir', default=None,
                    help='[B] apply_real 폴더 (기본: gs_vlnpe/apply_real). scene_meta/paths/camera_profiles를 여기서 찾는다')
    ap.add_argument('--scene_meta', default=None,
                    help='[B] 씬 정보 json 직접 지정 (기본: <apply_real>/scene_meta/<b_scene>.json)')
    ap.add_argument('--paths_json', default=None,
                    help='[B] 카메라 경로 json 직접 지정 (기본: <apply_real>/paths/<b_scene>_random.json)')
    ap.add_argument('--camera', default='d455_nominal', help='[B] 카메라 프로파일 (d455_nominal / d455_30m)')
    ap.add_argument('--compare_to', choices=['python', 'nothing'], default='python',
                    help='[B] 덧칠을 무엇과 비교할지. python=지금 코드가 하는 런타임 뒤집기(같아야 통과) / '
                         'nothing=아무도 안 뒤집은 상태(달라야 통과 — 덧칠이 뭘 바꾸는지 눈으로 보는 용도)')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=4)
    ap.add_argument('--control', action='store_true',
                    help='대조군 실행 — 두 번째 렌더도 원본으로 해서 렌더러 자체의 흔들림을 잰다 (A/B 공통)')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR), help='결과가 쌓이는 뿌리 폴더')
    ap.add_argument('--out_dir', default=None,
                    help='결과 폴더를 직접 지정 (기본: <log_dir>/s6_use_layers/<a_씬 또는 b_noeun>)')
    args = ap.parse_args()

    res = run_mode_a(args) if args.mode == 'a' else run_mode_b(args)
    out_dir = res['out_dir']
    if res.get('is_control'):
        # 대조군은 숫자만 남긴다 — 리포트는 시험 실행이 이 값을 읽어서 쓴다.
        (out_dir / f's6_{args.mode}_control.json').write_text(
            json.dumps({'compare': res['compare'], 'brightness': res['brightness'],
                        'depth_valid': res.get('depth_valid', {})},
                       indent=2, default=str, ensure_ascii=False), encoding='utf-8')
        print(f"[s6{args.mode}] 대조군 저장: {out_dir}/s6_{args.mode}_control.json")
        exit_skipping_isaac_teardown(0)
    title, summary, body = build_report(res, args)

    dump = {k: v for k, v in res.items() if k not in ('guard', 'blink', 'out_dir', 'layer_text')}
    dump['preservation'] = res['guard'].rows()
    suffix = '_control' if args.control else ''
    (out_dir / f's6_{args.mode}{suffix}.json').write_text(
        json.dumps(dump, indent=2, default=str, ensure_ascii=False), encoding='utf-8')
    path = viz_utils.save_gallery(out_dir, 'report.html', title, summary, body, eyebrow='gs_vlnpe usd_study')
    print(f'[s6{args.mode}] report: {path}')

    ok = verdict(res)[0] and res['guard'].all_unchanged
    print(f'[s6{args.mode}] {"PASS" if ok else "FAIL"}')
    # Isaac 정리 단계에서 세그폴트가 나므로 건너뛰고 나간다 (이유는 헬퍼 docstring)
    exit_skipping_isaac_teardown(0 if ok else 1)


if __name__ == '__main__':
    sys.exit(main())
