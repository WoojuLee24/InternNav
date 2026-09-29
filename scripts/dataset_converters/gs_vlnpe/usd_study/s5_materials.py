"""S5 — 머티리얼: 잔차 `1-SSIM = 0.096`에 머티리얼 몫이 있는가.

## 왜 이 스크립트가 있는가

`reports.md`의 "남은 차이는 조명으로 못 고친다" 절 결론은 이랬다:

    어떤 렌더 설정을 써도 최대 +0.034이고 남는 `1-SSIM = 0.096`은 **메쉬/텍스처 충실도**다.

그런데 그 결론에 도달하는 과정에서 만진 것은 전부 **조명·톤맵·노출**(`--rtx_ambient`,
`--film_iso`, Iray 노브, 감마, 슈퍼샘플링)이었고 **머티리얼은 한 번도 검사하지 않았다.**

S5-A로 실제 머티리얼을 열어보니 이렇다 (mp3d_pe `isaacsim_<hash>.usd`, 실측):

    implementationSource : sourceAsset
    sourceAsset(mdl)     : @OmniPBR.mdl@   (subIdentifier OmniPBR)
    diffuse_color_constant : (0.6, 0.6, 0.6)   <-- 텍스처에 **곱해지는** 알베도 계수
    diffuse_texture        : @/ssd/share/...jpg@   colorSpace = '' (미지정)
    reflection_roughness_constant : (아예 저작돼 있지 않음 -> MDL 기본값)

OmniPBR에서 `diffuse_color_constant`는 diffuse 텍스처에 곱해진다. 즉 **씬 전체 알베도가
60%로 눌려 있다.** 밝기를 맞추려고 ambient/노출을 올려온 것과 같은 축의 노브가 머티리얼 쪽에
숨어 있었다는 뜻이고, 이건 조명 스윕으로는 발견될 수 없다.

`colorSpace`가 미지정인 것도 후보다 — 렌더러 기본값에 맡겨져 있어서 sRGB/raw 해석이
GT 생성 당시와 다를 수 있다.

## 실험

전부 **override 레이어**로 걸고 원본은 건드리지 않는다. 같은 GT pose를 같은 렌더러 설정으로
렌더해 SSIM과 밝기를 baseline과 비교한다.

| 실험 | 바꾸는 것 |
|---|---|
| `baseline` | 원본 USD 그대로 (대조군) |
| `albedo` | `diffuse_color_constant` 0.6 -> 1.0 |
| `colorspace_srgb` | `diffuse_texture`의 `colorSpace` -> `sRGB` |
| `colorspace_raw` | `diffuse_texture`의 `colorSpace` -> `raw` |
| `roughness` | `reflection_roughness_constant` 신규 저작 (기본 -> 지정값) |

**해석 주의**: SSIM이 오르면 잔차의 원인 일부를 새로 찾은 것이고, 안 움직이면 기존
"메쉬/텍스처 충실도" 결론이 강화된다. 어느 쪽이든 결과다.

## 실행 (한 줄)

    timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s5_materials.py --scene 17DRP5sb8fy --dataset vln_n1 --rtx_ambient 6.0 --film_iso 70 --n_frames 6 --log_dir logs/gs-vlnpe/usd_study

`--rtx_ambient 6.0 --film_iso 70`은 `reports.md`의 **현재 권장 설정**이다(vln_n1 SSIM 0.876).
baseline이 그 값 근처로 나와야 비교가 유효하다 — 리포트가 그것도 같이 판정한다.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from usd_study_utils import (
    DEFAULT_LOG_DIR,
    DEFAULT_SCENE,
    DEFAULT_USD_ROOT,
    TABLE_CSS,
    PreservationGuard,
    esc,
    exit_skipping_isaac_teardown,
    clear_render_prims,
    glossary_html,
    glossary_note_html,
    mp3d_usd_variants,
    new_override_layer,
    pill,
    preservation_section,
    stat_row_html,
    table_html,
)

import viz_utils  # noqa: E402

# **`pxr`을 모듈 최상단에서 import하지 않는다.** `SimulationApp` 부팅 전에 `/opt/USD`의 standalone
# pxr이 먼저 로드되면 Omniverse 쪽 스키마 확장이 전부 import 실패한다(실측: `omni.usd`,
# `omni.UsdMdl`, `Semantics` 등이 "extension class wrapper ... has not been created yet"로 죽는다).
# 그래서 pxr은 부팅 이후에 함수 안에서 lazy import한다.

GS_VLNPE = Path(__file__).resolve().parent.parent
RECORDED_BASELINE_SSIM = {'vln_n1': 0.876, 'vln_pe': 0.775}  # reports.md 현재 권장 설정 값


# ---------------------------------------------------------------------------
# A. 머티리얼 덤프 (pure pxr — Isaac 부팅 전에 끝낸다)
# ---------------------------------------------------------------------------

def dump_materials(usd_path: Path) -> dict:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    shaders, inputs_seen = [], {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Shader':
            continue
        sh = UsdShade.Shader(prim)
        entry = {
            'path': str(prim.GetPath()),
            'impl_source': str(sh.GetImplementationSource()),
            'shader_id': str(sh.GetShaderId()) if sh.GetShaderId() else None,
            'source_asset_mdl': str(sh.GetSourceAsset('mdl')) if sh.GetSourceAsset('mdl') else None,
            'sub_identifier_mdl': str(sh.GetSourceAssetSubIdentifier('mdl')) or None,
            'inputs': {},
        }
        for i in sh.GetInputs():
            attr = i.GetAttr()
            entry['inputs'][i.GetBaseName()] = {
                'type': str(i.GetTypeName()), 'value': str(i.Get()), 'color_space': attr.GetColorSpace() or '(미지정)'}
            inputs_seen.setdefault(i.GetBaseName(), 0)
            inputs_seen[i.GetBaseName()] += 1
        shaders.append(entry)

    bindings = []
    for prim in stage.Traverse():
        if not prim.IsA(Usd.SchemaRegistry.GetTypeFromName('UsdGeomMesh')) and prim.GetTypeName() != 'Mesh':
            continue
        mat, rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        bindings.append({'mesh': str(prim.GetPath()), 'material': str(mat.GetPath()) if mat else None})
        if len(bindings) >= 8:
            break

    return {'shader_count': len(shaders), 'shaders_sample': shaders[:3], 'input_names': inputs_seen,
            'bindings_sample': bindings, 'all_shader_paths': [s['path'] for s in shaders]}


# ---------------------------------------------------------------------------
# override 레이어 authoring
# ---------------------------------------------------------------------------

def _new_override_layer(base_usd: Path, out_path: Path):
    """공용 헬퍼로 위임 — 상대 sublayer + `defaultPrim` 복사(둘 다 필수, 이유는 헬퍼 docstring)."""
    return new_override_layer(base_usd, out_path)


def build_experiment_layer(base_usd: Path, shader_paths, exp: str, out_dir: Path, args) -> dict:
    """실험 하나에 해당하는 override 레이어를 만든다. 반환에 실제로 몇 개 바꿨는지 담는다."""
    from pxr import Sdf, Usd, UsdShade

    layer = _new_override_layer(base_usd, out_dir / f'override_{exp}.usda')
    stage = Usd.Stage.Open(layer)
    changed, skipped = 0, 0

    for p in shader_paths:
        prim = stage.GetPrimAtPath(p)
        if not prim:
            skipped += 1
            continue
        sh = UsdShade.Shader(prim)
        if exp == 'albedo':
            inp = sh.GetInput('diffuse_color_constant') or sh.CreateInput('diffuse_color_constant', Sdf.ValueTypeNames.Color3f)
            inp.Set((args.albedo, args.albedo, args.albedo))
            changed += 1
        elif exp in ('colorspace_srgb', 'colorspace_raw'):
            inp = sh.GetInput('diffuse_texture')
            if inp is None:
                skipped += 1
                continue
            token = 'sRGB' if exp == 'colorspace_srgb' else 'raw'
            # colorSpace는 속성 메타데이터 — 값을 다시 쓰지 않아도 override 레이어에 opinion이 남는다.
            inp.GetAttr().SetColorSpace(token)
            changed += 1
        elif exp == 'roughness':
            inp = sh.GetInput('reflection_roughness_constant') or sh.CreateInput('reflection_roughness_constant', Sdf.ValueTypeNames.Float)
            inp.Set(float(args.roughness))
            changed += 1
        else:
            assert False, f'unreachable experiment={exp!r}'

    layer.Save()
    text = Path(layer.identifier).read_text(encoding='utf-8')
    return {'exp': exp, 'layer': layer.identifier, 'changed': changed, 'skipped': skipped,
            'head': '\n'.join(text.splitlines()[:20])}


# ---------------------------------------------------------------------------
# B. 렌더 비교 (기존 04_render_obs_isaac 을 import 해서 재사용)
# ---------------------------------------------------------------------------

def load_renderer_module():
    """`04_render_obs_isaac.py`를 모듈로 로드한다 (숫자로 시작해 일반 import 불가).

    **import 시점에 `SimulationApp`이 부팅된다** — 그게 그 모듈의 설계다. 그래서 A단계(pure pxr)를
    먼저 끝낸 뒤에 호출한다.
    """
    path = GS_VLNPE / '04_render_obs_isaac.py'
    spec = importlib.util.spec_from_file_location('render_obs_isaac_reused', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _clear_render_prims():
    """공용 헬퍼로 위임 — 이유는 `usd_study_utils.clear_render_prims` docstring."""
    clear_render_prims()


def render_variant(r4, usd_path: Path, gt, frame_idx, args) -> dict:
    """`build_renderer` + `render_along`을 그대로 재사용해 한 variant를 렌더한다."""
    poses = [gt['poses_c2w'][f] for f in frame_idx]
    _clear_render_prims()
    cas = r4.build_renderer(r4.RENDER_W, r4.RENDER_H, gt['k'], usd_path,
                            args.light, None, args.rtx_ambient, args.film_iso)
    rgb, depth = r4.render_along(cas, poses)
    return {'rgb': rgb, 'depth': depth}


def score(rgb_render, reals) -> dict:
    from skimage.metrics import structural_similarity as ssim_metric

    ssims = [float(ssim_metric(reals[i], rgb_render[i], channel_axis=2, data_range=255)) for i in range(len(reals))]
    mean_render = float(np.mean([r.mean() for r in rgb_render]))
    mean_real = float(np.mean([r.mean() for r in reals]))
    return {'ssim_median': float(np.median(ssims)), 'ssim_min': float(np.min(ssims)), 'ssim_all': ssims,
            'brightness_render': mean_render, 'brightness_real': mean_real,
            'brightness_gap': mean_render - mean_real}


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def build_report(dump, layers, results, imgs, guard, args):
    parts = [TABLE_CSS, glossary_note_html()]

    parts.append('<h2>A · 실제 머티리얼은 무엇인가</h2>')
    s0 = dump['shaders_sample'][0] if dump['shaders_sample'] else {}
    parts.append(table_html(['항목', '값'], [
        ['shader 개수', dump['shader_count']],
        ['implementationSource', s0.get('impl_source')],
        ['shader id (UsdPreviewSurface 계열이면 여기 값이 있다)', s0.get('shader_id') or '(없음 → MDL)'],
        ['sourceAsset(mdl)', s0.get('source_asset_mdl')],
        ['subIdentifier(mdl)', s0.get('sub_identifier_mdl')],
    ], mono_cols={1}))
    parts.append('<p><b>판정</b>: <code>UsdPreviewSurface</code>가 아니라 <b>MDL <code>OmniPBR</code></b>이다. '
                 'S2에서 <code>OmniPBR.mdl</code>이 미해결로 뜬 것은 <code>Ar</code>이 RTX의 MDL 검색경로를 '
                 '모르기 때문이고, 렌더는 정상 동작한다 — 즉 그 "미해결"은 도구 관점의 산물이다.</p>')
    if s0.get('inputs'):
        parts.append('<h3>저작된 입력값 (shader 1개 기준)</h3>')
        parts.append(table_html(['입력', '타입', '값', 'colorSpace'],
                                [[k, v['type'], v['value'], v['color_space']] for k, v in s0['inputs'].items()],
                                mono_cols={2}))
    parts.append('<h3>전체 shader에서 발견된 입력 이름</h3>')
    parts.append(table_html(['입력 이름', '몇 개 shader에 저작됨'], sorted(dump['input_names'].items())))
    parts.append('<p><code>reflection_roughness_constant</code>가 목록에 없으면 <b>아예 저작돼 있지 않다</b>는 뜻 '
                 '— MDL 기본값이 쓰인다.</p>')
    if dump['bindings_sample']:
        parts.append(table_html(['mesh', '바인딩된 material'],
                                [[b['mesh'], b['material']] for b in dump['bindings_sample']], mono_cols={0, 1}))

    parts.append('<h2>B · override 레이어로 바꿔 렌더 비교</h2>')
    rec = RECORDED_BASELINE_SSIM.get(args.dataset)
    base = results.get('baseline', {})
    if rec is not None and base:
        drift = abs(base['ssim_median'] - rec)
        parts.append(f'<p>대조군 유효성: baseline SSIM median <b>{base["ssim_median"]:.3f}</b> vs '
                     f'<code>reports.md</code> 기록값 <b>{rec}</b> (차이 {drift:.3f}) '
                     f'{pill(drift < 0.05, "비교 유효", "설정이 다름 — 해석 주의")}</p>')

    rows, classes = [], []
    for exp in ['baseline'] + [l['exp'] for l in layers]:
        r = results.get(exp)
        if not r:
            rows.append([exp, '—', '—', '—', '—', '—'])
            classes.append('')
            continue
        d = r['ssim_median'] - base.get('ssim_median', r['ssim_median'])
        rows.append([exp, f'{r["ssim_median"]:.4f}', f'{r["ssim_min"]:.4f}',
                     f'{d:+.4f}' if exp != 'baseline' else '(기준)',
                     f'{r["brightness_render"]:.1f}', f'{r["brightness_gap"]:+.1f}'])
        classes.append('differ' if exp != 'baseline' and abs(d) >= 0.005 else '')
    parts.append(table_html(['실험', 'SSIM median', 'SSIM min', 'baseline 대비', '렌더 밝기', 'GT와 밝기차'],
                            rows, row_classes=classes))
    parts.append(f'<p>GT 실제 밝기 = <b>{base.get("brightness_real", float("nan")):.1f}</b>. '
                 '<code>baseline 대비</code>가 ±0.005를 넘는 행만 강조돼 있다 — 그 아래는 노이즈로 본다.</p>')

    parts.append('<h3>각 실험의 override 레이어</h3>')
    for l in layers:
        parts.append(f'<p><b>{esc(l["exp"])}</b> — 바꾼 shader {l["changed"]}개 / 건너뜀 {l["skipped"]}개 '
                     f'· <code>{esc(Path(l["layer"]).name)}</code></p>')
        parts.append(f'<pre>{esc(l["head"])}\n…</pre>')

    if imgs:
        parts.append('<h3>같은 프레임 비교</h3>')
        for i, states in enumerate(imgs):
            parts.append(viz_utils.blink_widget_html(f's5_blink_{i}', states, title=f'frame {i}'))

    parts.append(preservation_section(guard))
    parts.append(glossary_html())

    best = max(((e, r['ssim_median']) for e, r in results.items() if e != 'baseline'), key=lambda x: x[1], default=(None, None))
    gain = (best[1] - base['ssim_median']) if best[0] and base else None
    summary = stat_row_html([
        ('shader', dump['shader_count']),
        ('머티리얼 종류', s0.get('sub_identifier_mdl') or '?'),
        ('diffuse_color_constant', (s0.get('inputs', {}).get('diffuse_color_constant', {}) or {}).get('value', '?')),
        ('baseline SSIM', f'{base.get("ssim_median", float("nan")):.4f}'),
        ('최고 실험', best[0] or '—'),
        ('최대 개선', f'{gain:+.4f}' if gain is not None else '—'),
        ('원본 보존', 'ALL UNCHANGED' if guard.all_unchanged else 'MUTATION'),
    ])
    return summary, ''.join(parts)


def main():
    ap = argparse.ArgumentParser(description='S5 — 머티리얼이 잔차에 기여하는지 검증')
    ap.add_argument('--scene', default=DEFAULT_SCENE)
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_n1')
    ap.add_argument('--usd_root', default=str(DEFAULT_USD_ROOT))
    ap.add_argument('--data_root', default=None, help='기본값은 dataset_utils.default_data_root')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--experiments', default='albedo,colorspace_srgb,colorspace_raw,roughness')
    ap.add_argument('--albedo', type=float, default=1.0, help='albedo 실험에서 쓸 diffuse_color_constant')
    ap.add_argument('--roughness', type=float, default=0.5)
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=6.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR))
    ap.add_argument('--skip_hash', action='store_true')
    ap.add_argument('--no_render', action='store_true', help='A단계와 레이어 생성만 하고 렌더는 생략')
    ap.add_argument('--only', default=None,
                    help='렌더할 variant만 콤마로 지정 (예: baseline / albedo). 한 프로세스에서 여러 개를 '
                         '렌더하면 RTX가 머티리얼을 캐시할 수 있어, 교차검증용으로 프로세스를 나눌 때 쓴다.')
    ap.add_argument('--tag', default='', help='출력 폴더 접미사 (프로세스 분리 실행 시 결과를 안 덮어쓰게)')
    args = ap.parse_args()

    variants = dict(mp3d_usd_variants(args.usd_root, args.scene))
    base_usd = variants.get('isaacsim')
    if base_usd is None:
        print('[fail] isaacsim_<hash>.usd 없음', file=sys.stderr)
        return 2

    out_dir = Path(args.log_dir) / 's5_materials' / f'{args.scene}_{args.dataset}{args.tag}'
    out_dir.mkdir(parents=True, exist_ok=True)

    guard = PreservationGuard([base_usd])
    if not args.skip_hash:
        guard.snapshot_before()

    # 렌더를 할 것이면 **여기서 먼저 부팅한다** — pxr을 건드리기 전에 Isaac이 올라와 있어야 한다.
    r4 = None
    if not args.no_render:
        print('[s5] Isaac 부팅 (04_render_obs_isaac import — pxr보다 먼저여야 한다)')
        r4 = load_renderer_module()

    print('[s5] A: 머티리얼 덤프')
    dump = dump_materials(base_usd)
    print(f'[s5]   shader {dump["shader_count"]}개, 입력 이름 {len(dump["input_names"])}종')

    exps = [e for e in args.experiments.split(',') if e]
    layers = [build_experiment_layer(base_usd, dump['all_shader_paths'], e, out_dir, args) for e in exps]
    for l in layers:
        print(f'[s5]   레이어 {l["exp"]}: shader {l["changed"]}개 변경, {l["skipped"]}개 건너뜀')

    results, imgs = {}, []
    if r4 is not None:
        print('[s5] B: 렌더 비교')
        import dataset_utils

        data_root = args.data_root or dataset_utils.default_data_root(args.dataset)
        gt = dataset_utils.load_gt_episode(data_root, args.scene, args.episode, args.dataset)
        n = len(gt['poses_c2w'])
        frame_idx = np.unique(np.linspace(0, n - 1, args.n_frames).astype(int)).tolist()

        scene_dir = Path(data_root) / args.scene
        rgb_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.rgb'
        reals = [dataset_utils.load_rgb_frame(rgb_dir, args.episode, f, args.dataset) for f in frame_idx]

        todo = [('baseline', base_usd)] + [(l['exp'], Path(l['layer'])) for l in layers]
        if args.only:
            keep = {x for x in args.only.split(',') if x}
            todo = [(e, p_) for e, p_ in todo if e in keep]
        rendered = {}
        for exp, path in todo:
            print(f'[s5]   렌더: {exp}')
            out = render_variant(r4, path, gt, frame_idx, args)
            rendered[exp] = out['rgb']
            results[exp] = score(out['rgb'], reals)
            print(f'[s5]     SSIM median {results[exp]["ssim_median"]:.4f}  밝기 {results[exp]["brightness_render"]:.1f}')

        import cv2

        img_dir = out_dir / 'frames'
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frame_idx):
            # blink_widget_html은 (label, image_path) 튜플 리스트를 받는다 (viz_utils 규약).
            states = []
            gt_p = img_dir / f'gt_f{f:04d}.jpg'
            cv2.imwrite(str(gt_p), cv2.cvtColor(reals[i], cv2.COLOR_RGB2BGR))
            states.append(('GT', gt_p))
            for exp in rendered:
                ep_p = img_dir / f'{exp}_f{f:04d}.jpg'
                cv2.imwrite(str(ep_p), cv2.cvtColor(rendered[exp][i], cv2.COLOR_RGB2BGR))
                states.append((f'{exp} (SSIM {results[exp]["ssim_all"][i]:.3f})', ep_p))
            imgs.append(states)

    if not args.skip_hash:
        guard.snapshot_after()

    (out_dir / 's5_materials.json').write_text(json.dumps(
        {'args': vars(args), 'material_dump': dump, 'layers': layers,
         'results': {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
         'preservation': guard.rows()}, indent=2, default=str, ensure_ascii=False), encoding='utf-8')

    summary, body = build_report(dump, layers, results, imgs, guard, args)
    path = viz_utils.save_gallery(out_dir, 'report.html', f'S5 · 머티리얼과 잔차 — {args.scene} / {args.dataset}',
                                  summary, body, eyebrow='gs_vlnpe usd_study')
    print(f'[s5] report: {path}')
    ok = guard.all_unchanged or args.skip_hash
    print(f'[s5] {"PASS" if ok else "FAIL"}')
    # Isaac 정리 단계에서 세그폴트가 나므로 건너뛰고 나간다 (이유는 헬퍼 docstring)
    exit_skipping_isaac_teardown(0 if ok else 1)


if __name__ == '__main__':
    sys.exit(main())
