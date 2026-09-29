"""S2 — asset resolution: 텍스처 절대경로 심링크를 USD-native로 대체할 수 있는가.

## 왜 이 스크립트가 있는가

`04_render_obs_isaac.py:132` `ensure_texture_symlink`는 `/ssd/share/Matterport3D/data/v1/scans/`
아래에 **컨테이너 밖 시스템 경로로 심링크를 만든다.** 파이프라인이 쓰는 `isaacsim_<hash>.usd`가
텍스처를 그 절대경로로 참조하기 때문이다. USD 안의 문제를 USD 밖(파일시스템)에서 푼 것이라
환경이 바뀌면 깨진다.

S1에서 원시 asset 문자열을 확인한 결과는 이랬다:

    fixed.usd                 ./textures/....jpg                        (상대 — 이미 portable)
    fixed_docker.usd          /isaac-sim/Matterport3D/...               (도커 절대경로)
    isaacsim_<hash>.usd       /ssd/share/Matterport3D/...               (박힌 절대경로 23개)
    isaacsim_<hash>_non_metric.usd  ./textures/....jpg                  (상대)

즉 복제본 4개의 차이 중 하나가 **정확히 이 asset path**이고, 파이프라인은 하필 절대경로 버전을
쓴다. 이 스크립트는 세 가지 해결법을 나란히 측정한다.

## 3단계

- **A** 자산별 원시 asset 문자열 + `Ar` resolve 결과 분류 (상대 / 절대-존재 / 절대-없음 / 미해결)
- **B** 해결법 3개 비교
    ① 심링크 (현재 방식) — 지금 실제로 존재하는지, 그것에 의존해 resolve되는지
    ② `Ar` resolver search path — **절대경로에는 원리적으로 안 먹는다**는 것을 확인 (기각도 결과다)
    ③ override 레이어에서 asset 속성을 상대경로로 재지정 — 원본 무변경으로 절대경로 제거
- **C** `UsdUtils.ComplianceChecker` 규격 검사

③의 판정 기준은 "심링크가 있든 없든, 모든 텍스처가 **repo 안의 로컬 경로로** resolve되는가"다.
심링크를 지우거나 만들지 않는다 — 시스템 상태를 바꾸지 않고 판정한다.

## 실행 (한 줄)

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s2_assets.py --scene 17DRP5sb8fy --log_dir logs/gs-vlnpe/usd_study
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from usd_study_utils import (
    DEFAULT_LOG_DIR,
    DEFAULT_SCENE,
    DEFAULT_USD_ROOT,
    MATTERPORT_TEXTURE_ROOT,
    TABLE_CSS,
    PreservationGuard,
    esc,
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

from pxr import Ar, Sdf, Usd, UsdUtils  # noqa: E402

SAMPLE = 6


# ---------------------------------------------------------------------------
# A. 원시 asset 문자열 수집 + resolve
# ---------------------------------------------------------------------------

def collect_asset_attrs(layer_path: Path):
    """레이어의 모든 Asset 타입 속성을 (prim path, 속성명, 원시 문자열)로 수집한다.

    `Sdf` 레벨에서 읽으므로 composition/resolve 이전의 **저작된 그대로**를 본다.
    """
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    found = []
    if layer is None:
        return found, None

    def walk(spec):
        for name, attr in spec.attributes.items():
            if attr.typeName == Sdf.ValueTypeNames.Asset and attr.default:
                found.append((str(spec.path), name, str(attr.default.path)))
        for child in spec.nameChildren.values():
            walk(child)

    for prim in layer.rootPrims.values():
        walk(prim)
    return found, layer


def classify(raw: str, anchor: Path) -> dict:
    """원시 asset 문자열 하나를 분류하고 resolve를 시도한다."""
    r = {'raw': raw, 'is_absolute': raw.startswith('/'), 'baked_prefix': raw.startswith(str(MATTERPORT_TEXTURE_ROOT))}
    resolver = Ar.GetResolver()
    try:
        # 앵커(레이어 위치) 기준으로 상대경로를 붙인 뒤 resolve — USD가 실제로 하는 것과 같다.
        anchored = resolver.CreateIdentifier(raw, Ar.ResolvedPath(str(anchor)))
        resolved = str(resolver.Resolve(anchored))
    except Exception as exc:  # noqa: BLE001
        r['error'] = repr(exc)
        resolved = ''
    r['resolved'] = resolved
    r['resolves'] = bool(resolved)
    r['exists'] = bool(resolved) and Path(resolved).exists()
    if raw.startswith('/'):
        r['target_exists_直'] = Path(raw).exists()
    r['kind'] = (
        'MDL/기타(비이미지)' if raw.lower().endswith(('.mdl', '.usd', '.usda', '.usdc'))
        else ('상대' if not raw.startswith('/') else ('절대-박힘' if r['baked_prefix'] else '절대-기타'))
    )
    return r


def describe_assets(name: str, usd_path: Path) -> dict:
    attrs, layer = collect_asset_attrs(usd_path)
    anchor = usd_path.resolve()
    rows = [dict(classify(raw, anchor), prim=p, attr=a) for p, a, raw in attrs]

    # sublayer/reference/payload 로 참조된 것도 따로 본다(텍스처와 성격이 다르다).
    try:
        sub, refs, payloads = UsdUtils.ExtractExternalReferences(str(usd_path))
        comp = {'sublayers': [str(s) for s in sub], 'references': [str(x) for x in refs], 'payloads': [str(x) for x in payloads]}
    except Exception as exc:  # noqa: BLE001
        comp = {'error': repr(exc)}

    kinds = Counter(r['kind'] for r in rows)
    return {
        'path': str(usd_path), 'attr_count': len(rows), 'kinds': dict(kinds),
        'n_absolute': sum(1 for r in rows if r['is_absolute']),
        'n_baked': sum(1 for r in rows if r['baked_prefix']),
        'n_unresolved': sum(1 for r in rows if not r['exists']),
        'unresolved_sample': [r['raw'] for r in rows if not r['exists']][:SAMPLE],
        'resolved_dirs': sorted({str(Path(r['resolved']).parent) for r in rows if r['resolved']}),
        'composition_refs': comp,
        'rows': rows,
    }


# ---------------------------------------------------------------------------
# B. 해결법 3개
# ---------------------------------------------------------------------------

def probe_symlink() -> dict:
    """① 현재 방식 — 심링크가 실제로 존재하는지. **만들지도 지우지도 않는다.**"""
    scans = MATTERPORT_TEXTURE_ROOT
    out = {'root': str(scans), 'root_exists': scans.exists(), 'entries': []}
    if scans.exists():
        for p in sorted(scans.glob('*'))[:20]:
            mm = p / 'matterport_mesh'
            out['entries'].append({'scene': p.name, 'is_symlink': mm.is_symlink(),
                                   'target': os.readlink(mm) if mm.is_symlink() else None,
                                   'resolves': mm.exists()})
    return out


def probe_ar_search_path(sample_raw: str, anchor: Path) -> dict:
    """② `Ar` search path — 절대경로에 먹히는지 원리 확인.

    `ArDefaultResolver`의 search path는 **relative** asset path를 찾을 때만 쓰인다.
    절대경로는 그대로 파일시스템으로 간다. 즉 박힌 절대경로는 이 방법으로 못 고친다 —
    그것을 "그렇다고 들었다"가 아니라 **측정으로** 확인한다.
    """
    res = {'sample': sample_raw}
    local_dir = str((anchor.parent / 'textures').resolve())
    try:
        ctx = Ar.DefaultResolverContext([local_dir])
        resolver = Ar.GetResolver()
        with Ar.ResolverContextBinder(ctx):
            ident = resolver.CreateIdentifier(sample_raw, Ar.ResolvedPath(str(anchor)))
            res['resolved_with_search_path'] = str(resolver.Resolve(ident))
        res['search_path'] = local_dir
        resolved = res['resolved_with_search_path']
        res['resolved_exists'] = bool(resolved) and Path(resolved).exists()
        # **판정 기준 주의**: "resolve 되었나"로 재면 안 된다 — 지금 심링크가 살아 있어서
        # 박힌 절대경로도 그대로 resolve되고 파일도 존재한다(B① 참고). 그건 search path가
        # 동작한 것이 아니라 심링크가 받아준 것이다. 진짜 기준은 **경로가 리다이렉트됐는가**다.
        res['redirected'] = bool(resolved) and not resolved.startswith(str(MATTERPORT_TEXTURE_ROOT))
        res['still_baked_after_search_path'] = bool(resolved) and resolved.startswith(str(MATTERPORT_TEXTURE_ROOT))
        res['works'] = res['redirected']
        res['resolve_only_via_symlink'] = res['resolved_exists'] and res['still_baked_after_search_path']
        # 대조군: 같은 파일을 basename만 주면 어디로 가는가 (앵커 기준 해석이 먼저 걸린다)
        base = Path(sample_raw).name
        with Ar.ResolverContextBinder(ctx):
            ident2 = resolver.CreateIdentifier(base, Ar.ResolvedPath(str(anchor)))
            res['basename_resolved'] = str(resolver.Resolve(ident2))
        res['basename_resolves'] = bool(res['basename_resolved'])
    except Exception as exc:  # noqa: BLE001
        res['error'] = repr(exc)
        res['works'] = False
    return res


def build_texture_override(base_usd: Path, rows, out_dir: Path) -> dict:
    """③ override 레이어로 asset 속성을 상대경로로 재지정. 원본 무변경.

    각 텍스처의 원시 절대경로에서 basename만 떼어, override 레이어 위치 기준 상대경로로
    다시 쓴다. 레이어를 옮겨도 같이 움직이도록 `os.path.relpath`를 쓴다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_path = out_dir / 'override_textures.usda'

    tex_dir = (base_usd.parent / 'textures').resolve()
    rel_prefix = os.path.relpath(tex_dir, layer_path.parent.resolve())

    # 공용 헬퍼 — 상대 sublayer + defaultPrim 복사까지 해 준다. defaultPrim이 없으면 이 레이어를
    # 검사용으로는 쓸 수 있어도 `UsdFileCfg`로 스폰할 수 없다(씬이 빈 채로 렌더된다).
    layer = new_override_layer(base_usd, layer_path)
    stage = Usd.Stage.Open(layer)

    retargeted, skipped = 0, []
    for r in rows:
        raw = r['raw']
        if r['kind'] == 'MDL/기타(비이미지)':
            skipped.append(raw)  # MDL은 텍스처가 아니라 머티리얼 정의 — S5에서 따로 본다
            continue
        prim = stage.GetPrimAtPath(r['prim'])
        if not prim:
            skipped.append(raw)
            continue
        attr = prim.GetAttribute(r['attr'])
        if not attr:
            skipped.append(raw)
            continue
        attr.Set(Sdf.AssetPath(f'{rel_prefix}/{Path(raw).name}'))
        retargeted += 1
    layer.Save()

    # 저장된 레이어를 새로 열어 전부 resolve되는지, 그리고 어디로 resolve되는지 확인한다.
    verify = Usd.Stage.Open(str(layer_path))
    resolver = Ar.GetResolver()
    checked, ok_local, still_baked, unresolved = 0, 0, 0, []
    repo_root = str(Path.cwd().resolve())
    for prim in verify.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            v = attr.Get()
            if v is None:
                continue
            checked += 1
            resolved = str(v.resolvedPath) if getattr(v, 'resolvedPath', '') else str(resolver.Resolve(v.path))
            if not resolved or not Path(resolved).exists():
                unresolved.append(str(v.path))
            elif resolved.startswith(str(MATTERPORT_TEXTURE_ROOT)):
                still_baked += 1
            elif resolved.startswith(repo_root):
                ok_local += 1
    return {
        'layer': str(layer_path), 'rel_prefix': rel_prefix, 'retargeted': retargeted,
        'skipped_count': len(skipped), 'skipped_sample': skipped[:SAMPLE],
        'checked': checked, 'resolved_local': ok_local, 'still_baked': still_baked,
        'unresolved': unresolved, 'unresolved_count': len(unresolved),
        # MDL(OmniPBR.mdl) 하나는 원래도 미해결이므로 그것만 남는 것을 성공으로 본다.
        'ok': still_baked == 0 and ok_local > 0 and all(u.lower().endswith('.mdl') for u in unresolved),
        'layer_head': '\n'.join((layer_path.read_text(encoding='utf-8')).splitlines()[:24]),
    }


# ---------------------------------------------------------------------------
# C. compliance
# ---------------------------------------------------------------------------

def compliance(usd_path: Path) -> dict:
    try:
        checker = UsdUtils.ComplianceChecker(arkit=False, skipARKitRootLayerCheck=True,
                                             rootPackageOnly=False, skipVariants=False, verbose=False)
        checker.CheckCompliance(str(usd_path))
        return {'errors': list(checker.GetErrors()), 'warnings': list(checker.GetWarnings()),
                'failed': list(checker.GetFailedChecks())}
    except Exception as exc:  # noqa: BLE001
        return {'error': repr(exc)}


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def build_report(descs, sym, ar, ov, comps, guard, args):
    parts = [TABLE_CSS, glossary_note_html()]

    parts.append('<h2>A · 자산별 asset 경로 실태</h2>')
    rows = [[n, d['attr_count'], d['n_absolute'], d['n_baked'], d['n_unresolved'],
             ', '.join(f'{k}={v}' for k, v in d['kinds'].items())] for n, d in descs]
    parts.append(table_html(['자산', 'asset 속성', '절대경로', f'박힌({MATTERPORT_TEXTURE_ROOT})', '미해결', '분류'], rows))
    for n, d in descs:
        parts.append(f'<h3>{esc(n)}</h3>')
        parts.append(table_html(['항목', '값'], [
            ['resolve된 디렉토리', ', '.join(d['resolved_dirs']) or '—'],
            ['미해결 샘플', ', '.join(d['unresolved_sample']) or '(없음)'],
            ['sublayers / references / payloads',
             f"{d['composition_refs'].get('sublayers')} / {d['composition_refs'].get('references')} / {d['composition_refs'].get('payloads')}"],
        ], mono_cols={1}))
        samp = d['rows'][:3]
        if samp:
            parts.append(table_html(['원시 문자열', '분류', 'resolve 결과', '존재'],
                                    [[r['raw'], r['kind'], r['resolved'] or '—', pill(r['exists'], 'O', 'X')] for r in samp],
                                    mono_cols={0, 2}))

    parts.append('<h2>B · 해결법 3개 비교</h2>')

    parts.append('<h3>① 심링크 (현재 방식)</h3>')
    parts.append(f'<p><code>{esc(sym["root"])}</code> 존재: {pill(sym["root_exists"], "있음", "없음")} '
                 '— 이 스크립트는 심링크를 만들지도 지우지도 않는다.</p>')
    if sym['entries']:
        parts.append(table_html(['scene', 'matterport_mesh가 심링크', 'target', 'resolve'],
                                [[e['scene'], pill(e['is_symlink'], 'O', 'X'), e['target'] or '—', pill(e['resolves'], 'O', 'X')]
                                 for e in sym['entries']], mono_cols={2}))
    else:
        parts.append('<p>등록된 항목 없음 — 아직 한 번도 렌더를 돌리지 않았거나 심링크가 지워진 상태.</p>')

    parts.append('<h3>② <code>Ar</code> resolver search path</h3>')
    works = bool(ar.get('works'))
    parts.append(table_html(['항목', '값'], [
        ['시험한 원시 문자열', ar.get('sample')],
        ['search path', ar.get('search_path')],
        ['resolve 결과', ar.get('resolved_with_search_path') or '—'],
        ['<b>경로가 리다이렉트됐나</b> (진짜 판정 기준)', pill(works, 'YES', 'NO')],
        ['resolve 자체는 성공했나', pill(bool(ar.get('resolved_exists')), 'YES', 'NO')],
        ['그 성공이 심링크 덕분인가', pill(not bool(ar.get('resolve_only_via_symlink')), 'NO', 'YES — search path 아님')],
        ['대조군: basename만 준 경우', ar.get('basename_resolved') or '—'],
        ['error', ar.get('error') or '—'],
    ], mono_cols={1}))
    parts.append('<p><b>해석</b>: <code>ArDefaultResolver</code>의 search path는 <i>상대</i> asset path를 찾을 때만 '
                 '쓰인다. 실측에서 resolve는 <b>성공했지만 경로가 <code>/ssd/share/...</code> 그대로</b>였다 — '
                 '즉 search path가 받아준 게 아니라 <b>지금 살아 있는 심링크가 받아준 것</b>이다(B① 참고). '
                 '"resolve 되었나"로 재면 이 착시에 걸린다. 박힌 절대경로는 이 방법으로 못 고치고, '
                 '심링크를 없애려면 ③이 필요하다 — <b>기각도 결과다.</b></p>')

    parts.append('<h3>③ override 레이어로 상대경로 재지정 <b>(원본 무변경)</b></h3>')
    if ov is None:
        parts.append('<p><span class="pill warn">SKIPPED</span></p>')
    else:
        parts.append(stat_row_html([
            ('판정', 'PASS' if ov['ok'] else 'FAIL'),
            ('재지정한 속성', ov['retargeted']),
            ('검사한 asset', ov['checked']),
            ('로컬로 resolve', ov['resolved_local']),
            ('여전히 박힌 경로', ov['still_baked']),
            ('미해결', ov['unresolved_count']),
        ]))
        parts.append(table_html(['항목', '값'], [
            ['override 레이어', ov['layer']],
            ['상대 prefix', ov['rel_prefix']],
            ['미해결로 남은 것', ', '.join(ov['unresolved']) or '(없음)'],
            ['건너뛴 것(샘플)', ', '.join(ov['skipped_sample']) or '(없음)'],
        ], mono_cols={1}))
        parts.append(f'<pre>{esc(ov["layer_head"])}\n…</pre>')
        parts.append('<p><b>해석</b>: 미해결로 남는 것이 <code>OmniPBR.mdl</code>뿐이면 성공이다 — '
                     '그건 텍스처가 아니라 MDL 머티리얼 정의이고, 원래도 이 자산에서 해결되지 않는다(S5에서 따로 본다).</p>')

    parts.append('<h2>C · ComplianceChecker</h2>')
    for n, c in comps:
        if 'error' in c:
            parts.append(f'<h3>{esc(n)} <span class="pill warn">실행 실패</span></h3><pre>{esc(c["error"])}</pre>')
            continue
        parts.append(f'<h3>{esc(n)} — error {len(c["errors"])} / warning {len(c["warnings"])} / failed {len(c["failed"])}</h3>')
        msgs = [['error', m] for m in c['errors'][:10]] + [['warning', m] for m in c['warnings'][:10]] + [['failed', m] for m in c['failed'][:10]]
        parts.append(table_html(['종류', '메시지'], msgs, mono_cols={1}) if msgs else '<p>깨끗함</p>')

    parts.append(preservation_section(guard))
    parts.append(glossary_html())

    summary = stat_row_html([
        ('② 경로 리다이렉트', 'NO' if not works else 'YES'),
        ('③ 판정', ('PASS' if ov['ok'] else 'FAIL') if ov else '—'),
        ('③ 로컬 resolve', ov['resolved_local'] if ov else '—'),
        ('③ 남은 박힌 경로', ov['still_baked'] if ov else '—'),
        ('심링크 현재 존재', 'O' if sym['root_exists'] else 'X'),
        ('원본 보존', 'ALL UNCHANGED' if guard.all_unchanged else 'MUTATION'),
    ])
    return summary, ''.join(parts)


def main():
    ap = argparse.ArgumentParser(description='S2 — asset resolution 실습')
    ap.add_argument('--scene', default=DEFAULT_SCENE)
    ap.add_argument('--usd_root', default=str(DEFAULT_USD_ROOT))
    ap.add_argument('--override_base', default='isaacsim',
                    help='③에서 텍스처 경로를 재지정할 자산 (기본: 파이프라인이 실제로 쓰는 isaacsim_<hash>.usd)')
    ap.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR))
    ap.add_argument('--skip_hash', action='store_true')
    args = ap.parse_args()

    variants = mp3d_usd_variants(args.usd_root, args.scene)
    if not variants:
        print('[fail] 자산 없음', file=sys.stderr)
        return 2
    print('[s2] 자산: ' + ', '.join(n for n, _ in variants))

    guard = PreservationGuard([p for _, p in variants])
    if not args.skip_hash:
        guard.snapshot_before()

    descs = []
    for name, path in variants:
        print(f'[s2] A: {name}')
        descs.append((name, describe_assets(name, path)))

    print('[s2] B①: 심링크 상태 확인')
    sym = probe_symlink()

    base_path = dict(variants).get(args.override_base)
    base_desc = dict(descs).get(args.override_base)
    ar, ov = {}, None
    if base_path is not None and base_desc is not None:
        abs_rows = [r for r in base_desc['rows'] if r['is_absolute'] and r['kind'] != 'MDL/기타(비이미지)']
        sample_raw = abs_rows[0]['raw'] if abs_rows else (base_desc['rows'][0]['raw'] if base_desc['rows'] else '')
        print('[s2] B②: Ar search path 시험')
        ar = probe_ar_search_path(sample_raw, base_path.resolve())
        print('[s2] B③: override 레이어 authoring')
        ov = build_texture_override(base_path, base_desc['rows'], Path(args.log_dir) / 's2_assets' / args.scene)

    print('[s2] C: ComplianceChecker')
    comps = [(n, compliance(p)) for n, p in variants]

    if not args.skip_hash:
        guard.snapshot_after()

    out_dir = Path(args.log_dir) / 's2_assets' / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {n: {k: v for k, v in d.items() if k != 'rows'} | {'rows_sample': d['rows'][:SAMPLE]} for n, d in descs}
    (out_dir / 's2_assets.json').write_text(json.dumps(
        {'args': vars(args), 'assets': slim, 'symlink': sym, 'ar_search_path': ar, 'override': ov,
         'compliance': dict(comps), 'preservation': guard.rows()}, indent=2, default=str, ensure_ascii=False), encoding='utf-8')

    summary, body = build_report(descs, sym, ar, ov, comps, guard, args)
    path = viz_utils.save_gallery(out_dir, 'report.html', f'S2 · asset resolution — {args.scene}',
                                  summary, body, eyebrow='gs_vlnpe usd_study')
    print(f'[s2] report: {path}')

    ok = (guard.all_unchanged or args.skip_hash) and (ov is None or ov['ok'])
    print(f'[s2] {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
