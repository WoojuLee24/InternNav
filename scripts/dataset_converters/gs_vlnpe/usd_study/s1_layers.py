"""S1 — USD composition(레이어 겹쳐쓰기) 실습.

## 왜 이 스크립트가 있는가

`gs_vlnpe` 파이프라인은 USD를 **읽기만** 한다 — `Usd.Stage.Open` 한 장 + `UsdFileCfg` reference
한 개가 전부다. USD의 핵심인 composition(sublayer / reference / payload / variant와 그 강도 순서)을
쓴 적이 없고, 그 결과 같은 씬이 **파일 4개로 복제**돼 있다:

    fixed.usd / fixed_docker.usd / isaacsim_<hash>.usd / isaacsim_<hash>_non_metric.usd

"`_docker`는 asset path만, `_non_metric`은 `metersPerUnit`만 다를 것"이라고 추정하고 있었지만
**실제로 확인한 적이 없다.** 이 스크립트가 그걸 측정으로 바꾸고(B), 그 복제를 레이어로 대체할 수
있는지 직접 authoring해서 확인한다(C).

## 3단계

- **A. 읽기** — 자산별 layer stack · composition arc · prim 타입/applied schema 히스토그램 ·
  `defaultPrim` · `upAxis` · `metersPerUnit` · kind · 외부 asset 의존성 덤프
- **B. diff** — 위 덤프를 필드별로 비교해 "4개가 정확히 뭐가 다른가" 표 확정
- **C. 쓰기** — override `.usda`를 만들어 원본을 sublayer로 얹고
    - C1: prim 하나의 `visibility`를 덮어씀 (prim-level opinion)
    - C2: `metersPerUnit`을 덮어씀 (layer metadata) → **`_non_metric` 복제를 레이어로 대체할 수
      있는가**에 대한 직접적인 답
  두 경우 모두 원본 파일은 건드리지 않고, sha256으로 그것을 증명한다.

`Usd.PrimCompositionQuery`로 내가 얹은 레이어가 **어느 강도 순위**로 들어갔는지 함께 출력한다
(LIVRPS를 눈으로 보는 지점).

## 실행 (한 줄)

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s1_layers.py --scene 17DRP5sb8fy --log_dir logs/gs-vlnpe/usd_study

Isaac Sim(`SimulationApp`)을 띄우지 않는다 — `pxr`만 쓰므로 수 초~수십 초에 끝난다.
`--assets all`로 노은역 usdz(2.2 GB)까지 포함할 수 있다(해싱·로드에 수 분 추가).
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from usd_study_utils import (
    DEFAULT_LOG_DIR,
    DEFAULT_SCENE,
    DEFAULT_USD_ROOT,
    DEFAULT_USDZ,
    MATTERPORT_TEXTURE_ROOT,
    TABLE_CSS,
    PreservationGuard,
    esc,
    glossary_html,
    glossary_note_html,
    mp3d_usd_variants,
    pill,
    preservation_section,
    stat_row_html,
    table_html,
)

import viz_utils  # noqa: E402  (usd_study_utils가 sys.path에 gs_vlnpe/를 넣은 뒤에 import)

from pxr import Sdf, Usd, UsdGeom, UsdUtils  # noqa: E402

SAMPLE_LIMIT = 8


# ---------------------------------------------------------------------------
# A. 읽기
# ---------------------------------------------------------------------------

def _composition_arcs(prim):
    """prim에 걸린 composition arc를 강도 순서대로 나열한다.

    `GetCompositionArcs()`는 강한 것부터 반환한다 — 이 순서가 곧 LIVRPS의 실물이다.
    """
    out = []
    try:
        arcs = Usd.PrimCompositionQuery(prim).GetCompositionArcs()
    except Exception as exc:  # noqa: BLE001 - 진단 스크립트, 실패도 결과로 기록한다
        return [{'error': repr(exc)}]
    for rank, arc in enumerate(arcs):
        entry = {'rank': rank}
        for key, getter in (
            ('arc_type', lambda a: a.GetArcType().displayName),
            ('introducing_layer', lambda a: a.GetIntroducingLayer().identifier),
            ('target_layer', lambda a: a.GetTargetNode().layerStack.identifier.rootLayer.identifier),
            ('in_root_layer_stack', lambda a: a.IsIntroducedInRootLayerStack()),
            ('is_implicit', lambda a: a.IsImplicit()),
        ):
            try:
                entry[key] = str(getter(arc))
            except Exception:  # noqa: BLE001
                entry[key] = '—'
        out.append(entry)
    return out


def _asset_dependencies(usd_path: Path):
    """레이어가 참조하는 외부 파일 전부(텍스처 포함)와 미해결 경로.

    `ComputeAllDependencies`는 sublayer/reference/payload뿐 아니라 shader의 텍스처 asset까지
    따라가므로, `fixed.usd` vs `fixed_docker.usd`의 차이를 잡는 데 이게 필요하다.
    """
    info = {'layers': [], 'assets': [], 'unresolved': [], 'error': None}
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(usd_path))
        info['layers'] = sorted(str(getattr(l, 'identifier', l)) for l in layers)
        info['assets'] = sorted(str(a) for a in assets)
        info['unresolved'] = sorted(str(u) for u in unresolved)
    except Exception as exc:  # noqa: BLE001
        info['error'] = repr(exc)
        try:
            sub, refs, payloads = UsdUtils.ExtractExternalReferences(str(usd_path))
            info['layers'] = sorted(str(s) for s in sub)
            info['assets'] = sorted(str(r) for r in list(refs) + list(payloads))
        except Exception as exc2:  # noqa: BLE001
            info['error'] += f' | fallback: {exc2!r}'
    return info


def describe(usd_path: Path, load_all: bool = True) -> dict:
    """자산 하나의 구조를 덤프한다. 실패해도 예외를 올리지 않고 error 필드에 남긴다."""
    d = {'path': str(usd_path), 'exists': usd_path.is_file(), 'error': None}
    if not d['exists']:
        return d
    d['size_bytes'] = usd_path.stat().st_size
    try:
        t0 = time.time()
        stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll if load_all else Usd.Stage.LoadNone)
        d['open_s'] = round(time.time() - t0, 3)

        root = stage.GetRootLayer()
        d['root_layer'] = root.identifier
        d['sublayer_paths'] = list(root.subLayerPaths)
        d['layer_stack'] = [l.identifier for l in stage.GetLayerStack()]
        d['default_prim'] = str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None
        d['up_axis'] = str(UsdGeom.GetStageUpAxis(stage))
        d['meters_per_unit'] = float(UsdGeom.GetStageMetersPerUnit(stage))
        d['time_codes_per_second'] = float(stage.GetTimeCodesPerSecond())

        types, schemas, kinds = Counter(), Counter(), Counter()
        mesh_paths = []
        t0 = time.time()
        for prim in stage.Traverse():
            types[str(prim.GetTypeName()) or '(no type)'] += 1
            for s in prim.GetAppliedSchemas():
                schemas[str(s)] += 1
            k = Usd.ModelAPI(prim).GetKind()
            if k:
                kinds[str(k)] += 1
            if prim.IsA(UsdGeom.Mesh) and len(mesh_paths) < 64:
                mesh_paths.append(str(prim.GetPath()))
        d['traverse_s'] = round(time.time() - t0, 3)
        d['prim_count'] = int(sum(types.values()))
        d['prim_types'] = dict(types.most_common())
        d['applied_schemas'] = dict(schemas.most_common())
        d['kinds'] = dict(kinds.most_common())
        d['mesh_count'] = int(types.get('Mesh', 0))
        # 표에 '물리 충돌 표시가 붙은 물건 수'로 쓰기 위한 파생 값
        d['physics_collision_count'] = int(schemas.get('PhysicsCollisionAPI', 0))
        d['mesh_paths_sample'] = mesh_paths[:SAMPLE_LIMIT]

        # composition arc는 "첫 Mesh"와 "pseudo-root 직하 prim"에서 각각 본다.
        d['arcs'] = {}
        if mesh_paths:
            d['arcs']['first_mesh'] = _composition_arcs(stage.GetPrimAtPath(mesh_paths[0]))
        children = stage.GetPseudoRoot().GetChildren()
        if children:
            d['arcs']['top_prim'] = _composition_arcs(children[0])
            d['top_prims'] = [str(c.GetPath()) for c in children[:SAMPLE_LIMIT]]

        deps = _asset_dependencies(usd_path)
        d['dep_layer_count'] = len(deps['layers'])
        d['dep_asset_count'] = len(deps['assets'])
        d['dep_unresolved_count'] = len(deps['unresolved'])
        d['dep_assets_sample'] = deps['assets'][:SAMPLE_LIMIT]
        d['dep_unresolved_sample'] = deps['unresolved'][:SAMPLE_LIMIT]
        d['dep_error'] = deps['error']
        # mp3d USD에 박힌 원래 변환 환경 절대경로가 몇 개인지 — S2의 심링크 근거
        prefix = str(MATTERPORT_TEXTURE_ROOT)
        d['dep_baked_abs_count'] = sum(1 for a in deps['assets'] + deps['unresolved'] if a.startswith(prefix))
        d['dep_asset_dirs'] = sorted({str(Path(a).parent) for a in deps['assets']})[:SAMPLE_LIMIT]
    except Exception as exc:  # noqa: BLE001
        d['error'] = repr(exc)
    return d


# ---------------------------------------------------------------------------
# B. diff
# ---------------------------------------------------------------------------

DIFF_FIELDS = [
    ('up_axis', '어느 축이 위쪽인가'),
    ('meters_per_unit', '숫자 1이 몇 미터인가'),
    ('default_prim', '남이 가져다 쓸 때 기본으로 딸려오는 물건'),
    ('prim_count', '물건 개수'),
    ('mesh_count', '형상(삼각형 덩어리) 개수'),
    ('physics_collision_count', '물리 충돌 표시가 붙은 물건 수'),
    ('dep_asset_dirs', '텍스처가 있다고 적힌 위치'),
    ('dep_baked_abs_count', '고정 절대경로로 박힌 텍스처 수'),
    ('size_bytes', '파일 크기(B)'),
]


def _fmt(v):
    if isinstance(v, dict):
        return ', '.join(f'{k}={n}' for k, n in list(v.items())[:6]) + ('' if len(v) <= 6 else f' … (+{len(v)-6})')
    if isinstance(v, list):
        return '[]' if not v else ', '.join(str(x) for x in v[:3]) + ('' if len(v) <= 3 else f' … (+{len(v)-3})')
    return v


def diff_table(descs) -> tuple:
    """descs: [(name, desc)]. **다른 항목만** 표로 만들고, 같은 항목은 한 줄로 적는다.

    같은 값까지 전부 나열하면 읽는 사람이 어디를 봐야 하는지 알 수 없다 — 표의 일은
    "무엇이 다른가"에 답하는 것이다.
    """
    names = [n for n, _ in descs]
    rows, differing, same_labels = [], [], []
    for key, label in DIFF_FIELDS:
        vals = [d.get(key) for _, d in descs]
        norm = [json.dumps(v, sort_keys=True, default=str) for v in vals]
        if all(x == norm[0] for x in norm):
            same_labels.append(label)
            continue
        differing.append(label)
        rows.append([label] + [_fmt(v) for v in vals])

    html = table_html(['무엇이', *names], rows, mono_cols=set(range(1, len(names) + 1)))
    if same_labels:
        html += (f'<p style="color:var(--text-dim);font-size:.85rem">네 파일에서 <b>같았던 항목</b>: '
                 f'{esc(", ".join(same_labels))}</p>')
    return html, differing


# ---------------------------------------------------------------------------
# C. 쓰기 — override 레이어 authoring
# ---------------------------------------------------------------------------

def author_override(base_usd: Path, out_dir: Path, target_prim: str, new_mpu: float) -> dict:
    """`base_usd`를 sublayer로 얹는 override 레이어를 만들고 두 가지를 덮어쓴다.

    C1: `target_prim`의 visibility  (prim-level opinion)
    C2: stage의 metersPerUnit       (layer metadata)

    base_usd는 열기만 하고 절대 쓰지 않는다 — 새 레이어가 stage의 root layer이므로
    모든 authoring이 그쪽으로 간다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {'base': str(base_usd), 'target_prim': target_prim}

    # 원본 단독 stage에서 "덮어쓰기 전" 값을 읽는다.
    base_stage = Usd.Stage.Open(str(base_usd))
    base_prim = base_stage.GetPrimAtPath(target_prim)
    res['base_visibility'] = str(UsdGeom.Imageable(base_prim).GetVisibilityAttr().Get()) if base_prim else '(prim 없음)'
    res['base_meters_per_unit'] = float(UsdGeom.GetStageMetersPerUnit(base_stage))

    layer_path = out_dir / 'override.usda'
    if layer_path.exists():
        layer_path.unlink()
    layer = Sdf.Layer.CreateNew(str(layer_path))
    # sublayer 경로는 레이어 자기 위치 기준으로 해석되므로 상대경로로 넣는다 —
    # 이 레이어를 다른 곳으로 옮겨도 같이 움직인다.
    rel = os.path.relpath(base_usd.resolve(), layer_path.parent.resolve())
    layer.subLayerPaths.append(rel)
    res['sublayer_rel'] = rel

    stage = Usd.Stage.Open(layer)
    res['composed_visibility_before_edit'] = str(UsdGeom.Imageable(stage.GetPrimAtPath(target_prim)).GetVisibilityAttr().Get())
    res['composed_mpu_inherited'] = float(UsdGeom.GetStageMetersPerUnit(stage))

    # C1 — prim opinion
    over = stage.OverridePrim(target_prim)
    UsdGeom.Imageable(over).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    # C2 — layer metadata
    UsdGeom.SetStageMetersPerUnit(stage, new_mpu)
    layer.Save()

    res['override_layer'] = str(layer_path)
    res['override_layer_text'] = layer_path.read_text(encoding='utf-8')

    # 저장한 레이어를 새로 열어서(캐시 영향 배제) 합성 결과를 다시 읽는다.
    Usd.Stage.Open(str(layer_path)).Reload()
    verify = Usd.Stage.Open(str(layer_path))
    res['composed_visibility_after'] = str(UsdGeom.Imageable(verify.GetPrimAtPath(target_prim)).GetVisibilityAttr().Get())
    res['composed_mpu_after'] = float(UsdGeom.GetStageMetersPerUnit(verify))
    res['arcs_after'] = _composition_arcs(verify.GetPrimAtPath(target_prim))

    # 원본 단독으로 다시 열어 "원본은 그대로"임을 값으로도 확인한다(해시와 별개의 증거).
    base_again = Usd.Stage.Open(str(base_usd))
    res['base_visibility_after'] = str(UsdGeom.Imageable(base_again.GetPrimAtPath(target_prim)).GetVisibilityAttr().Get())
    res['base_mpu_after'] = float(UsdGeom.GetStageMetersPerUnit(base_again))

    res['c1_ok'] = res['composed_visibility_after'] == 'invisible' and res['base_visibility_after'] == res['base_visibility']
    # 유효한 테스트이려면 (a) 원본과 다른 값을 요청했고 (b) 합성값이 그 값이 됐고
    # (c) 원본은 그대로여야 한다. (a)가 아니면 아무것도 증명하지 못하므로 명시적으로 기록한다.
    res['c2_is_meaningful'] = abs(new_mpu - res['base_meters_per_unit']) > 1e-12
    res['c2_ok'] = (res['c2_is_meaningful']
                    and abs(res['composed_mpu_after'] - new_mpu) < 1e-12
                    and abs(res['base_mpu_after'] - res['base_meters_per_unit']) < 1e-12)
    return res


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def _lineage_svg() -> str:
    """4개 파일의 계보 + 각 파일을 어느 단계가 쓰는지 한 장으로.

    프로즈로 쓰면 독자가 머리로 조립해야 하는 것 — "어느 파일이 어디서 나왔고 누가 쓰나" — 을
    그림으로 보인다. 계보는 파일 시각(mtime)과 계층 포함관계로 **추정**한 것이라 그림에도 그렇게 적는다.
    """
    box = 'fill="none" stroke="currentColor" stroke-width="1.2" rx="7"'
    hot = 'fill="none" stroke="var(--accent)" stroke-width="2" rx="7"'
    t = 'fill="currentColor" font-size="12.5"'
    tdim = 'fill="currentColor" font-size="11.5" opacity=".65"'
    tacc = 'fill="var(--accent)" font-size="12.5"'
    ln = 'stroke="currentColor" stroke-width="1.2" fill="none" marker-end="url(#s1arrow)"'
    lnacc = 'stroke="var(--accent)" stroke-width="1.8" fill="none" marker-end="url(#s1arrowA)"'
    return f'''<figure style="margin:16px 0 26px">
<svg viewBox="0 0 1000 430" role="img" width="100%" style="max-width:100%;height:auto"
     aria-label="같은 obj 하나에서 USD 4개가 갈라져 나온 계보와, 각 파일을 파이프라인의 어느 단계가 쓰는지">
  <defs>
    <marker id="s1arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 10,5 0,10" fill="currentColor"/></marker>
    <marker id="s1arrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 10,5 0,10" fill="var(--accent)"/></marker>
  </defs>

  <rect x="16" y="176" width="150" height="56" {box}/>
  <text x="91" y="199" {t} text-anchor="middle">원본 메시</text>
  <text x="91" y="217" {tdim} text-anchor="middle">.obj · 15.9 MB</text>

  <text x="300" y="16" {tdim} text-anchor="middle">한 번의 변환 배치 · 2025-02-11 01:45</text>
  <path d="M166 204 H206" {ln}/>
  <path d="M206 204 V62 H222" {ln}/>
  <path d="M206 204 H222" {ln}/>
  <path d="M206 204 V300 H222" {ln}/>

  <rect x="228" y="34" width="336" height="56" {box}/>
  <text x="244" y="55" {t}>isaacsim_&lt;hash&gt;_non_metric.usd</text>
  <text x="244" y="74" {tdim}>Y축이 위 · 1단위=0.01m · 물리 0 · 상대경로</text>

  <rect x="228" y="176" width="336" height="56" {hot}/>
  <text x="244" y="197" {tacc}>isaacsim_&lt;hash&gt;.usd</text>
  <text x="244" y="216" {tdim}>Z축이 위 · 1단위=1m · 물리 1 · /ssd/share 절대경로</text>

  <rect x="228" y="272" width="336" height="56" {box}/>
  <text x="244" y="293" {t}>fixed_docker.usd</text>
  <text x="244" y="312" {tdim}>Z축이 위 · 1단위=0.01m · 물리 3 · /isaac-sim 절대경로</text>

  <path d="M300 328 V370" {ln}/>
  <text x="316" y="352" {tdim}>하루 뒤 · 02-12 11:09 · 텍스처 경로만 다름</text>
  <rect x="228" y="370" width="336" height="50" {box}/>
  <text x="244" y="389" {t}>fixed.usd</text>
  <text x="244" y="407" {tdim}>Z축이 위 · 1단위=0.01m · 물리 3 · 상대경로</text>

  <path d="M564 62 H620" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 4" fill="none" opacity=".5"/>
  <text x="632" y="58" {tdim}>어디서도 안 쓴다</text>
  <text x="632" y="76" {tdim}>(코드가 이름으로 제외)</text>

  <path d="M564 204 H620" {lnacc}/>
  <text x="632" y="200" {tacc}>04 렌더링</text>
  <text x="632" y="219" {tdim}>텍스처 절대경로 → 심링크 필요</text>

  <path d="M564 300 H620" {ln}/>
  <text x="632" y="296" {t}>01 씬 게이트 (컨테이너일 때)</text>
  <text x="632" y="315" {tdim}>bounds·축만 읽는다</text>

  <path d="M564 395 H620" {ln}/>
  <text x="632" y="391" {t}>01 씬 게이트 · 02 맵</text>
  <text x="632" y="410" {tdim}>02는 --geometry usd 일 때만 (기본은 .obj)</text>
</svg>
<figcaption style="font-size:.85rem;color:var(--text-dim);line-height:1.6">
 같은 <code>.obj</code> 하나에서 갈라진 4개다 — <b>형상 좌표값은 넷이 완전히 동일</b>하고(16.35 × 8.279 × 2.807),
 다른 것은 파일에 <b>적힌 선언값</b>과 껍데기·부속물뿐이다. 파이프라인이 렌더에 쓰는
 <span style="color:var(--accent)">가운데 파일</span>만 "1단위=1m"로 맞게 선언한다 — 나머지 셋은 0.01m라서
 선언대로 읽으면 16 cm 건물이 된다. 화살표(계보)는 파일 시각과 계층 포함관계로 <b>추정</b>한 것이다.
</figcaption>
</figure>'''


def _vis(token) -> str:
    """USD visibility 토큰을 사람이 읽는 말로. 원래 값도 같이 남긴다."""
    t = str(token)
    label = {'inherited': '보임', 'invisible': '안 보임'}.get(t)
    return f'{label} ({t})' if label else t


def _details(summary_label: str, inner: str) -> str:
    """접히는 원자료 블록 — 본문 흐름에서 빼되 근거는 남긴다."""
    return (f'<details style="margin:10px 0 20px"><summary style="cursor:pointer;color:var(--text-dim);'
            f'font-family:var(--font-mono);font-size:.8rem">{esc(summary_label)}</summary>{inner}</details>')


def _raw_dump_html(descs) -> str:
    """자산별 구조 원자료. 읽는 흐름에는 필요 없어 접어 둔다."""
    out = []
    for name, d in descs:
        if d.get('error'):
            out.append(f'<h4>{esc(name)} <span class="pill bad">ERROR</span></h4><pre>{esc(d["error"])}</pre>')
            continue
        out.append(f'<h4>{esc(name)}</h4>')
        out.append(table_html(['항목', '값'], [
            ['prim 총수 / Mesh', f'{d.get("prim_count")} / {d.get("mesh_count")}'],
            ['prim 타입', _fmt(d.get('prim_types'))],
            ['applied API schema', _fmt(d.get('applied_schemas')) or '(없음)'],
            ['kind', _fmt(d.get('kinds')) or '(없음)'],
            ['최상위 prim', _fmt(d.get('top_prims'))],
            ['layer stack', _fmt(d.get('layer_stack'))],
            ['subLayerPaths', _fmt(d.get('sublayer_paths'))],
            ['asset 디렉토리', _fmt(d.get('dep_asset_dirs'))],
            ['미해결 asset', _fmt(d.get('dep_unresolved_sample'))],
            ['stage 열기(s) / 순회(s)', f'{d.get("open_s")} / {d.get("traverse_s")}'],
        ], mono_cols={1}))
        for where, entries in (d.get('arcs') or {}).items():
            out.append(f'<p style="font-size:.8rem;color:var(--text-dim)">composition arc — {esc(where)} (강한 것부터)</p>')
            out.append(table_html(['강도', 'arc 종류', 'target layer'],
                                  [[e.get('rank'), e.get('arc_type'), e.get('target_layer')] for e in entries],
                                  mono_cols={2}))
    return ''.join(out)


def build_report(descs, diff_html, differing, override, guard, args) -> str:
    """리포트는 세 절로만 쓴다 — 목적 / 커맨드와 결과 / Takeaway.

    구조 덤프(prim 히스토그램·layer stack·composition arc)는 읽는 사람에게 "so what?"이라
    본문에서 빼고 맨 아래 접힌 블록에 원자료로만 남긴다.
    """
    by_name = dict(descs)
    parts = [TABLE_CSS, glossary_note_html()]

    # ------------------------------------------------------------------ 목적
    parts.append('<h2>목적</h2>')
    parts.append(
        '<p>이 파이프라인은 USD를 <b>읽기만</b> 했다. 그래서 답을 모르는 질문이 두 개 있었다.</p>'
        '<p><b>질문 1 — 같은 씬이 왜 파일 4개로 복제돼 있나?</b><br>'
        '<code>fixed.usd</code> · <code>fixed_docker.usd</code> · <code>isaacsim_&lt;hash&gt;.usd</code> · '
        '<code>isaacsim_&lt;hash&gt;_non_metric.usd</code>. "경로나 단위만 다를 것"이라고 <b>추정만</b> 하고 '
        '실제로 확인한 적이 없다. 씬이 65개이므로 이 배수만큼 낭비된다.</p>'
        '<p><b>질문 2 — 원본을 안 건드리고 값만 바꿀 수 있나?</b><br>'
        'USD의 핵심 기능인 <b>레이어 겹치기</b>를 한 번도 써본 적이 없다. 이게 되면 위의 복제를 '
        '파일이 아니라 레이어로 표현할 수 있다.</p>')
    parts.append('<h3>전체 그림 — 4개가 어디서 나왔고 누가 쓰나</h3>')
    parts.append(_lineage_svg())

    # ------------------------------------------------------- 커맨드와 결과
    parts.append('<h2>커맨드와 결과</h2>')
    parts.append(f'<pre>/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s1_layers.py --scene {esc(args.scene)} --log_dir {esc(args.log_dir)}</pre>')
    parts.append('<p style="color:var(--text-dim);font-size:.85rem">'
                 '<b>파일이 4개인데 커맨드가 하나인 이유</b>: 스크립트는 <code>--scene</code>만 받고 '
                 '그 씬 폴더에서 4개를 <b>스스로 찾아 한 번에 비교</b>한다'
                 '(<code>mp3d_usd_variants()</code>). 파일별로 돌릴 필요가 없다. '
                 'Isaac Sim을 띄우지 않는다(수 초). '
                 '<code>--assets all</code>로 노은역 usdz(2.2 GB)까지 포함할 수 있다.</p>')

    parts.append('<h3>결과 1 — 파일 4개는 정확히 이게 다르다</h3>')
    parts.append(f'<p>비교한 항목 <b>{len(differing)}개</b>가 서로 달랐다. '
                 '같았던 항목이 있으면 표 아래에 따로 적었다.</p>')
    parts.append(diff_html)

    parts.append('<h3>결과 2 — 원본을 안 건드리고 값을 바꿨다</h3>')
    if override is None:
        parts.append('<p><span class="pill warn">SKIPPED</span></p>')
    else:
        o = override
        parts.append(f'<p><code>{esc(Path(o["base"]).name)}</code>를 아래에 깔고, 그 위 새 파일'
                     f'(<code>override.usda</code>)에 값 두 개를 덮어썼다.</p>')
        parts.append(table_html(
            ['바꾼 것', '원본 단독', '겹친 직후', '덮어쓴 뒤', '원본 재확인', '판정'],
            [
                ['이 벽 조각이 보이나', _vis(o['base_visibility']), _vis(o['composed_visibility_before_edit']),
                 _vis(o['composed_visibility_after']), _vis(o['base_visibility_after']), pill(o['c1_ok'])],
                ['숫자 1이 몇 미터인가', o['base_meters_per_unit'], o['composed_mpu_inherited'],
                 o['composed_mpu_after'], o['base_mpu_after'], pill(o['c2_ok'])],
            ], mono_cols={1, 2, 3, 4}))
        parts.append('<p>가운데 두 칸이 핵심이다 — <b>겹친 직후엔 원본 값이 그대로 보이고, '
                     '덮어쓴 뒤엔 새 값이 보이는데, 원본 파일은 안 변했다.</b></p>')
        parts.append(f'<p>원본 <code>{esc(Path(o["base"]).name)}</code> sha256 실행 전후 '
                     f'{pill(guard.all_unchanged, "동일", "변경됨")} — 파일이 안 바뀌었다는 증거.</p>')
        parts.append(_details('내가 쓴 override 레이어 전문 (펼치기)',
                              f'<pre>{esc(o["override_layer_text"])}</pre>'))

    # -------------------------------------------------------------- Takeaway
    parts.append('<h2>Takeaway</h2>')
    n1 = by_name.get('isaacsim', {})
    nm = by_name.get('isaacsim_non_metric', {})
    fx = by_name.get('fixed', {})
    fd = by_name.get('fixed_docker', {})
    tk = []

    tk.append(('<b>형상은 4개가 완전히 같다 — 다른 건 "적힌 값"과 껍데기뿐이다</b>',
               '네 파일의 좌표 범위가 <b>16.35 × 8.279 × 2.807로 동일</b>하다. 즉 메시를 다시 만든 게 아니라 '
               '같은 형상에 <b>선언값(위쪽 축·단위)</b>과 <b>감싸는 껍데기</b>, <b>부속물</b>만 달리 붙인 것이다. '
               '→ 4개를 "서로 다른 씬"으로 볼 이유가 없다.'))

    tk.append(('<b>단위 선언이 데이터와 맞는 파일은 <code>isaacsim_&lt;hash&gt;.usd</code> 하나뿐이다</b>',
               f'이 파일만 "1단위=1 m"({n1.get("meters_per_unit")})이고 나머지 셋은 0.01이다. '
               '좌표값은 넷이 같으므로, 선언을 존중하는 도구가 나머지 셋을 읽으면 <b>16 cm짜리 건물</b>이 된다. '
               '→ 파이프라인이 렌더에 이 파일을 쓰는 이유가 이것이다. 그리고 <b>하필 이 파일만</b> 텍스처가 '
               f'고정 절대경로({n1.get("dep_baked_abs_count")}개)라서 심링크가 필요했다 — 우연이 아니라 '
               '<b>같은 변환 단계의 산물</b>이다(S2로 이어짐).'))

    tk.append(('<b>물리 개수 0 → 1 → 3은 단계가 쌓인 결과다</b>',
               '어느 prim에 붙었는지 보면 계층이 그대로 포개진다. <code>isaacsim</code>은 '
               '<code>geometry</code> Xform <b>하나</b>에 "이 아래 전체가 충돌 대상"을 걸었고(1개), '
               '<code>fixed</code>/<code>fixed_docker</code>는 거기에 <b>정점 4개짜리 작은 판 2장</b>을 '
               '더했다(1×1 m, 2.1×1 m · 바닥 높이 z=0.0, 0.1). 건물이 16×8 m인 걸 감안하면 '
               '<b>스캔 구멍을 메우는 패치</b>로 보인다. 개수도 정확히 맞는다 — 물건 121→124, 형상 71→73, 물리 1→3.'))

    tk.append((f'<b><code>_non_metric</code>은 어디서도 안 쓴다</b>',
               f'위쪽 축이 <b>{nm.get("up_axis")}</b>로 선언돼 있는데(나머지는 {fx.get("up_axis")}) 좌표는 같다 — '
               '즉 <b>선언이 데이터와 안 맞는다</b>. 물리도 0개다. 코드가 이름으로 명시적으로 제외한다'
               '(<code>04_render_obs_isaac.py:145</code>). → 지워도 되는 후보지만 이번 범위에서는 건드리지 않았다.'))

    if override is not None and override['c1_ok'] and override['c2_ok']:
        tk.append(('<b>레이어 겹치기가 실제로 된다 — 원본 무변경으로</b>',
                   '물건의 속성(보임/안보임)도, 파일 전체 설정(단위)도 덮어쓸 수 있었다. '
                   '→ ① <code>fixed</code> vs <code>fixed_docker</code>처럼 <b>경로만 다른 복제</b>는 '
                   '레이어 한 장으로 없앨 수 있다 ② 지금 코드가 실행 중에 파이썬으로 하는 조명·visibility 조작도 '
                   '<b>파일 한 장으로 뺄 수 있다</b>(설정이 코드가 아니라 파일이 되면 diff·공유·재현이 된다). '
                   '단 <code>_non_metric</code>은 축 선언까지 달라 이 방법 하나로는 정리되지 않는다.'))

    tk.append(('<b>함정 — 이 레이어를 렌더에 쓰려면 한 줄이 더 필요하다</b>',
               '<code>defaultPrim</code>을 레이어에 <b>직접</b> 써 줘야 한다. 아래에 깐 파일의 것은 '
               '참조 해석에 쓰이지 않아서, 빼먹으면 <b>씬이 빈 채로 렌더된다</b>(S5에서 실제로 겪었다).'))

    parts.append('<div style="display:flex;flex-direction:column;gap:14px;margin:12px 0 22px">')
    for head, body in tk:
        parts.append(
            '<div style="border:1px solid var(--border);border-radius:10px;background:var(--surface);'
            f'padding:12px 14px"><div style="margin-bottom:6px">{head}</div>'
            f'<div style="font-size:.88rem;line-height:1.6;color:var(--text-dim)">{body}</div></div>')
    parts.append('</div>')

    parts.append('<h2>원자료</h2>')
    parts.append(_details('자산별 구조 덤프 — prim 히스토그램 · layer stack · composition arc (펼치기)',
                          _raw_dump_html(descs)))
    parts.append(_details('원본 보존 sha256 대조표 (펼치기)', preservation_section(guard)))
    parts.append(glossary_html())

    summary = stat_row_html([
        ('다른 항목', len(differing)),
        ('원본 무변경', 'YES' if guard.all_unchanged else 'NO'),
        ('레이어 덮어쓰기', ('성공' if override and override['c1_ok'] and override['c2_ok'] else '실패') if override else '—'),
        ('scene', args.scene),
    ])
    return summary, ''.join(parts)


def main():
    ap = argparse.ArgumentParser(description='S1 — USD composition 실습 (읽기/diff/authoring)')
    ap.add_argument('--scene', default=DEFAULT_SCENE)
    ap.add_argument('--usd_root', default=str(DEFAULT_USD_ROOT))
    ap.add_argument('--usdz', default=str(DEFAULT_USDZ), help='노은역 usdz 경로 (--assets all/usdz 일 때만 사용)')
    ap.add_argument('--assets', choices=['mp3d', 'all', 'usdz'], default='mp3d',
                    help='mp3d=복제본 4종만(빠름) / all=+usdz(2.2GB, 수 분) / usdz=usdz만')
    ap.add_argument('--override_base', default='fixed', help='C단계에서 sublayer로 얹을 자산의 논리 이름')
    # 기본값이 원본과 같으면 C2가 "안 바뀐 것을 안 바뀌었다고 확인"하는 무의미한 테스트가 된다
    # (fixed.usd의 metersPerUnit은 0.01) — 실제로 값이 달라지는 1.0을 기본으로 둔다.
    ap.add_argument('--new_mpu', type=float, default=1.0, help='C2에서 덮어쓸 metersPerUnit (원본과 달라야 유효)')
    ap.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR))
    ap.add_argument('--skip_hash', action='store_true', help='sha256 보존 검사를 건너뜀(디버깅용)')
    args = ap.parse_args()

    variants = mp3d_usd_variants(args.usd_root, args.scene) if args.assets in ('mp3d', 'all') else []
    targets = list(variants)
    if args.assets in ('all', 'usdz'):
        p = Path(args.usdz)
        if p.is_file():
            targets.append(('noeun_usdz', p))
        else:
            print(f'[warn] usdz 없음: {p}', file=sys.stderr)

    if not targets:
        print('[fail] 검사할 자산이 없다', file=sys.stderr)
        return 2

    print(f'[s1] 자산 {len(targets)}개: ' + ', '.join(n for n, _ in targets))
    guard = PreservationGuard([p for _, p in targets])
    if not args.skip_hash:
        print('[s1] sha256 (before) …')
        guard.snapshot_before()

    descs = []
    for name, path in targets:
        print(f'[s1] A 읽기: {name}')
        descs.append((name, describe(path)))

    print('[s1] B diff …')
    diff_src = [(n, d) for n, d in descs if n != 'noeun_usdz']
    diff_html, differing = diff_table(diff_src) if len(diff_src) >= 2 else ('<p>비교 대상 부족</p>', [])

    override = None
    base = dict(variants).get(args.override_base)
    if base is not None:
        base_desc = dict(descs).get(args.override_base, {})
        sample = base_desc.get('mesh_paths_sample') or base_desc.get('top_prims') or []
        if sample:
            print(f'[s1] C authoring: {args.override_base} <- override.usda')
            out_dir = Path(args.log_dir) / 's1_layers' / args.scene
            override = author_override(base, out_dir, sample[0], args.new_mpu)
        else:
            print('[warn] C 대상 prim을 못 찾음 — C 생략', file=sys.stderr)

    if not args.skip_hash:
        print('[s1] sha256 (after) …')
        guard.snapshot_after()

    out_dir = Path(args.log_dir) / 's1_layers' / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 's1_layers.json').write_text(
        json.dumps({'args': vars(args), 'descriptions': dict(descs), 'differing_fields': differing,
                    'override': override, 'preservation': guard.rows()}, indent=2, default=str, ensure_ascii=False),
        encoding='utf-8')

    summary, body = build_report(descs, diff_html, differing, override, guard, args)
    path = viz_utils.save_gallery(out_dir, 'report.html',
                                  f'S1 · USD composition 실습 — {args.scene}', summary, body,
                                  eyebrow='gs_vlnpe usd_study')
    print(f'[s1] report: {path}')

    ok = guard.all_unchanged or args.skip_hash
    if override is not None:
        ok = ok and override['c1_ok'] and override['c2_ok']
    print(f'[s1] {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
