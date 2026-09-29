"""S1~S5 실습 공용 유틸 — 자산 경로 해석, sha256 보존 검사, 표 HTML.

이 폴더의 스크립트는 **기존 gs_vlnpe 코드를 한 줄도 수정하지 않는다.** 리포트 HTML은
`gs_vlnpe/viz_utils.py`의 `save_gallery`를 그대로 재사용하고, 여기에는 그 위에 얹을
표 렌더러와 "원본이 안 변했다"를 증명하는 해시 헬퍼만 둔다.
"""

import hashlib
import html
import sys
import time
from pathlib import Path

# gs_vlnpe/ 를 import path에 올린다 (viz_utils 재사용).
_GS_VLNPE = Path(__file__).resolve().parent.parent
if str(_GS_VLNPE) not in sys.path:
    sys.path.insert(0, str(_GS_VLNPE))

DEFAULT_SCENE = '17DRP5sb8fy'
DEFAULT_USD_ROOT = Path('data/scene_data/mp3d_pe')
DEFAULT_USDZ = Path('data/GS_USDZ/Subway/noeun_station_collision.usdz')
DEFAULT_LOG_DIR = Path('logs/gs-vlnpe/usd_study')

# `04_render_obs_isaac.py:107` 의 상수와 같은 값 — mp3d_pe USD에 박혀 있는 원래 변환 환경의
# 절대경로. S1의 diff와 S2의 resolve 검사가 둘 다 이 접두어를 찾는다.
MATTERPORT_TEXTURE_ROOT = Path('/ssd/share/Matterport3D/data/v1/scans')


def scene_mesh_dir(usd_root, scene: str) -> Path:
    """`<usd_root>/<scene>/matterport_mesh/<hash>/` — 해시 폴더명은 씬마다 달라 glob으로 찾는다."""
    base = Path(usd_root) / scene / 'matterport_mesh'
    subdirs = sorted(p for p in base.glob('*') if p.is_dir())
    if not subdirs:
        raise FileNotFoundError(f'matterport_mesh 하위 해시 폴더를 못 찾음: {base}')
    return subdirs[0]


def mp3d_usd_variants(usd_root, scene: str):
    """같은 씬의 USD 복제본 4종을 논리 이름 -> 경로로 반환 (존재하는 것만)."""
    d = scene_mesh_dir(usd_root, scene)
    stem = d.name
    candidates = [
        ('fixed', d / 'fixed.usd'),
        ('fixed_docker', d / 'fixed_docker.usd'),
        ('isaacsim', d / f'isaacsim_{stem}.usd'),
        ('isaacsim_non_metric', d / f'isaacsim_{stem}_non_metric.usd'),
    ]
    return [(name, p) for name, p in candidates if p.is_file()]


def sha256_of(path, chunk_mb: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk_mb << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class PreservationGuard:
    """원본 자산이 실행 전후로 안 변했음을 증명하는 sha256 스냅샷.

    계획의 "기존 데이터 보존" 제약을 코드로 확인하는 장치다. 큰 usdz도 그대로 해싱한다
    (읽기 전용이므로 위험은 없고 시간만 든다).
    """

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths if Path(p).is_file()]
        self.before = {}
        self.after = {}
        self.elapsed_s = 0.0

    def snapshot_before(self):
        t0 = time.time()
        self.before = {str(p): (p.stat().st_size, sha256_of(p)) for p in self.paths}
        self.elapsed_s += time.time() - t0
        return self

    def snapshot_after(self):
        t0 = time.time()
        self.after = {str(p): (p.stat().st_size, sha256_of(p)) for p in self.paths}
        self.elapsed_s += time.time() - t0
        return self

    def rows(self):
        out = []
        for key in self.before:
            b, a = self.before[key], self.after.get(key)
            ok = a is not None and a == b
            out.append({'path': key, 'size': b[0], 'sha_before': b[1], 'sha_after': (a[1] if a else '—'), 'unchanged': ok})
        return out

    @property
    def all_unchanged(self) -> bool:
        return bool(self.before) and all(r['unchanged'] for r in self.rows())


# ---------------------------------------------------------------------------
# 표 HTML — viz_utils 의 CSS 클래스(pill good/bad/warn, stat-row)를 그대로 쓴다
# ---------------------------------------------------------------------------

TABLE_CSS = '''
<style>
table.us { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13px; }
table.us th, table.us td { border: 1px solid var(--border); padding: 6px 9px; text-align: left; vertical-align: top; }
table.us th { background: var(--surface-2); font-weight: 600; }
table.us td.mono, table.us th.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
table.us tr.differ td:first-child { color: var(--accent-strong); font-weight: 600; }
table.us td.wrap { max-width: 520px; word-break: break-all; }
</style>
'''


def esc(v) -> str:
    return html.escape(str(v))


def pill(ok: bool, good: str = 'PASS', bad: str = 'FAIL') -> str:
    return f'<span class="pill {"good" if ok else "bad"}">{esc(good if ok else bad)}</span>'


def table_html(headers, rows, row_classes=None, mono_cols=()) -> str:
    """rows: list of list. mono_cols: monospace로 렌더할 컬럼 인덱스 집합."""
    head = ''.join(f'<th>{esc(h)}</th>' for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = f' class="{row_classes[i]}"' if row_classes and row_classes[i] else ''
        cells = []
        for j, cell in enumerate(row):
            klass = 'mono wrap' if j in mono_cols else ''
            # 이미 HTML(pill 등)인 셀은 그대로 통과시킨다.
            text = cell if isinstance(cell, str) and cell.startswith('<span') else esc(cell)
            cells.append(f'<td class="{klass}">{text}</td>')
        body.append(f'<tr{cls}>{"".join(cells)}</tr>')
    return f'<table class="us"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def stat_row_html(pairs) -> str:
    items = ''.join(f'<div class="stat"><div class="label">{esc(k)}</div><div class="value">{esc(v)}</div></div>' for k, v in pairs)
    return f'<div class="stat-row">{items}</div>'


def preservation_section(guard: 'PreservationGuard') -> str:
    """리포트 맨 아래에 붙이는 '원본 무변경' 증거 절."""
    rows = [
        [Path(r['path']).name, f'{r["size"]:,}', r['sha_before'][:16] + '…', r['sha_after'][:16] + '…',
         pill(r['unchanged'], 'UNCHANGED', 'CHANGED')]
        for r in guard.rows()
    ]
    verdict = pill(guard.all_unchanged, 'ALL UNCHANGED', 'MUTATION DETECTED')
    return (
        f'<h2>원본 자산 보존 검사 {verdict}</h2>'
        f'<p>실행 전후 sha256 비교. 해싱에 {guard.elapsed_s:.1f}s 소요.</p>'
        + table_html(['파일', '크기(B)', 'sha256 (전)', 'sha256 (후)', '판정'], rows, mono_cols={2, 3})
    )


# ---------------------------------------------------------------------------
# 용어 절 — S1~S5 리포트 맨 아래에 공통으로 붙인다
# ---------------------------------------------------------------------------

_GLOSSARY = [
    ('문서 구조', [
        ('Stage', '겹쳐서 <b>합쳐진 최종 결과</b>. "열어놓은 장면"'),
        ('Prim', '문서 안의 <b>물건 하나</b>. 폴더처럼 경로로 쓴다 — 예: <code>/Root/geometry/chunk000_group000_sub002</code> = 벽 조각 하나'),
        ('Attribute', '그 물건의 <b>속성값</b> — <code>visibility</code>(보임/안보임), 텍스처 파일 경로 등'),
        ('Mesh', '삼각형으로 된 <b>실제 형상</b>'),
        ('defaultPrim', '남이 이 파일을 가져다 쓸 때 <b>기본으로 가져올 물건</b>'),
        ('upAxis', '<b>어느 축이 위쪽</b>인가 (Z / Y)'),
        ('metersPerUnit', '숫자 <b>1이 몇 미터</b>인가 (0.01=cm, 1.0=m)'),
        ('usdz', 'USD와 텍스처를 <b>zip처럼 한 파일로 묶은 것</b>'),
    ]),
    ('레이어 겹치기 (이 실습의 핵심)', [
        ('Layer', '<b>파일 한 장.</b> 여러 장을 겹쳐 stage가 된다 (포토샵 레이어와 같다)'),
        ('Sublayer', '<b>아래에 깔아둔 파일.</b> 위 파일이 아래 파일을 덮어쓴다'),
        ('Composition', '여러 장을 <b>겹쳐 하나로 합치는 규칙</b> 전체'),
        ('Composition arc', '<b>"이 물건의 값이 어느 파일에서 왔는가" 연결선.</b> 겹친 파일이 3장이면 arc 3개, <b>위에 있는 게 이긴다</b>'),
        ('Opinion', '각 레이어가 내는 <b>"이 값은 이거야"라는 주장.</b> 위 레이어 주장이 이긴다'),
        ('Override (<code>over</code>)', '위 레이어에서 <b>값 덮어쓰기.</b> 원본은 안 바뀐다'),
        ('Reference', '다른 파일을 <b>특정 위치에 끼워 넣기</b> (sublayer와 다름)'),
        ('Payload', '<b>"필요할 때 열기"</b> 방식 참조. 큰 파일을 미리 안 읽게 하는 용도'),
        ('LIVRPS', '겹쳤을 때 <b>누가 이기는지의 우선순위 규칙</b> 이름'),
    ]),
    ('규격 · 종류', [
        ('Schema', '그 물건이 <b>무슨 종류인가</b> 하는 규격'),
        ('Typed schema', '물건의 <b>종류 자체</b> (이건 Mesh다)'),
        ('Applied API schema', '나중에 <b>덧붙인 기능</b> (이건 물리 충돌도 한다)'),
        ('PhysicsCollisionAPI', '"이 물건은 <b>물리 충돌 대상</b>" 표시. 로봇이 부딪히는 대상'),
        ('Kind', '이 물건이 <b>부품(component)인지 조립품(assembly)인지</b> 표시'),
    ]),
    ('파일 경로 찾기', [
        ('Asset path', 'USD 안에 적힌 <b>"텍스처 이미지가 어디 있다"는 문자열</b>'),
        ('Resolve', '그 문자열로 <b>실제 파일을 찾아내는 것</b>'),
        ('Ar (resolver)', '그 찾기를 담당하는 모듈'),
        ('상대 / 절대경로', '<code>./textures/a.jpg</code>(파일 옆) / <code>/ssd/share/...</code>(고정 위치). <b>절대경로가 박히면 다른 환경에서 깨진다</b>'),
        ('Symlink', '파일시스템 <b>바로가기</b>. 지금 코드가 깨진 절대경로를 이걸로 우회하고 있다'),
    ]),
    ('표면 재질 · 렌더', [
        ('Material / Shader', '<b>표면 재질</b> 설정 (색·거칠기·텍스처)'),
        ('MDL / OmniPBR', 'NVIDIA의 재질 방식. <b>이 씬이 쓰는 것</b>'),
        ('UsdPreviewSurface', 'USD 표준 재질 방식. <b>이 씬은 이게 아니다</b>'),
        ('colorSpace', '이미지 <b>색을 어떻게 해석할지</b> (<code>sRGB</code>/<code>raw</code>). 틀리면 화면이 확 밝아진다'),
        ('Tonemap op', 'RTX가 <b>밝기 곡선을 어떤 방식으로 압축할지</b> 고르는 번호 0~7 (사진의 "필름 종류"). '
                       '0 Clamp · 1 Linear · 2 Reinhard · 3 Modified Reinhard · 4 HejlHableAlu · 5 HableUc2 · '
                       '<b>6 Aces</b> · <b>7 Iray</b>'),
        ('Render delegate', '실제로 그림을 그리는 <b>렌더 엔진 종류</b> (RTX Real-Time / Path Tracing)'),
        ('carb settings', 'Isaac Sim <b>내부 설정값 트리</b>. <code>/rtx/...</code> 같은 경로로 접근'),
    ]),
    ('측정 지표', [
        ('SSIM', '두 이미지가 <b>얼마나 닮았는지</b> 0~1 점수. 1이면 똑같다'),
        ('sha256', '파일 <b>지문</b>. 값이 같으면 파일이 안 바뀌었다는 증거'),
        ('잔차 <code>1-SSIM</code>', '아직 안 맞는 정도. SSIM 0.904 → 잔차 0.096'),
    ]),
]


def glossary_html() -> str:
    """리포트 맨 아래 용어 절. 표 셀에 HTML을 넣으므로 `esc`를 통과시키지 않는다."""
    parts = ['<h2 id="glossary">용어 — 모르는 단어가 나오면 여기</h2>',
             '<p>USD 파일은 <b>3D 장면을 적어둔 문서</b>다. 파워포인트 파일처럼 안에 물건들이 들어 있고, '
             '포토샵처럼 <b>여러 장을 겹쳐서</b> 하나의 결과를 만든다. 이 두 가지만 잡으면 나머지는 따라온다.</p>']
    for title, rows in _GLOSSARY:
        parts.append(f'<h3>{esc(title)}</h3>')
        body = ''.join(
            f'<tr><td class="mono" style="white-space:nowrap">{term}</td><td>{desc}</td></tr>'
            for term, desc in rows)
        parts.append('<table class="us"><thead><tr><th>용어</th><th>쉬운 말</th></tr></thead>'
                     f'<tbody>{body}</tbody></table>')
    return ''.join(parts)


def glossary_note_html() -> str:
    """리포트 상단에 두는 한 줄 안내."""
    return ('<p style="opacity:.8">용어가 낯설면 <a href="#glossary">맨 아래 「용어」 절</a>을 먼저 보세요 — '
            'prim · layer · composition arc 등을 쉬운 말로 정리해 뒀습니다.</p>')


def new_override_layer(base_usd, out_path):
    """`base_usd`를 sublayer로 깔고 그 위에 덮어쓸 빈 레이어를 만든다.

    두 가지를 반드시 해 준다 — 둘 다 빼먹으면 조용히 잘못 동작한다:

    1. **sublayer 경로를 상대경로로** 넣는다. 레이어를 다른 곳으로 옮겨도 같이 움직인다.
    2. **원본의 "파일 전체 설정"을 이 레이어에 직접 복사**한다 —
       `defaultPrim` · `upAxis` · `metersPerUnit`.

    2번이 왜 필요한가(실측): **물건(prim)과 그 값은 아래 깐 파일에서 올라오지만, 파일 전체
    설정은 안 올라온다.** `metersPerUnit=1.0`을 적어둔 원본 위에 아무것도 안 적은 레이어를
    얹고 열면 `upAxis=Y, metersPerUnit=0.01`(USD 기본값)로 읽힌다 — 물건 수는 121로 멀쩡한데도.
    `defaultPrim`을 빼먹으면 `UsdFileCfg`로 스폰할 때 **씬이 빈 채로 렌더된다**(S5에서 겪었다).

    반환: `Sdf.Layer` (호출자가 `Usd.Stage.Open(layer)`로 열어 authoring한 뒤 `layer.Save()`)
    """
    import os
    from pathlib import Path as _Path

    from pxr import Sdf

    base_usd, out_path = _Path(base_usd), _Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    layer = Sdf.Layer.CreateNew(str(out_path))
    layer.subLayerPaths.append(os.path.relpath(base_usd.resolve(), out_path.parent.resolve()))
    base_layer = Sdf.Layer.FindOrOpen(str(base_usd))
    if base_layer is not None:
        if base_layer.defaultPrim:
            layer.defaultPrim = base_layer.defaultPrim
        for field in ('upAxis', 'metersPerUnit'):
            if base_layer.pseudoRoot.HasInfo(field):
                layer.pseudoRoot.SetInfo(field, base_layer.pseudoRoot.GetInfo(field))
    return layer


def clear_render_prims():
    """`build_renderer`를 한 프로세스에서 여러 번 부르기 위한 정리.

    `build_renderer`는 `/World/Scene`만 지우고 `/World/RenderCamera`는 남긴다(원래 한 번만
    호출되는 함수라서). 두 번째 호출에서 `spawn_camera`가
    `ValueError: A prim already exists at path: '/World/RenderCamera'`로 죽는다(실측).
    기존 파일을 고치지 않기 위해 **호출자 쪽에서** 먼저 치운다. 부팅 이후에만 호출 가능.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for path in ('/World/RenderCamera', '/World/Scene'):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)


def exit_skipping_isaac_teardown(code: int):
    """결과를 다 쓴 뒤, Isaac 정리 단계를 건너뛰고 즉시 종료한다.

    **왜 필요한가**: Isaac을 띄운 스크립트가 정상적으로 `sys.exit()`을 하면 인터프리터 종료 →
    `atexit` → carb 플러그인 정리 순서로 가는데, 거기서 **세그폴트가 난다**(실측):

        atexit_callfuncs -> Py_FinalizeEx -> ... -> libcarb.scripting-python.plugin.so
        Segmentation fault (core dumped)
        There was an error running python

    작업은 이미 끝났고 파일도 다 저장된 뒤라 **결과에는 영향이 없지만**, 화면에 실패처럼 보인다.
    같은 계열의 문제가 `../reports.md`에도 기록돼 있다(`simulation_app.close()`가 수 분 걸려
    SIGKILL로 끊고, 성공 여부는 산출물 파일로 판단하라는 것).

    `os._exit()`은 `atexit`과 인터프리터 정리를 통째로 건너뛰므로 그 크래시를 안 만난다.
    대신 버퍼를 안 비워주므로 **직접 flush한 뒤** 부른다.
    """
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code))
