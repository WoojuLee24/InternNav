"""
gs_vlnpe 파이프라인 공용 시각화 유틸리티.

00~06 스크립트가 만드는 진단/오버레이 이미지는 원본 PNG/JPG로도 저장하고,
동시에 이미지를 base64로 인라인 임베드한 self-contained `report.html`을
함께 생성한다. 이 HTML은 외부 리소스에 의존하지 않으므로 Claude Artifact
도구로 그대로 publish해 모바일/웹 브라우저에서 링크로 열어볼 수 있다.

시각화 산출물은 전부 `logs/gs-vlnpe/<script_name>/...` 아래에 모은다
(`docs/execution-staged.md`의 "시각화 출력 컨벤션" 참고).

이 모듈이 제공하는 재사용 가능한 컴포넌트:
  - `blink_widget_html(widget_id, states, caption)` — 화살표(‹ ›)로 여러 이미지를
    같은 자리에서 번갈아 보여주는 "blink comparator". 정렬/정합 확인에 유용
    (두 이미지를 나란히 두는 것보다 번갈아 깜빡이며 보면 어긋남이 훨씬 잘 보인다).
  - `reference_button_html(label, entries)` — 클릭하면 공용 모달에 참고용
    이미지(예: 원본 rgb/depth)를 보여주는 버튼. 메인 흐름을 방해하지 않고
    "참고 자료"를 별도 창(모달)에서 볼 수 있게 한다.
  - `save_gallery(...)` — 위 컴포넌트들로 구성한 `body_html`을 페이지 뼈대에
    끼워 넣고 파일로 저장한다.

`summary_html`/`body_html`에서 쓸 수 있는 공용 CSS 클래스:
  - `<div class="stat-row">` 안에 `<div class="stat"><div class="label">..</div><div class="value">..</div></div>`
  - `<span class="pill good|bad|warn">텍스트</span>` — 판정 결과 배지
"""

import json
from pathlib import Path
import base64

LOG_ROOT = Path('logs/gs-vlnpe/apply_real')

_MIME_BY_EXT = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}

_STYLE = '''
:root {
  color-scheme: light dark;
  --bg: #0a0e13; --surface: #12181f; --surface-2: #1a222b; --border: rgba(255,255,255,0.10);
  --text: #e3e9ef; --text-dim: #8b97a3;
  --accent: #4fd1c5; --accent-strong: #7fe3da;
  --good: #57c785; --bad: #e2725b; --warn: #e0b34d;
  --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f5f7f9; --surface: #ffffff; --surface-2: #eef1f4; --border: rgba(10,20,30,0.12);
    --text: #131a20; --text-dim: #5b6771;
    --accent: #0f9e93; --accent-strong: #0c7a71;
    --good: #1f8a4c; --bad: #c1432a; --warn: #a97a12;
  }
}
:root[data-theme="dark"] {
  --bg: #0a0e13; --surface: #12181f; --surface-2: #1a222b; --border: rgba(255,255,255,0.10);
  --text: #e3e9ef; --text-dim: #8b97a3;
  --accent: #4fd1c5; --accent-strong: #7fe3da;
  --good: #57c785; --bad: #e2725b; --warn: #e0b34d;
}
:root[data-theme="light"] {
  --bg: #f5f7f9; --surface: #ffffff; --surface-2: #eef1f4; --border: rgba(10,20,30,0.12);
  --text: #131a20; --text-dim: #5b6771;
  --accent: #0f9e93; --accent-strong: #0c7a71;
  --good: #1f8a4c; --bad: #c1432a; --warn: #a97a12;
}

* { box-sizing: border-box; }
body {
  font-family: var(--font-ui); margin: 0; padding: 28px 20px 64px; background: var(--bg); color: var(--text);
  max-width: 880px; margin-inline: auto;
}
.eyebrow {
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 6px;
}
h1 { font-size: 1.3rem; font-weight: 650; margin: 0 0 18px; text-wrap: balance; }
h2 { font-size: 1.0rem; font-weight: 600; margin: 0; }

.summary {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 18px; margin-bottom: 24px; font-size: 0.88rem; line-height: 1.55;
}
.summary p { margin: 0 0 10px; }
.summary p:last-child { margin-bottom: 0; }
.summary table { border-collapse: collapse; width: 100%; font-size: 0.83rem; margin: 10px 0; font-variant-numeric: tabular-nums; }
.summary td, .summary th {
  border: 1px solid var(--border); padding: 6px 10px; text-align: left; font-family: var(--font-mono);
}
.summary th { color: var(--text-dim); font-weight: 600; }

.stat-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 14px; }
.stat {
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 14px; min-width: 120px;
}
.stat .label { font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 3px; }
.stat .value { font-family: var(--font-mono); font-size: 1.02rem; font-variant-numeric: tabular-nums; }

.pill {
  display: inline-block; font-family: var(--font-mono); font-size: 0.76rem; padding: 2px 9px;
  border-radius: 999px; border: 1px solid currentColor; line-height: 1.6;
}
.pill.good { color: var(--good); }
.pill.bad { color: var(--bad); }
.pill.warn { color: var(--warn); }

.grid { display: flex; flex-direction: column; gap: 18px; }
.card { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--surface); }
.caption {
  padding: 8px 12px; font-size: 0.78rem; font-family: var(--font-mono); color: var(--text-dim);
  background: var(--surface-2); border-bottom: 1px solid var(--border);
}
.card img { width: 100%; display: block; }

/* --- pair section (grouping of blink widgets + reference button) --- */
.pair-section {
  border: 1px solid var(--border); border-radius: 12px; background: var(--surface);
  padding: 14px 16px 16px; margin-bottom: 20px;
}
.pair-head {
  display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px;
}
.pair-cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
@media (max-width: 560px) { .pair-cols { grid-template-columns: minmax(0, 1fr); } }

/* --- blink comparator ---
   .blink 자체에 min-width:0을 명시하고 .blink-label에도 min-width:0을 줘야 한다: 긴 캡션
   텍스트가 들어오면(예: "cam2world backward-warp — err=33.08 valid_frac=0.80") flex item의
   기본 min-width(auto)가 content 너비를 강제해 카드 전체가 화면보다 넓어지고, 그 결과 이미지
   뷰포트가 화면 밖으로 밀려나 "이미지 크기/위치가 달라 보이는" 버그가 생긴다 — 실제로 겪은 문제. */
.blink { min-width: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface-2); }
.blink-viewport { width: 100%; line-height: 0; background: #000; }
.blink-viewport img { width: 100%; height: auto; display: block; }
.blink-bar { display: flex; align-items: center; gap: 8px; padding: 6px 8px; font-family: var(--font-mono); font-size: 0.74rem; }
.blink-label { flex: 1; min-width: 0; text-align: center; color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.blink-btn {
  font-family: var(--font-mono); font-size: 0.95rem; line-height: 1; padding: 4px 10px; border-radius: 6px;
  border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer;
}
.blink-btn:hover { border-color: var(--accent); color: var(--accent); }
.blink-btn:active { transform: translateY(1px); }
.blink-title { font-size: 0.78rem; color: var(--text-dim); margin-bottom: 6px; font-family: var(--font-mono); }

/* --- reference button + modal --- */
.ref-btn {
  font-family: var(--font-mono); font-size: 0.76rem; padding: 4px 11px; border-radius: 999px;
  border: 1px solid var(--accent); background: transparent; color: var(--accent); cursor: pointer;
}
.ref-btn:hover { background: var(--accent); color: var(--bg); }
dialog#ref-modal {
  border: 1px solid var(--border); border-radius: 12px; background: var(--surface); color: var(--text);
  padding: 0; max-width: min(92vw, 760px); width: 100%;
}
dialog#ref-modal::backdrop { background: rgba(0,0,0,0.6); }
.ref-modal-head {
  display: flex; align-items: center; justify-content: space-between; padding: 10px 14px;
  border-bottom: 1px solid var(--border); font-family: var(--font-mono); font-size: 0.82rem; color: var(--text-dim);
}
.ref-modal-head button {
  font-family: var(--font-mono); font-size: 1rem; background: none; border: none; color: var(--text);
  cursor: pointer; line-height: 1; padding: 2px 6px;
}
.ref-modal-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 10px; padding: 14px; }
@media (max-width: 560px) { .ref-modal-grid { grid-template-columns: minmax(0, 1fr); } }
.ref-modal-grid img { width: 100%; border-radius: 6px; display: block; }
.ref-label { font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim); margin-bottom: 4px; }
'''

_SHARED_SCRIPT = '''
document.querySelectorAll(".blink").forEach(function (el) {
  var states = JSON.parse(el.getAttribute("data-states"));
  var i = 0;
  var img = el.querySelector(".blink-img");
  var label = el.querySelector(".blink-label");
  function render() {
    img.src = states[i].src;
    label.textContent = states[i].label + "  (" + (i + 1) + "/" + states.length + ")";
  }
  el.querySelector(".blink-prev").addEventListener("click", function () {
    i = (i - 1 + states.length) % states.length; render();
  });
  el.querySelector(".blink-next").addEventListener("click", function () {
    i = (i + 1) % states.length; render();
  });
  render();
});

var refModal = document.getElementById("ref-modal");
var refGrid = document.getElementById("ref-modal-grid");
function openRefModal(entries) {
  refGrid.innerHTML = entries.map(function (e) {
    return '<div><div class="ref-label"></div><img></div>';
  }).join("");
  var cells = refGrid.querySelectorAll("div > div.ref-label");
  var imgs = refGrid.querySelectorAll("img");
  entries.forEach(function (e, idx) { cells[idx].textContent = e.label; imgs[idx].src = e.src; });
  if (typeof refModal.showModal === "function") { refModal.showModal(); } else { refModal.setAttribute("open", ""); }
}
document.querySelectorAll(".ref-btn").forEach(function (btn) {
  btn.addEventListener("click", function () {
    openRefModal(JSON.parse(btn.getAttribute("data-ref")));
  });
});
document.getElementById("ref-modal-close").addEventListener("click", function () {
  if (typeof refModal.close === "function") { refModal.close(); } else { refModal.removeAttribute("open"); }
});
refModal.addEventListener("click", function (ev) { if (ev.target === refModal) refModal.close(); });
'''

_REFERENCE_MODAL_HTML = '''
<dialog id="ref-modal">
  <div class="ref-modal-head"><span>reference</span><button type="button" id="ref-modal-close">&times;</button></div>
  <div class="ref-modal-grid" id="ref-modal-grid"></div>
</dialog>
'''


def image_to_data_uri(path) -> str:
    path = Path(path)
    mime = _MIME_BY_EXT.get(path.suffix.lower(), 'application/octet-stream')
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{mime};base64,{data}'


def blink_widget_html(widget_id: str, states, title: str = '') -> str:
    """화살표(‹ ›)로 states를 번갈아 보여주는 comparator.

    states: [(label:str, image_path), ...] — 2개 이상. 예: [("actual rgb(t15)", p1), ("cam2world warp err=41.6", p2)]
    """
    payload = json.dumps([{'label': label, 'src': image_to_data_uri(path)} for label, path in states])
    title_html = f'<div class="blink-title">{title}</div>' if title else ''
    return f'''{title_html}<div class="blink" id="{widget_id}" data-states='{payload}'>
      <div class="blink-viewport"><img class="blink-img" alt=""></div>
      <div class="blink-bar">
        <button type="button" class="blink-btn blink-prev">&lsaquo;</button>
        <span class="blink-label"></span>
        <button type="button" class="blink-btn blink-next">&rsaquo;</button>
      </div>
    </div>'''


def reference_button_html(label: str, entries) -> str:
    """entries: [(label:str, image_path), ...] — 클릭 시 공용 모달에 표시."""
    payload = json.dumps([{'label': lbl, 'src': image_to_data_uri(path)} for lbl, path in entries])
    return f'<button type="button" class="ref-btn" data-ref=\'{payload}\'>{label}</button>'


def build_gallery_html(title: str, summary_html: str, body_html: str, eyebrow: str = 'gs_vlnpe pipeline') -> str:
    return f'''<title>{title}</title>
<style>{_STYLE}</style>
<div class="eyebrow">{eyebrow}</div>
<h1>{title}</h1>
<div class="summary">{summary_html}</div>
{body_html}
{_REFERENCE_MODAL_HTML}
<script>{_SHARED_SCRIPT}</script>
'''


def save_gallery(out_dir, filename: str, title: str, summary_html: str, body_html: str,
                  eyebrow: str = 'gs_vlnpe pipeline') -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = build_gallery_html(title, summary_html, body_html, eyebrow=eyebrow)
    path = out_dir / filename
    path.write_text(html, encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# floorplan 캔버스 — 경로 비교 그림 (03 / 03b 공용)
# ---------------------------------------------------------------------------

# 경로 색 규약 — 리포트 전체에서 통일한다
GT_COLOR = (255, 230, 60)        # 노랑: 정답지
OURS_COLOR = (0, 255, 90)        # 초록: 우리 경로
FLOOR_COLOR = (90, 190, 255)     # 하늘: 기준선 ⓒ (루트가 완벽해도 남는 오차)
FLOOR_B_COLOR = (200, 120, 255)  # 보라: 기준선 ⓑ (refine 없음)
KNOT_COLOR = (255, 120, 60)      # 주황: 복원한 knot
FLOORPLAN_MIN_PX = 700


def floorplan_canvas(navigable, origin, cell_m: float, out_dir, min_px: int = FLOORPLAN_MIN_PX):
    """navigable floorplan 위에 world xy 경로를 그리는 `(base, draw, emit)` 묶음.

    2D 격자는 `arr[iy, ix]`이고 y를 뒤집지 않지만, **화면은 y가 아래로 증가**하므로 그리기 직전에만
    `h-1-iy`로 뒤집는다(`esdf_utils` 모듈 docstring의 좌표 규약 참고).

    `draw(img, xy, color, radius)` / `emit(img, name) -> Path`
    """
    import cv2
    import numpy as np

    from geometry_utils import save_jpg

    h, w = navigable.shape
    base = np.where(navigable[::-1][..., None], np.uint8([60, 90, 60]), np.uint8([38, 38, 38]))
    factor = max(1, int(np.ceil(min_px / max(h, w))))

    def draw(img, xy, color, radius=1):
        xy = np.asarray(xy, dtype=np.float64)
        if not len(xy):
            return img
        ij = ((xy[:, :2] - np.asarray(origin)[:2]) / cell_m).astype(int)
        keep = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        for x, y in ij[keep]:
            cv2.circle(img, (int(x), int(h - 1 - y)), radius, color, -1)
        return img

    def emit(img, name):
        return save_jpg(cv2.resize(img, None, fx=factor, fy=factor,
                                   interpolation=cv2.INTER_NEAREST), Path(out_dir) / name)

    return base, draw, emit


def line_chart(series, out_path, x_label: str = '', y_label: str = '',
               size=(900, 380), pad=56, legend_w=150):
    """간단한 꺾은선 차트 — `series`: `[(label, color, xs, ys), ...]`.

    matplotlib을 쓰지 않는다(리포트가 self-contained여야 하고 의존성을 늘리지 않는다).
    축 라벨은 ASCII만 쓸 것 — cv2.putText는 한글을 못 그린다.
    """
    import cv2
    import numpy as np

    from geometry_utils import save_jpg

    W, H = size
    img = np.full((H, W, 3), 24, dtype=np.uint8)
    xs_all = np.concatenate([np.asarray(s[2], dtype=np.float64) for s in series])
    ys_all = np.concatenate([np.asarray(s[3], dtype=np.float64) for s in series])
    x0, x1 = float(xs_all.min()), float(xs_all.max())
    y0, y1 = 0.0, float(ys_all.max()) * 1.08
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0

    # 범례는 플롯 영역 **밖** 오른쪽 여백에 둔다 — 곡선 위에 얹으면 가린다
    plot_w = W - 2 * pad - legend_w

    def px(x, y):
        return (int(pad + (x - x0) / (x1 - x0) * plot_w),
                int(H - pad - (y - y0) / (y1 - y0) * (H - 2 * pad)))

    for frac in np.linspace(0, 1, 5):                     # 가로 격자 + y 눈금
        gy = y0 + frac * (y1 - y0)
        cv2.line(img, px(x0, gy), px(x1, gy), (52, 58, 66), 1)
        cv2.putText(img, f'{gy:.2f}', (6, px(x0, gy)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (140, 150, 160), 1, cv2.LINE_AA)
    for s in reversed(series):     # 앞에 둔 계열이 위에 그려지게 — 강조 곡선을 먼저 넘길 수 있다
        xs, ys = np.asarray(s[2], dtype=np.float64), np.asarray(s[3], dtype=np.float64)
        pts = np.array([px(x, y) for x, y in zip(xs, ys)], dtype=np.int32)
        cv2.polylines(img, [pts], False, s[1], 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 3, s[1], -1, cv2.LINE_AA)
    for i, s in enumerate(series):                        # 범례 — 오른쪽 안쪽 (y축 라벨과 겹치지 않게)
        y = pad + 8 + i * 17
        lx = W - legend_w - 4
        cv2.line(img, (lx, y), (lx + 20, y), s[1], 2, cv2.LINE_AA)
        cv2.putText(img, s[0], (lx + 26, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (215, 222, 230), 1, cv2.LINE_AA)
    for x in np.asarray(series[0][2], dtype=np.float64):  # x 눈금
        cv2.putText(img, f'{x:g}', (px(x, y0)[0] - 10, H - pad + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 150, 160), 1, cv2.LINE_AA)
    cv2.putText(img, x_label, (pad + plot_w // 2 - 40, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (170, 180, 190), 1, cv2.LINE_AA)
    cv2.putText(img, y_label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (170, 180, 190), 1, cv2.LINE_AA)
    return save_jpg(img, Path(out_path))
