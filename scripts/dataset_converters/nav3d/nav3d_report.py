"""nav3d 단계 리포트 공용 — 같은 모양의 HTML(밝은·어두운 테마, 폰 폭)을 만든다.

쓰는 법:
    import nav3d_report as R
    body = R.header('nav3d 05a', '제목', '한 줄 설명') + R.section('판정', R.table(rows, cols))
    R.write(out_dir, '탭 제목', body)        # -> report.html (로컬) + report_page.html (Artifact 게시용)
"""

import html as H
import json
from pathlib import Path

CSS = """
/* 점검 보고서: 한 열 읽기 + 넓은 표·그림은 자기 칸에서 가로 스크롤 */
:root{--bg:#f6f7f9;--panel:#ffffff;--fg:#1c2330;--muted:#5d6878;--line:#d9dee6;--accent:#2b5fa8;
--ok:#1d7a46;--ok-bg:#e3f3ea;--ng:#b42318;--ng-bg:#fbe7e5;--info:#7a5b00;--info-bg:#fbf1d6;
--f-body:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12161d;--panel:#1a2029;--fg:#e4e8ee;--muted:#9aa5b4;--line:#2d3542;--accent:#7fa8e6;
--ok:#6fd39a;--ok-bg:#163526;--ng:#ff8a7e;--ng-bg:#3a1d1b;--info:#e8c35a;--info-bg:#352c12;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#12161d;--panel:#1a2029;--fg:#e4e8ee;--muted:#9aa5b4;--line:#2d3542;--accent:#7fa8e6;
--ok:#6fd39a;--ok-bg:#163526;--ng:#ff8a7e;--ng-bg:#3a1d1b;--info:#e8c35a;--info-bg:#352c12;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:15px;line-height:1.6}
.wrap{max-width:1180px;margin:0 auto;padding-inline:16px;padding-block:28px 64px;display:grid;gap:28px}
h1{font-size:1.75rem;line-height:1.25;margin:0;text-wrap:balance}
h2{font-size:1.3rem;margin:0 0 10px;text-wrap:balance;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:1.05rem;margin:18px 0 8px;text-wrap:balance}
p{margin:6px 0;max-width:72ch}.muted{color:var(--muted)}
.eyebrow{font-family:var(--f-mono);font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
section{min-width:0}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--panel);margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);white-space:nowrap;background:var(--bg)}
td.num,.mono{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:.82rem}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-family:var(--f-mono);font-size:.75rem;font-weight:600;letter-spacing:.04em}
.pill.ok{color:var(--ok);background:var(--ok-bg)}.pill.ng{color:var(--ng);background:var(--ng-bg)}.pill.info{color:var(--info);background:var(--info-bg)}
ul.find{margin:0;padding-left:1.2em;display:grid;gap:4px;max-width:80ch}
figure{margin:10px 0;min-width:0}figure img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
figcaption{font-size:.84rem;color:var(--muted);margin-top:4px;max-width:80ch}
details{border:1px solid var(--line);border-radius:6px;background:var(--panel);padding:6px 12px;margin:8px 0}
summary{cursor:pointer;color:var(--accent);font-size:.88rem}
pre{font-family:var(--f-mono);font-size:.78rem;white-space:pre-wrap;word-break:break-word;margin:6px 0}
a{color:var(--accent)}a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600'
         '&family=IBM+Plex+Sans+KR:wght@400;600;700&display=swap">')

esc = H.escape


def pill(v) -> str:
    if type(v).__name__ == 'bool_':          # numpy bool 도 PASS/FAIL 로
        v = bool(v)
    if v is True:
        return '<span class="pill ok">PASS</span>'
    if v is False:
        return '<span class="pill ng">FAIL</span>'
    return f'<span class="pill info">{esc(str(v))}</span>'


def table(rows: list, cols: list, num=(), raw=()) -> str:
    """rows = [dict]. raw 에 든 열은 HTML 그대로(배지 등), 나머지는 이스케이프."""
    h = '<tr>' + ''.join(f'<th>{esc(c)}</th>' for c in cols) + '</tr>'
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, '')
            v = v if c in raw else esc(str(v))
            cells.append(f'<td class="{"num" if c in num else ""}">{v}</td>')
        h += '<tr>' + ''.join(cells) + '</tr>'
    return f'<div class="tw"><table>{h}</table></div>'


def figure(src: str, caption: str) -> str:
    return f'<figure><img src="{esc(src)}" alt="{esc(Path(src).stem)}" loading="lazy"><figcaption>{esc(caption)}</figcaption></figure>'


def raw_json(title: str, obj) -> str:
    return (f'<details><summary>{esc(title)} (원자료 JSON)</summary>'
            f'<pre>{esc(json.dumps(obj, indent=1, ensure_ascii=False, default=str))}</pre></details>')


def bullets(items: list) -> str:
    return '<ul class="find">' + ''.join(f'<li>{esc(x)}</li>' for x in items) + '</ul>'


def header(eyebrow: str, title: str, lead_html: str) -> str:
    return f'<header><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1><p class="muted">{lead_html}</p></header>'


def section(title: str, inner: str) -> str:
    return f'<section><h2>{esc(title)}</h2>{inner}</section>'


def write(out_dir, tab_title: str, body: str) -> None:
    out_dir = Path(out_dir)
    head = f'<title>{esc(tab_title)}</title>\n{FONTS}\n<style>{CSS}</style>\n'
    page = f'<div class="wrap">{body}</div>'
    (out_dir / 'report_page.html').write_text(head + page)
    (out_dir / 'report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                                         '<meta name="viewport" content="width=device-width,initial-scale=1">'
                                         + head + '</head><body>' + page + '</body></html>')
