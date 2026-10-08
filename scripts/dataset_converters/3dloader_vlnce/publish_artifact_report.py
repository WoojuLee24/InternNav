"""Reusable — stage 디렉토리의 시각화를 self-contained Artifact 리포트로 만든다 (매 task 발행용).

`logs/embodiment_augment/<stage>/`의 `*.jpg`를 base64로 인라인하고, 있으면 `summary.html` 조각을
얹어 테마(light/dark) self-contained HTML(`artifact.html`)을 만든다. Artifact CSP(외부/상대경로
차단) 대응. 각 gate/task 스크립트는 시각화 jpg + summary.html만 남기면 이 스크립트가 발행본을 만든다.

실행:
  /usr/bin/python scripts/debugging/publish_artifact_report.py --stage_dir logs/embodiment_augment/g1 \
      --title "G1 — vln_ce pose↔mesh & renderer" --eyebrow "embodiment augmentation · gate"
그 후 Artifact 툴로 <stage_dir>/artifact.html publish.
"""

import argparse
import base64
import glob
from pathlib import Path

CSS = """
:root{
  --bg:#f6f7f9; --surface:#ffffff; --text:#1c2530; --muted:#5b6675; --border:#e2e6ec;
  --accent:#0f766e; --accent-soft:#d7ede9; --pass:#15803d; --fail:#b45309; --mono-bg:#f0f2f5;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0e1319; --surface:#161d26; --text:#e6ebf1; --muted:#93a0b0; --border:#26313d;
  --accent:#2dd4bf; --accent-soft:#123a37; --pass:#4ade80; --fail:#fbbf24; --mono-bg:#0b1016;}}
:root[data-theme="light"]{
  --bg:#f6f7f9; --surface:#ffffff; --text:#1c2530; --muted:#5b6675; --border:#e2e6ec;
  --accent:#0f766e; --accent-soft:#d7ede9; --pass:#15803d; --fail:#b45309; --mono-bg:#f0f2f5;}
:root[data-theme="dark"]{
  --bg:#0e1319; --surface:#161d26; --text:#e6ebf1; --muted:#93a0b0; --border:#26313d;
  --accent:#2dd4bf; --accent-soft:#123a37; --pass:#4ade80; --fail:#fbbf24; --mono-bg:#0b1016;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:40px 24px 80px}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 6px}
h1{font-size:28px;line-height:1.2;margin:0 0 8px;text-wrap:balance;font-weight:680}
.sub{color:var(--muted);margin:0 0 32px;font-size:15px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:22px 24px;margin:0 0 28px}
.card h2,.card h3{margin:0 0 12px}
/* troubleshooting: 기본 접힘. 본문은 "지금 맞는 결과"만 보이고, 틀렸던 측정·버그 이력은 여기로 */
details.trouble{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--fail);
  border-radius:14px;padding:0;margin:0 0 28px}
details.trouble>summary{cursor:pointer;padding:16px 24px;font-weight:620;color:var(--fail);
  list-style:none;display:flex;align-items:center;gap:8px}
details.trouble>summary::-webkit-details-marker{display:none}
details.trouble>summary::before{content:"▸";font-size:13px;transition:transform .15s}
details[open].trouble>summary::before{transform:rotate(90deg)}
details.trouble>summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
details.trouble .inner{padding:0 24px 20px;color:var(--muted);font-size:14px}
details.trouble .inner table{font-size:13px}
.summary table{border-collapse:collapse;width:100%;border:0;margin:0 0 14px;
  font-variant-numeric:tabular-nums;font-size:14px}
.summary th,.summary td{border:0;border-bottom:1px solid var(--border);padding:9px 12px;text-align:left}
.summary th{color:var(--muted);font-weight:600;font-size:12px;letter-spacing:.03em;text-transform:uppercase}
.summary td:nth-child(2){font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;color:var(--accent)}
.summary pre{background:var(--mono-bg);border:1px solid var(--border);border-radius:10px;padding:14px;
  overflow-x:auto;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12.5px;margin:0}
.summary p{margin:0 0 12px;color:var(--muted);font-size:14px}
.summary p b{color:var(--text)}
.intro{background:var(--accent-soft);border-color:transparent}
.intro h3{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0 0 6px}
.intro h3+p{margin:0 0 16px;font-size:14px;line-height:1.6;color:var(--text)}
.intro h3+p:last-child{margin-bottom:0}
.intro code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.92em}
.intro h3.concl{color:var(--pass)}
.intro h3.limit{color:var(--fail)}
.intro h3.concl+p,.intro h3.limit+p{padding-left:10px;border-left:2px solid currentColor}
.intro h3.concl+p{border-left-color:var(--pass)}
.intro h3.limit+p{border-left-color:var(--fail)}
.gallery{display:flex;flex-direction:column;gap:22px}
figure{margin:0;background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
figure img{display:block;width:100%;height:auto}
figcaption{padding:9px 14px;color:var(--muted);font-size:12.5px;
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;border-top:1px solid var(--border)}
"""


# 모든 리포트 맨 위에 공통으로 들어가는 소개. 리포트를 단독으로 열어도 "이게 무슨 작업의 일부인지"를
# 알 수 있어야 한다(링크만 받은 사람은 맥락이 없다).
PIPELINE_INTRO = (
    '학습 중 로봇의 <b>embodiment <code>e</code></b>(반경 <code>r_b</code>, 지면높이, 카메라 높이/pitch)를 '
    '바꿔 <b>정답 경로와 관측을 on-the-fly로 다시 만드는</b> 작업이다. 기존 VLN 데이터는 카메라 리그별로 '
    '미리 렌더해 저장하는 방식이라 <b><code>r_b</code> 축이 아예 없고</b> 리그를 추가할 때마다 수십 GB가 든다. '
    '→ 씬의 3D(mesh/occupancy)를 한 번 저장해두고, 학습 worker가 <code>e</code>에 맞춰 '
    '<code>occ → 2D → r_b dilation → A* → 경로</code>와 depth를 즉석에서 생성한다. '
    '목표는 "같은 장면인데 <code>e</code>가 다르면 정답도 달라야 한다"를 데이터로 만드는 것이다.'
)


# 여러 리포트에 반복 등장하는 **그림 표기 규약**. 리포트마다 다시 쓰지 않도록 공용화한다
# (이게 없으면 "무슨 색이 뭔지" 모르는 그림이 그냥 나열된다 — 실제로 8/12 리포트가 그랬다).
FIGURE_LEGEND = {
    'depth': ('<b>depth 그림</b>: turbo 컬러맵 — <span style="color:#3b6ff5">파랑=가까움</span> → '
              '초록 → 노랑 → <span style="color:#d33">빨강=멂</span>, 검정=값 없음(배경).'),
    'bev': ('<b>BEV 그림</b>(로봇 중심 위에서 본 점유도, 224 px = 10 m → 4.46 cm/px): '
            '검정=미관측(unknown), 회색=빈 공간(free), <span style="color:#d33">빨강=점유(occupied)</span>. '
            '로봇은 그림 중앙, 위쪽이 정면.'),
    'floorplan': ('<b>floorplan</b>(씬 전체를 위에서 본 그림): 초록=통행 가능, 검정=장애물/미관측. '
                  '경로·지점은 그림별 범례를 따른다.'),
    'err': ('<b>오차 패널</b>: <b>0~10 cm 고정</b> 스케일(검정=0, 밝을수록 큼) — 이미지마다 정규화하지 '
            '않으므로 패널끼리 직접 비교된다. <span style="color:#f0f">자홍 = 한쪽에만 지오메트리가 '
            '있는 픽셀</span>(= 잘려나감/누락). 패널 하단에 실제 median·max를 각인한다.'),
    # #04 전용 (`s2_mapcal`) — 다른 stage는 이 키를 쓰지 않으므로 #04의 색·버킷에 맞춰 둔다.
    'diffmap': ('<b>지도 비교</b>: <span style="color:#3c963c">초록=양쪽 다 통행 가능</span> · '
                '<span style="color:#28c">파랑 = 우리만 통행 가능(거짓 승인)</span> · '
                '<span style="color:#d33">빨강 = navmesh만 통행 가능 = 우리만 막음(거짓 기각)</span> · '
                '<span style="color:#f0f">자홍 = 파랑 중 "이 높이에 바닥이 없다"로 설명되는 것</span>. '
                '<b>어두운 곳은 비교 대상이 아니다</b>(GT 주변 2 m 밖). '
                '<span style="color:#ffe63c">노랑 선</span> = 원본 GT 궤적.'),
    'path': ('<b>경로 표기</b>: <span style="color:#fff;background:#333">흰=원본 GT 궤적</span> · '
             '<span style="color:#0dc">시안=사람 주석 waypoint</span> · '
             '<span style="color:#fc0">노랑=start</span> · <span style="color:#f0f">자홍=goal</span> · '
             '<span style="color:#f80">주황=우회가 일어난 지점</span>. 여러 r_b를 겹칠 때는 그림별 '
             '캡션의 색 표기를 따른다.'),
    'chart': ('<b>꺾은선 차트</b>: 축 라벨은 그림 안(ASCII), 계열 이름·색은 그림 오른쪽 범례. '
              'matplotlib을 쓰지 않아 리포트가 self-contained다.'),
}


def legend_card(keys, figtypes=()):
    """그림 규약을 **한 번만** 모아 카드로 — 공용 표기(컬러맵 등) + 이 리포트의 그림 종류별 설명."""
    items = [FIGURE_LEGEND[k] for k in keys if k in FIGURE_LEGEND]
    seen, types = set(), []
    for _pat, nm, desc in figtypes:                 # 같은 종류를 중복 출력하지 않는다
        if nm not in seen:
            seen.add(nm); types.append(f'<p><b>{nm}</b> — {desc}</p>')
    if not items and not types:
        return ''
    return ('<section class="card summary"><h3>그림 읽는 법</h3>'
            + ''.join(f'<p>{t}</p>' for t in items) + ''.join(types)
            + '<p style="opacity:.75">각 그림 아래에는 파일명과 위 종류 이름만 표시된다.</p></section>')


def parse_figtypes(spec):
    """`'pat|이름|설명;;...'` → [(pat, 이름, 설명)].

    설명은 **범례 카드에 한 번만** 쓰고, 각 그림에는 짧은 이름만 붙인다
    (같은 설명을 이미지마다 반복하면 읽는 사람이 피로하다 — 실제로 g1은 동일 캡션이 20번 반복됐다).
    """
    out = []
    for part in (spec or '').split(';;'):
        bits = [b.strip() for b in part.split('|')]
        if len(bits) >= 3:
            out.append((bits[0], bits[1], '|'.join(bits[2:])))
    return out


def _b64(path: Path) -> str:
    return 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()


def img_figure(path: Path, figtypes=()) -> str:
    """그림 하나. figcaption엔 **파일명 + 짧은 종류 이름**만(설명은 범례 카드에 한 번)."""
    name = next((nm for pat, nm, _d in figtypes if pat in path.name), '')
    body = f'<b>{path.name}</b>' + (f' · {name}' if name else '')
    return (f'<figure><img src="{_b64(path)}" alt="{path.name}">'
            f'<figcaption>{body}</figcaption></figure>')


def inline_body(body_html: str, stage: Path) -> str:
    """`body.html`의 상대경로 `<img src="x.jpg">`를 base64로 치환해 그대로 싣는다.

    **왜 필요한가**: 예전엔 `summary.html` + `*.jpg` 나열만 발행해서, 각 그림에 붙은 **설명·색 범례가
    통째로 빠졌다**(그림만 보고는 어떤 색이 무슨 뜻인지 알 수 없었다). body를 실으면 캡션이 함께 간다.
    Artifact는 CSP로 상대경로를 막으므로 반드시 인라인해야 한다.
    """
    import re

    def sub(m):
        p = stage / m.group(1)
        return f'src="{_b64(p)}"' if p.exists() else m.group(0)
    return re.sub(r'src="([^"]+\.jpg)"', sub, body_html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage_dir', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--eyebrow', default='embodiment augmentation · report')
    ap.add_argument('--sub', default='')
    ap.add_argument('--purpose', default='', help='이 리포트가 무엇을 확인하는지 1~2문장')
    # 수치는 `--result`, 그 수치가 뜻하는 한 문장은 `--conclusion` — 섞으면 결론이 표에 묻힌다.
    ap.add_argument('--result', default='', help='측정된 수치 (통과선 포함)')
    ap.add_argument('--conclusion', default='',
                    help='측정 결과 무엇이 결론인가 1~2문장. **수치를 포함해라** — 결론만 있고 근거가 '
                         '없으면 나중에 검증이 안 된다')
    ap.add_argument('--limits', default='',
                    help='이 리포트가 답하지 **못한** 것 1~2문장. 비워두지 마라 — 한계를 안 쓰면 '
                         '읽는 사람이 이 결과를 실제보다 넓게 해석한다')
    ap.add_argument('--figtypes', default='',
                    help="그림 종류: 'pat|이름|설명;;...' — 설명은 범례에 한 번, 그림엔 이름만")
    ap.add_argument('--legend', default='',
                    help='공용 그림 규약 키 콤마열: depth,bev,floorplan')
    ap.add_argument('--trouble_title', default='troubleshooting — 틀렸던 측정과 버그 이력 (펼치기)',
                    help='stage_dir/trouble.html이 있을 때 접힌 섹션의 제목')
    args = ap.parse_args()

    stage = Path(args.stage_dir)
    summary = (stage / 'summary.html').read_text(encoding='utf-8') if (stage / 'summary.html').exists() else ''
    # `trouble.html`이 있으면 **접힌 섹션**으로 붙인다 — 틀렸던 측정·버그 이력은 본문에서 빼고 여기에.
    tr = (stage / 'trouble.html')
    trouble = (f'<details class="trouble"><summary>{args.trouble_title}</summary>'
               f'<div class="inner">{tr.read_text(encoding="utf-8")}</div></details>') if tr.exists() else ''
    body_p = stage / 'body.html'
    imgs = sorted(p for p in (Path(x) for x in glob.glob(str(stage / '*.jpg'))))
    figt = parse_figtypes(args.figtypes)
    if body_p.exists():
        # body가 있으면 그것을 본문으로 쓴다(그림+설명이 짝지어져 있다).
        section, n = inline_body(body_p.read_text(encoding='utf-8'), stage), 'body'
    else:
        section, n = ''.join(img_figure(p, figt) for p in imgs), 'gallery'

    content = (
        # <title>은 갤러리·브라우저 탭에 표시되는 이름이다. 없으면 파일명(artifact)이 그대로 노출돼
        # 리포트를 구분할 수 없다(실제로 그렇게 됐다).
        f'<title>{args.title}</title>'
        f'<style>{CSS}</style>'
        f'<div class="wrap">'
        f'<p class="eyebrow">{args.eyebrow}</p>'
        f'<h1>{args.title}</h1>'
        + (f'<p class="sub">{args.sub}</p>' if args.sub else '')
        + (f'<section class="card intro">'
           f'<h3>전체 파이프라인의 목적</h3><p>{PIPELINE_INTRO}</p>'
           + (f'<h3>이 리포트의 목적</h3><p>{args.purpose}</p>' if args.purpose else '')
           # 결론·한계는 **인트로에 함께 둔다** — 맨 아래에 두면 그림에 묻혀 아무도 안 읽는다.
           + (f'<h3>실험 결과</h3><p>{args.result}</p>' if args.result else '')
           + (f'<h3 class="concl">결론</h3><p>{args.conclusion}</p>' if args.conclusion else '')
           + (f'<h3 class="limit">한계 — 이 리포트가 답하지 못한 것</h3><p>{args.limits}</p>'
              if args.limits else '')
           + f'</section>')
        + (f'<section class="card summary">{summary}</section>' if summary else '')
        + legend_card([k.strip() for k in args.legend.split(',') if k.strip()], figt)
        + (f'<section class="card summary">{section}</section>' if section else '')
        + trouble
        + '</div>'
    )
    out = stage / 'artifact.html'
    out.write_text(content, encoding='utf-8')
    print(f'[publish] {out}  ({len(imgs)} images, source={n})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
