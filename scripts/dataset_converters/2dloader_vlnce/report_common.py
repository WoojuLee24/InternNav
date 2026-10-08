"""리포트 공통 조각 — 범례·용어표·파라미터표.

**규칙(3dloader R1의 교훈)**: 범례에는 **그림에 실제로 그려진 색만** 넣는다. 계획이 기각돼 선이
없는 `r_b`에 색을 배정하면 그림과 범례가 어긋난다. 그래서 `legend_table`은 호출부가 "그렸다"고
확인한 항목만 받는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent / '3dloader_vlnce'), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from viz2d import (  # noqa: E402
    ADJ_COLOR, CSPACE_COLOR, GOAL_COLOR, GT_COLOR, NEW_COLOR, OBST_COLOR, START_COLOR,
)

BEV_UNKNOWN, BEV_FREE, BEV_OCC = (0, 0, 0), (90, 90, 90), (255, 70, 70)


def swatch(rgb, shape: str = 'box') -> str:
    """색 견본. `shape`: box(면) / ring(빈 원) / dot(채운 원) / line(선) / none(색 없음)."""
    if rgb is None:
        return '<span style="opacity:.6">—</span>'
    c = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
    if shape == 'ring':
        css = f'width:14px;height:14px;border-radius:50%;border:2px solid {c};background:transparent'
    elif shape == 'dot':
        css = f'width:14px;height:14px;border-radius:50%;background:{c}'
    elif shape == 'line':
        css = f'width:22px;height:3px;background:{c}'
    else:
        css = f'width:14px;height:14px;border-radius:3px;border:1px solid #8888;background:{c}'
    return f'<span style="display:inline-block;{css}"></span>'


def legend_table(rows, title: str = '범례 — 그림의 색과 기호') -> str:
    """rows: `[(rgb|None, shape, 이름, 뜻), ...]`. 그림에 **실제로 그린 것만** 넘길 것."""
    body = ''.join(
        f'<tr><td style="text-align:center">{swatch(rgb, shape)}</td>'
        f'<td><b>{name}</b></td><td>{meaning}</td></tr>' for rgb, shape, name, meaning in rows)
    return (f'<h4>{title}</h4>'
            f'<table border=1 cellpadding=6><tr><th style="width:44px">색</th><th>이름</th>'
            f'<th>뜻</th></tr>{body}</table>')


def gate_note(what: str, why: str, criterion: str = '') -> str:
    """**이 리포트가 무엇을 검증하는가**를 맨 앞에 못 박는다 (게이트마다 다름)."""
    c = (f'<p style="margin-top:10px"><b>통과 기준</b> — {criterion}</p>' if criterion else '')
    return (f'<h4>이 리포트가 검증하는 것</h4>'
            f'<p><b>무엇을</b> — {what}</p><p><b>왜</b> — {why}</p>{c}')


def bev_rows(with_cspace: bool = False):
    """모든 BEV 그림에 공통인 3색 + 기본 마커. `with_cspace`면 r_b 팽창분 색도 넣는다."""
    if with_cspace:
        return [
            (BEV_FREE, 'box', 'A* 통행 가능 (회색) — <b>이 격자에서 경로를 골랐다</b>',
             '17.9 cm 조립 격자 한 칸. "여유 ≥ r_b인 세밀 칸이 하나라도 있고(any), 16칸 전부가 '
             '관측된 free(all)"일 때만 통행 가능. 네모가 굵게 보이는 것은 실제 격자가 그만큼 성기기 때문'),
            (CSPACE_COLOR, 'box', 'A*가 못 쓴 칸 (어두운 빨강)',
             '비어 있다고 관측됐지만 <b>이 로봇에겐 여유가 부족하거나</b>(r_b) 그 조립 칸 안에 '
             '미관측이 섞여 제외된 곳. <b>r_b가 커질수록 넓어진다 — 패널마다 지도가 다른 이유</b>'),
            (BEV_OCC, 'box', '점유 (빨강)',
             '높이 밴드 (h_nav, h_b] 안에 depth 반환이 있는 칸 = 장애물 자체. r_b와 무관하다'),
            (BEV_UNKNOWN, 'box', '미관측 (검정)',
             '이 프레임 depth로는 아무 정보가 없는 칸. 시야각 밖이거나 물체 뒤'),
            (START_COLOR, 'dot', '로봇 원점 (흰 점)', '지도 중심. 로봇이 서 있는 자리, 전방은 위쪽'),
        ]
    return [
        (BEV_UNKNOWN, 'box', 'BEV unknown (검정)',
         '이 프레임 depth로는 아무 정보가 없는 칸. 시야각 밖이거나 물체 뒤'),
        (BEV_FREE, 'box', 'BEV free (회색)',
         '통행 가능하다고 관측된 칸. (a) 점유 셀로 가는 광선 위 + (b) 바닥 반환이 찍힌 칸'),
        (BEV_OCC, 'box', 'BEV occupied (빨강)',
         '높이 밴드 (h_nav, h_b] 안에 depth 반환이 있는 칸 = 이 로봇에게 장애물'),
        (START_COLOR, 'dot', '로봇 원점 (흰 점)', 'BEV 중심. 로봇이 서 있는 자리, 전방은 위쪽'),
    ]


def path_rows(with_gt: bool = True, with_new: bool = False, with_obst: bool = False):
    rows = []
    if with_gt:
        rows.append((GT_COLOR, 'dot', '원본 GT path (노랑)',
                     '데이터셋에 기록된 실제 이동 경로를 robot 좌표로 옮긴 것'))
    if with_new:
        rows.append((NEW_COLOR, 'line', '재계획 path (초록)', '이 e에서 새로 만든 GT 경로'))
    if with_obst:
        rows.append((OBST_COLOR, 'line', '합성 장애물 footprint (주황)',
                     '넣은 3D 박스의 밑면 다각형. BEV 빨강이 이 안에 들어와야 정합'))
    return rows


def goal_rows(with_orig: bool = True, with_adj: bool = False):
    rows = []
    if with_orig:
        rows.append((GOAL_COLOR, 'ring', '저장된 원본 pixel goal (자홍 빈 원)',
                     'parquet `goal.<rig>`의 [u,v]. 640×480 룩다운 이미지 좌표'))
    if with_adj:
        rows.append((ADJ_COLOR, 'dot', '재투영/조정된 goal (채운 원)',
                     '우리가 3D goal을 다시 투영했거나 e에 맞춰 옮긴 위치'))
    return rows


def status_glossary() -> str:
    return (
        '<h4>상태값 뜻</h4>'
        '<table border=1 cellpadding=6><tr><th>필드</th><th>값</th><th>뜻</th></tr>'
        '<tr><td rowspan=2><code>status</code><br>(샘플 최종)</td><td><code>ok</code></td>'
        '<td>이 e로 GT를 만들었다 — 학습에 그대로 쓸 수 있다</td></tr>'
        '<tr><td><code>rejected:*</code></td><td>이 embodiment로는 만들 수 없어 버린다. '
        '<code>astar_failed</code>=관측된 통행 가능 영역에 경로가 없음 / '
        '<code>path_leaves_free</code>=경로가 점유·미관측 칸을 지나 폐기 / '
        '<code>goal_not_navigable</code>=목표 칸을 이 r_b로 못 밟음 / '
        '<code>occluded</code>·<code>out_of_image</code>=새 goal 픽셀이 안 보임</td></tr>'
        '<tr><td rowspan=4><code>goal</code><br>(라벨 판정)</td><td><code>unchanged</code></td>'
        '<td>원본 goal에 <b>실제로 도달했다</b>(끝점 오차 ≤ 10 cm) — 라벨을 건드리지 않는다</td></tr>'
        '<tr><td><code>adjusted</code></td><td>도달하지 못해 경로를 따라 물러났고, 원본 goal이 '
        '<b>관측된 칸</b>이었다 — 이 로봇은 거기까지 못 가므로 <b>라벨도 갱신</b>한다</td></tr>'
        '<tr><td><code>unobserved</code></td><td>물러났지만 원본 goal이 미관측이었다. 못 본 곳을 '
        '위험하다 할 근거가 없으므로 <b>라벨은 그대로</b> 두고 계획 목표만 물린다</td></tr>'
        '<tr><td><code>no_safe_point</code></td><td>최소 전진거리(<code>min_goal_m</code>) 이상 갈 수 '
        '있는 지점이 하나도 없다 → 기각</td></tr>'
        '</table>'
        '<p><b>후퇴는 clearance가 아니라 도달 가능성으로 한다</b> — clearance만 보면 큰 로봇은 goal이 '
        '가까이 당겨져 쉽게 성공하고 작은 로봇은 먼 goal 그대로 실패하는 <b>뒤집힌 결과</b>가 나온다'
        '(실측 17DRP f29). 도달 가능성 기준이면 작은 로봇이 더 멀리, 큰 로봇이 더 일찍 멈춘다.</p>')


def param_glossary(**vals) -> str:
    """이번 실행의 파라미터 표. 값은 키워드로 넘긴다(없으면 그 행을 빼지 않고 '—')."""
    rows = [
        ('r_b', '로봇 반경 [m]. `truncate_navigable(esdf, r_b)`의 임계 — 여유가 이보다 작은 칸은 통행 불가'),
        ('h_nav', '밟고 넘을 수 있는 높이 [m]. BEV 높이 밴드의 아래끝(`z_min`)'),
        ('h_b', '로봇 높이 [m]. BEV 높이 밴드의 위끝(`z_max`)'),
        ('unknown', '미관측 칸 정책. `nontraversable`=여유 계산엔 안 넣되 경로는 못 지나감(기본) / '
                    '`free`=완전 무시 / `block`=장애물 취급'),
        ('goal_adjust', 'goal이 이 r_b로 못 갈 때 옮기는 방식. `retreat`=원본 경로를 따라 후퇴(기본) / '
                        '`shift`=거리 유지·방위 회전 / `nearest`=최소 변위'),
        ('preset', '`<높이>cm_<pitch1>_<pitch2>`. **depth·pose·pixel goal은 전부 pitch_2(룩다운) rig 기준**, '
                   'pitch_1은 System 2가 보는 FPV RGB'),
        ('obstacle', '합성 장애물 사용 여부'),
        ('seed', '장애물 샘플링 난수 시드'),
    ]
    body = ''.join(f'<tr><td><code>{k}</code></td><td>{vals.get(k, "—")}</td><td>{d}</td></tr>'
                   for k, d in rows if k in vals)
    return (f'<h4>이번 실행 파라미터</h4><table border=1 cellpadding=6>'
            f'<tr><th>이름</th><th>값</th><th>뜻</th></tr>{body}</table>')


def pipeline_note() -> str:
    return (
        '<h4>이 파이프라인이 하는 일</h4>'
        '<p><b>씬 mesh·3D occupancy·렌더러 없이</b>, 그 프레임의 RGB/depth 한 장만으로 '
        '(a) 장애물을 합성하고 (b) embodiment <code>e=(r_b, h_nav, h_b)</code>를 반영해 '
        '(c) GT path와 pixel goal을 다시 만든다.</p>'
        '<pre style="white-space:pre-wrap">depth ─(장애물 합성)→ depth\' ─depth_to_bev_occ_ros2(z_min=h_nav, z_max=h_b)→ BEV\n'
        '      → ESDF → truncate_navigable(r_b) → A* → refine → thin → spline → 새 GT path\n'
        '      → goal 판정 → 새 pixel goal</pre>'
        '<p>BEV는 <b>학습·평가가 실제로 쓰는 그 함수</b>(<code>depth_to_bev_occ_ros2</code>)로 만든다. '
        '관측을 만드는 격자와 GT를 만드는 격자가 같아서 train==eval 불변식이 유지되고, 정보 누수도 없다 '
        '(둘 다 같은 depth 한 장에서 나온다).</p>')
