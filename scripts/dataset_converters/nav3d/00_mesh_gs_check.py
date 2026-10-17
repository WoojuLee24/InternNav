r"""00 — 씬 mesh(충돌·depth 용)와 3DGS(RGB 용)가 **얼마나 잘 겹치는지**, **바닥에 구멍이 얼마나 있는지** 검사한다.

    usdz ─┬─ mesh.ply            (충돌 mesh — navmesh·depth 가 쓴다)
          └─ <scene>.nurec       (3DGS — Isaac 이 RGB 를 그린다. gzip + msgpack, 가우시안 1,960 만 개)
       -> A 정합 (A1 전체 강체 어긋남 · A1b 표면 간격 · A2 국소 어긋남 · A3 GS 없는 mesh · A4 mesh 없는 GS)
       -> B 바닥·계단 (B1 mesh 구멍 · B2 GS 구멍 · B3 바닥 높이 차)
       -> 그림 · ply · csv · summary.json · report.html

## 용어
- **불투명 가우시안**: opacity = sigmoid(density) ≥ `--op_min`(0.5). 반투명 안개·떠다니는 점은 정합 판단에서 뺀다.
- **표면 가우시안**: 불투명 + 가장 긴 축 ≤ `--surf_smax`(0.05 m). 위치가 표면에 붙어 있다고 볼 수 있는 것.
- **d**: 가우시안 중심 → mesh 최근접점 거리. **s(부호 거리)**: d 를 그 면의 법선 방향으로 잰 것(+ = 면 앞, 방 쪽).
- **δ(표면 간격)**: 모든 면에서 똑같이 GS 가 mesh 뒤(−)/앞(+)에 있는 양. 강체 이동과 따로 푼다 —
  안 그러면 바닥 면적이 천장보다 커서 "바닥·천장 둘 다 뒤" 가 "z 로 밀림" 으로 잘못 읽힌다(노은역 실측).
- **바닥 띠**: 층(01 verify.json 의 storeys) 높이 범위 ± `--band`(0.3 m).
- **셀 분류** (바닥 띠 위 1 m 에서 아래로 쏜 광선이 처음 맞은 mesh 면으로):
  바닥(띠 안 · 수평) / 덮임(띠 위에서 맞음 · 띠 안 수직면 = 기둥·벽·물건) / 트임(띠 아래 **위향** 면 = 계단·선로·아래층) /
  뚫림(안 맞음 또는 띠 아래 **아래향** 면 = 아래층 천장 뒷면이 보임).
- **mesh 구멍**: 바깥(격자 테두리)과 이어지지 않은 뚫림 덩어리 중, 둘레의 ≥ 50 % 가 바닥인 것.
  둘레가 주로 벽이면 "닫힌 미스캔 영역"으로 따로 센다(창고·막힌 방).
- **GS 바닥 α**: 바닥 높이 −0.25 ~ +0.15 m 의 가우시안을 위에서 내려다보며 쌓은 불투명도(0~1). < 0.5 면 **GS 구멍**(RGB 에서 바닥이 비침).
- **GS 바닥 높이**: 위에서 쌓은 α 가 0.5 에 처음 닿는 높이. B3 = GS 바닥 높이 − mesh 바닥 높이.
- **내접 지름**: 구멍 안에 들어가는 가장 큰 원 지름. ≥ 0.10 m 면 발·바퀴가 빠질 크기.
- **통행 구역**: 01 navmesh 를 r_b + 0.05 m 넓히고 안쪽 빈 곳을 채운 영역 (구멍 때문에 navmesh 가 깎인 자리도 포함).

## 판정 기준
| | 묻는 것 | 기준 | 왜 |
|---|---|---|---|
| G0 | GS 변환이 항등 · 쿼터니언 순서 | xform 항등 · offset 0 · 납작한 표면 가우시안의 짧은 축 ∥ 면 법선 (\|cos\| 중앙 ≥ 0.80, 무작위 0.5) | 아니면 이하 숫자가 다 틀림 |
| A1 | GS 를 mesh 에 맞추는 강체 변환 | 이동 ≤ 0.02 m · 회전 ≤ 0.05° | 0.05° 는 60 m 끝에서 5 cm. 2 cm = depth·RGB 경계 1~2 px(3 m) |
| A1b | 표면 간격 δ | \|δ\| ≤ 0.03 m | 로봇 발 높이·벽 여유 오차로 무시할 수 있는 크기 |
| A2 | 1 m 복셀별 표면 가우시안 중앙 d | 중앙 d > 0.10 m 인 복셀 ≤ 2 % | 전체 중앙 d ≈ 0.04 m 의 두 배 이상 = 국소 뒤틀림 |
| A3 | mesh 표면 점 0.15 m 안 가우시안 16 개의 누적 α = 1 − Π(1 − oᵢ) ≥ 0.5 인가 (층 바닥 −0.3 ~ +2.5 m) | 못 미치는 면적 ≤ 5 % | 0.15 = 표면 간격 + 퍼짐. 누적으로 보는 이유: 반투명 여러 개가 겹쳐 불투명하게 그려진다 |
| A4 | mesh 에서 0.30 m 넘게 떨어진 불투명 가우시안 덩어리 | **보고** + 통행 구역 위(바닥 +0.1~+1.8 m) 덩어리 목록 | RGB 엔 보이는데 충돌 mesh 엔 없음 = 유리·표지판·뜬 점. 자동 판정 불가 |
| B1 | mesh 구멍 (바닥 + 계단) | 통행 구역·계단 안 · 내접 ≥ 0.10 m 인 것 0 개 | 로봇이 빠지거나 depth 가 뚫림 |
| B2 | GS 구멍 (mesh 바닥·디딤판 위) | 층마다 · 계단 전체 각각 ≤ 1 % | RGB 에서 바닥이 비쳐 보임 |
| G0c | mesh ↔ GS 무게중심 거리 | 보고 | GS 가 mesh 밖까지 퍼져 판정에 못 씀 |
| B3 | GS 바닥 높이 − mesh 바닥 높이 | \|중앙\| ≤ 0.03 m · p95 \|차\| ≤ 0.10 m | RGB 의 바닥과 로봇 발(충돌)이 같은 높이여야 |

사용 (internnav-habitat-navmesh 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/00_mesh_gs_check.py --scene noeun_station
"""

import argparse
import csv
import gzip
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nav3d_utils as nu  # noqa: E402

SCENES = {
    # usdz 안 mesh.ply == data/scene_data/noeun_station/noeun_station_mid.ply (01 이 쓰는 것과 같다).
    # gauss.usda: xformOp 항등, omni:nurec:offset 0 -> GS 좌표 = mesh 좌표 (G0 가 매번 확인).
    'noeun_station': {'usdz': 'data/GS_USDZ/Subway/noeun_station_collision.usdz',
                      'nav01': 'scripts/dataset_converters/nav3d/logs/01_build_navmesh/noeun_station'},
}
SH_C0 = 0.28209479177387814
KERNEL_MIN_RESPONSE = 0.0113        # nurec config density_kernel_min_response — 이보다 작은 기여는 렌더러도 버린다

# 지표·기준표 — report.html 의 "지표와 기준" 절이 이것을 그대로 그린다 (docstring 표와 같은 내용).
# 신뢰도: 명확 = 계산 정의가 하나로 정해짐 / 근사 = 정의는 명확하나 렌더러와 다른 근사 / 휴리스틱 = 임계값·반경이 결과를 크게 좌우
METRICS = [
    ('G0', '좌표 규약', 'gauss.usda xformOp 가 항등, omni:nurec:offset = 0. 쿼터니언 순서: 납작한 표면 가우시안의 짧은 축과 붙은 면 법선의 |cos| 중앙값을 wxyz/xyzw 로 비교',
     '항등 · 고른 순서 |cos| ≥ 0.80 (무작위 방향 0.5)', '틀리면 아래 숫자가 전부 틀림', '명확'),
    ('G0c', '무게중심 거리 (참고)', 'mesh 면적가중 무게중심 ↔ GS 무게중심 (전체 · opacity 가중 · mesh 상자 안 opacity 가중)',
     '보고만', 'GS 는 mesh 보다 넓게(바깥 건물·하늘) 퍼져 있어 정합 판정에 못 씀 — 상자 안 값만 참고', '명확(해석 주의)'),
    ('A1', '강체 어긋남', '표면 가우시안(opacity ≥ 0.5, 최대축 ≤ 0.05 m, |s| < 0.15 m) 에 대해 min Σ Huber(n·(ω×p + t) + δ + s) — t 이동, ω 회전',
     '|t| ≤ 0.02 m · |ω| ≤ 0.05°', '2 cm ≈ 3 m 거리 depth·RGB 경계 1~2 px, 0.05° ≈ 60 m 끝에서 5 cm', '명확'),
    ('A1b', '표면 간격 δ', 'A1 과 같이 푼 δ: 모든 면에서 GS 가 mesh 뒤(+)에 있는 공통 거리. 방향별(바닥·천장·±x·±y) 부호 거리 중앙값도 보고',
     '|δ| ≤ 0.03 m', '로봇 발 높이·벽 여유 오차로 무시할 크기', '명확'),
    ('A2', '국소 어긋남', '1 m 복셀마다 표면 가우시안(d < 0.30) 중심→mesh 거리 d 의 중앙값 (가우시안 ≥ 30 개인 복셀만)',
     '중앙 d > 0.10 m 인 복셀 ≤ 2 %', '0.10 = 전체 중앙 0.045 의 두 배. 반사 바닥 가우시안이 섞임', '휴리스틱'),
    ('A3', 'GS 가 안 덮은 mesh', 'mesh 표면 점(1/0.01 m²)마다 0.15 m 안 가까운 가우시안 16 개의 누적 α = 1 − Π(1 − oᵢ). 층 바닥 −0.3 ~ +2.5 m 만 판정',
     'α < 0.5 인 면적 ≤ 5 %', '시점 무시·반경/개수 임의 — 하한을 바꾸면 21/14/10 % 로 흔들림. 정확히는 카메라 렌더 비교 필요', '휴리스틱'),
    ('A4', 'mesh 없는 GS', 'mesh 에서 d ≥ 0.30 m 인 불투명 가우시안 → 0.5 m 복셀 덩어리. 통행 구역 위 바닥 +0.1 ~ +1.8 m 에 30 개 이상이면 "통행 구역 위"',
     '보고 (목록)', '유리·표지판·뜬 점·반사 가우시안을 자동으로 못 가름', '명확(해석 수동)'),
    ('B1', '바닥·계단 mesh 구멍', '바닥: 5 cm 격자, 층 띠 위 1 m 에서 아래로 쏜 광선이 띠 안 바닥을 못 맞춘 칸 중 격자 바깥과 안 이어지고 둘레 ≥ 50 % 가 바닥인 덩어리. '
     '계단: 01 navmesh 의 층 사이 삼각형(높이 폭 ≥ 1 m 조각) 칸마다 navmesh 높이 +1 m 에서 아래로, navmesh −0.35 ~ +0.25 m 에 위향 면(디딤판)이 없는 칸. 내접 지름 = 들어가는 가장 큰 원',
     '통행 구역 안 · 내접 ≥ 0.10 m 인 것 0 개', '0.10 m = 발·바퀴가 빠지는 크기(2 셀). 계단 ±: 챌판 0.17 m 톱니가 navmesh 경사면에서 벗어나는 폭. 바닥 가장자리에서 바깥과 이어진 구멍은 못 잡음', '명확'),
    ('B2', '바닥·계단 GS 구멍', 'mesh 바닥(계단은 디딤판) 칸에서 그 높이 −0.25 ~ +0.15 m 가우시안을 위에서 쌓은 α (2D 응답 = o·exp(−½dᵀΣxy⁻¹d)). 층마다, 계단은 전체를 한 구역으로',
     '구역마다 α < 0.5 면적 ≤ 1 %', '수직 투영 근사 — 비스듬히 보는 카메라와 다를 수 있음. 계단을 한 구역으로 모으면 작은 조각의 높은 비율이 희석됨', '근사'),
    ('B3', '바닥 높이 차', 'GS 바닥 높이(위에서 쌓은 α 가 0.5 에 닿는 z) − mesh 바닥 z, 통행 구역 바닥 칸',
     '|중앙| ≤ 0.03 m · p95 |차| ≤ 0.10 m', 'RGB 바닥과 로봇 발(충돌) 높이 일치', '근사'),
]

# 판정 기준 프로파일 (--criteria). strict = 처음 정한 기준(근거는 METRICS). relaxed = 노은역 실측(2026-10-14)을 보고
# 통과하도록 정한 완화 기준 — 측정 뒤에 정한 값이라 "현 상태 기록"이지 품질 보증이 아니다. B1 은 숫자뿐 아니라 정의가 바뀐다.
# B1·B2 는 바닥(층 띠)과 계단(01 navmesh 의 층 사이 삼각형, 높이 폭 ≥ 1 m 조각)을 합쳐 판정한다 (2026-10-15 사용자 결정).
CRITERIA = {
    'strict': {'A1_t': 0.02, 'A1_rot': 0.05, 'A1b': 0.03, 'A2': 0.02, 'A3': 0.05,
               'B1_mode': 'count', 'B1': 0, 'B2': 0.01, 'B3_med': 0.03, 'B3_p95': 0.10},
    'relaxed': {'A1_t': 0.02, 'A1_rot': 0.05, 'A1b': 0.05, 'A2': 0.10, 'A3': 0.15,
                'B1_mode': 'area_frac', 'B1': 0.01, 'B2': 0.03, 'B3_med': 0.06, 'B3_p95': 0.15},
}

# 완화 기준의 근거 (물리적 기준점). 숫자는 노은역 실측(2026-10-14)을 덮도록 고른 것이므로, 근거는 "이 정도면 쓸 수 있다"는
# 설명이지 측정 전 설계가 아니다. 로봇 = RBQ10 (턱 0.20 m, r_b 0.25 m), 카메라 = D455 (높이 약 0.6 m~1.25 m).
RELAXED_WHY = {
    'G0': '엄격과 같음 — 틀리면 이하 숫자가 다 틀린다.',
    'A1': '엄격과 같음 — 2 cm ≈ 3 m 거리 depth·RGB 경계 1~2 px. 노은역 0.95 cm · 0.021° 로 여유 큼.',
    'A1b': '≤ 5 cm: D455 depth 오차(2.5 m 에서 약 2 % ≈ 5 cm)와 같은 크기라 센서로 구분이 안 되는 수준. 로봇 턱 0.20 m 의 1/4.',
    'A2': '≤ 10 %: 1 m 공간의 90 % 이상이 10 cm 안에서 맞으면 통과. 반사 바닥 가우시안이 섞이는 휴리스틱이라 넉넉히 잡음.',
    'A3': '≤ 15 %: 휴리스틱 지표 — opacity 하한에 따라 10~21 % 로 흔들리는 폭의 위쪽. 정밀 판정은 카메라 렌더 비교로 해야 함.',
    'B1': '개수 → 면적: navmesh 가 구멍 둘레를 r_b 만큼 깎아 계획 경로는 구멍을 피하므로, 영향은 개수보다 잃는 면적. 1 % = 통행 면적 100 m² 당 구멍 1 m².',
    'B2': '≤ 3 %: 화면만의 문제(충돌·depth 는 정상). 통행 바닥 100 m² 당 GS 얼룩 3 m² 까지 허용.',
    'B3': '|중앙| ≤ 6 cm · p95 ≤ 15 cm: p95 가 로봇 턱 0.20 m 보다 작음. 1.25 m 높이 카메라에서 바닥 6 cm 차 ≈ 3 m 앞 바닥 위치 약 15 cm 차.',
}


def criterion_text(k, c):
    """판정 항목 k 의 기준 c 를 사람이 읽는 글로."""
    if k == 'G0':
        return '항등 · 고른 순서 |cos| ≥ 0.80'
    elif k == 'A1':
        return f"|t| ≤ {c['A1_t']} m · 회전 ≤ {c['A1_rot']}°"
    elif k == 'A1b':
        return f"|δ| ≤ {c['A1b']} m"
    elif k == 'A2':
        return f"어긋난 복셀 ≤ {100 * c['A2']:g} %"
    elif k == 'A3':
        return f"α<0.5 면적 ≤ {100 * c['A3']:g} %"
    elif k == 'B1':
        if c['B1_mode'] == 'count':
            return f"바닥 통행 구역 + 계단의 내접 ≥ 0.10 m 구멍 ≤ {c['B1']} 개"
        elif c['B1_mode'] == 'area_frac':
            return f"mesh 구멍 면적 ≤ (바닥 통행 구역 + 계단) 면적의 {100 * c['B1']:g} %"
        else:
            assert False, f"unreachable B1_mode={c['B1_mode']!r}"
    elif k == 'B2':
        return f"층마다(바닥 통행 구역) · 계단 전체(디딤판) 각각 ≤ {100 * c['B2']:g} %"
    elif k == 'B3':
        return f"|중앙| ≤ {c['B3_med']} m · p95 ≤ {c['B3_p95']} m"
    else:
        assert False, f"unreachable k={k!r}"


# 바닥 셀 분류 코드
OUT, FLOOR, COVER, OPEN, SEE, UNDER = 0, 1, 2, 3, 4, 5


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True, help=f'{list(SCENES)} 또는 --usdz 와 함께 임의 이름')
    ap.add_argument('--usdz', default=None, help='mesh.ply + *.nurec 이 든 usdz. 생략 시 SCENES 표')
    ap.add_argument('--nav01', default=None, help='01 출력 폴더(verify.json·navmesh) — 층과 통행 구역. "none" 이면 mesh 로 층 찾기')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/00_mesh_gs')
    ap.add_argument('--op_min', type=float, default=0.5, help='불투명 가우시안 opacity 하한')
    ap.add_argument('--surf_smax', type=float, default=0.05, help='표면 가우시안 최대 축 [m]')
    ap.add_argument('--near_m', type=float, default=0.30, help='이보다 멀면 "mesh 없는 GS"(A4), 가까우면 A1·A2 대상')
    ap.add_argument('--fit_m', type=float, default=0.15, help='A1 강체 맞춤에 쓰는 점의 |s| 상한')
    ap.add_argument('--cover_m', type=float, default=0.15, help='A3: mesh 점에서 이 안에 불투명 가우시안이 없으면 "GS 없음"')
    ap.add_argument('--voxel_m', type=float, default=1.0, help='A2 국소 어긋남 복셀')
    ap.add_argument('--local_bad_m', type=float, default=0.10, help='A2: 복셀 중앙 d 가 이보다 크면 어긋남')
    ap.add_argument('--cell_m', type=float, default=0.05, help='B 바닥 격자 (navmesh cell_size 와 같게)')
    ap.add_argument('--band', type=float, default=0.30, help='바닥 띠 = 층 높이 범위 ± band')
    ap.add_argument('--floor_cos', type=float, default=0.85, help='바닥 면 판정 |n_z| 하한 (약 32°)')
    ap.add_argument('--hole_min_diam', type=float, default=0.10, help='B1·B2: 이 이상 내접 지름이면 "발이 빠질 구멍"')
    ap.add_argument('--alpha_min', type=float, default=0.5, help='B2: GS 바닥 α 하한')
    ap.add_argument('--agent_radius', type=float, default=0.25, help='통행 구역 = navmesh + r_b + 0.05 (01 과 같은 r_b)')
    ap.add_argument('--agent_height', type=float, default=1.8, help='A4: 통행 구역 위 이 높이까지의 가우시안 덩어리를 "가로막는 것"으로 본다')
    ap.add_argument('--criteria', default='strict', choices=list(CRITERIA), help='판정 기준 프로파일. 리포트에는 두 프로파일 결과가 나란히 나온다')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 읽기
def load_usdz(path: str):
    """usdz -> (mesh trimesh, GS dict, G0 정보). GS: P(N,3) S(N,3)=exp(scale) Q(N,4) O(N,) C(N,3) 0~1."""
    import msgpack
    import trimesh
    z = zipfile.ZipFile(path)
    names = z.namelist()
    nurec = [n for n in names if n.endswith('.nurec')]
    assert len(nurec) == 1 and 'mesh.ply' in names, names
    gu = z.read('gauss.usda').decode() if 'gauss.usda' in names else ''
    m = re.search(r'xformOp:transform = \((.*?)\)\s*\n', gu)
    xf = np.array([float(v) for v in re.findall(r'-?[\d.]+(?:e-?\d+)?', m.group(1))]).reshape(4, 4) if m else np.eye(4)
    mo = re.search(r'omni:nurec:offset = \(([^)]*)\)', gu)
    off = np.array([float(v) for v in mo.group(1).split(',')]) if mo else np.zeros(3)
    g0 = {'nurec': nurec[0], 'xform_identity': bool(np.allclose(xf, np.eye(4))), 'offset': off.tolist()}

    mesh = trimesh.load(io.BytesIO(z.read('mesh.ply')), file_type='ply', process=False)
    with z.open(nurec[0]) as f:
        d = msgpack.unpackb(gzip.GzipFile(fileobj=f).read(), raw=False)
    sd, p = d['nre_data']['state_dict'], '.gaussians_nodes.gaussians.'
    cfg = d['nre_data']['config']['layers']['gaussians']
    assert cfg['density_activation'] == 'sigmoid' and cfg['scale_activation'] == 'exp', cfg
    dt = {16: np.float16, 32: np.float32}[cfg['precision']]

    def arr(k):
        return np.frombuffer(sd[p + k], dt).reshape(sd[p + k + '.shape']).astype(np.float32)
    P = arr('positions')
    gs = {'P': P, 'S': np.exp(arr('scales')), 'Q': arr('rotations'),
          'O': 1.0 / (1.0 + np.exp(-arr('densities')[:, 0])),
          'C': np.clip(0.5 + SH_C0 * arr('features_albedo'), 0, 1)}
    g0['n_gaussians'] = int(len(P))
    g0['precision_bits'] = int(cfg['precision'])
    g0['nre_version'] = d['nre_data'].get('version')
    return mesh, gs, g0


def quat_to_R(q: np.ndarray, order: str) -> np.ndarray:
    """(N,4) -> (N,3,3). order 'wxyz' 또는 'xyzw'."""
    if order == 'wxyz':
        w, x, y, z = q.T
    elif order == 'xyzw':
        x, y, z, w = q.T
    else:
        assert False, f'unreachable order={order!r}'
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def raycaster(mesh):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    return sc


def closest(sc, fn, P, chunk=4_000_000):
    """P -> (d, s, face normal). s = 법선 방향 부호 거리 (+ = 면 앞)."""
    import open3d as o3d
    d, s, n = (np.empty(len(P), np.float32), np.empty(len(P), np.float32), np.empty((len(P), 3), np.float32))
    for i in range(0, len(P), chunk):
        q = sc.compute_closest_points(o3d.core.Tensor(P[i:i + chunk].astype(np.float32)))
        dv = P[i:i + chunk] - q['points'].numpy()
        nn = fn[q['primitive_ids'].numpy()]
        d[i:i + chunk], s[i:i + chunk], n[i:i + chunk] = np.linalg.norm(dv, axis=1), (dv * nn).sum(1), nn
    return d, s, n


def storeys_from(nav01, mesh, args):
    if nav01 and (Path(nav01) / 'verify.json').exists():
        st = json.load(open(Path(nav01) / 'verify.json'))['storeys']
        return [{'z_range': s['z_range'], 'z': s['z']} for s in st], 'nav01'
    fn, fa = mesh.face_normals, mesh.area_faces
    up = fn[:, 2] >= np.cos(np.radians(nu.UP_COS_DEG))
    lv = nu.find_levels(mesh.triangles_center[up, 2], fa[up], min_area_m2=30.0)
    _, st = nu.group_storeys(lv, 1.5)
    return [{'z_range': s['z_range'], 'z': s['z']} for s in st], 'mesh'


# ---------------------------------------------------------------------------
# G0
def g0_quat_order(gs, d, n, args):
    """납작한 표면 가우시안의 가장 짧은 축이 붙은 면의 법선과 나란해야 한다 — 쿼터니언 순서 확인.

    바닥만 보면 못 가른다: 항등에 가까운 wxyz 를 xyzw 로 읽으면 x 축 180° 회전이 되어 수직축이 그대로 수직이다.
    그래서 벽·천장 포함 모든 면에서 |cos(짧은 축, 법선)| 중앙값을 두 순서로 비교한다 (무작위 방향 = 0.5).
    """
    Ss = np.sort(gs['S'], 1)
    sel = np.where((gs['O'] > args.op_min) & (d < 0.08) & (Ss[:, 0] < 0.1 * Ss[:, 1]))[0]
    sel = np.random.default_rng(args.seed).choice(sel, min(len(sel), 300_000), replace=False)
    out = {}
    for order in ('wxyz', 'xyzw'):
        R = quat_to_R(gs['Q'][sel], order)
        ax = R[np.arange(len(sel)), :, gs['S'][sel].argmin(1)]      # 가장 짧은 축 = R 의 그 열
        out[order] = float(np.median(np.abs((ax * n[sel]).sum(1))))
    out['n'] = int(len(sel))
    out['order'] = max(('wxyz', 'xyzw'), key=lambda k: out[k])
    return out


# ---------------------------------------------------------------------------
# A 정합
def a1_rigid(P, s, n, args):
    """min Σ ρ(n·(ω×(p−c)) + n·t + δ + s). 결과 (ω, t) 는 GS 에 적용해 mesh 에 맞추는 변환, δ 는 남는 표면 간격.

    부호: s = 가우시안이 면 앞(+)·뒤(−). GS 를 n·t 만큼 옮기면 s 가 s + n·t. 잔차 → 0 이 되게 푼다.
    δ 는 "GS 를 법선 방향으로 δ 만큼 앞으로 보내면 맞는다" — δ > 0 이면 GS 가 mesh 뒤에 있다.
    """
    k = np.abs(s) < args.fit_m
    P, s, n = P[k].astype(np.float64), s[k].astype(np.float64), n[k].astype(np.float64)
    c = P.mean(0)
    A = np.c_[np.cross(P - c, n), n, np.ones(len(P))]
    b = -s
    w = np.ones(len(P))
    for _ in range(5):                                       # Huber IRLS
        sw = np.sqrt(w)
        x = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
        r = A @ x - b
        w = np.minimum(1.0, 0.02 / np.maximum(np.abs(r), 1e-9))
    om, t, dl = x[:3], x[3:6], x[6]
    # 같은 점으로 δ 없이 풀면 (= 흔한 point-to-plane ICP) 이동이 어떻게 오염되나
    x2 = np.linalg.lstsq(A[:, :6] * np.sqrt(w)[:, None], b * np.sqrt(w), rcond=None)[0]
    res = {'n': int(len(P)), 'center': c.tolist(), 'rot_deg': np.degrees(om).tolist(),
           'rot_norm_deg': float(np.degrees(np.linalg.norm(om))), 'trans_m': t.tolist(),
           'trans_norm_m': float(np.linalg.norm(t)), 'surface_offset_m': float(dl),
           'resid_mad_m': float(np.median(np.abs(r))),
           'without_offset_term': {'trans_m': x2[3:6].tolist(), 'rot_norm_deg': float(np.degrees(np.linalg.norm(x2[:3])))}}
    res['passed_rigid'] = bool(res['trans_norm_m'] <= 0.02 and res['rot_norm_deg'] <= 0.05)
    res['passed_offset'] = bool(abs(dl) <= 0.03)
    return res


def by_orientation(s, n):
    out = {}
    for name, c in [('floor(+z)', n[:, 2] > .9), ('ceiling(-z)', n[:, 2] < -.9), ('+x', n[:, 0] > .9), ('-x', n[:, 0] < -.9),
                    ('+y', n[:, 1] > .9), ('-y', n[:, 1] < -.9)]:
        if c.sum() > 100:
            out[name] = {'n': int(c.sum()), 's_p25': float(np.percentile(s[c], 25)), 's_med': float(np.median(s[c])),
                         's_p75': float(np.percentile(s[c], 75))}
    return out


def a2_local(P, d, s, args):
    v = np.floor(P / args.voxel_m).astype(np.int64)
    key, inv, cnt = np.unique(v, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind='stable')
    splits = np.cumsum(cnt)[:-1]
    dm = np.array([np.median(x) for x in np.split(d[order], splits)])
    sm = np.array([np.median(x) for x in np.split(s[order], splits)])
    ok = cnt >= 30
    key, cnt, dm, sm = key[ok], cnt[ok], dm[ok], sm[ok]
    bad = dm > args.local_bad_m
    # 26-이웃 덩어리
    comps = voxel_components(key[bad])
    clusters = []
    for ci in np.unique(comps):
        kk = key[bad][comps == ci]
        w = cnt[bad][comps == ci]
        ctr = ((kk + 0.5) * args.voxel_m * w[:, None]).sum(0) / w.sum()
        clusters.append({'n_voxels': int(len(kk)), 'center': ctr.round(2).tolist(),
                         'min': (kk.min(0) * args.voxel_m).tolist(), 'max': ((kk.max(0) + 1) * args.voxel_m).tolist(),
                         'median_d_m': float(np.median(dm[bad][comps == ci])),
                         'median_s_m': float(np.median(sm[bad][comps == ci]))})
    clusters.sort(key=lambda c: -c['n_voxels'])
    frac = float(bad.mean()) if len(bad) else 0.0
    return {'n_voxels': int(len(key)), 'bad_voxels': int(bad.sum()), 'bad_frac': frac, 'passed': bool(frac <= 0.02),
            'voxel_d_median_m': float(np.median(dm)), 'clusters': clusters[:30]}, (key, dm, sm, bad)


def voxel_components(keys):
    from scipy import ndimage
    if len(keys) == 0:
        return np.zeros(0, int)
    lo = keys.min(0)
    g = np.zeros(keys.max(0) - lo + 1, bool)
    g[tuple((keys - lo).T)] = True
    lab, _ = ndimage.label(g, structure=np.ones((3, 3, 3)))
    return lab[tuple((keys - lo).T)]


def a3_coverage(mesh, tree, O_all, storeys, args):
    """mesh 표면 점(1 점 / 0.01 m²)마다 cover_m 안 가까운 가우시안 16 개의 누적 불투명도 α = 1 − Π(1 − oᵢ).

    가우시안 하나의 opacity 로 보면 안 된다 — 반투명 여러 개가 겹쳐 불투명하게 그려진다(실측: 하한 0.5 / 0.2 / 0.05 에
    "GS 없음" 21 / 14 / 10 %). α < alpha_min 이면 그 mesh 면은 RGB 에 (거의) 안 그려진다.
    판정은 로봇 시야 높이(각 층 바닥 −0.3 ~ +2.5 m)만, 전체는 보고.
    """
    import trimesh
    n = int(mesh.area / 0.01)
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=args.seed)

    def alpha_at(q):
        dd, ii = tree.query(q, k=16, distance_upper_bound=args.cover_m, workers=-1)
        ok = np.isfinite(dd)
        o = np.where(ok, O_all[np.minimum(ii, len(O_all) - 1)], 0.0)
        return 1 - np.prod(1 - np.minimum(o, 0.999), axis=1), np.where(ok[:, 0], dd[:, 0], 1.0)
    al, dd = alpha_at(pts)
    va, _ = alpha_at(np.asarray(mesh.vertices))
    unc = al < args.alpha_min
    zone = np.zeros(len(pts), bool)
    for S in storeys:
        zone |= (pts[:, 2] >= S['z_range'][0] - 0.3) & (pts[:, 2] <= S['z_range'][1] + 2.5)
    nz = mesh.face_normals[fid, 2]
    fz = float(unc[zone].mean()) if zone.any() else 0.0
    return {'n_points': n, 'mesh_area_m2': float(mesh.area), 'uncovered_frac_all': float(unc.mean()),
            'uncovered_area_m2_all': float(unc.mean() * mesh.area), 'robot_zone_area_m2': float(zone.mean() * mesh.area),
            'uncovered_frac_robot_zone': fz, 'uncovered_area_m2_robot_zone': float(unc[zone].sum() / n * mesh.area),
            'by_orientation_robot_zone': {k: float(unc[zone & c].mean()) for k, c in
                                          [('floor', nz > .9), ('ceiling', nz < -.9), ('wall', np.abs(nz) < .3)] if (zone & c).any()},
            'alpha_p10': float(np.percentile(al, 10)), 'nearest_p50': float(np.median(dd)),
            'passed': bool(fz <= 0.05)}, (pts, al, va)


def a4_off_mesh(P, O, C, d, storeys, walk, args):
    """mesh 에서 near_m 넘게 떨어진 불투명 가우시안 -> 0.5 m 복셀 덩어리. 통행 구역 위에 뜬 것 = 가로막는 후보."""
    far = d >= args.near_m
    vs = 0.5
    v = np.floor(P[far] / vs).astype(np.int64)
    key, inv, cnt = np.unique(v, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    keep = cnt >= 20
    comps = np.full(len(key), -1)
    comps[keep] = voxel_components(key[keep])
    gcomp = comps[inv]
    clusters = []
    Pf, Cf = P[far], C[far]
    for ci in np.unique(comps[keep]):
        m = gcomp == ci
        p = Pf[m]
        lo, hi = p.min(0), p.max(0)
        ctr = p.mean(0)
        on_walk, s_id = False, -1
        for si, S in enumerate(storeys):
            z0 = S['z_range'][0] - args.band
            z1 = S['z_range'][1] + args.band
            if not (z0 - 0.5 <= ctr[2] <= z1 + args.agent_height):
                continue
            # 가로막는 높이(바닥 +0.1 ~ +키)에 있고 통행 구역 위인 가우시안이 30 개 이상
            g = walk.get(si)
            if g is None:
                continue
            ix, iy = g.cell(p[:, :2])
            inb = (ix >= 0) & (iy >= 0) & (ix < g.mask.shape[1]) & (iy < g.mask.shape[0])
            zf = np.full(len(p), np.nan)
            zf[inb] = g.floor_z[iy[inb], ix[inb]]
            hit = inb & np.isfinite(zf)
            hit[hit] &= g.mask[iy[hit], ix[hit]] & (p[hit, 2] > zf[hit] + 0.1) & (p[hit, 2] < zf[hit] + args.agent_height)
            if hit.sum() >= 30:
                on_walk, s_id = True, si
                break
        clusters.append({'n': int(m.sum()), 'center': ctr.round(2).tolist(), 'min': lo.round(2).tolist(), 'max': hi.round(2).tolist(),
                         'size_m': (hi - lo).round(2).tolist(), 'color': Cf[m].mean(0).round(2).tolist(),
                         'median_d_m': float(np.median(d[far][m])), 'on_walkable': on_walk, 'storey': s_id})
    clusters.sort(key=lambda c: (-int(c['on_walkable']), -c['n']))
    n_op = int(len(P))
    return {'n_opaque': n_op, 'n_far': int(far.sum()), 'far_frac': float(far.mean()),
            'n_clusters': len(clusters), 'n_clusters_on_walkable': int(sum(c['on_walkable'] for c in clusters)),
            'clusters': clusters[:40]}, far


# ---------------------------------------------------------------------------
# B 바닥
class Grid:
    def __init__(self, lo, hi, cell):
        self.o = np.floor(np.asarray(lo) / cell) * cell
        self.c = cell
        self.shape = (int(np.ceil((hi[1] - self.o[1]) / cell)) + 1, int(np.ceil((hi[0] - self.o[0]) / cell)) + 1)

    def centers(self):
        ys, xs = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        return np.c_[self.o[0] + (xs.ravel() + 0.5) * self.c, self.o[1] + (ys.ravel() + 0.5) * self.c]

    def cell(self, xy):
        return (np.floor((xy[:, 0] - self.o[0]) / self.c).astype(np.int64), np.floor((xy[:, 1] - self.o[1]) / self.c).astype(np.int64))

    def extent(self):
        return [self.o[0], self.o[0] + self.shape[1] * self.c, self.o[1], self.o[1] + self.shape[0] * self.c]


def classify_floor(sc, fn, mesh, S, args):
    """층 하나 -> Grid, 분류 (H,W), mesh 바닥 z (H,W)."""
    import open3d as o3d
    z0, z1 = S['z_range'][0] - args.band, S['z_range'][1] + args.band
    tc = mesh.triangles_center
    up = (np.abs(fn[:, 2]) >= args.floor_cos) & (tc[:, 2] >= z0) & (tc[:, 2] <= z1)
    xy = mesh.triangles[up][:, :, :2].reshape(-1, 2)
    g = Grid(xy.min(0) - 1.0, xy.max(0) + 1.0, args.cell_m)
    C = g.centers()
    zs = z1 + 1.0
    rays = np.c_[C, np.full(len(C), zs), np.zeros((len(C), 2)), -np.ones(len(C))].astype(np.float32)
    h = sc.cast_rays(o3d.core.Tensor(rays))
    t = h['t_hit'].numpy()
    pid = h['primitive_ids'].numpy()
    hit = np.isfinite(t)
    zh = np.where(hit, zs - t, np.nan)
    nz = np.zeros(len(C))
    nz[hit] = fn[pid[hit], 2]
    cls = np.full(len(C), SEE, np.uint8)
    inb = hit & (zh >= z0) & (zh <= z1)
    cls[inb & (np.abs(nz) >= args.floor_cos)] = FLOOR
    cls[inb & (np.abs(nz) < args.floor_cos)] = COVER
    cls[hit & (zh > z1)] = COVER
    cls[hit & (zh < z0) & (nz >= 0.5)] = OPEN
    # 뚫림 중 아래향 면(천장 뒷면)에 맞은 셀: 그 면을 지나 계속 쏴서 위향 면이 나오면 "지붕 아래 트임"(UNDER).
    # 지상층에서 내려가는 계단 아래쪽은 지붕(경사 천장) 밑이라 위에서 보면 천장 뒷면만 맞는다 — 노은역 S2 출구 계단.
    # 구멍 판정은 SEE 와 똑같이 다룬다(구멍 아래로 아래층 바닥이 보이는 경우도 여기). 그림·면적 보고만 나눈다.
    k = np.where((cls == SEE) & hit)[0]
    zc = zh[k].copy()
    for _ in range(6):
        if len(k) == 0:
            break
        r2 = np.c_[C[k], zc - 0.01, np.zeros((len(k), 2)), -np.ones(len(k))].astype(np.float32)
        h2 = sc.cast_rays(o3d.core.Tensor(r2))
        t2 = h2['t_hit'].numpy()
        ok = np.isfinite(t2)
        k, zc, t2, p2 = k[ok], zc[ok], t2[ok], h2['primitive_ids'].numpy()[ok]
        zc = zc - 0.01 - t2
        up_hit = fn[p2, 2] >= 0.5
        cls[k[up_hit]] = UNDER
        k, zc = k[~up_hit], zc[~up_hit]
    flipped = int((inb & (nz <= -args.floor_cos)).sum())
    zf = np.where(cls == FLOOR, zh, np.nan)
    return g, cls.reshape(g.shape), zf.reshape(g.shape), flipped


def fill_floor_z(cls, zf):
    """바닥이 아닌 셀에 가장 가까운 바닥 셀의 z 를 채운다 (구멍 셀의 기준 높이)."""
    from scipy import ndimage
    if not (cls == FLOOR).any():
        return zf
    _, (iy, ix) = ndimage.distance_transform_edt(cls != FLOOR, return_indices=True)
    return zf[iy, ix]


def gs_floor_alpha(gs, R2, g, zref, args, tile_m=8.0):
    """위에서 내려다본 바닥 띠(zref −0.25 ~ +0.15) 가우시안의 불투명도 α 와 α=0.5 높이. 셀마다.

    가우시안 2D 응답 = O · exp(−½ dᵀ Σxy⁻¹ d) (Σxy = R S² Rᵀ 의 xy 블록), 3σ · KERNEL_MIN_RESPONSE 에서 자름.
    """
    H, W = g.shape
    alpha = np.zeros((H, W), np.float32)
    zgs = np.full((H, W), np.nan, np.float32)
    zlo, zhi = np.nanmin(zref) - 0.3, np.nanmax(zref) + 0.2
    P, O = gs['P'], gs['O']
    sel = np.where((P[:, 2] >= zlo) & (P[:, 2] <= zhi) & (O >= 0.02)
                   & (P[:, 0] >= g.o[0] - 1) & (P[:, 0] <= g.o[0] + W * g.c + 1)
                   & (P[:, 1] >= g.o[1] - 1) & (P[:, 1] <= g.o[1] + H * g.c + 1))[0]
    Ps, Os, Rs, Ss = P[sel], O[sel], R2[sel], gs['S'][sel]
    # Σxy
    M = Rs[:, :2, :] * Ss[:, None, :]                       # (n,2,3) = (R S) 의 xy 행
    Sxy = M @ M.transpose(0, 2, 1) + (0.3 * g.c) ** 2 * np.eye(2)   # 셀보다 작은 가우시안이 셀 사이로 빠지지 않게
    det = Sxy[:, 0, 0] * Sxy[:, 1, 1] - Sxy[:, 0, 1] ** 2
    inv = np.stack([Sxy[:, 1, 1], -Sxy[:, 0, 1], Sxy[:, 0, 0]], 1) / det[:, None]
    rad = np.minimum(np.ceil(3 * np.sqrt(np.maximum(Sxy[:, 0, 0], Sxy[:, 1, 1])) / g.c), 6).astype(int)
    cx, cy = g.cell(Ps[:, :2])
    nty, ntx = int(np.ceil(H * g.c / tile_m)), int(np.ceil(W * g.c / tile_m))
    tcell = int(tile_m / g.c)
    for ty in range(nty):
        for tx in range(ntx):
            y0, x0 = ty * tcell, tx * tcell
            y1, x1 = min(y0 + tcell, H), min(x0 + tcell, W)
            k = np.where((cy >= y0 - 6) & (cy < y1 + 6) & (cx >= x0 - 6) & (cx < x1 + 6))[0]
            if len(k) == 0:
                continue
            E_cell, E_z, E_a = [], [], []
            for r in np.unique(rad[k]):
                kk = k[rad[k] == r]
                oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
                oy, ox = oy.ravel(), ox.ravel()
                iy = cy[kk, None] + oy[None]
                ix = cx[kk, None] + ox[None]
                dx = g.o[0] + (ix + 0.5) * g.c - Ps[kk, 0, None]
                dy = g.o[1] + (iy + 0.5) * g.c - Ps[kk, 1, None]
                q = inv[kk, 0, None] * dx * dx + 2 * inv[kk, 1, None] * dx * dy + inv[kk, 2, None] * dy * dy
                a = Os[kk, None] * np.exp(-0.5 * q)
                ok = (a >= KERNEL_MIN_RESPONSE) & (iy >= y0) & (iy < y1) & (ix >= x0) & (ix < x1)
                ii = np.where(ok)
                iyy, ixx = iy[ii], ix[ii]
                zr = zref[iyy, ixx]
                zz = Ps[kk, 2][ii[0]]
                inband = (zz >= zr - 0.25) & (zz <= zr + 0.15)
                E_cell.append((iyy * W + ixx)[inband])
                E_z.append(zz[inband])
                E_a.append(np.minimum(a[ii][inband], 0.99))
            if not E_cell:
                continue
            ce, ze, ae = np.concatenate(E_cell), np.concatenate(E_z), np.concatenate(E_a)
            if len(ce) == 0:
                continue
            o = np.lexsort((-ze, ce))                          # 셀별, 위에서 아래로
            ce, ze, ae = ce[o], ze[o], ae[o]
            la = np.log1p(-ae)
            cs = np.cumsum(la)
            first = np.r_[True, ce[1:] != ce[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(ce)), 0))
            base = np.where(start > 0, cs[start - 1], 0.0)
            acc = 1 - np.exp(cs - base)                        # 셀 안에서 위부터 쌓은 α
            last = np.r_[ce[1:] != ce[:-1], True]
            alpha.flat[ce[last]] = acc[last]
            reach = acc >= args.alpha_min
            rr = np.where(reach)[0]
            if len(rr):
                c_r = ce[rr]
                f = np.r_[True, c_r[1:] != c_r[:-1]]
                zgs.flat[c_r[f]] = ze[rr[f]]
    return alpha, zgs, int(len(sel))


def components(mask, cls, zf_fill, g, walk_zone, alpha, kind, args, si):
    """구멍 덩어리 목록."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    edt = ndimage.distance_transform_edt(mask)
    objs = ndimage.find_objects(lab)
    out = []
    for i, sl in enumerate(objs, 1):
        sl2 = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        m = lab[sl2] == i
        ring = ndimage.binary_dilation(m, structure=np.ones((3, 3))) & ~m
        rc = cls[sl2][ring]
        floor_frac = float((rc == FLOOR).mean()) if len(rc) else 0.0
        area = float(m.sum() * g.c ** 2)
        diam = float(max(2 * edt[sl2][m].max() - 1, 1) * g.c)
        yy, xx = np.where(m)
        yy, xx = yy + sl2[0].start, xx + sl2[1].start
        cx, cy = g.o[0] + (xx.mean() + 0.5) * g.c, g.o[1] + (yy.mean() + 0.5) * g.c
        on_walk = bool(walk_zone[yy, xx].any()) if walk_zone is not None else None
        gs_fill = float((alpha[yy, xx] >= args.alpha_min).mean())
        out.append({'kind': kind, 'storey': si, 'x': round(float(cx), 2), 'y': round(float(cy), 2),
                    'z': round(float(np.nanmedian(zf_fill[yy, xx])), 2), 'area_m2': round(area, 4),
                    'inscribed_diam_m': round(diam, 3), 'ring_floor_frac': round(floor_frac, 2),
                    'on_walkable': on_walk, 'gs_filled_frac': round(gs_fill, 2),
                    'bbox_xy': [round(g.o[0] + xx.min() * g.c, 2), round(g.o[1] + yy.min() * g.c, 2),
                                round(g.o[0] + (xx.max() + 1) * g.c, 2), round(g.o[1] + (yy.max() + 1) * g.c, 2)]})
    return out, lab


def walkable_mask(nav01, g, S, args, storeys=()):
    """01 navmesh 를 이 층 격자에 래스터 -> (navmesh, 통행 구역, 계단 = 어느 층 띠에도 안 드는 navmesh 삼각형)."""
    from scipy import ndimage
    if not nav01:
        return None, None, None
    nv = sorted(Path(nav01).glob('*.navmesh'))
    if not nv:
        return None, None, None
    pf = nu.load_pathfinder(str(nv[0]))
    tris = nu.navmesh_triangles(pf)
    zc = tris[:, :, 2].mean(1)
    z0, z1 = S['z_range'][0] - args.band, S['z_range'][1] + args.band
    tr = tris[(zc >= z0) & (zc <= z1)]
    nav = nu.rasterize_tris_xy(tr, g.o, g.c, g.shape)
    r = int(np.ceil((args.agent_radius + 0.05) / g.c))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    zone = ndimage.binary_fill_holes(ndimage.binary_dilation(nav, structure=(yy ** 2 + xx ** 2) <= r * r))
    inany = np.zeros(len(tris), bool)
    for T in storeys:
        inany |= (zc >= T['z_range'][0] - args.band) & (zc <= T['z_range'][1] + args.band)
    stairs = nu.rasterize_tris_xy(tris[~inany], g.o, g.c, g.shape) if (~inany).any() else np.zeros(g.shape, bool)
    return nav, zone, stairs


def stair_check(sc, fn, gs, R_all, nav01, storeys, args):
    """계단 (B1·B2 에 합산) — 01 navmesh 중 어느 층 띠에도 안 드는 삼각형(= 계단·경사, 01 의 연결부) 위의 mesh 디딤판과 GS.

    바닥 검사(B1·B2)는 층 띠 안만 보므로 계단 경사 구간이 빠진다. 여기서는 navmesh 높이를 기준으로 칸마다 쏜다.
    반환: (stats 목록, 구멍 목록, 그림 재료) 또는 None.
    """
    if not nav01:
        return None
    nv = sorted(Path(nav01).glob('*.navmesh'))
    if not nv:
        return None
    import open3d as o3d
    from scipy import ndimage
    tris = nu.navmesh_triangles(nu.load_pathfinder(str(nv[0])))
    zc = tris[:, :, 2].mean(1)
    inany = np.zeros(len(tris), bool)
    for T in storeys:
        inany |= (zc >= T['z_range'][0] - args.band) & (zc <= T['z_range'][1] + args.band)
    st = tris[~inany]
    if len(st) == 0:
        return None
    comp = nu.tri_components(st, np.ones(len(st), bool))
    xy = st[:, :, :2].reshape(-1, 2)
    g = Grid(xy.min(0) - 0.5, xy.max(0) + 0.5, args.cell_m)
    navz = np.full(g.shape, np.nan)
    cid = np.full(g.shape, -1)
    for t, c in zip(st, comp):                               # 삼각형마다 bbox 칸 + 무게중심 좌표로 z 보간
        ix0, iy0 = g.cell(t[:, :2].min(0, keepdims=True))
        ix1, iy1 = g.cell(t[:, :2].max(0, keepdims=True))
        yy, xx = np.mgrid[iy0[0]:iy1[0] + 1, ix0[0]:ix1[0] + 1]
        q = np.c_[g.o[0] + (xx.ravel() + 0.5) * g.c, g.o[1] + (yy.ravel() + 0.5) * g.c]
        a, b, cc = t[0, :2], t[1, :2], t[2, :2]
        M = np.array([b - a, cc - a]).T
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        uv = np.linalg.solve(M, (q - a).T).T
        w = np.c_[1 - uv.sum(1), uv]
        ok = (w >= -1e-6).all(1)
        navz[yy.ravel()[ok], xx.ravel()[ok]] = w[ok] @ t[:, 2]
        cid[yy.ravel()[ok], xx.ravel()[ok]] = c
    mask = np.isfinite(navz)
    C = g.centers()[mask.ravel()]
    z0 = navz[mask] + 1.0
    h = sc.cast_rays(o3d.core.Tensor(np.c_[C, z0, np.zeros((len(C), 2)), -np.ones(len(C))].astype(np.float32)))
    t = h['t_hit'].numpy()
    pid = h['primitive_ids'].numpy()
    hit = np.isfinite(t)
    rel = np.where(hit, 1.0 - t, np.nan)                    # 맞은 높이 − navmesh 높이
    nz = np.where(hit, fn[np.where(hit, pid, 0), 2], 0.0)
    cl = np.full(len(C), SEE, np.uint8)
    tread = hit & (rel >= -0.35) & (rel <= 0.25) & (np.abs(nz) >= 0.5)
    cl[tread] = FLOOR
    cl[hit & (rel > 0.25)] = COVER
    cl[hit & (rel >= -0.35) & (rel <= 0.25) & (np.abs(nz) < 0.5)] = COVER
    cls = np.full(g.shape, OUT, np.uint8)
    cls[mask] = cl
    tz = np.full(g.shape, np.nan)
    tz[mask] = np.where(tread, z0 - t, np.nan)
    zref = fill_floor_z(cls, tz)
    zref[~mask] = np.nan
    alpha, zgs, _ = gs_floor_alpha(gs, R_all, g, zref, args)
    mesh_hole = mask & (cls == SEE)
    gs_hole = (cls == FLOOR) & (alpha < args.alpha_min)
    holes = []
    for kind, m_ in (('stair mesh', mesh_hole), ('stair gs', gs_hole)):
        hs, lab = components(m_, cls, zref, g, mask, alpha, kind, args, -1)
        for h_ in hs:
            yy, xx = g.cell(np.array([[h_['x'], h_['y']]]))[::-1]
            h_['connector'] = int(cid[min(max(yy[0], 0), g.shape[0] - 1), min(max(xx[0], 0), g.shape[1] - 1)])
            h_['flag'] = 'CRITICAL' if h_['inscribed_diam_m'] >= args.hole_min_diam else 'minor'
        holes += hs
    stats = []
    for c in np.unique(comp):
        m_ = cid == c
        if m_.sum() * g.c ** 2 < 1.0:                        # 1 m² 미만 조각은 뺌
            continue
        dz = (zgs - tz)[m_ & np.isfinite(zgs) & np.isfinite(tz)]
        ys, xs = np.where(m_)
        stats.append({'connector': int(c), 'area_m2': float(m_.sum() * g.c ** 2),
                      'center_xy': [round(float(g.o[0] + (xs.mean() + 0.5) * g.c), 2), round(float(g.o[1] + (ys.mean() + 0.5) * g.c), 2)],
                      'z_range': [round(float(np.nanmin(navz[m_])), 2), round(float(np.nanmax(navz[m_])), 2)],
                      'tread_frac': float((cls[m_] == FLOOR).mean()), 'cover_frac': float((cls[m_] == COVER).mean()),
                      'mesh_hole_m2': float((mesh_hole & m_).sum() * g.c ** 2),
                      'n_mesh_holes_big': sum(h_['kind'] == 'stair mesh' and h_['flag'] == 'CRITICAL' and h_['connector'] == c for h_ in holes),
                      'gs_hole_frac': float((gs_hole & m_).sum() / max((cls[m_] == FLOOR).sum(), 1)),
                      'gs_dz_median_m': float(np.median(dz)) if len(dz) else None,
                      'bbox': [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]})
    # 이름 붙이기: 01 연결부(계단) 중심 3 m 안이면 그 번호, 아니면 높이 폭 ≥ 1 m 는 "01 에 없는 계단·경사", 작으면 "층 안 경사·조각"
    conns = json.load(open(Path(nav01) / 'verify.json')).get('connectors', []) if (Path(nav01) / 'verify.json').exists() else []
    for S_ in stats:
        S_['z_span_m'] = round(S_['z_range'][1] - S_['z_range'][0], 2)
        near = [c for c in conns if np.hypot(*(np.array(c['center_xy']) - S_['center_xy'])) < 3.0]
        if near:
            S_['kind'] = f"01 C{near[0]['id']} {near[0]['kind']}"
        elif S_['z_span_m'] >= 1.0:
            S_['kind'] = 'level change not in 01 connectors'
        else:
            S_['kind'] = 'slope / fragment'
        S_['gated'] = S_['z_span_m'] >= 1.0
    stats.sort(key=lambda x: (not x['gated'], -x['area_m2']))
    return stats, holes, (g, cls, alpha, tz, mask)


def fig_stairs(out, scene, stats, fig_in, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g, cls, alpha, tz, mask = fig_in
    if not stats:
        return None
    img = np.ones(cls.shape + (3,))
    img[cls == FLOOR] = CLS_COLORS['floor']
    img[cls == COVER] = CLS_COLORS['cover']
    img[(cls == SEE) & mask] = CLS_COLORS['mesh_hole_both']
    img[(cls == FLOOR) & (alpha < args.alpha_min)] = CLS_COLORS['gs_hole']
    n = len(stats)
    fig, axs = plt.subplots(n, 3, figsize=(16, 4.2 * n), squeeze=False)
    for r, S in enumerate(stats):
        x0, x1, y0, y1 = S['bbox']
        sl = (slice(max(y0 - 10, 0), y1 + 11), slice(max(x0 - 10, 0), x1 + 11))
        e = [g.o[0] + sl[1].start * g.c, g.o[0] + sl[1].stop * g.c, g.o[1] + sl[0].start * g.c, g.o[1] + sl[0].stop * g.c]
        axs[r, 0].imshow(img[sl], origin='lower', extent=e, interpolation='nearest')
        axs[r, 0].set_title(f"piece {S['connector']} [{S['kind']}] @({S['center_xy'][0]:.0f},{S['center_xy'][1]:.0f}) z {S['z_range'][0]:+.1f}~{S['z_range'][1]:+.1f}\n"
                            f"gray tread {100 * S['tread_frac']:.1f}% / dark cover / red MESH hole {S['mesh_hole_m2']:.2f} m2 / blue GS hole", fontsize=9)
        im = axs[r, 1].imshow(np.where(mask, alpha, np.nan)[sl], origin='lower', extent=e, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        plt.colorbar(im, ax=axs[r, 1], fraction=0.046)
        axs[r, 1].set_title(f"GS alpha on treads (< {args.alpha_min} = GS hole {100 * S['gs_hole_frac']:.1f}%)", fontsize=9)
        im = axs[r, 2].imshow(tz[sl], origin='lower', extent=e, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=axs[r, 2], fraction=0.046)
        axs[r, 2].set_title('mesh tread z [m] (steps should look banded)', fontsize=9)
        for ax in axs[r]:
            ax.set_aspect('equal')
    fig.tight_layout()
    fn = 'stairs.png'
    fig.savefig(out / fn, dpi=90)
    plt.close(fig)
    return fn


class WalkInfo:
    def __init__(self, g, zone, floor_z):
        self.o, self.c, self.mask, self.floor_z = g.o, g.c, zone, floor_z
        self.cell = g.cell


# ---------------------------------------------------------------------------
# 그림
CLS_COLORS = {'out': (1, 1, 1), 'under': (0.80, 0.72, 0.95), 'floor': (0.82, 0.82, 0.82), 'cover': (0.35, 0.35, 0.35), 'open': (0.70, 0.85, 1.0),
              'unscanned': (1.0, 0.92, 0.55), 'mesh_hole_gs_ok': (0.85, 0.0, 0.85), 'mesh_hole_both': (0.9, 0.0, 0.0),
              'gs_hole': (0.0, 0.35, 1.0)}
CLS_LABELS = {'out': 'nothing at this height (outside scan)', 'floor': 'floor OK (mesh + GS)',
              'cover': 'covered (wall/column/object/stairs going up)', 'open': 'opening (stairs / lower level)',
              'under': 'lower level seen through ceiling (stairs down / storey below)',
              'unscanned': 'enclosed unscanned (walls around)', 'mesh_hole_gs_ok': 'MESH hole, GS looks solid',
              'mesh_hole_both': 'MESH + GS hole', 'gs_hole': 'GS hole (mesh floor OK)'}


def floor_rgb(cls, mesh_hole, unscanned, gs_hole, alpha, args):
    img = np.ones(cls.shape + (3,))
    img[cls == FLOOR] = CLS_COLORS['floor']
    img[cls == COVER] = CLS_COLORS['cover']
    img[cls == OPEN] = CLS_COLORS['open']
    img[cls == UNDER] = CLS_COLORS['under']
    img[unscanned] = CLS_COLORS['unscanned']
    img[gs_hole] = CLS_COLORS['gs_hole']
    img[mesh_hole & (alpha >= args.alpha_min)] = CLS_COLORS['mesh_hole_gs_ok']
    img[mesh_hole & (alpha < args.alpha_min)] = CLS_COLORS['mesh_hole_both']
    return img


def fig_floor(out, scene, si, S, g, img, nav, holes, args, stairs=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    H, W = img.shape[:2]
    fw = 18
    fh = max(6, fw * H / W + 1.5)
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.imshow(img, origin='lower', extent=g.extent(), interpolation='nearest')
    if nav is not None:
        ax.contour(np.linspace(g.extent()[0], g.extent()[1], W), np.linspace(g.extent()[2], g.extent()[3], H),
                   nav.astype(float), levels=[0.5], colors=['#1a9e3a'], linewidths=0.8)
    if stairs is not None and stairs.any():
        ax.contourf(np.linspace(g.extent()[0], g.extent()[1], W), np.linspace(g.extent()[2], g.extent()[3], H),
                    stairs.astype(float), levels=[0.5, 1.5], colors='none', hatches=['////'])
        ax.contour(np.linspace(g.extent()[0], g.extent()[1], W), np.linspace(g.extent()[2], g.extent()[3], H),
                   stairs.astype(float), levels=[0.5], colors=['#8a2be2'], linewidths=1.0)
    for h in holes:
        if h['flag'] in ('CRITICAL', 'MAJOR'):
            col = '#d00' if h['flag'] == 'CRITICAL' else '#e80'
            rr = max(0.6, h['inscribed_diam_m'] * 2)
            ax.add_patch(plt.Circle((h['x'], h['y']), rr, fill=False, ec=col, lw=1.6))
            ax.text(h['x'] + rr, h['y'] + rr, h['id'], color=col, fontsize=8, weight='bold')
    hand = [Patch(facecolor=CLS_COLORS[k], edgecolor='#999', label=v) for k, v in CLS_LABELS.items()]
    hand.append(Patch(facecolor='none', edgecolor='#1a9e3a', label='navmesh (01) outline, this storey'))
    hand.append(Patch(facecolor='none', edgecolor='#8a2be2', hatch='////', label='stairs = 01 navmesh between storeys'))
    hand.append(Patch(facecolor='none', edgecolor='#d00', label=f'CRITICAL: walkable & diam >= {args.hole_min_diam} m'))
    hand.append(Patch(facecolor='none', edgecolor='#e80', label='MAJOR: diam >= thr, off walkable'))
    ax.legend(handles=hand, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_title(f'{scene} storey S{si} z {S["z_range"][0]:+.2f}~{S["z_range"][1]:+.2f}  floor holes  (cell {g.c} m)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    fn = f'floor_S{si}.png'
    fig.savefig(out / fn, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return fn


def fig_floor_offset(out, scene, si, S, g, dz, alpha, nav, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H, W = dz.shape
    fig, axs = plt.subplots(1, 2, figsize=(22, max(5, 11 * H / W + 1.5)))
    im = axs[0].imshow(dz, origin='lower', extent=g.extent(), cmap='RdBu_r', vmin=-0.15, vmax=0.15, interpolation='nearest')
    plt.colorbar(im, ax=axs[0], fraction=0.03, label='GS floor z - mesh floor z [m]  (blue = GS below)')
    axs[0].set_title(f'S{si} B3 floor height offset')
    im = axs[1].imshow(np.where(alpha > 0, alpha, np.nan), origin='lower', extent=g.extent(), cmap='magma', vmin=0, vmax=1,
                       interpolation='nearest')
    plt.colorbar(im, ax=axs[1], fraction=0.03, label='GS floor alpha (top-down, floor band)')
    axs[1].set_title(f'S{si} B2 GS floor opacity  (< {args.alpha_min} = GS hole)')
    for ax in axs:
        if nav is not None:
            ax.contour(np.linspace(g.extent()[0], g.extent()[1], W), np.linspace(g.extent()[2], g.extent()[3], H),
                       nav.astype(float), levels=[0.5], colors=['#1a9e3a'], linewidths=0.6)
        ax.set_aspect('equal')
    fn = f'floor_offset_S{si}.png'
    fig.savefig(out / fn, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return fn


def fig_hole_crops(out, scene, crops, args):
    """심각한 구멍 확대: 분류 · GS α · mesh z."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not crops:
        return None
    n = len(crops)
    fig, axs = plt.subplots(n, 3, figsize=(13, 4.0 * n), squeeze=False)
    for r, (h, g, img, alpha, zf) in enumerate(crops):
        half = 2.0
        ext = [h['x'] - half, h['x'] + half, h['y'] - half, h['y'] + half]
        ix0, iy0 = g.cell(np.array([[ext[0], ext[2]]]))
        ix1, iy1 = g.cell(np.array([[ext[1], ext[3]]]))
        sl = (slice(max(iy0[0], 0), iy1[0]), slice(max(ix0[0], 0), ix1[0]))
        e2 = [g.o[0] + sl[1].start * g.c, g.o[0] + sl[1].stop * g.c, g.o[1] + sl[0].start * g.c, g.o[1] + sl[0].stop * g.c]
        axs[r, 0].imshow(img[sl], origin='lower', extent=e2, interpolation='nearest')
        axs[r, 0].set_title(f'{h["id"]} {h["flag"]} {h["kind"]}  area {h["area_m2"]:.3f} m2  diam {h["inscribed_diam_m"]:.2f} m', fontsize=9)
        im = axs[r, 1].imshow(alpha[sl], origin='lower', extent=e2, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        plt.colorbar(im, ax=axs[r, 1], fraction=0.046)
        axs[r, 1].set_title('GS floor alpha', fontsize=9)
        im = axs[r, 2].imshow(zf[sl], origin='lower', extent=e2, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=axs[r, 2], fraction=0.046)
        axs[r, 2].set_title(f'mesh floor z (S{h["storey"]})  ({h["x"]:.1f}, {h["y"]:.1f}, {h["z"]:.2f})', fontsize=9)
    fig.tight_layout()
    fn = 'hole_crops.png'
    fig.savefig(out / fn, dpi=90)
    plt.close(fig)
    return fn


def fig_align_topdown(out, scene, storeys, P, d, far, Pfar, cover_pts, cover_alpha, a4c, args):
    """층마다 위에서 본 정합: (왼) 0.25 m 기둥별 표면 가우시안 중앙 d, (오) GS 없는 mesh 점(파랑) · mesh 없는 GS(빨강)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = []
    for si, S in enumerate(storeys):
        z0, z1 = S['z_range'][0] - args.band, S['z_range'][1] + 3.0
        k = (P[:, 2] >= z0) & (P[:, 2] <= z1)
        if k.sum() < 1000:
            continue
        xy = P[k, :2]
        lo, hi = xy.min(0), xy.max(0)
        c = 0.25
        nx, ny = int((hi[0] - lo[0]) / c) + 1, int((hi[1] - lo[1]) / c) + 1
        ix = ((xy[:, 0] - lo[0]) / c).astype(int)
        iy = ((xy[:, 1] - lo[1]) / c).astype(int)
        lin = iy * nx + ix
        o = np.argsort(lin, kind='stable')
        u, st, cnt = np.unique(lin[o], return_index=True, return_counts=True)
        med = np.array([np.median(x) for x in np.split(d[k][o], st[1:])])
        img = np.full(nx * ny, np.nan)
        img[u[cnt >= 5]] = med[cnt >= 5]
        img = img.reshape(ny, nx)
        ext = [lo[0], lo[0] + nx * c, lo[1], lo[1] + ny * c]
        fig, axs = plt.subplots(1, 2, figsize=(24, max(6, 12 * ny / nx + 1.5)))
        im = axs[0].imshow(img, origin='lower', extent=ext, cmap='turbo', vmin=0, vmax=0.2, interpolation='nearest')
        plt.colorbar(im, ax=axs[0], fraction=0.03, label='median GS->mesh dist [m]: blue = aligned, red = off')
        axs[0].set_title(f'S{si} A2 local alignment (z {z0:+.1f}~{z1:+.1f}); >= {args.local_bad_m} m = misaligned')
        kc = (cover_pts[:, 2] >= z0) & (cover_pts[:, 2] <= z1)
        kk = kc & (cover_alpha < args.alpha_min)
        axs[1].scatter(cover_pts[kc & ~kk, 0][::20], cover_pts[kc & ~kk, 1][::20], s=0.2, c='#bbbbbb', rasterized=True,
                       label=f'mesh covered by GS (OK, alpha >= {args.alpha_min})')
        axs[1].scatter(cover_pts[kk, 0], cover_pts[kk, 1], s=0.4, c='#0050ff', rasterized=True, label=f'A3 mesh without GS (alpha < {args.alpha_min})')
        kf = (Pfar[:, 2] >= z0) & (Pfar[:, 2] <= z1)
        sub = np.where(kf)[0][::3]
        axs[1].scatter(Pfar[sub, 0], Pfar[sub, 1], s=0.3, c='#e00000', rasterized=True, label=f'A4 GS without mesh (d >= {args.near_m} m)')
        for i, cl in enumerate(a4c):
            if cl['on_walkable'] and cl['storey'] == si:
                axs[1].add_patch(plt.Rectangle(cl['min'][:2], cl['size_m'][0], cl['size_m'][1], fill=False, ec='k', lw=1.2))
                axs[1].text(cl['max'][0], cl['max'][1], f'G{i}', fontsize=8, weight='bold')
        from matplotlib.patches import Patch
        hd, lb = axs[1].get_legend_handles_labels()
        lg = axs[1].legend(hd + [Patch(facecolor='none', edgecolor='k', label='A4 cluster over walkable area')],
                           lb + ['A4 cluster over walkable area'], loc='upper right', markerscale=15)
        lg.legend_handles[-1].set_visible(True)
        axs[1].set_xlim(ext[0], ext[1])
        axs[1].set_ylim(ext[2], ext[3])
        axs[1].set_title(f'S{si} A3 (blue) / A4 (red) ; black box = A4 cluster over walkable area')
        for ax in axs:
            ax.set_aspect('equal')
        fn = f'align_S{si}.png'
        fig.savefig(out / fn, dpi=100, bbox_inches='tight')
        plt.close(fig)
        figs.append(fn)
    return figs


def fig_hist(out, d, s, res_a1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].hist(np.clip(d, 0, 0.5), bins=100, color='#357')
    axs[0].set_xlabel('GS->mesh distance d [m] (opaque gaussians, clipped 0.5)')
    axs[0].set_yscale('log')
    k = np.abs(s) < 0.3
    axs[1].hist(s[k], bins=120, color='#753')
    axs[1].axvline(-res_a1['surface_offset_m'], color='k', ls='--', label=f'-delta = {-res_a1["surface_offset_m"]:+.3f} m')
    axs[1].set_xlabel('signed distance s [m]  (+ = in front of mesh face, - = behind)')
    axs[1].legend()
    fn = 'dist_hist.png'
    fig.tight_layout()
    fig.savefig(out / fn, dpi=100)
    plt.close(fig)
    return fn


def write_ply(path, pts, rgb):
    import trimesh
    pc = trimesh.PointCloud(np.asarray(pts, np.float32), colors=(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    pc.export(path)


def colormap(v, vmax, name='turbo'):
    import matplotlib
    return matplotlib.colormaps[name](np.clip(v / vmax, 0, 1))[:, :3]


def html3d(out, mesh, va, Pfar, holes, args, seed=0):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    V = np.asarray(mesh.vertices)
    k = rng.choice(len(V), min(len(V), 150_000), replace=False)
    tr = [go.Scatter3d(x=V[k, 0], y=V[k, 1], z=V[k, 2], mode='markers', name='mesh vertex (color = GS alpha, A3)',
                       marker=dict(size=1.2, color=va[k], colorscale='Magma', cmin=0, cmax=1,
                                   colorbar=dict(title='GS alpha near mesh')))]
    kf = rng.choice(len(Pfar), min(len(Pfar), 60_000), replace=False) if len(Pfar) else []
    if len(Pfar):
        tr.append(go.Scatter3d(x=Pfar[kf, 0], y=Pfar[kf, 1], z=Pfar[kf, 2], mode='markers', name='A4 GS without mesh',
                               marker=dict(size=1.2, color='red')))
    hh = [h for h in holes if h['flag'] in ('CRITICAL', 'MAJOR')]
    if hh:
        tr.append(go.Scatter3d(x=[h['x'] for h in hh], y=[h['y'] for h in hh], z=[h['z'] + 0.1 for h in hh], mode='markers+text',
                               text=[h['id'] for h in hh], name='floor holes (critical/major)',
                               marker=dict(size=6, color=['#d00' if h['flag'] == 'CRITICAL' else '#e80' for h in hh], symbol='x')))
    f = go.Figure(tr)
    f.update_layout(scene=dict(aspectmode='data'), height=900, legend=dict(itemsizing='constant'))
    fn = 'mesh_gs_3d.html'
    f.write_html(out / fn, include_plotlyjs=True)
    t = (out / fn).read_text()                                 # plotly 는 doctype 을 안 쓴다 -> quirks 모드 방지
    if not t.lstrip().lower().startswith('<!doctype'):
        (out / fn).write_text('<!doctype html>\n' + t.replace('<head>', '<head><meta name="viewport" content="width=device-width,initial-scale=1">', 1))
    return fn


# ---------------------------------------------------------------------------
REPORT_CSS = """
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
.lvl{font-size:.78rem;color:var(--muted);white-space:nowrap}
ul.find{margin:0;padding-left:1.2em;display:grid;gap:4px;max-width:80ch}
figure{margin:10px 0;min-width:0}figure img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
figcaption{font-size:.84rem;color:var(--muted);margin-top:4px;max-width:80ch}
details{border:1px solid var(--line);border-radius:6px;background:var(--panel);padding:6px 12px;margin:8px 0}
summary{cursor:pointer;color:var(--accent);font-size:.88rem}
pre{font-family:var(--f-mono);font-size:.78rem;white-space:pre-wrap;word-break:break-word;margin:6px 0}
a{color:var(--accent)}a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

FIG_CAPTIONS = {
    'floor_S': '층 바닥 지도 (5 cm 칸). 빨강 원 = 통행 구역 안 내접 ≥ 0.10 m 구멍, 주황 원 = 통행 구역 밖. 칠한 색: 보라 = mesh 구멍(GS 는 막혀 보임), 빨강 = 둘 다 구멍, 파랑 = GS 구멍, 연보라 = 천장 너머 아래층(내려가는 계단·아래층), 빗금 = 01 navmesh 계단.',
    'crops': '큰 구멍 확대 (4 m × 4 m): 분류 · GS 바닥 α · mesh 바닥 높이.',
    'stairs': 'B1·B2 계단 내역, 조각마다: 디딤판 분류(회색 디딤판 · 진회색 챌판/난간 · 빨강 mesh 구멍 · 파랑 GS 구멍) · 디딤판 위 GS α · 디딤판 높이(계단이면 띠 모양).',
    'floor_offset_': '왼쪽 B3: GS 바닥 높이 − mesh 바닥 높이 (파랑 = GS 가 아래). 오른쪽 B2: 위에서 쌓은 GS 바닥 α (0.5 미만 = GS 구멍).',
    'align_': '왼쪽 A2: 0.25 m 기둥마다 표면 가우시안 → mesh 거리 중앙값 (파랑 = 잘 맞음, 0.10 m 이상 = 어긋남). 오른쪽: 회색 = GS 가 덮은 mesh, 파랑 = A3 GS 가 안 덮은 mesh, 빨강 = A4 mesh 없는 GS.',
    'hist': 'GS → mesh 거리 분포와 부호 거리(− = GS 가 mesh 면 뒤). 점선 = 공통 표면 간격 δ.',
}


def write_report(out, scene, res, figs):
    import html as H

    def tbl(rows, cols, num=()):
        h = '<tr>' + ''.join(f'<th>{H.escape(c)}</th>' for c in cols) + '</tr>'
        for r in rows:
            h += '<tr>' + ''.join(f'<td class="{"num" if c in num else ""}">{r.get(c, "")}</td>' for c in cols) + '</tr>'
        return f'<div class="tw"><table>{h}</table></div>'

    def pill(v):
        if v is True:
            return '<span class="pill ok">PASS</span>'
        if v is False:
            return '<span class="pill ng">FAIL</span>'
        return f'<span class="pill info">{H.escape(str(v))}</span>'

    def fig(key):
        if not figs.get(key):
            return ''
        cap = next((c for k, c in FIG_CAPTIONS.items() if key.startswith(k)), '')
        return f'<figure><img src="{figs[key]}" alt="{H.escape(key)}" loading="lazy"><figcaption>{H.escape(key)} — {H.escape(cap)}</figcaption></figure>'

    def raw(title, obj):
        return f'<details><summary>{H.escape(title)} (원자료 JSON)</summary><pre>{H.escape(json.dumps(obj, indent=1, ensure_ascii=False, default=str))}</pre></details>'

    f3 = lambda x: f'{x:.3f}' if isinstance(x, float) else x
    act = res['criteria']
    other = 'strict' if act == 'relaxed' else 'relaxed'
    kor = {'strict': '엄격', 'relaxed': '완화'}
    gates = [{'검증': H.escape(k), '결과': pill(v[act]), '값': H.escape(str(v['value'])), '기준': H.escape(v['criterion_' + act]),
              '근거': H.escape(RELAXED_WHY.get(v['id'], '') if act == 'relaxed' else ''),
              f'{kor[other]} 기준 (참고)': f"{pill(v[other])} {H.escape(v['criterion_' + other])}"} for k, v in res['gates'].items()]
    crit_by_id = {v['id']: v for v in res['gates'].values()}
    metrics = [{'ID': m[0], '지표': H.escape(m[1]), '계산': H.escape(m[2]),
                '엄격 기준': H.escape(crit_by_id[m[0]]['criterion_strict'] if m[0] in crit_by_id and m[0] not in ('A4', 'G0c') else m[3]),
                '완화 기준': H.escape(crit_by_id[m[0]]['criterion_relaxed'] if m[0] in crit_by_id and m[0] not in ('A4', 'G0c') else '같음'),
                '근거·한계': H.escape(m[4]), '신뢰도': f'<span class="lvl">{H.escape(m[5])}</span>'} for m in METRICS]
    a1 = res['a1']
    cent = res['g0'].get('centroid', {})
    orient = [{'면 방향': k, 'n': v['n'], 's p25': f"{v['s_p25']:+.3f}", 's 중앙': f"{v['s_med']:+.3f}", 's p75': f"{v['s_p75']:+.3f}"}
              for k, v in res['a1_orient'].items()]
    a2c = [{k: f3(v) for k, v in c.items()} for c in res['a2']['clusters'][:15]]
    a4 = [dict({k: f3(v) for k, v in c.items()}, id=f'G{i}') for i, c in enumerate(res['a4']['clusters'])][:25]
    bst = [{'층': f"S{b['storey']}", 'z': f"{b['z_range'][0]:+.2f}~{b['z_range'][1]:+.2f}", '바닥 m²': f"{b['floor_m2']:.0f}",
            '통행 구역 m²': f"{b['walk_zone_m2']:.0f}" if b['walk_zone_m2'] else '-',
            'mesh 구멍 m² (통행)': f"{b['mesh_hole_m2']:.2f} ({b['mesh_hole_walk_m2']:.2f})",
            'GS 구멍 통행 %': f"{100 * b['gs_hole_walk_frac']:.2f}", '닫힌 미스캔 m²': f"{b['unscanned_m2']:.1f}",
            '천장 너머 아래층 m²': f"{b.get('under_roof_m2', 0):.0f}",
            '바닥 높이차 중앙 / p95': f"{b['floor_dz_median_m']:+.3f} / {b['floor_dz_abs_p95_m']:.3f}" if b['floor_dz_median_m'] is not None else '-',
            'CRITICAL': b['critical']} for b in res['b_storeys']]
    holes = [{k: f3(v) for k, v in h.items()} for h in res['holes'] if h['flag'] == 'CRITICAL'][:40]
    stairs = [{'조각': x['connector'], '종류': x['kind'], '판정 대상': '예' if x['gated'] else '아니오', 'z': f"{x['z_range'][0]:+.1f}~{x['z_range'][1]:+.1f}",
               '면적 m²': f"{x['area_m2']:.1f}", '디딤판 %': f"{100 * x['tread_frac']:.1f}", 'mesh 구멍 m²': f"{x['mesh_hole_m2']:.2f}",
               'mesh 구멍(≥0.10)': x['n_mesh_holes_big'], 'GS 구멍 %': f"{100 * x['gs_hole_frac']:.1f}",
               'GS−mesh 높이 중앙': f"{x['gs_dz_median_m']:+.3f}" if x['gs_dz_median_m'] is not None else '-'} for x in res.get('stairs', [])]
    n_pass = {p_: sum(v[p_] is True for v in res['gates'].values()) for p_ in ('strict', 'relaxed')}
    n_fail = {p_: sum(v[p_] is False for v in res['gates'].values()) for p_ in ('strict', 'relaxed')}
    crit_mesh = sum(1 for h in res['holes'] if h['flag'] == 'CRITICAL' and h['kind'].startswith('mesh'))
    findings = [
        f"전체 위치는 맞다: 강체 이동 {a1['trans_norm_m'] * 100:.1f} cm · 회전 {a1['rot_norm_deg']:.3f}°.",
        f"mesh 가 GS 보다 방 쪽으로 부풀어 있다: 공통 표면 간격 δ {a1['surface_offset_m'] * 100:+.1f} cm (바닥·천장 약 5 cm, 벽 약 2.5 cm). "
        f"바닥에서는 로봇 발(충돌 mesh)이 RGB 바닥보다 약 {-np.mean([b['floor_dz_median_m'] for b in res['b_storeys'] if b['floor_dz_median_m'] is not None]) * 100:.1f} cm 위.",
        f"바닥 mesh 구멍 중 통행 구역 안 · 내접 ≥ 0.10 m 인 것 {crit_mesh} 개 — 대부분 지상층(S2)과 서쪽 계단 홀·출구 쪽 가장자리. 대합실·승강장 가운데는 깨끗하다.",
        (lambda g_: f"계단(B1·B2 에 합산) 조각 {len(g_)} 개 중 mesh 구멍 0 · GS 구멍 ≤ 1 % 인 것 {sum(x['n_mesh_holes_big'] == 0 and x['gs_hole_frac'] <= 0.01 for x in g_)} 개. "
                    f"면적 큰 넷: " + ', '.join(f"{x['kind']} ({x['area_m2']:.0f} m², GS 구멍 {100 * x['gs_hole_frac']:.1f} %)" for x in sorted(g_, key=lambda x: -x['area_m2'])[:4]) + '.')(
            [x for x in res.get('stairs', []) if x['gated']]),
        f"A3(GS 가 안 덮은 mesh, 로봇 시야 높이 {100 * res['a3']['uncovered_frac_robot_zone']:.1f} %)는 기하 근사 지표다. 판정용으로는 카메라 렌더 비교가 필요하다.",
    ]
    body = f"""<div class="wrap">
<header><div class="eyebrow">nav3d 00 · 씬 자산 점검 · {H.escape(scene)}</div>
<h1>노은역 mesh ↔ 3DGS 정합과 바닥 구멍</h1>
<p class="muted">충돌 mesh(navmesh·depth 용)와 3DGS(RGB 용)가 같은 자리에 있는지, 바닥·계단에 구멍이 있는지. <b>{kor[act]} 기준 {n_pass[act]} 통과 · {n_fail[act]} 미달</b> (참고: {kor[other]} 기준 {n_pass[other]} 통과 · {n_fail[other]} 미달).
입력 <span class="mono">{H.escape(res['inputs']['usdz'])}</span> · 층·통행 구역 <span class="mono">{H.escape(str(res['inputs']['nav01']))}</span></p></header>
<section><h2>요약</h2><ul class="find">{''.join(f'<li>{H.escape(x)}</li>' for x in findings)}</ul></section>
<section><h2>판정 — {kor[act]} 기준</h2>
<p class="muted">완화 기준(<span class="mono">--criteria relaxed</span>)은 노은역 실측을 본 뒤 정한 값이고, 근거 칸은 그 값이 왜 쓸 만한지의 물리적 기준점이다.
엄격 기준(처음 정한 값)은 참고 칸에 둔다. B1 은 개수 → 면적 비율로 정의가 바뀐다. B1·B2 는 바닥과 계단을 합친 값이다(계단 내역은 B 절 표).</p>
{tbl(gates, ['검증', '결과', '값', '기준', '근거', f'{kor[other]} 기준 (참고)'])}</section>
<section><h2>지표와 기준</h2>
<p class="muted">신뢰도 — 명확: 계산 정의가 하나로 정해짐 · 근사: 정의는 명확하나 렌더러와 다른 근사 · 휴리스틱: 임계값·반경이 결과를 크게 좌우.
기준값은 이 스크립트가 정한 값이며 업계 표준이 아니다. 모두 인자로 바꿀 수 있다.</p>
{tbl(metrics, ['ID', '지표', '계산', '완화 기준', '엄격 기준', '근거·한계', '신뢰도'])}</section>
<section><h2>G0 좌표 · 무게중심</h2>
<p>GS 변환 항등 {res['g0']['xform_identity']} · offset {res['g0']['offset']} · 쿼터니언 {res['g0']['quat_order']['order']}
(|cos| wxyz {res['g0']['quat_order']['wxyz']:.3f} / xyzw {res['g0']['quat_order']['xyzw']:.3f}) · 위치 fp16 (mesh 끝에서 반 칸 {res['g0']['fp16_half_step_at_mesh_extent_m'] * 100:.1f} cm).</p>
<p>무게중심 거리 (mesh 면적가중 기준): GS 전체 <b>{cent.get('dist_gs_all', float('nan')):.2f} m</b> · opacity 가중 {cent.get('dist_gs_opacity_weighted', float('nan')):.2f} m ·
mesh 상자 안 opacity 가중 {cent.get('dist_gs_in_mesh_box_opacity_weighted', float('nan')):.2f} m.
GS 범위 {cent.get('gs_bbox')} 가 mesh 범위 {cent.get('mesh_bbox')} 보다 넓고(바깥 건물·하늘), 스캔 밀도도 달라 이 값은 정합 오차가 아니다 — 정합은 A1 로 본다.</p>
{raw('G0', res['g0'])}</section>
<section><h2>A 정합</h2>
<h3>A1 강체 · A1b 표면 간격</h3>
<p>GS 를 mesh 에 맞추는 이동 <span class="mono">{[round(x, 4) for x in a1['trans_m']]}</span> m (|t| {a1['trans_norm_m']:.4f}) · 회전 {a1['rot_norm_deg']:.4f}° ·
표면 간격 δ <b>{a1['surface_offset_m']:+.4f} m</b> (+ = GS 가 mesh 뒤). δ 항 없이 풀면 이동이 <span class="mono">{[round(x, 4) for x in a1['without_offset_term']['trans_m']]}</span> 로 오염된다.</p>
{tbl(orient, ['면 방향', 'n', 's p25', 's 중앙', 's p75'], num=('n', 's p25', 's 중앙', 's p75'))}
{fig('hist')}{raw('A1', a1)}
<h3>A2 국소 어긋남</h3><p>{res['a2']['bad_voxels']}/{res['a2']['n_voxels']} 복셀 ({100 * res['a2']['bad_frac']:.2f} %) 이 중앙 d &gt; {res['args']['local_bad_m']} m. 덩어리 큰 순:</p>
{tbl(a2c, ['n_voxels', 'center', 'min', 'max', 'median_d_m', 'median_s_m'])}
<h3>A3 GS 가 안 덮은 mesh · A4 mesh 없는 GS</h3>
<p>A3 로봇 시야 높이 {100 * res['a3']['uncovered_frac_robot_zone']:.2f} % ({res['a3']['uncovered_area_m2_robot_zone']:.0f} m²), 전체 {100 * res['a3']['uncovered_frac_all']:.2f} %.
A4 불투명 가우시안의 {100 * res['a4']['far_frac']:.2f} %, 덩어리 {res['a4']['n_clusters']} 개, 통행 구역 위 {res['a4']['n_clusters_on_walkable']} 개.
(노은역 실측: 대합실 가운데 빨간 점의 54 % 는 바닥 아래 반사 가우시안, 33 % 는 1.8 m 위 표지판·천장 설비.)</p>
{tbl(a4, ['id', 'on_walkable', 'storey', 'n', 'center', 'size_m', 'color', 'median_d_m'])}
{''.join(fig(k) for k in figs if k.startswith('align_'))}
</section>
<section><h2>B 바닥 · 계단</h2>
{tbl(bst, list(bst[0].keys()) if bst else [])}
<h3>바닥 구멍 — CRITICAL (통행 구역 안 · 내접 ≥ {res['args']['hole_min_diam']} m), 전체는 holes.csv</h3>
<p class="muted">mesh 구멍 = 로봇이 빠지거나 depth 가 뚫림. GS 구멍 = RGB 에서 바닥이 비쳐 보임(충돌은 정상). B1 판정에는 mesh 구멍만 센다.</p>
{tbl(holes, ['id', 'kind', 'storey', 'x', 'y', 'z', 'area_m2', 'inscribed_diam_m', 'ring_floor_frac', 'gs_filled_frac'])}
{''.join(fig(k) for k in figs if k.startswith('floor_S'))}{fig('crops')}
<h3>B1·B2 의 계단 내역 (01 navmesh 중 층 띠 밖 삼각형)</h3>
<p class="muted">층 띠 밖이라 바닥 검사가 못 보는 계단 경사 구간을 navmesh 높이 기준으로 본다. 높이 폭 ≥ 1 m 조각만 B1·B2 에 합산. 01 연결부 중심 3 m 안이면 그 번호를 붙였다.</p>
{tbl(stairs, list(stairs[0].keys()) if stairs else [])}
{fig('stairs')}
{''.join(fig(k) for k in figs if k.startswith('floor_offset_'))}
</section>
<section><h2>파일</h2><p>3D 보기: <a href="{figs.get('html3d', '')}">{figs.get('html3d', '')}</a> (mesh 꼭짓점 색 = 주변 GS α, 빨강 = mesh 없는 GS, × = 큰 구멍).
로컬 PLY: {H.escape(', '.join(res['plys']))} · CSV: holes.csv, gs_without_mesh_clusters.csv · summary.json</p>
<p class="muted">명령: <span class="mono">/usr/bin/python scripts/dataset_converters/nav3d/00_mesh_gs_check.py --scene {H.escape(scene)}</span> · 걸린 시간 {res['elapsed_s']:.0f} s</p></section>
</div>"""
    head = (f'<title>노은역 mesh-GS 점검</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@400;600;700&display=swap">'
            f'\n<style>{REPORT_CSS}</style>\n')
    (out / 'report_page.html').write_text(head + body)              # Artifact 게시용 (문서 뼈대는 게시 때 붙음)
    (out / 'report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                                     '<meta name="viewport" content="width=device-width,initial-scale=1">' + head + '</head><body>' + body + '</body></html>')


def main():
    args = parse()
    t0 = time.time()
    usdz = args.usdz or SCENES[args.scene]['usdz']
    nav01 = args.nav01 or SCENES.get(args.scene, {}).get('nav01')
    if nav01 == 'none':
        nav01 = None
    out = Path(args.out_dir) / args.scene
    out.mkdir(parents=True, exist_ok=True)
    print(f'[00] {args.scene}: {usdz}')

    mesh, gs, g0 = load_usdz(usdz)
    fn = np.asarray(mesh.face_normals)
    lo, hi = mesh.bounds
    print(f'  mesh {len(mesh.vertices):,} v / {len(mesh.faces):,} f, area {mesh.area:.0f} m² · GS {g0["n_gaussians"]:,} ({time.time() - t0:.0f} s)')

    # 대상: mesh 상자(+0.5 m) 안의 불투명 가우시안
    P = gs['P']
    inbox = np.all((P >= lo - 0.5) & (P <= hi + 0.5), axis=1)
    mc_ = (mesh.triangles_center * mesh.area_faces[:, None]).sum(0) / mesh.area
    cent = {'mesh_area_weighted': mc_.round(3).tolist(), 'gs_all': P.mean(0).round(3).tolist(),
            'gs_opacity_weighted': (np.average(P, axis=0, weights=gs['O'])).round(3).tolist(),
            'gs_in_mesh_box_opacity_weighted': np.average(P[inbox], axis=0, weights=gs['O'][inbox]).round(3).tolist()}
    for k in ('gs_all', 'gs_opacity_weighted', 'gs_in_mesh_box_opacity_weighted'):
        cent['dist_' + k] = float(np.linalg.norm(np.array(cent[k]) - mc_))
    cent['gs_bbox'] = [P.min(0).round(1).tolist(), P.max(0).round(1).tolist()]
    cent['mesh_bbox'] = [lo.round(1).tolist(), hi.round(1).tolist()]
    g0['centroid'] = cent
    print(f"  G0c centroid dist: GS all {cent['dist_gs_all']:.2f} m · opacity-weighted {cent['dist_gs_opacity_weighted']:.2f} m · "
          f"in mesh box {cent['dist_gs_in_mesh_box_opacity_weighted']:.2f} m")
    op = inbox & (gs['O'] >= args.op_min)
    idx = np.where(op)[0]
    sc = raycaster(mesh)
    d, s, n = closest(sc, fn, P[idx])
    print(f'  opaque in box {len(idx):,} (of {inbox.sum():,} in box, {len(P):,} total) · closest {time.time() - t0:.0f} s')

    # G0
    sub = {k: v[idx] for k, v in gs.items()}
    qo = g0_quat_order(sub, d, n, args)
    g0['quat_order'] = qo
    absmax = np.abs(np.r_[lo, hi]).max()
    g0['fp16_half_step_at_mesh_extent_m'] = float(np.spacing(np.float16(absmax)) / 2) if g0['precision_bits'] == 16 else 0.0
    g0['passed'] = bool(g0['xform_identity'] and np.allclose(g0['offset'], 0) and qo[qo['order']] >= 0.8 and qo['wxyz'] != qo['xyzw'])

    # A1 · A2
    surf = sub['S'].max(1) <= args.surf_smax
    near = d < args.near_m
    rng = np.random.default_rng(args.seed)
    fit = np.where(surf & near)[0]
    fit = rng.choice(fit, min(len(fit), 3_000_000), replace=False)
    a1 = a1_rigid(sub['P'][fit], s[fit], n[fit], args)
    a1_orient = by_orientation(s[surf & near], n[surf & near])
    print(f'  A1 t {np.round(a1["trans_m"], 4)} |t| {a1["trans_norm_m"]:.4f} m · rot {a1["rot_norm_deg"]:.4f}° · δ {a1["surface_offset_m"]:+.4f} m')
    a2, a2_vox = a2_local(sub['P'][surf & near], d[surf & near], s[surf & near], args)
    print(f'  A2 bad voxels {a2["bad_voxels"]}/{a2["n_voxels"]} ({100 * a2["bad_frac"]:.2f} %)')

    # A3
    from scipy.spatial import cKDTree
    storeys, st_src = storeys_from(nav01, mesh, args)
    a3sel = inbox & (gs['O'] >= 0.02)
    tree = cKDTree(P[a3sel])
    a3, (cpts, calpha, valpha) = a3_coverage(mesh, tree, gs['O'][a3sel], storeys, args)
    del tree
    print(f'  A3 mesh without GS: robot zone {100 * a3["uncovered_frac_robot_zone"]:.2f} % ({a3["uncovered_area_m2_robot_zone"]:.0f} m²) · '
          f'all {100 * a3["uncovered_frac_all"]:.2f} % · {time.time() - t0:.0f} s')

    # B 바닥 (A4 가 통행 구역을 쓰므로 먼저)
    R_all = quat_to_R(gs['Q'], qo['order'])
    holes, b_st, figs, crops, walk = [], [], {}, [], {}
    for si, S in enumerate(storeys):
        g, cls, zf, flipped = classify_floor(sc, fn, mesh, S, args)
        zfill = fill_floor_z(cls, zf)
        nav, zone, stairs = walkable_mask(nav01, g, S, args, storeys)
        if zone is not None:
            walk[si] = WalkInfo(g, zone, zfill)
        alpha, zgs, ng = gs_floor_alpha(gs, R_all, g, zfill, args)
        from scipy import ndimage
        # 바깥 = 격자 테두리와 이어진 뚫림/트임
        lab, _ = ndimage.label((cls == SEE) | (cls == OPEN) | (cls == UNDER), structure=np.ones((3, 3)))
        border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        outside = np.isin(lab, border[border > 0])
        enclosed = ((cls == SEE) | (cls == UNDER)) & ~outside
        cand, lab_c = components(enclosed, cls, zfill, g, zone, alpha, 'mesh', args, si)
        mesh_hole = np.zeros_like(enclosed)
        unscanned = np.zeros_like(enclosed)
        for i, h in enumerate(cand, 1):
            if h['ring_floor_frac'] >= 0.5:
                mesh_hole |= lab_c == i
            else:
                unscanned |= lab_c == i
                h['kind'] = 'unscanned'
        gs_hole = (cls == FLOOR) & (alpha < args.alpha_min)
        gh, _ = components(gs_hole, cls, zfill, g, zone, alpha, 'gs', args, si)
        hs = [h for h in cand if h['kind'] == 'mesh'] + gh
        for h in hs:
            big = h['inscribed_diam_m'] >= args.hole_min_diam
            h['flag'] = 'CRITICAL' if (big and h['on_walkable']) else 'MAJOR' if big else 'minor'
            if h['kind'] == 'mesh':
                h['kind'] = 'mesh (GS solid)' if h['gs_filled_frac'] >= 0.5 else 'mesh+GS'
        holes += hs
        # 통계
        fl = cls == FLOOR
        zone_ = zone if zone is not None else np.ones_like(fl)
        dz = zgs - zf
        ok = fl & zone_ & np.isfinite(dz)
        b = {'storey': si, 'z_range': S['z_range'], 'grid': list(g.shape), 'n_gauss_floorband': ng,
             'floor_m2': float(fl.sum() * g.c ** 2), 'flipped_floor_cells': flipped,
             'walk_zone_m2': float(zone.sum() * g.c ** 2) if zone is not None else None,
             'mesh_hole_m2': float(mesh_hole.sum() * g.c ** 2), 'mesh_hole_walk_m2': float((mesh_hole & zone_).sum() * g.c ** 2),
             'unscanned_m2': float(unscanned.sum() * g.c ** 2), 'under_roof_m2': float((cls == UNDER).sum() * g.c ** 2),
             'gs_hole_m2': float(gs_hole.sum() * g.c ** 2), 'gs_hole_walk_m2': float((gs_hole & zone_).sum() * g.c ** 2),
             'gs_hole_walk_frac': float((gs_hole & zone_).sum() / max((fl & zone_).sum(), 1)),
             'floor_dz_median_m': float(np.median(dz[ok])) if ok.any() else None,
             'floor_dz_abs_p95_m': float(np.percentile(np.abs(dz[ok]), 95)) if ok.any() else None,
             'n_mesh_holes': sum(h['kind'].startswith('mesh') for h in hs), 'n_gs_holes': len(gh),
             'critical': sum(h['flag'] == 'CRITICAL' for h in hs)}
        b_st.append(b)
        img = floor_rgb(cls, mesh_hole, unscanned, gs_hole, alpha, args)
        for h in hs:
            if h['flag'] != 'minor':
                h['_g'] = (g, img, alpha, zf)
        # 그림 id 는 아래에서 붙인 뒤 그린다
        b['_fig'] = (g, img, nav, dz, alpha, stairs)
        print(f'  B S{si}: floor {b["floor_m2"]:.0f} m² · mesh hole {b["mesh_hole_m2"]:.2f} m² (walk {b["mesh_hole_walk_m2"]:.2f}) · '
              f'GS hole walk {100 * b["gs_hole_walk_frac"]:.2f} % · dz med {b["floor_dz_median_m"]} · critical {b["critical"]} · {time.time() - t0:.0f} s')

    order = {'CRITICAL': 0, 'MAJOR': 1, 'minor': 2}
    holes.sort(key=lambda h: (order[h['flag']], -h['area_m2']))
    for i, h in enumerate(holes):
        h['id'] = f'H{i}'
    for si, b in enumerate(b_st):
        g, img, nav, dz, alpha, stairs = b.pop('_fig')
        figs[f'floor_S{si}'] = fig_floor(out, args.scene, si, storeys[si], g, img, nav, [h for h in holes if h['storey'] == si], args, stairs)
        figs[f'floor_offset_S{si}'] = fig_floor_offset(out, args.scene, si, storeys[si], g, dz, alpha, nav, args)
    crops = [(h, *h['_g']) for h in holes if '_g' in h][:12]
    figs['crops'] = fig_hole_crops(out, args.scene, crops, args)
    for h in holes:
        h.pop('_g', None)

    # 계단 (B1·B2 에 합산)
    sr = stair_check(sc, fn, gs, R_all, nav01, storeys, args)
    stair_stats, stair_holes = ([], []) if sr is None else sr[:2]
    if sr is not None:
        figs['stairs'] = fig_stairs(out, args.scene, stair_stats, sr[2], args)
        stair_holes.sort(key=lambda h: (h['flag'] != 'CRITICAL', -h['area_m2']))
        for i, h in enumerate(stair_holes):
            h['id'] = f'S{i}'
        for S_ in stair_stats:
            S_.pop('bbox')
        print('  stairs (-> B1·B2): ' + ' · '.join(f"{x['connector']}[{x['kind']}] tread {100 * x['tread_frac']:.1f}% mesh hole {x['mesh_hole_m2']:.2f} m² "
                                          f"GS hole {100 * x['gs_hole_frac']:.1f}%" for x in stair_stats))

    # A4
    a4, far = a4_off_mesh(sub['P'], sub['O'], sub['C'], d, storeys, walk, args)
    print(f'  A4 GS without mesh {100 * a4["far_frac"]:.2f} % · clusters {a4["n_clusters"]} (over walkable {a4["n_clusters_on_walkable"]})')

    # 그림 · 파일
    figs['hist'] = fig_hist(out, d, s, a1)
    figs.update({f'align_S{i}': f for i, f in enumerate(fig_align_topdown(out, args.scene, storeys, sub['P'][surf & near],
                                                                          d[surf & near], far, sub['P'][far], cpts, calpha, a4['clusters'], args))})
    figs['html3d'] = html3d(out, mesh, valpha, sub['P'][far], holes, args)
    import trimesh
    mc = mesh.copy()
    mc.visual = trimesh.visual.ColorVisuals(mc, vertex_colors=(colormap(valpha, 1.0, 'magma') * 255).astype(np.uint8))
    mc.export(out / 'mesh_gs_alpha.ply')
    write_ply(out / 'gs_without_mesh.ply', sub['P'][far], sub['C'][far])
    hp = [h for h in holes if h['flag'] != 'minor']
    plys = ['mesh_gs_alpha.ply (vertex color = A3 GS α, magma: 검정 0 = GS 없음 · 노랑 1)', 'gs_without_mesh.ply (GS color)']
    with open(out / 'holes.csv', 'w', newline='') as f:
        cols = ['id', 'flag', 'kind', 'storey', 'x', 'y', 'z', 'area_m2', 'inscribed_diam_m', 'ring_floor_frac', 'on_walkable',
                'gs_filled_frac', 'bbox_xy']
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for h in holes:
            w.writerow({c: h[c] for c in cols})
    with open(out / 'gs_without_mesh_clusters.csv', 'w', newline='') as f:
        cols = ['id', 'on_walkable', 'storey', 'n', 'center', 'min', 'max', 'size_m', 'color', 'median_d_m']
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for i, c in enumerate(a4['clusters']):
            w.writerow({'id': f'G{i}', **{k: c[k] for k in cols[1:]}})
    if hp:
        write_ply(out / 'floor_holes.ply', np.array([[h['x'], h['y'], h['z']] for h in hp]),
                  np.array([[1, 0, 0] if h['flag'] == 'CRITICAL' else [1, 0.5, 0] for h in hp], float))
        plys.append('floor_holes.ply (red = critical, orange = major)')

    # 판정 — 측정값은 한 번, 기준은 프로파일마다
    crit_mesh = [h for h in holes if h['flag'] == 'CRITICAL' and h['kind'].startswith('mesh')]
    gs_walk = sum(b['gs_hole_walk_m2'] for b in b_st)
    dzm = [b['floor_dz_median_m'] for b in b_st if b['floor_dz_median_m'] is not None]
    dzp = [b['floor_dz_abs_p95_m'] for b in b_st if b['floor_dz_abs_p95_m'] is not None]
    gs_frac = max([b['gs_hole_walk_frac'] for b in b_st] or [0])
    gst = [x for x in stair_stats if x['gated']]               # 계단 = 높이 폭 ≥ 1 m 조각
    gst_ids = [x['connector'] for x in gst]
    st_big = [h for h in stair_holes if h['kind'] == 'stair mesh' and h['flag'] == 'CRITICAL' and h['connector'] in gst_ids]
    st_area = sum(x['area_m2'] for x in gst)
    st_mh = sum(x['mesh_hole_m2'] for x in gst)
    st_tread = sum(x['area_m2'] * x['tread_frac'] for x in gst)
    st_gs_frac = sum(x['area_m2'] * x['tread_frac'] * x['gs_hole_frac'] for x in gst) / st_tread if st_tread else 0.0
    mh_walk = sum(b['mesh_hole_walk_m2'] for b in b_st)
    walk_m2 = sum(b['walk_zone_m2'] or 0 for b in b_st)
    mh_frac = (mh_walk + st_mh) / (walk_m2 + st_area) if (walk_m2 + st_area) else 0.0
    n_big = len(crit_mesh) + len(st_big)

    def judge(k, c):
        if k == 'G0':
            return bool(g0['passed'])
        elif k == 'A1':
            return bool(a1['trans_norm_m'] <= c['A1_t'] and a1['rot_norm_deg'] <= c['A1_rot'])
        elif k == 'A1b':
            return bool(abs(a1['surface_offset_m']) <= c['A1b'])
        elif k == 'A2':
            return bool(a2['bad_frac'] <= c['A2'])
        elif k == 'A3':
            return bool(a3['uncovered_frac_robot_zone'] <= c['A3'])
        elif k == 'B1':
            if c['B1_mode'] == 'count':
                return n_big <= c['B1']
            elif c['B1_mode'] == 'area_frac':
                return bool(mh_frac <= c['B1'])
            else:
                assert False, f"unreachable B1_mode={c['B1_mode']!r}"
        elif k == 'B2':
            return bool(gs_frac <= c['B2'] and st_gs_frac <= c['B2'])
        elif k == 'B3':
            return bool(dzm and max(abs(x) for x in dzm) <= c['B3_med'] and max(dzp) <= c['B3_p95'])
        else:
            assert False, f"unreachable k={k!r}"

    values = {
        'G0': ('G0 GS 좌표·쿼터니언', f"xform 항등 {g0['xform_identity']} · |cos(짧은 축, 법선)| 중앙 wxyz {qo['wxyz']:.3f} / xyzw {qo['xyzw']:.3f} -> {qo['order']}"),
        'A1': ('A1 강체 어긋남', f"|t| {a1['trans_norm_m']:.4f} m · rot {a1['rot_norm_deg']:.4f}°"),
        'A1b': ('A1b 표면 간격 δ', f"{a1['surface_offset_m']:+.4f} m (+ = GS 가 mesh 뒤)"),
        'A2': ('A2 국소 어긋남 복셀', f"{a2['bad_voxels']}/{a2['n_voxels']} ({100 * a2['bad_frac']:.2f} %)"),
        'A3': ('A3 GS 없는 mesh (로봇 시야 높이)', f"{100 * a3['uncovered_frac_robot_zone']:.2f} % ({a3['uncovered_area_m2_robot_zone']:.0f} m²) · 전체 {100 * a3['uncovered_frac_all']:.2f} %"),
        'B1': ('B1 바닥·계단 mesh 구멍', f"내접 ≥ 0.10 m 구멍 {n_big} 개 (바닥 {len(crit_mesh)} · 계단 {len(st_big)}) · "
                                       f"면적 {mh_walk + st_mh:.2f} m² / {walk_m2 + st_area:.0f} m² ({100 * mh_frac:.2f} %; 바닥 {mh_walk:.2f} · 계단 {st_mh:.2f})"),
        'B2': ('B2 바닥·계단 GS 구멍', f"바닥 층 최대 {100 * gs_frac:.2f} % (합 {gs_walk:.2f} m²) · 계단 {100 * st_gs_frac:.2f} % "
                                     f"(디딤판 {st_tread:.0f} m², 조각 {len(gst)} 개 중 최대 {100 * max([x['gs_hole_frac'] for x in gst] or [0]):.1f} %)"),
        'B3': ('B3 바닥 높이 차 (GS − mesh)', f"층별 중앙 {[round(x, 3) for x in dzm]} · p95|차| {[round(x, 3) for x in dzp]}"),
    }
    gates = {}
    for k, (name, val) in values.items():
        st_, rl_ = judge(k, CRITERIA['strict']), judge(k, CRITERIA['relaxed'])
        gates[name] = {'id': k, 'passed': judge(k, CRITERIA[args.criteria]), 'strict': st_, 'relaxed': rl_, 'value': val,
                       'criterion': criterion_text(k, CRITERIA[args.criteria]),
                       'criterion_strict': criterion_text(k, CRITERIA['strict']), 'criterion_relaxed': criterion_text(k, CRITERIA['relaxed'])}
    for name, val, crit in [
            ('A4 mesh 없는 GS (통행 구역 위 덩어리)', f"불투명의 {100 * a4['far_frac']:.2f} % · 덩어리 {a4['n_clusters']} · 통행 구역 위 {a4['n_clusters_on_walkable']}", '보고 — 목록 확인'),
            ('G0c 무게중심 거리 (참고)', f"GS 전체 {cent['dist_gs_all']:.2f} m · opacity 가중 {cent['dist_gs_opacity_weighted']:.2f} m · mesh 상자 안 {cent['dist_gs_in_mesh_box_opacity_weighted']:.2f} m",
             '보고 (GS 가 mesh 밖까지 퍼져 판정 불가)')]:
        gates[name] = {'id': name.split()[0], 'passed': '보고', 'strict': '보고', 'relaxed': '보고', 'value': val, 'criterion': crit,
                       'criterion_strict': crit, 'criterion_relaxed': crit}
    res = {'scene': args.scene, 'criteria': args.criteria, 'criteria_profiles': CRITERIA, 'inputs': {'usdz': usdz, 'nav01': nav01, 'storeys_from': st_src}, 'args': vars(args), 'g0': g0,
           'a1': a1, 'a1_orient': a1_orient, 'a2': a2, 'a3': a3, 'a4': a4, 'b_storeys': b_st,
           'holes': holes, 'stairs': stair_stats, 'stair_holes': stair_holes, 'gates': gates, 'plys': plys, 'figures': figs, 'elapsed_s': time.time() - t0}
    json.dump(res, open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False, default=str)
    write_report(out, args.scene, res, figs)
    print(f'  gates [{args.criteria}]: ' + ' · '.join(f"{v['id']} {v['passed']}" for v in gates.values()))
    print('  strict / relaxed: ' + ' · '.join(f"{v['id']} {v['strict']}/{v['relaxed']}" for v in gates.values() if v['strict'] != '보고'))
    print(f'  -> {out}/report.html · summary.json · holes.csv  ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
