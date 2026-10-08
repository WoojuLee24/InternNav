"""모든 stage 리포트를 **목적·결론·한계**를 붙여 한 번에 재발행한다.

## 왜 스크립트로 두나
`publish_artifact_report.py`는 범용 발행기라 stage별 문장을 모른다. 그 문장을 커밋되지 않는 셸 히스토리에
두면 재발행할 때마다 다시 쓰게 되고, 그러면 결론·한계가 리포트마다 어긋난다. 여기 한 곳에 모아둔다.

## 규칙 (실제로 겪은 문제에서 나온 것)
- **수치는 `실험 결과`, 결론은 한 문장.** 섞으면 결론이 표에 묻히고, 근거 없는 결론은 검증이 안 된다.
- **한계를 비우지 않는다.** 안 쓰면 읽는 사람이 결과를 실제보다 넓게 해석한다.
- 재실행이 필요한(수치가 낡은) 리포트는 한계에 **그 사실을 적는다.**

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/publish_all.py [--only s1_navmesh,perf]
"""

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PUB = _HERE / 'publish_artifact_report.py'
LOGS = Path('logs/embodiment_augment')

# stage -> (title, sub, purpose, conclusion, limits, legend[, result])
# `result`(7번째)는 선택 — 있으면 '실험 결과' 카드로 따로 나가고, 결론은 한 문장만 남는다.
# `legend`는 발행기의 공용 `FIGURE_LEGEND` 키 콤마열 — 그림 읽는 법을 카드로 한 번만 낸다.
REPORTS = {
    'g1': (
        '#00 · 카메라를 집 3D 좌표에 놓을 수 있나',
        's8pcmisQ38h · ep0 · 125cm_0deg · 6프레임 — 저장 depth vs Open3D 렌더 (구 G1)',
        '이 파이프라인은 카메라 pose·intrinsics·좌표 규약이 맞는지 확인한다. 이를 위해 <b>Open3D '
        'renderer</b>로 <code>data/scene_data/mp3d_n1</code>의 <code>obj</code> 에셋을 로드해서 depth를 '
        'rendering하고 <b>저장된 depth와 비교</b>한다.',
        '카메라 pose·intrinsics·좌표 규약이 맞다.',
        '남은 꼬리는 <b>실루엣(깊이 급변) 픽셀</b>이다 — 정합 오차가 아니라 픽셀 귀속 문제이고, 단일 '
        '픽셀 max가 수 m까지 튄다(mesh에 없는 거울·창으로 추정하나 <b>원인을 특정하지 않았다</b>). '
        '<b>RGB는 비교하지 않았다</b>(v1은 depth/BEV만). habitat 렌더러와의 교차 비교는 이 리포트 밖이다 '
        '— <b>#02</b>에서 따로 측정했다. 씬 2채 · 에피소드 1개 · 6프레임.',
        'depth,err',
        '정합 residual <b>0.0003 m</b>(통과선 0.01), 렌더 depth vs 저장 depth <b>median 0.0005 m</b> · '
        'p90 0.0009 m · |err|&gt;10 cm 픽셀 <b>0.04%</b> → 좌표 규약·하드코딩 K(fx 388.19)·pose 해석이 '
        '모두 맞다. 17DRP에서도 median이 같은 0.0005 m라 <b>씬에 의존하지 않는다</b>. '
        '이전 판의 <code>Z_OFFSET_M=0.20</code>은 fudge였고 0.0으로 정정하니 median 0.0035→0.0005 m로 떨어졌다.'),
    'depth_sources': (
        '#02 · depth 세 소스 비교 — 저장 / 우리 / habitat',
        's8pcmisQ38h · ep0 · 125cm_0deg · 6프레임 · 범위 5·10·15·20 m (구 G1c)',
        '이 리포트는 우리 렌더러가 <b>데이터셋을 만든 habitat 렌더러와 같은 depth를 내는지</b> 확인한다. '
        '우리는 Open3D로 <code>mp3d_n1/*.obj</code>를 렌더하고 데이터셋은 habitat이 '
        '<code>mp3d_ce/*.glb</code>를 렌더했으므로 <b>에셋도 렌더러도 다르다</b>. #00은 "저장 ↔ 우리"만 '
        '봤다. 이를 위해 같은 pose에서 세 소스를 <b>5·10·15·20 m</b> 범위별로 비교한다.',
        '에셋과 렌더러가 달라도 지오메트리는 같다. 그리고 <b>#00이 보고한 잔차는 우리 오차가 아니라 '
        '저장 포맷의 버림이다.</b>',
        '씬 1채 · 에피소드 1개 · 6프레임 — 씬마다 에셋 품질이 다를 수 있다. '
        '<b>RGB는 비교하지 않았다</b>(v1은 depth/BEV만). max가 수 m로 튀는 픽셀이 남는데 #00이 꼬리로 '
        '보고한 <b>실루엣·거울/창</b> 픽셀로 보이나 <b>원인을 특정하지 않았다</b>. '
        '저장 depth와의 마스크 불일치 1.8~2.0%도 원인을 나누지 않았다(저장 0값 = 측정 없음 vs 렌더 배경).',
        'depth,err',
        '<b>우리 렌더 ↔ habitat 렌더가 median 0.00~0.01 mm</b> · p90 0.03~0.05 mm로 사실상 동일하다. '
        '<b>범위를 5→20 m로 넓혀도 median이 커지지 않는다</b>. 마스크 불일치 0.00~0.02%. '
        '두 렌더 모두 저장 depth와는 median 0.50 mm 차이인데, 부호가 <b>99.3~100% 양수</b>이고 분포가 '
        '0~+1 mm 균일하다 → <b>저장 포맷의 버림(truncation)</b>이다(round면 ±0.5로 대칭이어야 한다).'),
    's1_navmesh': (
        '#03 · 데이터셋이 쓴 지도에 두 경로를 올린다',
        '17DRP5sb8fy 14 + s8pcmisQ38h 9 에피소드 · 125cm_0deg · r_b=0.10 고정 (구 S1)',
        '이 폴더는 <b>로봇 크기를 바꿨을 때 정답 경로를 다시 그려주는 코드</b>다. 그러면 순서가 정해진다 '
        '— <b>큰 로봇용을 새로 그리기 전에, 원래 크기에서 원래 정답을 그대로 뽑아낼 수 있는지 먼저 '
        '확인해야 한다.</b> 게다가 우리 지도에서 원본 정답이 "벽을 뚫는" 증상이 보고돼 있었다. '
        '이를 위해 배포된 <code>.navmesh</code>(= 데이터셋이 실제로 쓴 지도)에 <b>ⓐ 원본 정답</b>과 '
        '<b>ⓑ waypoint를 <code>find_path</code>로 이어 만든 경로</b>를 같이 올려, 경로를 5 cm 간격으로 '
        '채워 점마다 "여기 서 있을 수 있나"를 묻는다.',
        '<b>"벽을 뚫는" 문제는 데이터셋의 지도에서 재현되지 않는다.</b> 17DRP는 완전 정합(위반 0), '
        's8은 2450점 중 14점(0.57%)이 벗어나는데 <b>깊이가 최대 3.7 cm로 격자 한 칸(5 cm)보다 얕다</b> '
        '— 관통이 아니라 경계에 걸친 이산화 오차다. <b>원 증상은 우리 쪽 문제였다</b>(#04 / #05).',
        '⚠️ <b>이전 판의 결론을 철회했다.</b> "GT가 경계를 여유 0으로 스친다"는 '
        '<b><code>clearance min</code>(최악의 한 점)을 전형값으로 읽은 오류</b>였고, 이번 판에서는 '
        '<code>clearance</code> 분석 자체를 지웠다 — navmesh는 이미 반경 0.10만큼 깎인 지도라 그 위에서 '
        '"여유 ≥ 반경"을 또 요구하면 <b>반경을 두 번 센다</b>. · 씬 2채 · <b>계단 에피소드 제외</b>'
        '(s8은 14개 중 5개) — <code>get_topdown_view</code>가 단일 높이 절단이라 다층 경로를 그릴 수 '
        '없다. · <code>is_navigable</code>은 <b>수평 1 cm · 수직 0.5 m 맹점</b>이 있다. · '
        '<b>우리 플래너와 우리 occ 맵은 이 리포트 범위 밖</b>이다(#04 / #05). · 게이트 D는 자기가 쓰는 '
        'r_b&gt;0.10 지도가 옳은지 검증하지 않는다(#05 W6의 일).',
        # `path`를 뺐다 — 그 공용 범례는 "흰=GT · 주황=우회지점"인데 #03은 "노랑=GT · 주황=waypoint 연결"이다.
        # #03의 summary가 자체 범례를 싣는다. `path`는 #05 등 6개 stage가 공유하므로 수정하지 않는다.
        'floorplan',
        '배포된 <code>.navmesh</code>는 <b>habitat 기본 설정 그대로</b>다(<code>agent_radius</code> 0.10 · '
        '<code>agent_height</code> 1.50 · <code>agent_max_climb</code> 0.20) → <b>"원본 파라미터를 '
        '따른다"는 자유 파라미터가 0개</b>라는 뜻이다. waypoint <b>474/474 · 531/531</b> 유효, '
        'leg <code>find_path</code>가 다음 waypoint 0.5 m 안 도달 <b>69/69 · 47/47 = 100%</b>'
        '(논문 자신의 기준), 5 cm로 채운 경로 점 <b>ⓐ 100% / 99.43% · ⓑ 100% / 100%</b> 유효. '
        '<b>"뚫는다"의 분해</b>: J1(habitat 직접 답) 밖 <b>0 / 14점</b>(최대 깊이 <b>3.7 cm</b> &lt; 한 칸) · '
        'J1 통과인데 J2(5 cm 래스터) 밖 <b>0 / 58점이고 전부 깊이 정확히 1칸</b> → 남은 불일치는 '
        '<b>격자 반올림</b>이다(2칸 이상이 하나라도 나오면 반증인데, 나오지 않았다). '
        '⚠️ <b>ⓑ의 재현 오차는 코너 절단만이 아니다</b> — 17DRP median 12.9 cm는 코너 절단이지만 '
        's8 median <b>56.4 cm</b>는 <b>측지 최단선이 장애물의 반대편으로 도는 것</b>이고, 그때도 '
        '<b>길이비는 1.002</b>다. 즉 <b>길이비 ≈ 1은 "같은 경로"의 증거가 아니다.</b> → 회랑 중심선은 '
        '<code>find_path</code>가 아니라 <b>GT 서브궤적이어야 한다</b>. '
        '<code>recompute_navmesh</code> 재현 오차 0.02% / 0.15%.'),
    's2_mapcal': (
        '#04 · 우리 지도가 habitat 지도와 같은 판정을 내리나',
        '17DRP5sb8fy 14 + s8pcmisQ38h 9 에피소드 · 125cm_0deg · r_b=0.10 고정 (구 S2)',
        '학습 중에 정답 경로를 즉석에서 다시 그리려면 <b>habitat 없이 도는 지도</b>가 필요하다'
        '(Isaac/VLN-PE도 같은 지도를 쓴다). #03이 기준(데이터셋의 지도)을 세웠으니, 우리가 3D 스캔으로 '
        '만든 지도가 그 기준과 <b>같은 판정을 내리는지</b>, 다르면 <b>왜 다른지</b>, 그리고 <b>어떻게 '
        '맞출 수 있는지</b>를 본다. 값은 전부 배포 navmesh 직독(자유 파라미터 0개). 그 위에 <b>ⓐ 원본 '
        '정답</b>과 <b>ⓑ pathfollower(0.25 m/15° 이산 액션)로 이은 경로</b>를 올려 정답이 성립하는지도 '
        '확인한다.',
        '<b>두 지도가 다른 이유는 질문의 구조다</b> — 밴드는 "장애물이 없나", recast는 "걸을 수 있는 '
        '바닥이 있나". 다층 씬 거짓 승인의 <b>66.6%가 "바닥 없음"</b>이었다. 질문을 바꾼 '
        '<b>recast-like v2</b>(칸마다 바닥 + despike + 시작점 연결성)가 s8을 <b>75.83 → 92.30%</b>로 '
        '올리고 GT·follower를 안전하게 유지한다. → <b>default는 navmesh 래스터화 유지</b>(정의상 100% '
        '캐시가 이미 있다), v2는 <b>navmesh가 없는 씬용 최선 후보</b>.',
        '⚠️ <b>게이트 2개 실패.</b> <b>C</b>(밴드 거짓 승인의 F+R 설명률 ≥90%): 86.1/84.4% — S 버킷 '
        '13~15%(recast의 <code>region_min_size</code>·ledge·도달불가 island) 미해명. <b>G1</b>(v2의 F '
        '소멸): s8 잔여 F 745 > 기준 200 — 남은 축은 <code>.house</code> region별 바닥높이. · 씬 2채 · '
        '<b>계단 에피소드 제외</b>(s8 5/14) — 단일 높이 절단의 한계는 어떤 후보도 못 고친다. · 비교는 '
        '<b>GT 주변 2 m 안</b>만. · follower <code>goal_radius</code>(0.25 m)는 데이터셋 생성값 미상. · '
        '<b>판정 기준이 여전히 habitat</b>(상한이 habitat). · 이전 판의 <code>floor_exists</code> 후보·'
        '두께 스윕·게이트 D는 <b>v2로 대체돼 삭제</b>(기록: reports.md #04).',
        'floorplan,diffmap',
        '<b>(1) 원인 분해</b>: 밴드 맵 거짓 승인 중 s8 <b>66.6%가 F(바닥 없음)</b>, 17DRP는 <b>R(경계 '
        '1칸 = 격자 반올림, 불가침)이 80%</b> → 다층 씬 문제라는 예측과 일치. '
        '<b>(2) recast-like v2</b>: 칸마다 바닥(창 0.5 m) + 자기 바닥 앵커 headroom + 단차 0.2 m + '
        '<b>despike</b>(스캔 구멍 kf 보간 — 가짜 seam 제거) + <b>시작점 연결성</b>("바닥처럼 생긴 면" '
        '탈락) = s8 <b>92.30%</b>(모든 후보 중 최고) · 17DRP 96.23% · GT 이탈 <b>0/0</b> · follower 이탈 '
        '58/15점 @ ≤1칸 · 거짓 기각 895. '
        '<b>(3) pathfollower</b>: 코너는 크게 개선(s8 재현 56.4→8.8 cm)하지만 분포가 두 덩어리 — '
        's8 9개 중 4개는 <b>경로가 장애물 반대편으로 갈라진다</b>(이산 액션도 못 고침) → 회랑 중심선은 '
        'GT 서브궤적이어야 한다. '
        '<b>(4)</b> GT가 우리 지도를 벗어난 점 <b>0</b> — "정답이 우리 지도를 뚫는다"는 원 증상은 재현되지 '
        '않았다. 좌표 정합 사전 게이트 <b>1.0000</b>.'),
    'waypoints_17drp_occ': (
        '#05 · 원본 우선, 못 지나갈 때만 보정 — 17DRP · occ',
        '17DRP5sb8fy · 14 에피소드 · 125cm_0deg · 밴드 h_nav 0.20 / h_obs 1.50 (구 W)',
        '정답 경로는 사람이 쓴 문장과 짝이라 이유 없이 흔들면 안 된다. 이 리포트는 <b>"지나갈 수 있으면 '
        '원본 그대로, 못 지나갈 때만 밀어내기 → 우회 → 멈춤"</b> 사다리가 성립하는지 확인한다. 이를 위해 '
        '<code>r_b</code>를 키우며 구간(leg)별로 원본을 그대로 썼는지(<b>W5</b>)와 habitat이 막은 구간을 '
        '우리가 승인했는지(<b>W6</b>)를 센다.',
        '사다리가 성립한다 — 기본 반경에서는 <b>원본을 한 점도 안 건드리고</b>, 보정은 반경이 커질 때만 '
        '걸린다.',
        '<b>거짓 기각 47건</b>이 남는다 — 구간을 각각 독립으로 계획해 이음새가 생기고, 첫 막힘에서 경로를 '
        '끊기 때문이다(habitat은 구간을 독립 판정). 이전 구간 끝점에서 이어 계획하면 줄어들 것이나 미구현. '
        '그리고 <b>실제 학습 코드에 붙이지 않았고 모델 성능은 측정하지 않았다</b>.',
        'floorplan,path',
        '<b>W5 identity PASS</b> — r_b=0.1에서 <b>69/69 구간 전부 원본 그대로</b>(<code>none</code>), '
        '보정한 구간 <b>0개</b>, 원본과의 거리 <b>0.00 cm</b>. 보정은 필요할 때만 걸린다(r_b=0.2에서 '
        '밀어내기 2, r_b=0.3에서 우회 5). <b>W6 거짓 승인 2</b>/276 — r_b=0.30에서 2건.'),
    'waypoints_occ': (
        '#05 · 원본 우선, 못 지나갈 때만 보정 — s8 · occ',
        's8pcmisQ38h · 14 에피소드 · 125cm_0deg · 밴드 h_nav 0.20 / h_obs 1.50 (구 W)',
        '위와 같은 게이트를 <b>두 번째 씬</b>(더 크고 다층)에서 확인한다 — 사다리가 씬 하나의 우연이 '
        '아닌지 보기 위해서다.',
        '두 번째 씬에서도 사다리가 성립한다. 그리고 이 씬에서는 <b>밴드 지도(occ)가 navmesh 래스터화보다 '
        '낫다</b> — #04가 잰 1셀 양자화가 여기서 실제로 손해를 낸다.',
        '계단 에피소드 <b>14개 중 5개(36%)를 제외</b>했다 — 단일 바닥 높이로 2D 투영이 안 되기 때문이다. '
        '거짓 기각 45건은 위와 같은 원인. 학습·성능 미측정.',
        'floorplan,path',
        '<b>W5 identity PASS</b> — <b>47/47 구간 전부 원본 그대로</b>, 보정 <b>0개</b>, 원본과의 거리 '
        '<b>0.00 cm</b>. <b>W6 거짓 승인 0</b>/188. 같은 씬을 navmesh 래스터화로 돌리면 격자 1셀 오차로 '
        '<b>5개 구간</b>을 건드린다.'),
    'waypoints_17drp': (
        '#05 · 원본 우선, 못 지나갈 때만 보정 — 17DRP · navmesh',
        '17DRP5sb8fy · 14 에피소드 · 125cm_0deg · navmesh 래스터화 (구 W)',
        '같은 게이트를 <b>지도만 바꿔</b> 돌린다. occ 옵션과 나란히 두어 "habitat 판정과 100% 일치시키면 '
        '무엇을 얻고 무엇을 잃는가"를 수치로 본다.',
        'habitat 판정과 <b>완전히 일치</b>하는 대신 <b>쓸 수 있는 샘플을 더 버린다</b> — 안전과 수량의 '
        '교환이고, 어느 쪽을 쓸지는 씬에 따라 다르다.',
        '<b>habitat에 묶인다</b> — offline 캐시(21 KB/맵)라 worker는 habitat 없이 돌지만 Isaac/VLN-PE는 '
        '별도 지도가 필요하다. 그리고 지도가 단일 높이 절단이라 다층 씬에서 에피소드별 바닥 높이가 '
        '정확해야 한다.',
        'floorplan,path',
        '<b>거짓 승인 0</b>/276 (occ는 2건) → habitat 판정과 완전 일치. W5도 PASS(<b>69/69</b> · '
        '0.00 cm). 대가는 <b>거짓 기각 58건</b>(occ 47).'),
    'waypoints': (
        '#05 · 원본 우선, 못 지나갈 때만 보정 — s8 · navmesh',
        's8pcmisQ38h · 14 에피소드 · 125cm_0deg · navmesh 래스터화 (구 W)',
        '위와 같은 비교를 <b>큰 다층 씬</b>에서. 래스터화는 연속 지도를 5 cm 격자로 자르므로, 그 '
        '<b>격자 양자화</b>가 어디서 문제를 만드는지 본다.',
        '큰 씬에서는 래스터화의 <b>1셀 양자화가 5개 구간을 건드려 W5가 CHECK</b>다. 원인이 #04가 잰 '
        '1셀 오차와 <b>같은 것</b>이라 새 문제는 아니다.',
        '기준을 느슨하게 해서 100%로 만들지 <b>않았다</b> — 1셀 오차의 출처를 아는 것이 낫다고 판단. '
        '계단 에피소드 제외, 학습·성능 미측정.',
        'floorplan,path',
        '<b>거짓 승인 0</b>/188이지만 <b>W5는 CHECK</b> — <b>37/42</b>만 원본 그대로이고 5개 구간을 '
        '건드렸다(원본과의 거리 0.58 cm). 실패 5건은 <b>전부 "지도 밖"</b>이고, #04가 정량화한 66/8459점 · '
        '<b>최대 깊이 1셀(0.050 m)</b>과 같은 원인이다. 2건은 밀어내기가 흡수했다.'),
    'g2': (
        '#05b · 로봇 크기가 정말 경로를 바꾸나',
        '17DRP5sb8fy · 125cm_0deg · r_b 스윕 경로 오버레이 (구 G2)',
        '이 작업의 전제 — <b>"반경만 바꿔도 정답 경로가 달라진다"</b> — 가 사실인지 확인한다. 거짓이면 '
        '이 파이프라인 자체가 의미가 없다. 이를 위해 같은 에피소드에서 <code>r_b</code>만 바꿔 계획한 '
        '경로를 겹쳐 그리고 성공 개수를 센다.',
        '<b>달라진다 — 전제가 성립한다.</b> 덤으로 이 리포트에서 장애물을 관통하는 생성 경로를 발견해 '
        '원인을 제거했다.',
        '여기 경로는 <b>start→goal 순수 A*</b>다 — 사람이 쓴 문장을 모르므로 다른 방으로 돌아버릴 수 있다. '
        '실제로 <code>reference_path</code>가 최단거리보다 긴 에피소드가 절반 가까이다. <b>#05가 이 방식을 '
        '대체했다</b> — 이 리포트는 "크기가 경로를 바꾼다"는 전제 확인용으로만 인용해라.',
        'floorplan,path',
        '계획 성공이 <code>r_b</code>에 <b>단조 감소</b>한다(0.1→6/7, 0.18→4/7, 0.35→0/7). 그리고 '
        '<b>장애물 관통 경로 4/7을 발견</b>해 원인(coarse 격자 A*의 셀 중심 웨이포인트)을 제거하고 '
        '라벨 유효성 4중 기각을 넣었다.'),
    'modes': (
        '#06b · 로봇 크기를 관측에 넣는 두 방식',
        '17DRP5sb8fy · ep0 · 125cm_0deg — (A) 카메라 따라감 vs (B) 고정 + BEV 팽창',
        '"반경만 바꿨는데 왜 관측이 변하나"에 답이 두 가지다. 이 리포트는 어느 쪽이 <b>"같은 장면 × 다른 '
        '크기" 비교를 성립시키는지</b> 고른다. 이를 위해 (A) 카메라를 로봇 크기에 따라 옮기는 방식과 '
        '(B) 카메라를 고정하고 BEV 장애물만 팽창시키는 방식을 같은 프레임에서 나란히 렌더한다.',
        '<b>(B) 카메라 고정 + BEV 장애물 팽창을 채택한다.</b> (A)는 관측 차이를 "위치가 달라서"로도 '
        '설명할 수 있어 비교 자체가 성립하지 않는다.',
        '<b>수치가 낡았다</b> — <code>Z_OFFSET_M</code> 정정(0.20→0.0) 이전에 측정해 바닥 높이가 0.2 m '
        '어긋난 상태다. 재실행 대기. 그리고 BEV 점유율만 봤고 <b>학습 효과는 측정하지 않았다</b>.',
        'depth,bev',
        '(B)에서 BEV 점유가 <b>7.6%→10.2%</b>로 커져 큰 로봇에게 좁은 틈이 막힌 것으로 보인다. '
        '(A)는 카메라가 함께 움직여 같은 지점의 관측이 두 요인(크기·위치)으로 동시에 변한다.'),
    'pixelgoal_shift': (
        '#07 · 화면 목표점 조정 — shift',
        '17DRP5sb8fy · ep0 · 125cm_0deg — 진행 방향 수직으로만 최소 이동 (채택)',
        '모델이 실제로 배우는 라벨은 <b>"화면에서 어디를 향해 갈까"</b>다. 로봇이 커져 그 지점 자체가 '
        '막히면 지점을 옮겨야 하는데, 이 리포트는 <b>어떻게 옮기면 목적지 의미가 유지되는지</b> 고른다. '
        '이를 위해 원본 재현, 재투영 오차, 시야 이탈, <code>r_b</code> 단조성, 도달 오차를 잰다.',
        '<b>shift를 채택한다</b> — 원본을 정확히 재현하고 <code>r_b</code>에 단조다. 진행 방향 수직으로만 '
        '밀어 틈 중앙에 정렬하므로 목적지를 유지하며 통과한다.',
        '이 경로는 <b>사람 주석 waypoint를 쓰지 않는다</b>(순수 A*) — #05의 방식이 아직 이쪽에 적용되지 '
        '않았다. 그리고 <code>Z_OFFSET_M</code> 정정 이전 수치라 <b>재실행 대기</b>다.',
        'depth,floorplan',
        '원본 재현 <b>0.00 px</b>(26/26), 재투영 오차 2.25 px, 좌우 시야 이탈 <b>0.0%</b>, '
        '<code>r_b</code>에 <b>단조</b>, 도달 오차 <b>2.6 cm</b>.'),
    'pixelgoal_retreat': (
        '#07 · 화면 목표점 조정 — retreat',
        '17DRP5sb8fy · ep0 · 125cm_0deg — 경로를 따라 뒤로 물러남 (shift의 폴백)',
        'shift가 불가능할 때의 대안을 잰다. 이 리포트는 <b>"큰 로봇은 덜 간다"는 해석이 자연스러운지</b>를 '
        '같은 지표로 확인한다.',
        '<b>shift가 실패할 때의 폴백으로 채택한다.</b> 방향과 경로 의미는 유지되지만 목적지가 뒤로 밀린다.',
        'shift와 같은 한계 — 사람 주석 미적용, <code>Z_OFFSET_M</code> 정정 이전 수치로 재실행 대기. '
        '목적지가 밀리면 "…에서 멈춰라" 문장과 어긋날 수 있는데 그 영향은 측정하지 않았다.',
        'depth,floorplan',
        '<code>r_b</code>에 <b>단조</b>이고 도달 오차 <b>4.6 cm</b>(shift 2.6 cm).'),
    'pixelgoal_nearest': (
        '#07 · 화면 목표점 조정 — nearest (대조군)',
        '17DRP5sb8fy · ep0 · 125cm_0deg — 방향 무관 최소 변위 (비단조라 기각)',
        '<b>"가장 가까운 안전한 점으로 옮기면 되지 않나"</b>라는 가장 단순한 안을 <b>대조군</b>으로 둔다 — '
        'shift/retreat가 왜 필요한지 보이려면 단순한 안이 실패하는 지점을 수치로 남겨야 한다.',
        '<b>기각한다</b> — 보존율은 가장 높지만 <code>r_b</code>에 단조가 아니다. 로봇이 커졌는데 목표점이 '
        '더 가까워지지 않는 경우가 생긴다.',
        '대조군이므로 채택 대상이 아니다. 같은 재실행 대기(<code>Z_OFFSET_M</code>).',
        'depth,floorplan',
        '보존율은 세 방식 중 <b>가장 높지만</b> <code>r_b</code>에 <b>단조가 아니다</b>. '
        '도달 오차 <b>7.1 cm</b>(shift 2.6 · retreat 4.6).'),
    'perf': (
        '#08 · 학습을 느리게 만들지 않나',
        's8pcmisQ38h · T=12 · num_workers/batch 스케일링 (구 R2)',
        '즉석 생성이 GPU를 기다리게 만들면 이 방식 자체를 쓸 수 없다. 이 리포트는 <b>실제 학습 조건</b>'
        '(T=12 · num_workers 4 · batch 16)에서 sample 1개 비용과 worker 스케일링을 측정한다.',
        'batch 16을 <b>1.01 s</b>에 준비한다. 다만 <b>GPU step을 재지 않았으므로 "병목이 아니다"라고 '
        '말할 수는 없다</b> — 이 리포트는 비용만 확정했다.',
        '⚠️ <b>GPU step 시간을 측정하지 않아 병목 여부는 미결이다.</b> "1.01 s면 충분하다"고 말할 근거가 '
        '없다 — GPU step이 1 s보다 짧으면 병목이 된다. 실제 학습 루프에서 재야 답이 나온다(P-B2 이후). '
        '씬 1채·하드웨어 1종(24 CPU)만.',
        'chart',
        'sample 1개 <b>390 ms</b>(지배 요인은 Open3D 렌더 <b>339 ms</b>). 4 workers에서 '
        '<b>15.8 samples/s</b> → batch 16 준비 <b>1.01 s</b>. 부모에서 렌더러를 만들면 fork 후 worker가 '
        '데드락하고, worker당 <code>torch.set_num_threads(1)</code>이 필요하다(둘 다 실측).'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='콤마열 stage만 발행 (기본: 전부)')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    want = [s.strip() for s in args.only.split(',') if s.strip()] or list(REPORTS)

    ok, skip, fail = [], [], []
    for stage in want:
        if stage not in REPORTS:
            fail.append((stage, 'REPORTS에 없음')); continue
        d = LOGS / stage
        if not d.is_dir():
            skip.append(stage); continue
        title, sub, purpose, concl, limits, legend, *rest = REPORTS[stage]
        result = rest[0] if rest else ''
        cmd = [sys.executable, str(PUB), '--stage_dir', str(d), '--title', title, '--sub', sub,
               '--purpose', purpose, '--result', result, '--conclusion', concl,
               '--limits', limits, '--legend', legend]
        if args.dry_run:
            print(f'[dry] {stage}'); ok.append(stage); continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        (ok if r.returncode == 0 else fail).append(stage if r.returncode == 0
                                                   else (stage, r.stderr.strip()[-200:]))
    print(f'발행 {len(ok)} · 산출물 없음 {len(skip)} · 실패 {len(fail)}')
    if skip:
        print('  없음:', ', '.join(skip))
    for f in fail:
        print('  실패:', f)
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
