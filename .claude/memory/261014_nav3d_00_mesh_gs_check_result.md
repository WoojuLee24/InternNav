# nav3d 00 — 노은역 mesh ↔ 3DGS 정합 · 바닥 구멍 검사 (2026-10-14)

새 스크립트 `scripts/dataset_converters/nav3d/00_mesh_gs_check.py` (habitat 컨테이너, 약 45 s). 기준·결과표는 `nav3d/README.md` 00 절.
리포트: `nav3d/logs/00_mesh_gs/noeun_station/report.html`.

## 명령
`/usr/bin/python scripts/dataset_converters/nav3d/00_mesh_gs_check.py --scene noeun_station`

## 결과 요약
- A1 강체 PASS (이동 0.0095 m, 회전 0.021°). **A1b 표면 간격 δ +0.036 m FAIL**: GS 가 모든 면에서 mesh 뒤 — 바닥·천장 −5 cm, 벽 −2.5 cm.
  B3 도 같음(GS 바닥이 mesh 바닥보다 4.3~5.0 cm 아래). = 충돌 mesh 가 방 쪽으로 부풀어 있음(평행 이동 아님).
- A2 국소 어긋남 7.5 % 복셀 (서쪽 계단 홀) · A3 GS 없는 mesh 로봇 시야 11.6 % · A4 mesh 없는 GS 1.3 % (통행 구역 위 덩어리 1, 7 cm)
- B1 통행 구역 mesh 구멍(내접 ≥ 0.10 m) 58 개 (S2 지상 44) · B2 GS 바닥 구멍 S0 1.2 % / S1 0.97 % / S2 2.7 %. 가장자리에 몰림.

## 측정 함정 (고침)
- nurec = gzip+msgpack, positions/scales/rotations/densities fp16 (19.6 M). opacity = sigmoid(density), scale = exp.
- 쿼터니언 순서를 바닥만으로 검사하면 못 가른다(항등 근처 wxyz 를 xyzw 로 읽으면 x 180° 회전 → 수직축 그대로). 모든 면 법선으로 비교 → wxyz.
- 강체 맞춤에 표면 간격 δ 항이 없으면 바닥 면적 > 천장 면적이라 "z +3 cm 밀림"으로 오판.
- A3 를 가우시안 하나의 opacity 로 재면 하한(0.5/0.2/0.05)에 따라 21/14/10 % 로 흔들림 → 0.15 m 안 16 개 누적 α 로 바꿈.
- mesh 기둥은 속이 빈 껍데기 → 위에서 쏘면 "닫힌 미스캔"으로 분류(구멍 아님).

## 다음 / 확인 필요
- δ 5 cm 를 고칠지(mesh 를 법선 반대로 깎기 or 그대로) 결정. S0 (−0.2, 6.6) GS 바닥 구멍은 렌더로 반사 바닥인지 확인.

- 2026-10-15 계단을 B1·B2 에 합산(B4 판정 삭제). 계단 GS 구멍은 전체 한 구역(0.79 %) — 작은 조각 33 % 가 희석됨.
