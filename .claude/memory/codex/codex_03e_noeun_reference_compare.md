# Codex 03e — 노은역 생성 GT 단계 간 비교

## 원칙

원본 03e의 R2R instruction 매칭은 VLN-PE 전용이므로 노은역에는 존재하지 않는다. HTML을 따로 만들지 않고, 박사님의 overlay renderer, 표 및 `save_gallery` 구조에 노은역 입력만 연결했다.

## 노은역 비교 대상

- 노랑: 03 최종 path GT
- 하늘: refine/smoothing 전 A* waypoint 경로
- 빨강: 04가 실제 저장한 camera pose

## 결과

- 04 pose와 03 path GT 같은 루트: 20/20
- A* 원경로와 최종 path GT chamfer 중앙값: 0.101m
- HTML: `logs/gs-vlnpe/codex_noeun/03e_compare_reference_path/noeun_station_mid_random/report.html`

이 결과는 03에서 만든 경로 GT가 04 렌더 단계의 pose 파일까지 일관되게 전달됐음을 확인한다.
