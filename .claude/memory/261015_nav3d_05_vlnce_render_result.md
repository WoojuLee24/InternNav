# nav3d 05a·05b — vln_ce GT 재렌더 + 생성 경로 렌더 (habitat, 2026-10-15)

vln_ce 는 habitat 으로 렌더한다(사용자 지시 — 릴리스가 habitat 산출). 렌더러 = pull 받은 `instruction_labeling/render`(RigSim, 640×480 · hfov 79 · 1.25 m · pitch 0/30).
Isaac 재렌더(`gs_vlnpe/05_export_vlnce` + `--episodes_json` 추가, 출력 `vln_ce_render_v2`)는 참고용으로만 남김(`nav3d/05x_*`, 5/6 — 60cm rig 정합 4~6 cm 는 이전부터).

## 05a GT 재렌더 — 4/5 (https://claude.ai/artifact/KZUstJLiBGStTCWdn3TPje)
`p1r_replay_verify` 685 ep(계단 156) · 39,852 프레임: pose 최대 0.08 mm / 0.002°, SSIM 최소 0.9745, 계단−평지 SSIM +0.0001.
**H4 depth FAIL(89 % 가 mm 일치 ≥ 0.98)**: 불일치 픽셀은 **우리 depth = 0(표면 못 맞춤) & 릴리스 값 있음**(3 씬 99.7~99.9 %, 17DRP 64.6 %), 릴리스는 불일치 픽셀에서 0 이 없다.
→ 릴리스가 depth 구멍을 채웠거나 mesh 뒷면을 그렸다(미확정). labeling 팀 p1r 메모(평균 99.6 %)는 평균이라 가려졌다. 생성 데이터 depth 구멍이 릴리스보다 많아지는 분포 차이.

## 05b 생성 경로 렌더 — 5/5 (https://claude.ai/artifact/XCqVmJW9onakLonWyKPU9k)
nav3d 03 wp·sg 56 경로(계단 18) -> 0.5 m 경유점 -> GreedyGeodesicFollower(0.25 m/15°) -> render_trajectory/export_scene -> `vln_ce_nav3d/traj_data/r2r/nav3d_{wp,sg}/`.
완주 56/56(끝점 ≤ 0.23 m) · 벗어남 ≤ 0.21 · 빠뜨림 ≤ 0.24 m · 충돌 0 · 계단 |Δz| ≤ 0.067 m · 형식 56/56. 지시문은 GT 문장 임시(다시 생성 필요).
함정: P4 를 "가장 가까운 프레임" z 로 재면 프레임 간격 0.25 m 때문에 0.22 m 로 부풀음 → 걸은 위치를 경로에 투영해 비교. numpy bool 이 PASS 로 안 찍히던 리포트 버그.

## 환경
- `internnav-isaac` 컨테이너 새로 띄움(scikit-image 설치, numpy 1.26.4 고정). git 은 컨테이너 root 로(.git 일부가 root 소유였음 → chown 1000).
- `git pull origin feature/input_v0.1` -> ee4e4fa (로컬 변경과 겹침 없음).

## depth 구멍 채우기 A 적용 (2026-10-15)
- 원인: 릴리스 depth 는 habitat no-hit(0) 구멍을 사후 채움(구멍 RGB 는 릴리스도 검정 → 뒷면 아님). 범위·값 규칙은 완전 재현 못 함 — 생성기 코드 없음(dialog_mp3d.fill_small_holes 는 다른 용도).
- `nav3d/depth_fill.py` = 3×3 최소 이웃 × 19 (uint16 mm). 05b `--depth_fill min19`(기본, 원본은 depth_raw.<rig>/), 05c 뷰어 4 상태, 05a H4b·H4c.
- 685 ep 전수(5 프레임 간격): 1 m 초과 픽셀 평균 0.512 → 0.144 %, 0 위치 불일치 0.521 → 0.143 %, 1 cm 안 ≥ 0.98 ep 652 → 664/685. **H4b 96.93 % · H4c 프레임 최대 8.2 % 로 둘 다 FAIL**.
- 반례: 1LXt ep19 f12 — 화면 56 % 가 하나의 큰 구멍(no-hit), 릴리스는 3.6 % 만 채움 → 우리 규칙이 5.8 % 를 잘못 채움(1 cm 안 0.9738 → 0.9585). 거리 20 px 규칙도 테두리 규칙도 아님.
