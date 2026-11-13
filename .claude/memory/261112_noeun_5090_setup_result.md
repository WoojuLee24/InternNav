# 노은역 ABC·DAgger 준비 — RTX 5090 (universe 데스크톱) 결과 (2026-11-12)

- 서버: RTX 5090 1장 (sm 12.0, 32 GB, 드라이버 590.48). 상시 사용 ~7.4 GB (teleop_view.py ~5.9 GB + 화면 ~1.2 GB).
- 코드: InternNav main 8e6a45b, InternUtopia main 92b9353 (둘 다 origin/main 과 같음, 로컬 미커밋 수정 유지. pull 전 백업 = InternNav `stash@{0}`).
- 경로: WS=/home/universe/gd_project/modules/gd_vln/workspace, DATA=/media/TrainDataset, CKPT=/media/LTDataset/checkpoints.
- 컨테이너: `noeun-isaac` (torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090, TRITON_OVERRIDE_ARCH=sm120), `noeun-habitat` (torch2.9.0-cuda13.0-habitat-jazzy-5090). 기존 internnav-habitat 은 정지.
- 81 서버 :8041 (noeun-habitat 에서 백그라운드).

## 시험 (84 ABC, v6 ep 0·20)
- 전체 149 s (Isaac 시작 포함, 첫 실행). ep 당 약 35~60 s 추정 (로그에 ep 별 시간 없음, 파일 시각으로 추정). A5000 ~110 s.
- 가지 끝 목표 거리: 0.07~0.24 m (A5000 0.06~0.29 m). 18 가지 모두 목표 도착 True.
- GPU: 렌더 중 최대 18.6 GB (상시 7.4 GB 제외 렌더 하나 ~11.2 GB; A5000 기록 5.5~8.5 GB 보다 큼).
  → 렌더 2 개 동시면 ~30 GB (teleop 켠 채로 빠듯). DAgger(모델 18 GB + 렌더 11 GB) 는 teleop 꺼도 ~30 GB.
- 85 그림 8 장: logs/abc_smoke_viz/ (frames_* 6, top_* 2). 검은 사진 없음.
  흐림: frames_abc_v6_s2_202.jpg (B 가지, 목표 지나 돌아오는 f64~f73) — 기둥·에스컬레이터에 너무 붙어 GS 가 뭉개짐.
