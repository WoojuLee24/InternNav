# ply2usdz 결과 (2026-10-29)

원본 3DGS PLY + 충돌 mesh → `<name>_collision.usdz` (GS `.nurec` + 숨긴 충돌 `mesh.ply`/`mesh.usd` 한 파일, data/GS_USDZ 와 같은 형식).

## 명령
`scripts/dataset_converters/ply2usdz/run_ply2usdz.sh <gs.ply> <mesh.obj|mesh.ply> <out.usdz> --nurec-name <이름>`

| 인자 | 뜻 | 왜 |
|---|---|---|
| gs.ply | 원본 3DGS PLY (`f_rest` 45개) | 3DGRUT ply_to_usd 입력 |
| mesh | GS 와 같은 좌표계 충돌 mesh | usdz 안 mesh.ply·mesh.usd 로 들어감 |
| --nurec-name | usdz 안 `.nurec` 이름 | 기존 파일과 같게 (노은역 `noeun_station`) |
| --near-clip | 카메라에서 이 거리(m) 안 GS 미렌더, 기본 1e-8 | 학습 데이터·3DGRUT 최신 기본값과 같음 |

환경: usd2usdz `setup_3dgrut_venv.sh` 로 만든 `~/git/3dgrut-venv` (3DGRUT 16b7b19 + 축 패치), 이미지 `dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090`.

## 노은역 비교 (`sim_docker/workspace/NoeunStation/`)

| 항목 | 기존 `noeun_station_collision.usdz` | sim_docker `-reproduce` | InternNav `-internnav` |
|---|---|---|---|
| `.nurec` (GS·near_clip) | 기준 | 같음 | 같음 |
| `mesh.ply` | 기준 | 데이터 같음 (trimesh 주석 1줄) | **바이트 같음** |
| `mesh.usd` | 기준 | 바이트 같음 | USD 내용 같음 (바이너리 배치 36 B 차이) |
| `default.usda`·`gauss.usda` | 기준 | 같음 | `timeCodesPerSecond = 24` 한 줄 없음 (3DGRUT 16b7b19 출력, 기본값 24 라 의미 같음) |
| 64바이트 정렬·ZIP32 | — | OK | OK |
