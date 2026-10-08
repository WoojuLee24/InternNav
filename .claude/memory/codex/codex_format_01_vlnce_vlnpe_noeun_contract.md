# Codex 포맷 조사 01 — VLN-CE·VLN-PE·노은역 계약 비교

작성일: 2026-08-14 (UTC)

## 결론

노은역 04 결과는 **VLN-PE와 의미 수준에서는 대응하는 RGB/depth/pose를 포함하지만, 파일 저장 포맷은 동일하지 않다.** VLN-CE는 애초에 동일한 offline trajectory 저장 포맷이 아니라 Habitat가 episode JSON과 scene asset을 읽고 RGB/depth 관측을 온라인 생성하는 평가 환경이다.

## 코드 및 실파일 검사 결과

### VLN-PE 실데이터 (`17DRP5sb8fy`, episode 0)

- parquet 행: 87
- pose: `observation.camera_position` 3-vector + `observation.camera_orientation` quaternion
- intrinsic/extrinsic 행렬 컬럼: 실제 parquet에는 없음
- RGB: episode당 하나의 `.npy`, `(87,256,256,3)`, `uint8`
- depth: episode당 하나의 `.npy`, `(87,256,256)`, `float32`, 실측 범위 `0.020985...~1.0`
- 저장 depth는 정규화 값이며 저장소 로더는 `meters = value × 10`; `1.0`은 10 m clip/invalid로 처리한다.

### 노은역 Isaac D455 결과 (episode 0)

- 프레임 수: RGB 285 / depth 285 / extrinsic 285 — 일대일 대응
- RGB: 프레임별 JPG, `480×270`, RGB
- depth: 프레임별 PNG, `(270,480)`, `uint16`, 실측 raw `0~10000`
- depth 단위: `meters = raw × 0.001`; `10000`은 10 m clip
- intrinsic: episode당 `(3,3) float32` NPY
- extrinsic/pose: 프레임별 `(4,4) float32` NPY

### VLN-CE 평가 계약

`scripts/eval/configs/vln_r2r_mini.yaml`은 Habitat RGB/depth 센서를 모두 `640×480`, hfov 79°, depth `0~10 m`로 설정한다. scene은 MP3D `.glb`와 `.navmesh`이고, trajectory/instruction은 gzip JSON 계열이다. 관측 영상은 Habitat에서 실행 중 생성되므로 노은역 폴더와 파일 단위 비교하는 것은 범주가 맞지 않는다.

## 호환성 판정표

| 비교 항목 | VLN-PE ↔ 노은역 | VLN-CE ↔ 노은역 |
|---|---|---|
| RGB 의미 | 대응 | 대응 |
| depth 의미(광학 카메라 depth) | 대응 | 대응 |
| 최대 depth | 둘 다 10 m | 설정상 10 m |
| 해상도 | 불일치: 256² vs 480×270 | 불일치: 640×480 vs 480×270 |
| depth dtype/scale | 불일치: float32/0.1 m 환산 vs uint16/0.001 m | Habitat runtime float 관측과 불일치 |
| pose 표현 | 불일치: position+quaternion parquet vs 4×4 NPY | Habitat agent/sensor state와 별도 변환 필요 |
| 파일 레이아웃 | 불일치 | 직접 비교 대상 아님 |
| LiDAR 필드 | 없음 | 없음 |

## 중요한 해석

1. 노은역 산출물의 `extrinsic` 파일명은 실제 내용이 프레임별 카메라 pose인지 mount extrinsic인지 명확한 계약이 필요하다. 기존 VLN-N1에서는 `action[t]`가 world pose이고 `camera_extrinsic`은 고정 mount 변환이므로 이름 혼동이 모델 입력 오류로 이어질 수 있다.
2. VLN-PE와 VLN-CE 어느 기본 포맷에도 LiDAR 필드가 없다. LiDAR를 추가한다면 기존 포맷 일치 문제가 아니라 **공통 확장 schema**를 새로 정의하는 일이다.
3. D455의 `depth_scale=0.001`은 depth PNG 양자화 단위다. LiDAR range 단위나 LiDAR profile과는 별개다.

## 현재 제한

로컬의 `data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i` 기본 경로는 현재 존재하지 않아 이번 단계에서 VLN-N1 실파일을 재검사하지 못했다. 과거 00~04가 사용한 결과와 코드 계약은 남아 있지만, 이번 LiDAR/포맷 판정에서는 이 부재를 숨기지 않고 별도 제한으로 기록한다.

## 다음 구현 방향

- 기존 RGB/depth 포맷은 그대로 보존한다.
- LiDAR는 `lidar/<frame>.npz`처럼 임의 추가하기 전에, 센서 profile·좌표계·timestamp·유효 거리·반환값 필드를 먼저 하나의 명시적 계약으로 정한다.
- VLN-PE scene USD와 노은역 USDZ에 동일한 Isaac Sim LiDAR 설정을 적용해 동일 schema로 저장한다.
- VLN-CE `.glb`는 Isaac Sim에 직접 넣는 경로가 현재 평가 코드에 없으므로, USD 변환/재질·scale 검증을 거친 뒤 같은 LiDAR replay 경로를 사용해야 공정한 비교가 된다.
