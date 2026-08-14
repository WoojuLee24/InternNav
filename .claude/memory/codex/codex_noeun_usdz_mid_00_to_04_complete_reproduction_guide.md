# Codex 노은역 USDZ 중간층 데이터셋 00~04 완전 재현 가이드

작성 기준일: 2026-08-14
작업 디렉터리: `/ws/src/InternNav`
대상 코드: `scripts/dataset_converters/gs_vlnpe`

## 1. 이 문서의 목적과 결론

이 문서는 노은역 USDZ 한 개에서 `noeun_station_mid`라는 논리적 중간층을 정의하고, 원본
GS-VLNPE 변환기의 00→01→02→03→04 흐름을 이용해 경로 GT와 RGB-D 관측 GT를 만든 과정을
처음부터 다시 실행할 수 있게 설명한다. 기존 단계별 메모와 실제 코드, JSON, HTML, 생성 파일을
서로 대조해 작성했다.

가장 중요한 결론은 다음과 같다.

1. `noeun_station_mid`는 별도로 잘라 저장한 USDZ 파일이 아니다.
2. 원본 `data/noeun_station/noeun_station_collision.usdz`는 수정되지 않았다.
3. `noeun_station_mid`는 원본 USDZ 안에서 `floor_z=-0.05 m` 주변의 상향 수평면으로 계산한
   XY 범위와, 바닥 아래 0.20 m부터 위 1.60 m까지의 Z 범위를 기록한 **논리적 scene ID/ROI**다.
4. USDZ mesh 자체가 공간 geometry GT다. USDZ에 기존 VLN episode GT가 없었던 것이지,
   geometry GT가 없었던 것이 아니다.
5. 02가 geometry로 occupancy/ESDF를 만들고, 03이 그 위에서 20개 경로 GT를 생성하며, 04가
   경로를 따라 RGB·depth·camera pose를 렌더링한다.
6. 최종 사용 대상은 Isaac Sim으로 만든
   `noeun_station_mid_random_isaac_d455_nominal`이다. 기본 Open3D 04도 실행했지만 RGB 재질이
   없어 최종 데이터로는 부적합하다.

## 1.1 왜 승강장·중간층·상층으로 나뉘었고, 왜 중간층부터 했는가

### 당시 바닥 탐색 결과물

00~04를 적용하기 전에 원본 USDZ가 실제로 몇 개의 주요 보행층을 포함하는지 먼저 조사했다.
그때 만든 결과물이 다음 디렉터리에 그대로 남아 있다.

```text
logs/gs-vlnpe/noeun_floor_probe/
├── level_low_xy.png
├── level_mid_xy.png
└── level_high_xy.png
```

파일명 그대로 원본 mesh에서 검출한 낮은 층, 중간층, 높은 층의 XY 분포를 따로 그린
Matplotlib 결과다. 이 세 이미지가 `noeun_station_mid`를 만들기 전에 이미 low/mid/high 세
높이대를 분리해 조사했다는 직접 산출물이다. 다만 당시 일회성 probe 생성 script와 숫자 로그는
현재 저장소에 남아 있지 않다. 따라서 아래 설명은 남아 있는 세 이미지, 현재 USDZ mesh, 확정된
01 metadata를 다시 대조해 동일 결과가 나오도록 복원한 절차다. 확인되지 않은 일회성 명령을
과거의 정확한 원문처럼 꾸며 쓰지 않는다.

### 바닥 후보를 찾을 때 실제로 찾은 것

바닥을 찾는다는 것은 mesh vertex의 Z 최솟값을 찾는 것이 아니다. 최솟값은 선로 아래 구조물,
벽의 끝, 스캔 잡음일 수 있기 때문이다. 대신 사람이 걸을 수 있는 넓은 면에 가까운 triangle을
찾았다.

1. USDZ package 내부의 `mesh.ply` triangle mesh를 읽는다.
2. 각 triangle의 face normal, centroid, surface area를 계산한다.
3. `normal_z >= cos(15°)`인 면만 남긴다.
   - 위를 향하는 거의 수평인 면을 남긴다.
   - 벽과 급경사는 제거한다.
   - normal이 아래를 향하는 천장 아랫면도 제거한다.
4. 남은 면의 centroid Z를 높이축에 누적한다.
5. 단순 face 개수만 세지 않고 surface area도 함께 본다.
   - 작은 triangle이 많이 쪼개진 곳이 과대평가되는 것을 막기 위해서다.
   - 넓고 연속된 역사 바닥은 큰 누적 면적으로 나타난다.
6. Z축에서 서로 수 m 떨어진 큰 수평면 군집을 찾는다.
7. 각 높이 군집의 triangle을 XY 평면에 투영해 공간 형태와 연속성을 확인한다.
8. 그 결과를 `level_low_xy.png`, `level_mid_xy.png`, `level_high_xy.png`로 각각 저장했다.

즉 층을 나눈 1차 근거는 **상향 수평면의 Z 높이 군집**, 2차 근거는 **각 군집의 XY 공간
분포와 넓이**였다. Z histogram의 peak 하나만 보고 층을 확정한 것이 아니라, 해당 높이의 면이
역사 평면에서 실제로 넓고 연속적인지도 XY 그림으로 확인했다.

### 원본 mesh에서 다시 확인한 세 높이대

현재 원본 USDZ에 같은 조건을 다시 적용했다. 5 cm Z bin으로 상향면 면적 분포를 보고, 서로
분리된 큰 군집의 대표 높이를 중심으로 `±0.45 m`를 probe한 결과는 다음과 같다.

| probe 이름 | 대표 floor Z | 선택 상향 face | 선택 면적 | face centroid Z 범위 | 수직 순서상 해석 |
|---|---:|---:|---:|---:|---|
| low | 약 -5.30 m | 14,891 | 642.6150 m² | -5.7496~-4.8501 m | 승강장/아래층 |
| mid | -0.05 m | 14,618 | 798.1859 m² | -0.4996~+0.3994 m | 중간층 |
| high | 약 +6.80 m | 18,414 | 547.9675 m² | +6.3627~+7.2500 m | 상층/위층 |

세 군집 사이의 대표 높이 차이는 low→mid 약 5.25 m, mid→high 약 6.85 m다. 하나의 두꺼운
바닥이나 몇 cm 스캔 잡음으로 설명할 수 없는 간격이며, 원본 USDZ가 서로 다른 주요 수평층을
포함한다는 근거다.

5 cm 단위 면적 histogram에서도 다음 구간들이 강하게 나타났다.

- 승강장/아래층 계열: 약 `-5.50~-5.05 m`
- 중간층 계열: 약 `-0.60~+0.40 m`, 그중 `-0.05~0.00 m` bin이 99.547 m²로 전체 5 cm bin 중 가장 큼
- 상층 계열: 약 `+6.65~+6.95 m`

여기서 `±0.45 m` band가 넓어 보이는 이유는 스캔 mesh가 이상적인 한 장의 평면이 아니고,
바닥 두께·완만한 경사·계단/연결부·재구성 오차가 함께 있기 때문이다. 대표 floor Z는 이 band의
모든 Z를 평균낸 값이 아니라, 해당 보행층을 나타내는 기준 높이다.

### 승강장·중간층·상층이라는 의미를 붙인 방법

Z 값만으로 “이 면은 승강장이다”라는 건축 의미를 자동 판정할 수는 없다. 먼저 low/mid/high
높이 군집을 기하적으로 분리하고, 각 군집을 XY로 그려 노은역 공간 배치에서 어느 부분을
차지하는지 확인해 의미를 대응했다.

- 가장 낮은 큰 보행면 군집은 승강장/아래층
- 0 m 부근의 넓고 연속적인 군집은 중간층
- 가장 높은 큰 보행면 군집은 상층/위층

따라서 “승강장, 중간, 위 층이 나왔다”는 말은 파일 이름을 임의로 세 개 만든 것이 아니라,
상향 수평면이 Z축에서 세 개의 큰 군집으로 분리되고 각 XY 배치도 서로 다른 층을 나타낸다는
뜻이다. `noeun_floor_probe`의 세 이미지가 바로 이 의미 대응을 육안으로 검토하기 위한 결과다.

### 왜 `floor_z=-0.05 m`를 중간층 대표값으로 정했는가

중간층의 상향 수평면은 넓게 보면 약 `-0.60~+0.40 m`에 분포한다. 그중 5 cm bin 기준 가장 큰
수평면 면적이 `-0.05~0.00 m`에 있었고, 확정 메타의 `-0.05±0.45 m` 선택은 14,618 face와
798.1859 m²의 넓은 연속 영역을 얻는다. 이 영역의 XY bounds도 역사 중간층을 충분히 포함하면서
원본 USDZ bounds 안에 완전히 들어간다. 그래서 `-0.05 m`를 중간층 planning의 기준 바닥으로
사용했다.

중요하게도 `floor_z=-0.05`는 바닥 전체가 수학적으로 정확히 같은 Z라는 주장이 아니다. 02의
navigation slice, obstacle height band, 03의 camera/body Z를 일관되게 계산하기 위한 logical
level의 기준면이다.

### 중간층부터 데이터셋을 만든 결정

floor probe 단계에서는 승강장/아래층, 중간층, 상층/위층 세 후보가 모두 나왔다. 이후 사용자의
요청으로 전체 층을 한 번에 처리하지 않고 **중간층부터** 00→04 pipeline을 완성하기로 했다.
그 결정에 따라 다음 이름과 값이 고정됐다.

```text
scene_id = noeun_station_mid
source_scene_id = noeun_station
logical_level = mid
floor_z = -0.05 m
source geometry = data/noeun_station/noeun_station_collision.usdz
```

따라서 현재 20개 episode는 노은역 전체 층 데이터셋도, 승강장 데이터셋도 아니다. 원본 USDZ
안에서 선택한 **중간층 logical ROI**의 경로와 관측 데이터셋이다. low와 high는 probe 결과만
남아 있으며, 동일 절차로 별도 scene ID·floor Z·ROI를 확정하기 전에는 현재 mid ESDF나 path를
그 층에 그대로 사용하면 안 된다.

## 2. 확정 사실과 역사적으로 복원할 수 없는 부분

### 2.1 코드와 산출물로 확정되는 사실

- 기준 바닥 높이는 `-0.05 m`다.
- 선택식은 face normal이 Z축에서 15° 이내인 상향 면이며, face centroid Z가 기준 높이에서
  `±0.45 m` 안에 드는 면이다.
- 이 식으로 선택된 face는 14,618개, 합계 면적은 798.1859255685 m²다.
- 선택 면의 XY 경계에 1.0 m padding을 주어 logical ROI를 만들었다.
- 00부터 04 및 03b~03e의 현재 산출물과 HTML이 실제로 존재한다.
- 03은 20/20 경로를 만들었고 04 Isaac은 그 20개 경로에서 총 8,378 frame을 만들었다.

### 2.2 반드시 구분해야 하는 사실

현재 `00_inspect_vln_n1.py`와 `01_prepare_scene.py`는 USDZ 전체에서 층을 자동으로 찾아
“이것이 중간층이다”라고 명명하는 알고리즘이 아니다. `--floor_z -0.05`를 먼저 입력받고, 그
주변에 충분한 상향 수평면이 있는지와 좌표계·스케일·ROI 경계를 검증한다.

기존 메타에는 “Noeun Station MID에서 확정한 중간층 기준 바닥 z=-0.05 m”라고 기록돼 있지만,
그 값을 최초로 선택한 사람/AI의 원시 실험 로그나 층별 후보 비교표는 저장소에서 찾지 못했다.
따라서 “코드가 자동으로 모든 층을 비교해 -0.05를 골랐다”고 설명하면 사실과 다르다.

다만 이 문서 작성 시 USDZ 내부 `mesh.ply`를 독립적으로 다시 계산했다. 15° 이내 상향 face의
Z 분포에서 `-0.05~0.00 m` 구간은 1,728 face, 99.547 m²로 5 cm bin 중 가장 큰 면적이었다.
`-0.05±0.45 m` 전체 선택 결과도 현재 메타와 정확히 같은 14,618 face, 798.1859255685 m²였다.
즉 `-0.05`는 임의의 빈 높이가 아니라 실제로 큰 보행 가능 수평면 군집을 관통하는 값이다.
원본에는 약 `-5.5~-5.1 m`, `-0.6~0.4 m`, `6.65~6.95 m` 등 여러 수평면 군집이 있어 다층
구조라는 사실도 확인된다. “중간층”이라는 의미는 여기서 선택한 0 m 부근 논리층을 가리킨다.

## 3. 입력 USDZ의 정확한 정체

입력 파일:

`data/noeun_station/noeun_station_collision.usdz`

식별 정보:

| 항목 | 값 |
|---|---:|
| 파일 크기 | 2,336,320,706 bytes |
| SHA-256 | `3a4719b101f3c5e4a0b4cbfba5904be26b781e85f6022dac364727d64044fcd1` |
| USD up axis | Z |
| metersPerUnit | 1.0 |
| mesh vertex | 384,207 |
| mesh triangle | 665,812 |
| 전체 bounds min | `[-63.8869438171, -58.9734878540, -9.1222715378]` m |
| 전체 bounds max | `[184.5661926269, 57.4124221802, 59.5262184143]` m |

USDZ는 ZIP 기반 패키지이며 내부에는 다음 파일이 있다.

- `default.usda`: package 진입 stage
- `noeun_station.nurec`: 대용량 재구성 데이터
- `gauss.usda`: Gaussian/reconstruction 관련 USD 구성
- `mesh.ply`: collision/geometry 처리에 재사용되는 triangle mesh
- `mesh.usd`: USD mesh 표현

`mesh.ply`는 geometry는 있지만 최종 RGB 렌더에 필요한 색/UV/texture 정보가 충분하지 않다.
이 때문에 Open3D 04는 깊이 검증에는 유용하지만, 최종 RGB는 Isaac Sim에서 USDZ 전체를 로드해
렌더링한 결과를 사용해야 한다.

입력 파일이 같은지 재확인하는 명령:

```bash
sha256sum data/noeun_station/noeun_station_collision.usdz
```

## 4. `noeun_station_mid`가 만들어진 방법

### 4.1 이름의 의미

- `noeun_station`: 원본 공간 자산의 ID
- `mid`: 원본 다층 공간 중 0 m 부근을 사용할 logical level
- `noeun_station_mid`: 이후 02~04가 같은 ROI와 산출물을 연결할 때 쓰는 scene key

이 이름으로 새 geometry 파일을 만든 것이 아니다. 실제 메타의 `mesh_path`는 `null`이고,
`usd_path`는 계속 원본 USDZ를 가리킨다.

### 4.2 중간층 face 선택식

각 triangle face `i`에 대해 다음 조건을 동시에 만족하면 중간층 바닥 후보로 선택한다.

```text
normal_z(i) >= cos(15°)
abs(face_centroid_z(i) - (-0.05)) <= 0.45 m
```

첫 조건은 위를 향하며 수평에서 최대 15° 기울어진 면을 허용한다. 벽, 천장의 아랫면, 심하게
기울어진 구조를 제외하기 위한 조건이다. 두 번째 조건은 기준 바닥 주변 Z band를 정한다.

### 4.3 ROI 계산식

선택된 모든 triangle vertex의 XY 최소·최대값은 다음과 같다.

```text
floor XY min = [-32.2199516296, -24.8250007629] m
floor XY max = [ 32.8250007629,  38.3705673218] m
```

XY 양쪽에 1.0 m padding을 주고 Z는 기준 바닥 아래 0.20 m, 위 1.60 m로 잡았다.

```text
ROI min = [-33.2199516296, -25.8250007629, -0.25] m
ROI max = [ 33.8250007629,  39.3705673218,  1.55] m
```

이 ROI는 원본 USDZ bounds 안에 완전히 포함된다. 01은 이 숫자를
`scene_meta/noeun_station_mid.json`에 기록할 뿐 원본 USDZ를 crop하거나 overwrite하지 않는다.

### 4.4 이 방식의 의미와 한계

- `±0.45 m`는 완전히 평평한 한 장의 바닥만 고르는 값이 아니라 스캔 오차, 완만한 변화,
  경사 연결부를 포함하는 probe band다.
- XY bounds는 선택된 모든 면의 전체 외접 사각형이다. 그 사각형 내부가 전부 걸을 수 있다는
  뜻은 아니다. 실제 장애물과 free space 판정은 02의 occupancy/ESDF가 한다.
- `headroom=1.60 m`는 3D voxelization ROI 높이다. 카메라 높이나 D455 최대 거리가 아니다.
- 다른 USDZ에 그대로 적용할 때는 `-0.05`를 복사하면 안 된다. 해당 자산의 수평면 Z 분포와
  의도한 층을 먼저 확인한 뒤 `--floor_z`를 지정해야 한다.

## 5. 전체 데이터 흐름

```text
원본 noeun_station_collision.usdz
  ↓ 00: 자산/좌표계/mesh 계약 검사
target_schema.json
  ↓ 01: -0.05 m 층의 logical ROI 정의
scene_meta/noeun_station_mid.json
  ↓ 02: ROI geometry voxelization → obstacle/free space/ESDF
esdf/noeun_station_mid.npz + json
  ↓ 03: navigable map에서 start/goal 샘플 → A* → refine → smooth
paths/noeun_station_mid_random.json (20 episode 경로 GT)
  ↓ 03b~03e: 재현성·refine·parameter·reference overlay 검증
verify/compare/gridsearch JSON + HTML
  ↓ 04 Isaac: 각 경로를 따라 USDZ 렌더링
obs/noeun_station_mid_random_isaac_d455_nominal/
  ├─ camera_config.json
  └─ episode_000000 ... episode_000019
     ├─ rgb/frame_XXXX.jpg
     ├─ depth/frame_XXXX.png
     └─ extrinsic/frame_XXXX.npy
```

실제 의존 순서는 `00 → 01 → 02 → 03 → 03b/03c/03d → 04 → 03e`로 이해하면 가장 명확하다.
03e는 04 pose overlay도 사용하므로 04 뒤에 실행하는 것이 온전하다. 03b~03d는 품질 검증이며
04의 직접 입력은 03의 path JSON이다.

## 6. 실행 환경과 공통 규칙

모든 명령은 저장소 루트 `/ws/src/InternNav`에서 실행한다. Conda나 mamba는 사용하지 않는다.
USD/Isaac API가 필요한 단계는 저장소에 연결된 Isaac Sim Python을 사용한다.

```bash
cd /ws/src/InternNav
```

공통 경로:

```text
script root = scripts/dataset_converters/gs_vlnpe
output root = scripts/dataset_converters/gs_vlnpe/apply_real
log root    = logs/gs-vlnpe/apply_real
input USDZ  = data/noeun_station/noeun_station_collision.usdz
scene ID    = noeun_station_mid
```

같은 output root로 재실행하면 같은 이름의 JSON·이미지·episode 결과가 갱신될 수 있다. 기존 결과를
보존하며 시험하려면 모든 단계에서 `--out_dir`와 `--log_dir`를 동일한 별도 루트로 바꿔야 한다.
단계마다 서로 다른 output root를 쓰면 다음 단계가 앞 단계 산출물을 찾지 못하므로 일관되게 바꾼다.

## 7. Stage 00 — USDZ 자산과 데이터 계약 검사

### 목적

원본의 `00_inspect_vln_n1.py`는 VLN-N1/VLN-PE의 frame, camera convention, mesh 정합을 검사한다.
노은역 분기는 `--usd_path`를 지정했을 때 기존 dataset episode 검사 대신 새 USDZ가 이후 pipeline의
공간 GT로 사용 가능한지를 검사하도록 최소 확장했다.

### 실제 처리

1. Isaac `SimulationApp`과 USD API로 package를 연다.
2. USD stage의 up axis, meters-per-unit, world bounds, mesh prim을 읽는다.
3. triangle mesh를 로드한다.
4. `floor_z=-0.05`, 15°, ±0.45 m 조건의 후보 face 수를 센다.
5. 다음 여섯 gate를 모두 만족해야 PASS다.
   - USD readable
   - Z-up
   - metersPerUnit=1
   - finite이고 양의 크기를 갖는 bounds
   - mesh prim 존재
   - 중간층 후보 face 존재
6. `target_schema.json`, 후보면 이미지, 원본 gallery 형식 HTML을 저장한다.

### 실행 명령

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/00_inspect_vln_n1.py --scene noeun_station_mid --usd_path data/noeun_station/noeun_station_collision.usdz --floor_z -0.05 --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

### 확인할 산출물과 결과

- JSON: `scripts/dataset_converters/gs_vlnpe/apply_real/target_schema.json`
- HTML: `logs/gs-vlnpe/apply_real/00_inspect_vln_n1/noeun_station_mid/report.html`
- 결과: 여섯 gate 모두 PASS, triangle 665,812, 후보 face 14,618

00은 USDZ를 변환하지 않으며 경로 GT도 만들지 않는다. 이후 단계가 사용할 입력 계약을 확정하는
검사 단계다.

## 8. Stage 01 — 중간층 logical scene metadata 생성

### 목적

원본 `01_prepare_scene.py`가 하던 scene 사용 가능 여부 판정과 `scene_meta` 계약을 새 USDZ에
적용한다. `--usd_path`가 없으면 기존 Matterport/VLN 경로가 그대로 작동하고, 지정한 경우에만
새 scene branch가 실행된다.

### 실제 처리

1. 원본 USDZ mesh와 stage bounds를 읽는다.
2. 앞서 설명한 상향 수평면 식으로 후보를 선택한다.
3. 선택 면의 XY bounds, 면 수, 면적을 계산한다.
4. XY 1 m padding, Z `floor-0.20`부터 `floor+1.60`으로 ROI를 만든다.
5. Z-up, 1 m/unit, 유효 ROI, source bounds 내부 여부를 gate로 검사한다.
6. 원본 USDZ 경로와 logical ROI만 JSON에 기록한다.

### 실행 명령

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/01_prepare_scene.py --scene noeun_station_mid --usd_path data/noeun_station/noeun_station_collision.usdz --floor_z -0.05 --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

### 확인할 산출물과 결과

- JSON: `scripts/dataset_converters/gs_vlnpe/apply_real/scene_meta/noeun_station_mid.json`
- ROI 이미지: `logs/gs-vlnpe/apply_real/01_prepare_scene/noeun_station_mid/middle_floor_roi.jpg`
- HTML: `logs/gs-vlnpe/apply_real/01_prepare_scene/noeun_station_mid/report.html`
- 결과: PASS, 14,618 face, 798.1859 m²

`mesh_path=null`은 mesh가 없다는 뜻이 아니다. 이 logical scene이 별도 mesh 파일을 만들지 않고
`usd_path`의 원본 geometry를 사용한다는 뜻이다. 신규 scene이라 기존 episode에서 측정할
`gt_robot_params`, `gt_camera_z`, `frame_alignment`가 이 시점에 null인 것도 정상이다.

## 9. Stage 02 — occupancy, free map, ESDF 생성

### 목적

`02_build_freemap_esdf.py`가 01의 logical ROI 안에서 USD geometry를 5 cm voxel grid로 바꾸고,
로봇이 지날 수 있는 2D navigable map과 장애물까지의 signed/Euclidean distance 정보를 만든다.

### 중요 수정 이유

초기 구현은 거대한 다층 source 전체에서 고정 개수 surface point를 샘플한 다음 중간층 ROI만
잘랐다. 그러면 중간층 sample density가 원본 전체 크기에 종속되어 구멍이 생길 수 있다.
현재 `esdf_utils.voxelize_surface(..., bounds=...)`는 먼저 ROI와 triangle bounds가 겹치는 face를
선택한 뒤 그 submesh를 샘플한다. 이는 새 scene을 안정적으로 처리하기 위한 핵심 수정이다.

또한 standalone Isaac Python에서 USD API를 쓰려면 `SimulationApp`이 살아 있어야 한다. app을
mesh 로드 직후 닫으면 프로세스가 정상 종료처럼 보이면서 산출물이 생기지 않는 문제가 있었기
때문에, 현재 02는 작업 완료까지 app 참조를 유지한다.

### 기준 파라미터

- voxel size: 0.05 m
- grid shape: `[1345, 1308, 40]`
- grid origin: `[-33.4, -26.0, -0.4]` m
- floor: -0.05 m
- navigation slice height `h_nav`: 0.10 m
- robot radius `r_b`: 0.25 m
- 신규 scene reference body/camera height `ref_h_b`: 0.875 m
- geometry source: USDZ

`ref_h_b=0.875`는 obstacle을 만들 때 사용할 높이 band 기준이다. D455 depth scale이나 카메라
intrinsic과는 관계없다.

### 실행 명령

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/02_build_freemap_esdf.py --scene noeun_station_mid --geometry usd --ref_h_b 0.875 --scene_meta_dir scripts/dataset_converters/gs_vlnpe/apply_real --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

### 확인할 산출물과 결과

- NPZ: `scripts/dataset_converters/gs_vlnpe/apply_real/esdf/noeun_station_mid.npz`
- JSON: `scripts/dataset_converters/gs_vlnpe/apply_real/esdf/noeun_station_mid.json`
- HTML: `logs/gs-vlnpe/apply_real/02_build_freemap_esdf/noeun_station_mid/report.html`
- occupancy fraction: 0.01013146
- navigable fraction: 0.879639
- max ESDF: 27.6688 m
- geometry sanity: PASS

JSON의 external GT clearance 또는 pointcloud overlap 결과가 `null`/`UNVERIFIED`인 것은 FAIL이
아니다. 이 단계 전에는 새 노은역 trajectory GT와 외부 GT pointcloud가 없으므로 해당 비교를
건너뛴 것이다. geometry 자체 gate는 통과했다.

## 10. Stage 03 — 20개 경로 GT 생성

### 목적

`03_sample_gt_paths.py --mode random`이 02의 navigable map에서 start/goal을 뽑고 충돌 없는
경로를 생성한다. 이것이 노은역에 처음 생긴 episode-level trajectory GT다.

### 실제 처리 순서

1. ESDF와 navigable mask를 읽는다.
2. seed 0의 deterministic random sampling으로 start/goal 후보를 고른다.
3. 0.20 m A* grid에서 경로를 찾는다.
4. 원래 5 cm grid/ESDF에서 waypoint를 장애물에서 더 안전한 위치로 refine한다.
5. waypoint 간격을 약 0.8 m로 정리한다.
6. cubic smoothing 후 0.05 m 간격의 조밀한 trajectory를 만든다.
7. robot radius와 hard collision gate를 검사한다.
8. 실패 후보는 원본 retry/resample 흐름으로 버리고 대체한다.
9. episode별 body height `h_b`, pitch, start, goal, A* waypoint, refined waypoint,
   최종 path를 JSON에 기록한다.

### 실제 파라미터

| 파라미터 | 값 |
|---|---:|
| seed | 0 |
| episode | 20 |
| robot radius | 0.25 m |
| refine radius | 0.10 m |
| voxel cell | 0.05 m |
| A* cell | 0.20 m |
| h_nav | 0.10 m |
| refine mode | argmax |
| downsample mode | any |
| waypoint spacing | 0.8 m |
| smooth | cubic |
| smooth output step | 0.05 m |

### 실행 명령

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/03_sample_gt_paths.py --scene noeun_station_mid --mode random --num_episodes 20 --esdf_dir scripts/dataset_converters/gs_vlnpe/apply_real/esdf --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

### 확인할 산출물과 결과

- JSON: `scripts/dataset_converters/gs_vlnpe/apply_real/paths/noeun_station_mid_random.json`
- HTML: `logs/gs-vlnpe/apply_real/03_sample_gt_paths/noeun_station_mid_random/report.html`
- 결과: 20/20 성공, hard collision 0
- rejected resample: 1개 후보를 폐기하고 다시 뽑음
- path length median: 14.4067 m
- path length range: 2.26~31.90 m
- trajectory minimum clearance: 0.1581 m
- 최종 `passed=true`

`distribution_ok=false`는 Matterport/R2R 분포와 같은지를 보는 advisory 항목이며 collision/path
성공 gate가 아니다. 노은역 고유 공간에서 만든 20개 경로라는 점 때문에 분포가 다를 수 있다.

## 11. Stage 03b~03e — 경로 GT 품질 검증

사용자가 말한 03의 b, c, d, e도 모두 실행했다. 이들은 별도 경로 생성기가 아니라 03 결과가
재현 가능하고 parameter 선택이 타당한지 보여주는 검증/비교 단계다.

### 11.1 03b — 재현성 검증

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/03b_verify_reproduction.py --scene noeun_station_mid --mode random --num_episodes 20 --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

- 03 path JSON을 다시 읽어 동일 seed/계산으로 재현한다.
- 최초에는 19/20이었는데 03은 scan coverage mask를 적용하고 03b는 빠뜨린 것이 원인이었다.
- 두 단계의 navigable 계산을 일치시킨 뒤 F1 결정론 20/20 PASS가 됐다.
- knot knee는 19/20, median knot spacing은 0.89 m다.
- JSON: `apply_real/verify/noeun_station_mid.json`
- HTML: `logs/gs-vlnpe/apply_real/03b_verify_reproduction/noeun_station_mid/report.html`

### 11.2 03c — refine 방식 비교

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/03c_compare_refine.py --scene noeun_station_mid --mode random --num_episodes 20 --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

- argmax refine와 min-move 계열을 같은 20개 경로에서 비교한다.
- 둘 다 radius gate를 통과했다.
- argmax는 기준 결과와 Chamfer 0, min-move는 약 0.0944 m여서 argmax를 유지했다.
- JSON: `apply_real/compare/noeun_station_mid.json`
- HTML: `logs/gs-vlnpe/apply_real/03c_compare_refine/noeun_station_mid/report.html`

### 11.3 03d — parameter grid search

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/03d_grid_search_params.py --scenes noeun_station_mid --mode random --num_episodes 20 --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

검토 대상은 A* grid, refine radius, connectivity, clearance tie-break weight, waypoint spacing,
cubic output step, h_nav, robot radius, downsample mode다. 기존 메모에는 “8축”이라고 적힌 곳이
있지만 실제 나열 항목은 9개이므로 이 문서는 항목을 그대로 열거한다. 모든 candidate 실행과
regression gate가 통과했다.

- JSON: `apply_real/gridsearch/results.json`
- HTML: `logs/gs-vlnpe/apply_real/03d_grid_search_params/report.html`

### 11.4 03e — reference/path/04 pose overlay 비교

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/03e_compare_reference_path.py --scene noeun_station_mid --dataset noeun_generated --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

노은역에는 R2R instruction에 대응하는 외부 reference path가 없으므로, 03 내부 단계와 04가 실제
저장한 pose를 overlay한다.

- 노랑: 03 최종 path GT
- 청록: refine/smoothing 전 A* 계열 path
- 빨강: 04 저장 camera pose
- 04 pose와 03 route 일치: 20/20
- raw A*와 final GT median Chamfer: 0.101 m
- HTML: `logs/gs-vlnpe/apply_real/03e_compare_reference_path/noeun_station_mid_random/report.html`

## 12. Stage 04 — Isaac Sim D455 nominal RGB-D 관측 GT

### 목적

`04_render_obs_isaac.py`가 03의 20개 path를 읽고, 원본 USDZ를 Isaac Sim에 올린 뒤 각 pose에서
RGB, depth, camera pose를 렌더링한다. 이 결과가 현재 노은역 중간층 데이터셋의 기준 관측이다.

### D455 nominal profile

| 항목 | 값 |
|---|---:|
| image size | 480×270 |
| fx, fy | 240, 240 px |
| cx, cy | 240, 135 px |
| depth dtype | uint16 |
| depth unit | 0.001 m/raw |
| stored valid depth | 0.1~10.0 m |
| render near/far | 0.05/12.0 m |

intrinsic matrix:

```text
K = [[240,   0, 240],
     [  0, 240, 135],
     [  0,   0,   1]]
```

### 0.0001과 0.001의 정확한 해석

#### 원래 VLN-N1 D435i dataset camera

원본 파이프라인에서 `--camera dataset`을 사용하면 새 nominal camera가 아니라 VLN-N1의
`matterport3d_d435i` parquet에 저장된 camera intrinsic을 읽는다. 보존된 `target_schema.json`에서
확인한 episode 0의 값은 다음과 같다.

| 항목 | 원래 VLN-N1 / D435i dataset 값 |
|---|---:|
| image size | 480×270 |
| fx | 355.81463623046875 px |
| fy | 351.68701171875 px |
| cx | 240.0 px |
| cy | 135.0 px |
| depth dtype | uint16 |
| depth unit | 0.0001 m/raw = 0.1 mm/raw |
| uint16 유효 표현 상한 | 6.5534 m |
| 기존 04 생성 저장 cutoff | 3.0 m |
| 기존 render near/far | 0.05/10.0 m |

```text
K_D435i_dataset = [[355.81463623,   0.0, 240.0],
                    [  0.0,       351.68701172, 135.0],
                    [  0.0,         0.0,         1.0]]
```

약 6.5 m 한계는 `65534 raw × 0.0001 m/raw = 6.5534 m`로 생긴다. PNG depth는 uint16이며
이 pipeline은 0을 invalid로 두고 유효 최대 raw를 65534로 제한한다. 따라서 기존 scale로 10 m를
저장하면 overflow가 발생한다. 사용자가 기억한 “약 6.5 m까지밖에 표현하지 못했다”는 것은
이 저장 포맷의 수치적 표현 상한이다. 실제 기존 04는 이보다 보수적인 3.0 m cutoff를 적용했다.

- 6.5534 m: 기존 encoding이 이론적으로 저장할 수 있는 절대 최대 거리
- 3.0 m: 기존 04가 실제 생성물에 적용한 유효 depth cutoff

이 값은 실물 D435i의 공식 측정 가능 거리가 아니라 InternNav VLN-N1 dataset/04 저장 규약이다.

#### 노은역 D455 nominal과의 직접 비교

| 항목 | 원래 VLN-N1 / D435i dataset | 노은역 D455 nominal |
|---|---:|---:|
| 용도 | 기존 dataset GT 호환 | 노은역 synthetic 관측 생성 |
| profile 선택 | `--camera dataset` | `--camera d455_nominal` |
| image size | 480×270 | 480×270 |
| fx, fy | 355.8146, 351.6870 px | 240, 240 px |
| cx, cy | 240, 135 px | 240, 135 px |
| depth dtype | uint16 | uint16 |
| depth unit | 0.0001 m/raw | 0.001 m/raw |
| 한 raw step | 0.1 mm | 1 mm |
| uint16 유효 표현 상한 | 6.5534 m | 65.534 m |
| pipeline 저장 cutoff | 3.0 m | 10.0 m |
| render near/far | 0.05/10.0 m | 0.05/12.0 m |
| calibration 성격 | dataset parquet에 저장된 K | 특정 실물 장비가 아닌 nominal K |

D455 nominal은 `65534 raw × 0.001 m/raw = 65.534 m`까지 숫자로 표현할 수 있으므로 10 m
cutoff를 overflow 없이 저장한다. 10 m는 raw 10000이며 실제 결과에서도 최대 raw가 10000이었다.
대신 양자화 간격은 기존 0.1 mm/raw에서 1 mm/raw로 커진다. 10 m 범위를 확보하기 위해 표현
범위와 양자화 단위 사이의 이 trade-off를 선택한 것이다.

`0.001`은 **저장 단위/scale**이지 D455의 정확도가 10 m까지 보장된다는 뜻이 아니다. 65.534 m도
파일 형식의 이론적 상한일 뿐 센서 성능이 아니다. 실제 D455 수집 시에는 장비별
`get_intrinsics()`와 `get_depth_scale()` 실측값을 저장해야 하므로 profile 이름도 `nominal`이다.

### 실행 명령

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/04_render_obs_isaac.py --scene noeun_station_mid --mode random --num_episodes 20 --camera d455_nominal --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

### 저장 구조

```text
scripts/dataset_converters/gs_vlnpe/apply_real/obs/
└── noeun_station_mid_random_isaac_d455_nominal/
    ├── camera_config.json
    ├── episode_000000/
    │   ├── rgb/frame_0000.jpg ...
    │   ├── depth/frame_0000.png ...
    │   └── extrinsic/frame_0000.npy ...
    ├── ...
    └── episode_000019/
```

`extrinsic`이라는 디렉터리 이름만 보고 일반적인 world-to-camera matrix라고 가정하면 안 된다.
이 pipeline은 N1 action pose 호환 convention을 저장하고 `pose_convention=cam2world_gl` 변환 규약을
사용한다. 소비 코드에서는 기존 `geometry_utils.action_to_c2w(...)` 흐름을 따라야 한다.

### 실제 결과

- episode: 20/20 PASS
- 총 frame: 8,378
- episode별 frame 수:
  `285, 344, 282, 611, 72, 682, 487, 383, 708, 407, 159, 845, 65, 420, 181, 417, 72, 463, 583, 912`
- 모든 episode에서 RGB/depth/extrinsic 개수 일치
- depth shape/dtype: 270×480, uint16
- 최대 raw depth: 10000 = 10.000 m
- pose step violation: 0
- 전체 mesh-anchor median: 0.00462 m
- 대략적인 저장 용량: 3.1 GB
- HTML: `logs/gs-vlnpe/apply_real/04_render_obs_isaac/noeun_station_mid_random_d455_nominal/report.html`

당시 모든 파일과 report가 저장된 뒤 Isaac shutdown hook이 오래 기다리는 현상이 있어 session을
수동 종료했다. 이는 데이터 생성 실패가 아니며 저장물 전수 검사로 완결성을 확인했다.

## 13. 기본 Open3D 04를 실행한 이유와 사용하면 안 되는 부분

Isaac 이외의 원본 기본 renderer도 다음과 같이 실행했다.

```bash
EGL_PLATFORM=surfaceless /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/apply_real/04_render_obs.py --scene noeun_station_mid --mode random --num_episodes 20 --usd_path data/noeun_station/noeun_station_collision.usdz --camera d455_nominal --out_dir scripts/dataset_converters/gs_vlnpe/apply_real --log_dir logs/gs-vlnpe/apply_real
```

Open3D 0.19가 USDZ를 직접 읽지 못해 코드가 package 내부 `mesh.ply`를 임시 디렉터리에 꺼내
렌더링한다. geometry/depth 관점에서는 8,378 frame, uint16, 10 m cutoff, pose step 0,
mesh-anchor 약 0.00001 m로 좋았다. 그러나 PLY에 색/UV/texture가 없어 RGB가 거의 흰색이었다.

- RGB mean 범위: 249.03~254.75
- RGB std 범위: 1.02~11.24
- 최종 판정: RGB 데이터셋으로는 FAIL
- HTML: `logs/gs-vlnpe/apply_real/04_render_obs/noeun_station_mid_random/report.html`

따라서 이 결과는 depth/geometry 진단용이며, 최종 노은역 관측은 반드시 Isaac 결과를 사용한다.

## 14. 처음부터 재현할 때의 실행 순서

다음 명령을 저장소 루트에서 순서대로 실행한다. Conda는 필요하지 않으며 사용하지 않는다.

1. USDZ SHA-256을 확인한다.
2. Stage 00 명령을 실행하고 target schema와 HTML이 PASS인지 확인한다.
3. Stage 01 명령을 실행하고 face 14,618, area 798.1859 m², ROI가 같은지 확인한다.
4. Stage 02 명령을 실행하고 ESDF NPZ/JSON과 geometry sanity PASS를 확인한다.
5. Stage 03 명령을 실행하고 20/20, hard collision 0인지 확인한다.
6. Stage 03b, 03c, 03d를 실행해 경로 재현성과 parameter gate를 확인한다.
7. Stage 04 Isaac 명령을 실행하고 20 episode, 8,378 frame, stream count 일치를 확인한다.
8. Stage 03e를 실행해 03 path와 04 pose가 20/20 같은 route인지 확인한다.
9. 각 단계의 `report.html`을 열어 수치뿐 아니라 시각화도 확인한다.

03이 GT path를 만드는 것은 맞지만, 그렇다고 00~02보다 03을 먼저 실행하는 것은 아니다. 새
USDZ에는 기존 episode GT가 없어도 00은 asset contract를 검사하고, 01은 logical ROI를 만들며,
02는 geometry만으로 ESDF를 만들 수 있다. 그 ESDF가 있어야 03이 path GT를 생성할 수 있다.

## 15. 단계별 PASS 체크리스트

### 00

- [ ] 입력 SHA-256 일치
- [ ] USD readable
- [ ] Z-up, metersPerUnit=1
- [ ] mesh prim/triangle 존재
- [ ] 중간층 후보 14,618 face
- [ ] report.html 생성

### 01

- [ ] `floor_z=-0.05`
- [ ] angle 15°, band ±0.45 m
- [ ] selected area 798.1859 m²
- [ ] logical ROI가 source bounds 내부
- [ ] `usd_path`는 원본, `mesh_path=null`
- [ ] report.html 생성

### 02

- [ ] grid 1345×1308×40, voxel 0.05 m
- [ ] occupancy와 ESDF 배열 생성
- [ ] obstacle fraction이 0도 1도 아님
- [ ] navigable fraction 0.879639 부근
- [ ] geometry sanity PASS
- [ ] report.html 생성

### 03 및 03b~03e

- [ ] path 20/20
- [ ] hard collision 0
- [ ] seed 0 및 parameter 기록 존재
- [ ] 03b deterministic F1 20/20
- [ ] 03c argmax 선택 근거 존재
- [ ] 03d candidate/regression 통과
- [ ] 03e 03 route/04 pose 20/20 일치
- [ ] 각 report.html 생성

### 04 Isaac

- [ ] `camera_config.json`이 d455_nominal
- [ ] K, 480×270, 0.001 m/raw, cutoff 10 m
- [ ] episode 000000~000019 존재
- [ ] RGB/depth/extrinsic frame 수가 episode마다 동일
- [ ] 총 8,378 frame
- [ ] depth uint16, 270×480, raw ≤10000
- [ ] pose step violation 0
- [ ] mesh-anchor millimeter 수준
- [ ] report.html 생성

## 16. 최종 산출물의 역할

| 산출물 | GT/역할 |
|---|---|
| 원본 USDZ mesh | 공간 표면 geometry GT |
| scene_meta JSON | 중간층 logical ROI와 좌표계 계약 |
| ESDF NPZ/JSON | voxel occupancy, obstacle, free/navigable, clearance |
| path JSON | 20개 episode의 trajectory GT |
| RGB JPG | Isaac USDZ visual observation |
| depth PNG | D455 nominal 1 mm/raw의 synthetic depth GT |
| extrinsic NPY | 각 frame의 pipeline pose/action convention |
| report HTML | 각 단계 수치·시각 품질 검증 기록 |

## 17. 원본 코드에서 벗어난 범위

수정 방향은 새로운 노은역 전용 복제 script를 여러 개 쌓는 대신 원본과 같은 파일 이름과 기존
default 흐름을 유지하고, 새 scene에 필요한 명시적 option branch만 더하는 것이었다.

- 00: `--usd_path`, `--floor_z`일 때 새 USDZ asset 검사 branch
- 01: `--usd_path`일 때 logical level metadata 생성 branch
- 02: `--ref_h_b`, USD app lifecycle, ROI-first sampling
- 03: GT가 없는 new scene용 random path 생성과 원래 retry/gate 재사용
- 04: camera profile 분리, D455 nominal, new-scene USDZ 로딩/visibility helper
- 기존 N1/D435i default profile과 기존 dataset 경로는 유지

즉 새 데이터에 꼭 필요한 입력 분기와 camera profile은 추가했지만, A*, refine, smoothing, report
gallery, frame 저장 규약 같은 핵심 pipeline은 원본 GS-VLNPE 구조를 재사용했다.

## 18. 자주 생기는 오해

1. **“노은역 GT가 없다”**
   틀린 표현이다. 처음에 기존 VLN trajectory/observation GT가 없었을 뿐 USDZ mesh는 공간 GT다.
   03과 04를 거쳐 path와 observation GT도 생성했다.

2. **“noeun_mid.usdz를 새로 만들었다”**
   아니다. `noeun_station_mid`는 logical scene ID이며 원본 USDZ는 그대로다.

3. **“01이 자동으로 중간층을 발견했다”**
   아니다. `-0.05`를 지정하고 그 주변 surface와 bounds가 유효한지 검증했다. 다만 독립 Z
   histogram에서도 이 높이는 가장 큰 5 cm 상향면 area bin에 해당한다.

4. **“0.001로 바꾸면 D455가 65 m 또는 10 m에서 정확하다”**
   아니다. 0.001은 depth raw unit이고, 10 m는 이 데이터 생성의 cutoff다.

5. **“Open3D 04가 더 정합하니 그것을 써야 한다”**
   geometry 정합은 좋지만 texture가 없어 RGB가 흰색이므로 최종 관측에는 부적합하다.

6. **“03을 먼저 만들어야 하므로 00~02는 실행할 수 없다”**
   아니다. 현재 new-scene 흐름에서 00~02는 외부 trajectory GT 없이 geometry로 실행 가능하고,
   02 결과가 있어야 03 경로 GT를 만들 수 있다.

## 19. 관련 근거 파일

이 통합 문서는 다음 실제 파일을 기준으로 작성했다.

- `scripts/dataset_converters/gs_vlnpe/apply_real/00_inspect_vln_n1.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/01_prepare_scene.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/02_build_freemap_esdf.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/03_sample_gt_paths.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/03b_verify_reproduction.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/03c_compare_refine.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/03d_grid_search_params.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/03e_compare_reference_path.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/04_render_obs.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/04_render_obs_isaac.py`
- `scripts/dataset_converters/gs_vlnpe/esdf_utils.py`
- `scripts/dataset_converters/gs_vlnpe/camera_profiles.py`
- `scripts/dataset_converters/gs_vlnpe/usdz_scene_utils.py`
- `scripts/dataset_converters/gs_vlnpe/apply_real/target_schema.json`
- `scripts/dataset_converters/gs_vlnpe/apply_real/scene_meta/noeun_station_mid.json`
- `scripts/dataset_converters/gs_vlnpe/apply_real/esdf/noeun_station_mid.json`
- `scripts/dataset_converters/gs_vlnpe/apply_real/paths/noeun_station_mid_random.json`
- `scripts/dataset_converters/gs_vlnpe/apply_real/obs/noeun_station_mid_random_isaac_d455_nominal/camera_config.json`
- `.claude/memory/codex_00_noeun_asset_schema.md`부터
  `.claude/memory/codex_04_noeun_d455_observation_gt.md`까지의 단계별 실행 기록

이 문서의 숫자와 명령은 2026-08-14 현재 위 코드 및 보존된 산출물 기준이다. 이후 parameter나
입력 USDZ가 바뀌면 SHA-256, ROI, ESDF grid, path 수치, frame 수가 달라질 수 있으므로 새 실행의
JSON과 HTML을 함께 보존해야 한다.
