# ply2usdz — 원본 3DGS PLY → sim 배경 usdz

PortalCam(XGRIDS LCC)에서 내보낸 3DGS PLY와 충돌 mesh를 Isaac Sim 배경 파일 `<이름>_collision.usdz` 하나로 만든다.
노은역·월드컵역·학습 데이터(`data/GS_USDZ`)와 같은 형식이라 sim(`GUIDEDOG_BACKGROUND_USD`)과
데이터 변환(`nav3d/noeun/11_check_assets.py` 등)에 그대로 쓴다.

## 용어

| 용어 | 뜻 |
|---|---|
| 3DGS PLY | Gaussian splat 점 데이터 파일 (보이는 배경). 바닥·벽 같은 면 정보는 없다 |
| 충돌 mesh | GS 와 같은 좌표계의 삼각형 mesh (`.obj`). 로봇이 밟고 LiDAR·depth 가 맞는 면 |
| NuRec | Isaac Sim 이 GS 를 렌더하는 NVIDIA 형식. usdz 안 `<이름>.nurec` |
| near_clip | 카메라에서 이 거리(m) 안쪽 GS 는 그리지 않는 설정 |
| 64바이트 정렬 · ZIP32 | usdz(=무압축 zip) 규격. OpenUSD 는 ZIP64 헤더를 못 읽어 파일이 4 GiB 를 넘으면 안 된다 |

## 입력 (2개)

| 입력 | 파일 | 조건 |
|---|---|---|
| ① GS | `point_cloud.ply` | 표준 3DGS PLY, `f_rest` 45개(SH 3차) |
| ② 충돌 mesh | `<이름>.obj` (또는 삼각형 `.ply`) | ① 과 같은 좌표계 |

LCC 내보내기 폴더(장면 하나)에서의 위치:

```
<장면>_ply/
  ply-result/point_cloud/iteration_*/point_cloud.ply    ← ① GS (장면 전체)
  ply-result/point_cloud/iteration_*/environment.ply    ✗ 미리보기용(점 수만 개), 쓰지 않는다
  mesh-files/<이름>.obj                                  ← ② 충돌 mesh
```

## 출력 (1개)

`<이름>_collision.usdz` — 위치·이름은 실행할 때 정한다. 안에 든 파일:

| 파일 | 내용 | 어디서 |
|---|---|---|
| `default.usda` | 루트 (/World) | ① |
| `<이름>.nurec` | GS 데이터 + 렌더 설정 (near_clip 1e-8) | ① |
| `gauss.usda` | GS Volume. proxy → /World/mesh | ① |
| `mesh.ply` | 충돌 mesh (삼각형) | ② |
| `mesh.usd` | 같은 mesh 의 USD. 충돌 켬 · 화면에 숨김 | ② |

## 준비 (처음 한 번)

3DGRUT venv 를 만든다 (기본 `~/git/3dgrut-venv`, 3DGRUT `16b7b19` + PortalCam 좌표 패치, 10~30분).
`setup_3dgrut_venv.sh` · `requirements-3dgrut.txt` 는 usd2usdz 에서 가져온 사본이다.

```bash
scripts/dataset_converters/ply2usdz/setup_3dgrut_venv.sh
```

이미지 `dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090` 과 GPU 가 필요하다.

## 실행 (호스트, InternNav 폴더에서)

```bash
scripts/dataset_converters/ply2usdz/run_ply2usdz.sh <입력 ① gs.ply> <입력 ② mesh.obj> <출력 이름_collision.usdz>
```

| 구분 | 인자 | 뜻 | 왜 필요한가 |
|---|---|---|---|
| 입력 ① | `gs.ply` | 원본 3DGS PLY (`point_cloud.ply`) | 보이는 배경 |
| 입력 ② | `mesh` | 충돌 mesh `.obj` 또는 삼각형 `.ply` | PLY 에는 밟을 바닥·LiDAR 가 맞을 면이 없다 (좌표계 일치는 스크립트가 확인하지 않음) |
| 출력 | `<이름>_collision.usdz` | 결과 파일 | `<이름>` 이 usdz 안 `.nurec` 이름이 된다. 이미 있으면 덮어쓴다 |
| 옵션 | `--nurec-name` | `.nurec` 이름을 따로 정할 때 | 렌더링과 무관. 보통 안 준다 |
| 옵션 | `--near-clip` | 기본 1e-8 | 3DGRUT 최신 기본값 · 노은역 학습 데이터와 같음 |

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `VENV` | `~/git/3dgrut-venv` | 3DGRUT venv |
| `IMAGE` | `dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090` | 실행 이미지 |
| `MEM` | `80g` | 컨테이너 메모리 상한. 2천만 점대 장면은 수십 GB 를 쓴다 |

### 예) ETRI 병합 장면 (결과는 ETRI 폴더 바로 아래)

```bash
E=/home/universe/data/XGRIDS_Potalcam_Export/ETRI; n=261021_ETRI_MERGE_BUILDING1; scripts/dataset_converters/ply2usdz/run_ply2usdz.sh $E/${n}_ply/ply-result/point_cloud/iteration_100/point_cloud.ply $E/${n}_ply/mesh-files/*.obj $E/${n}_collision.usdz
```

### 예) 노은역 (기존 `noeun_station_collision.usdz` 를 덮어쓰지 않게 하위 폴더로)

```bash
D=/home/universe/data/XGRIDS_Potalcam_Export/NoeunStation; scripts/dataset_converters/ply2usdz/run_ply2usdz.sh $D/point_cloud.ply $D/noeun_station_collision_unzipped/mesh.ply $D/ply2usdz/noeun_station_collision.usdz
```

## sim 에서 보기 (sim + InternNav + gdm 한 번에)

`XGRIDS_Potalcam_Export/` 는 sim 컨테이너에서 `/isaac-sim/workspace/` 로 보인다 (sim_docker `.env` 의 `SIM_SCENE_STORE` + 링크).

```bash
GUIDEDOG_BACKGROUND_USD=/isaac-sim/workspace/NoeunStation/noeun_station_collision-internnav.usdz scripts/sim/run_vln_sim.sh -m a --map noeun --gdm
```

```bash
GUIDEDOG_BACKGROUND_USD=/isaac-sim/workspace/ETRI/261021_ETRI_MERGE_BUILDING1_collision.usdz scripts/sim/run_vln_sim.sh -m a --map none --sim indoor --gdm
```

- ETRI 병합 장면은 런처에 전용 씬 이름이 없어 `--sim indoor` 설정을 빌린다. 스폰 위치가 안 맞으면 `GUIDEDOG_INIT_X/Y/Z/YAW` 로 옮긴다.
- sim 이 이미 떠 있으면 그대로 쓴다 → 배경을 바꿀 때는 먼저 `scripts/sim/run_vln_sim.sh stop --gdm`.

## 단계 (`ply2usdz.py`)

1. 3DGRUT `ply_to_usd` — PLY → NuRec usdz (좌표 그대로)
2. `.nurec` 의 near_clip 을 1e-8 로 (16b7b19 템플릿 값 0.2 를 바꿈)
3. mesh → `mesh.ply` + `mesh.usd`(충돌 켬 · 숨김), `gauss.usda` 에 mesh 참조와 Volume proxy 추가
   (3DGRUT `add_mesh_to_usdz` 와 같은 구성. 16b7b19 에는 이 스크립트가 없어 직접 구현)
4. 64바이트 정렬 · ZIP32 로 다시 씀. 4 GiB 를 넘으면 중단

## 검증 (2026-10-29)

노은역 원본 `point_cloud.ply` → 기존 `noeun_station_collision.usdz`(3dgs2usd) 와 비교:

| 항목 | 결과 |
|---|---|
| `.nurec` (GS · near_clip) | 같음 |
| `mesh.ply` | 바이트 같음 |
| `mesh.usd` | USD 내용 같음 (바이너리 배치만 36 B 차이) |
| `default.usda` · `gauss.usda` | `timeCodesPerSecond = 24` 한 줄 없음 (USD 기본값이 24 라 의미 같음) |
| 정렬 · ZIP32 | OK |

sim_docker `third_party/usd2usdz_convert.sh`(3DGRUT `a37ef72`) 결과와도 GS · mesh 가 같다.

ETRI 병합 장면 (mesh 정점·면 수 obj 와 같음, 정렬 · ZIP32 OK):

| 장면 | Gaussian | mesh 면 | usdz | 시간 |
|---|---:|---:|---:|---:|
| 261021_ETRI12_MERGE_1F2F | 18,718,725 | 303,443 | 2.22 GB | 41 s |
| 261021_ETRI12_MERGE_3FOUT | 22,782,107 | 1,702,592 | 2.75 GB | 49 s |
| 261021_ETRI_MERGE_BUILDING1 | 26,598,638 | 2,640,661 | 3.23 GB | 58 s |

자세한 기록: `.claude/memory/ply2usdz_result.md`.

## 주의

- 결과 usdz 가 4 GiB 를 넘으면 Isaac Sim 이 못 연다 → 스크립트가 멈춘다. Gaussian 1개 ≈ 115 B 라 약 3,500만 개가 한계.
- `f_rest` 가 45개가 아닌 PLY(SH 0~2차)는 시험하지 않았다. 필요하면 usd2usdz `gs_to_ply.py` 로 45개로 맞춘 뒤 넣는다.
- 컨테이너는 root 로 돈다(venv python 이 isaac-sim 사용자 전용 `/isaac-sim` 을 가리킴). 결과 파일 소유자는 실행한 사용자로 되돌린다.
