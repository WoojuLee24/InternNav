# P-A + G4 결과 — mesh 타일 빌더 & 무손실 검증

작성 2026-08-14. 신규 `scripts/dataset_converters/gs_vlnpe/05_build_mesh_tiles.py`,
리포트 `scripts/debugging/01b_pa_report.py` → `logs/embodiment_augment/pa/` + Artifact.

## 05 mesh 타일 빌더 (offline)
씬 mesh를 stride=R(기본 6m) grid로 crop, 각 타일=중심±(R+margin) 박스 안 face submesh, **.ply(지오메트리만)**
저장. v1은 depth만 쓰므로 텍스처 불필요(v2에서 텍스처 타일). 출력 `data/embodiment_aug/tiles/<scene>/tile_gx_gy.ply` + `index.json`.
- 17DRP5sb8fy: grid 4×3 → **12 타일, 4.4~8.2 MB/타일** (full obj 199MB).
- s8pcmisQ38h(21m): grid 5×3 → **15 타일, 4.3~10 MB/타일**.
- → worker가 카메라 주변 **타일 1개(~4-10MB)** 만 로드 → feasibility 문서의 "full mesh 수백MB 상주" ✗ 해소.
- `tile_for_xy(index, x, y)`로 카메라 xy → 타일 매핑.

## G4 — 타일 crop 무손실
tile_0_0(ep0 커버)에서 렌더한 depth vs **full 씬** 렌더 depth: **median 0.00000 m**(완전 일치).
→ margin(=depth clip 5m)이 카메라 시야를 완결시켜 crop이 무손실. 타일링 아키텍처 검증됨.

## occupancy
vln_ce는 vln_n1과 **동일 mesh(mp3d)** → 기존 vln_n1 occ npz(`gs_vlnpe/logs/esdf/<scene>.npz`) 재사용.
전체 61 vln_ce 씬은 02를 mesh로 1회 실행하면 됨(3D occ라 다층 포함, augmenter가 T_sf2mesh z로 층 선택).

## 남은 P-A 조각 (P-B에서)
- **에피소드별 T_sf2mesh 캐시**: `vlnce_align.align_episode`를 vln_ce 전 에피소드에 배치 실행해 저장
  (augmenter가 로드). 방법은 G1에서 확립(순수 GT 해석적).
- 전체 61씬 02 occ + 05 타일 배치(현재 2씬만).

## 다음
- **G3**: on-the-fly 예산(타일 로드+occ→2D→A*+refine+spline+depth 렌더) 실측 <30ms/sample.
- **P-B**: EmbodimentAugmenter(depth+BEV) + dataloader hook + config threading + docs.
