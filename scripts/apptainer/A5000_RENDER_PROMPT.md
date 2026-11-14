# A5000 사진 찍기 분담 준비 — a6000 data2 세션용 (2026-11-14)

## 상황 (6000pro 서버 = 이 저장소 /data/users/dnwn24/src/InternNav, 브랜치 noeun-landmark-house, HEAD e94000c)
- 노은역 랜드마크 경로 1,752개(학습 1,549 · 평가 A `eval` 103 · 평가 B `eval_seen` 100)를 만들었다.
  RBQ 걷기(42단계)는 6000pro GPU 0·1에서 진행 중이며 약 15시에 끝난다.
- 끝나면 `scripts/apptainer/noeun_landmark_pipeline.sh` 가 split마다 성공 경로를 골라
  (`logs/noeun/paths_landmark_ext/walk/sel_{train,eval,eval_seen}.json`: 학습 1200 · 평가 A 80 · 평가 B 80)
  51단계 사진(vln_pe · vln_ce 두 형식)을 찍는다.
- 사진은 약 16시간 걸린다(Blackwell 기준 vln_pe 0.36 s/장 · vln_ce 0.23 s/장 · 경로당 약 130장).
  그래서 일부를 A5000(GPU 4장 사용 가능)에 맡긴다. 나누는 비율은 A5000 속도를 잰 뒤 6000pro 세션에서 정한다.
- 걷기가 끝나기 전에는 A5000에서 실제 데이터를 렌더하지 않는다. 지금 할 일은 **준비와 시험**뿐이다.

## 할 일
1. **코드: GitHub push → A5000 pull** (6000pro에서 커밋, 브랜치 noeun-landmark-house)
   - 수정: `scripts/dataset_converters/nav3d/noeun/73_goal_candidates.py` · `74_make_paths.py` · `scripts/dataset_converters/nav3d/utils/path_quality.py`
   - 새 파일: `scripts/dataset_converters/nav3d/noeun/74b_make_paths_parallel.py` · `74c_extend_paths.py` · `scripts/apptainer/noeun_landmark_pipeline.sh` · `scripts/apptainer/A5000_RENDER_PROMPT.md` · `.claude/memory/261114_noeun_paths_render_result.md`
   - 커밋 메시지·push 전에 사용자에게 diff 요약을 보여주고 확인받는다.
2. **git에 없는 것은 Gdrive** (`gdrive:InternNav/data/noeun_eval/landmark_render_20261114/` 같은 새 폴더)
   - 이미 Gdrive에 있음: `gdrive:InternNav/data/noeun_eval/landmark_house_20261113/rbq10_assets.tar` (RBQ usd · 정책 · policy_runtime.py)
   - 올릴 것:
     - 역 모델 `/data/users/dnwn24/data/GS_USDZ/noeun_station_collision.usdz` (A5000에 이미 있으면 크기·md5 비교 후 생략)
     - 시험용 걷기 결과 `logs/env_cmp/v5/walk/rtest/walk_p3.json` (3 ep) 와 비교용 6000pro 렌더 `logs/env_cmp/v5/walk/rtest/out_p3/` (vln_pe) · `ce_p3/` (vln_ce)
     - 도구 `logs/env_cmp/tools/frames.py` (렌더 npy → 시작·중간·끝 PNG)
     - 걷기가 끝나면(15시 이후): `logs/noeun/paths_landmark_ext/walk/sel_*.json` · `paths_meshfloor_straight_v5.json`
3. **A5000 준비** (`ssh dnwn24@10.254.92.150 -p 20022`, 비밀번호는 사용자에게 받는다 — 파일·로그에 적지 않는다)
   - 저장소 pull, `rbq10_assets.tar` 를 `<workspace>/assets/` 에 풀기 (`assets/rbq10/usd/rbq10.usd` 확인), usdz 를 `/ws/assets/GS_USDZ/` 로 보이게 연결.
   - Isaac 컨테이너 확인. 51단계는 `scikit-image` 가 필요하다. 6000pro 에서는 컨테이너 밖 폴더에 설치해 연결했다:
     `python.sh -m pip install --no-deps --target <pylib> scikit-image==0.25.2 lazy_loader imageio tifffile` 후 `-B <pylib>:/pylib` · `PYTHONPATH=$PYTHONPATH:/pylib`.
     6000pro 의 Isaac 실행 앞부분 예: `logs/env_cmp/isaac_walk_cmd.txt` (bind · env 목록 참고).
   - GPU 4장이 다른 사람 작업에 쓰이고 있지 않은지 `nvidia-smi` 로 확인.
4. **시험 렌더** (A5000 GPU 1장, 3 ep):
   `/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/51_render_rbq.py --walk_json <walk_p3.json> --format vlnpe --out_root <tmp>/out_p3 --log_dir <tmp>/log`
   그리고 `--format vlnce` 도. 기록할 것:
   - 속도 (장/초, 형식별). 6000pro: vln_pe 244장 86 s · vln_ce 244장 56 s (같은 3 ep).
   - 6000pro 결과와 비교: 같은 ep 의 RGB 평균 차이 · 깊이 차이 (frames.py 로 PNG 를 만들어 눈으로도). 6000pro 값: ep0 106장, RGB 평균 191.4.
   - GPU 4장 동시 실행이 되는지 (6000pro 는 한 GPU 에 3개 동시는 거의 안 빨라졌다 → GPU 당 1개 권장).
5. 결과(속도 · 비교 · 준비 완료 여부)를 6000pro 세션에 넘길 짧은 요약으로 정리한다. 실제 분담 렌더는 6000pro 세션이 걷기 결과를 보고 지시한다.

## 지킬 것
- 6000pro 의 실행 중인 작업(걷기 · 파이프라인 스크립트)은 건드리지 않는다. 6000pro GPU 는 0·1 만.
- 커밋·push 는 사용자 확인 후. 데이터 삭제 금지. 비밀번호를 파일·커밋·로그에 남기지 않는다.
