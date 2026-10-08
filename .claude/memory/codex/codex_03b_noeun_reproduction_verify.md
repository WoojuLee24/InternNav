# Codex 03b — 노은역 경로 재현 역검증

## 원칙

HTML을 별도 작성하지 않았다. 박사님이 작성한 `03b_verify_reproduction.py`의 기존 분석, 시각화, `save_gallery` 코드를 실행해 생성했다. 입력만 parquet 대신 `03`의 노은역 random 경로 GT로 연결했다.

## 수정 및 발견 사항

- `--mode random`에서 `paths/noeun_station_mid_random.json`을 읽는다.
- 최초 실행은 F1 19/20이었다. 03은 scan coverage mask를 쓰는데 03b 재계획에는 빠져 있어 한 경로가 다른 corridor로 갔다.
- 03b의 navigable 계산을 03과 동일하게 맞춘 후 F1 20/20이 됐다.

## 실행 결과

- F1 결정성: 20/20 동일
- knot 무릎: 19/20
- knot 간격 중앙값: 0.89m
- 출력 JSON: `noeun_pipeline_codex/verify/noeun_station_mid.json`
- HTML: `logs/gs-vlnpe/codex_noeun/03b_verify_reproduction/noeun_station_mid/report.html`
