# 노은역 ↔ VLN-PE/VLN-CE 포맷 검증

`validate_noeun_vln_formats.py`는 노은역 04 Isaac 20개 episode, 실제 VLN-PE episode,
VLN-CE gzip episode JSON과 Habitat sensor config를 함께 읽어 저장 계약과 직접 loader 호환성을
비교한다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/format_validation/validate_noeun_vln_formats.py
```

출력 `noeun_vln_format_validation.json`의 최상위 `status=PASS`는 비교 입력을 정상적으로 전수/표본
검증했다는 뜻이다. 실제 포맷 동일 여부는 `compatibility.*.exact_storage_format_same`에서 확인한다.

검증 뒤 사람이 보는 HTML을 생성한다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/format_validation/generate_format_report.py
```

결과는 같은 폴더의 `report.html`과 `report_assets/`에 저장된다.
