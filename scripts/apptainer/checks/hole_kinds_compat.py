import sys; sys.path.insert(0,'scripts/dataset_converters/nav3d/utils')
from path_quality import load_holes
c='scripts/dataset_converters/nav3d/logs/noeun/11_check_assets/noeun_station/holes.csv'
a=load_holes(c); b=load_holes(c,('mesh',)); g=load_holes(c,('mesh','gs'))
print(f'기본(=mesh): {len(a)}   명시 mesh: {len(b)}   mesh+gs: {len(g)}')
assert len(a)==len(b) and (a==b).all(), '기본값이 기존 동작과 다르다'
print('기본값 = 기존 동작 동일 확인')
