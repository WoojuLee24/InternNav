"""mini · full 데이터셋 루트 → 릴리스 · 원본 R2R · 씬 경로 (vlnce 51 렌더 · common/label_paths 가 같이 쓴다).

두 데이터셋은 루트만 다르고 구조가 같다:
    <루트>/vln_ce/traj_data/r2r/<씬>/          릴리스 학습 데이터 (LeRobot)
    <루트>/vln_ce/raw_data/r2r/<split>/         원본 R2R json.gz
    <루트>/scene_data/mp3d_ce/mp3d/<씬>/        habitat glb · navmesh
"""

from pathlib import Path

ROOTS = {'mini': 'data/InternData-N1-v0.5-mini', 'full': 'data/InternData-N1'}
MINI = ROOTS['mini']


def release_root(root: str) -> str:
    return f'{root}/vln_ce/traj_data/r2r'


def raw_root(root: str) -> str:
    return f'{root}/vln_ce/raw_data/r2r'


def scene_root(root: str) -> str:
    return f'{root}/scene_data/mp3d_ce/mp3d'


def apply_data_root(cfg: dict, root: str) -> dict:
    """instruction_labeling render.yaml 설정(load_cfg 결과)의 릴리스 · 원본 · 씬 경로를 root 로 바꾼다 (제자리에서, 같은 dict 를 돌려준다).
    이미지 기준 파일(jpeg 양자화 표 · parquet 스키마)도 같은 루트의 같은 파일로."""
    R = Path(root).resolve()
    cfg['paths']['release_root'] = R / 'vln_ce' / 'traj_data' / 'r2r'
    cfg['paths']['raw_root'] = R / 'vln_ce' / 'raw_data' / 'r2r'
    cfg['paths']['scene_root'] = R / 'scene_data' / 'mp3d_ce' / 'mp3d'
    for k in ('jpeg_qtables_ref', 'schema_ref'):
        if cfg.get('image', {}).get(k, '').startswith(MINI + '/'):
            cfg['image'][k] = str(R / cfg['image'][k][len(MINI) + 1:])
    return cfg
