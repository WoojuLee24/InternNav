"""Helpers isolated to raw/collision USDZ scenes.

No function in this module mutates the source USDZ.  Visibility changes are
authored only on the composed runtime stage used by Isaac sensors.
"""

import json
from pathlib import Path


def load_new_scene(out_dir, scene: str, load_usd_mesh):
    meta_path = Path(out_dir) / 'scene_meta' / f'{scene}.json'
    if not meta_path.is_file():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get('scene_type') != 'new_scene':
        return None
    if not meta.get('passed', False):
        raise ValueError(f'new_scene preparation failed: {meta_path}')
    usd_path = Path(meta['usd_path'])
    if not usd_path.is_file():
        raise FileNotFoundError(f'new_scene USD not found: {usd_path}')
    return usd_path, load_usd_mesh(str(usd_path)), meta


def expose_collision_meshes_for_rendering(stage, usd_path, usd_geom) -> int:
    """Make hidden collision meshes visible on a runtime stage only."""
    if not Path(usd_path).name.endswith('_collision.usdz'):
        return 0
    changed = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith('/World/Scene'):
            continue
        if not prim.IsA(usd_geom.Mesh):
            continue
        if 'PhysicsCollisionAPI' not in set(prim.GetAppliedSchemas()):
            continue
        imageable = usd_geom.Imageable(prim)
        if imageable.GetVisibilityAttr().Get() == usd_geom.Tokens.invisible:
            imageable.GetVisibilityAttr().Set(usd_geom.Tokens.inherited)
            changed += 1
    return changed
