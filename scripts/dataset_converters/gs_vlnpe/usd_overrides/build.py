"""덧칠 레이어(.usda)를 이 폴더에 만든다 — 원본 USD는 건드리지 않는다.

왜 "모아두기"가 아니라 "빌더"인가
---------------------------------
`.usda` 안의 sublayer 경로는 **그 파일 위치 기준 상대경로**다. 그래서 다른 곳에서 만든
`.usda`를 이 폴더로 `cp`하면 sublayer가 끊어져 조용히 빈 씬이 된다. 항상 최종 위치에서
새로 만들어야 한다 — 그게 이 스크립트다.

만들어진 파일은 레포에 체크인해 두고 파이프라인에 그대로 먹인다:

    ... /04_render_obs_isaac.py --usd_override scripts/dataset_converters/gs_vlnpe/usd_overrides/<파일>

커맨드
------
목록 보기:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --list

노은역 레이어 만들기:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --recipe noeun_collision_visible

mp3d 텍스처 레이어를 전 씬에 대해 만들기:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --recipe mp3d_textures --scene all

만들어진 레이어가 원본과 합성되는지 확인만 하기 (쓰기 없음):
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_overrides/build.py --verify_only

Isaac Sim을 띄우지 않는다 — `pxr`만 쓰므로 수 초에 끝난다.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GS_VLNPE = HERE.parent
USD_STUDY = GS_VLNPE / 'usd_study'
for p in (str(HERE), str(USD_STUDY)):
    if p not in sys.path:
        sys.path.insert(0, p)

import s2_assets  # noqa: E402
from recipes import RECIPES  # noqa: E402
from usd_study_utils import (  # noqa: E402
    PreservationGuard,
    mp3d_usd_variants,
    new_override_layer,
    scene_mesh_dir,
)

SCRIPT_NAME = 'usd_overrides/build'


def build_visibility(spec: dict, out_path: Path) -> dict:
    """숨은(`visibility=invisible`) 충돌 메시를 보이게 덮어쓴다."""
    from pxr import Usd, UsdGeom

    base = Path(spec['base'])
    stage = Usd.Stage.Open(str(base))
    targets = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
        and 'PhysicsCollisionAPI' in set(prim.GetAppliedSchemas())
        and UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
    ]
    layer = new_override_layer(base, out_path)
    ov_stage = Usd.Stage.Open(layer)
    for path in targets:
        UsdGeom.Imageable(ov_stage.OverridePrim(path)).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    layer.Save()
    return {'targets': len(targets), 'detail': f'숨은 충돌 메시 {len(targets)}개를 보이게'}


def build_textures(spec: dict, out_path: Path, scene: str) -> dict:
    """텍스처 asset 속성을 이 레이어 위치 기준 상대경로로 재지정한다."""
    variants = dict(mp3d_usd_variants(spec['usd_root'], scene))
    if spec['variant'] not in variants:
        raise FileNotFoundError(f"{scene}에 {spec['variant']} 변종이 없다 (있는 것: {list(variants)})")
    base = variants[spec['variant']]
    desc = s2_assets.describe_assets(spec['variant'], base)
    # s2_assets는 파일명을 `override_textures.usda`로 고정해 쓴다 — 만든 뒤 원하는 이름으로 옮긴다.
    res = s2_assets.build_texture_override(base, desc['rows'], out_path.parent)
    made = out_path.parent / 'override_textures.usda'
    if made.resolve() != out_path.resolve():
        if out_path.exists():
            out_path.unlink()
        made.rename(out_path)
    return {'targets': res['retargeted'],
            'detail': f"텍스처 {res['retargeted']}개 재지정 · 로컬 resolve {res['resolved_local']} · "
                      f"아직 박힌 경로 {res['still_baked']}"}


def base_of(spec: dict, scene: str = None) -> Path:
    """레시피가 덧칠하는 원본 파일 경로 — 보존 검사와 verify가 같이 쓴다."""
    kind = spec['kind']
    if kind == 'visibility':
        return Path(spec['base'])
    elif kind == 'textures':
        return dict(mp3d_usd_variants(spec['usd_root'], scene))[spec['variant']]
    else:
        assert False, f'unreachable kind={kind!r}'


def verify(out_path: Path) -> dict:
    """만든 레이어를 열어 원본이 실제로 합성돼 올라오는지 본다 (sublayer가 끊기면 여기서 걸린다)."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(out_path))
    prims = sum(1 for _ in stage.Traverse())
    root = stage.GetRootLayer()
    return {
        'prims': prims,
        'defaultPrim': str(root.defaultPrim),
        'metersPerUnit': root.pseudoRoot.GetInfo('metersPerUnit') if root.pseudoRoot.HasInfo('metersPerUnit') else None,
        'sublayer_ok': prims > 1,
    }


def scenes_for(spec: dict, arg_scene: str) -> list:
    if '{scene}' not in spec['out']:
        return [None]
    if arg_scene and arg_scene != 'all':
        return [arg_scene]
    root = Path(spec['usd_root'])
    return sorted(p.name for p in root.glob('*') if (p / 'matterport_mesh').is_dir())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='레시피 목록과 현재 만들어진 파일 상태만 출력')
    ap.add_argument('--recipe', default='all', help=f"만들 레시피 이름 또는 all (있는 것: {', '.join(RECIPES)})")
    ap.add_argument('--scene', default=None, help="[mp3d_textures] 씬 ID 또는 all. 생략하면 이미 만들어진 것만 갱신")
    ap.add_argument('--out_dir', default=str(HERE), help='레이어가 만들어질 폴더 (기본: 이 폴더)')
    ap.add_argument('--verify_only', action='store_true', help='만들지 않고, 이미 있는 레이어의 합성 여부만 검사')
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    names = list(RECIPES) if args.recipe == 'all' else [args.recipe]
    for n in names:
        if n not in RECIPES:
            print(f'[{SCRIPT_NAME}] 그런 레시피가 없다: {n} (있는 것: {", ".join(RECIPES)})', file=sys.stderr)
            raise SystemExit(2)

    if args.list:
        print(f'[{SCRIPT_NAME}] out_dir = {out_root}')
        for n in names:
            spec = RECIPES[n]
            made = sorted(out_root.glob(spec['out'].replace('{scene}', '*')))
            print(f'\n  {n}  ({spec["kind"]})')
            print(f'    {spec["doc"]}')
            print(f'    출력 규칙: {spec["out"]}')
            print(f'    만들어진 파일: {len(made)}개' + (f' (예: {made[0].name})' if made else ' — 아직 없음'))
        return

    total, bad = 0, []
    for n in names:
        spec = RECIPES[n]
        for scene in scenes_for(spec, args.scene):
            rel = spec['out'].format(scene=scene) if scene else spec['out']
            out_path = out_root / rel

            if args.verify_only:
                if not out_path.is_file():
                    continue
                v = verify(out_path)
                total += 1
                if not v['sublayer_ok']:
                    bad.append(rel)
                print(f"[{SCRIPT_NAME}] {'OK  ' if v['sublayer_ok'] else 'FAIL'} {rel}  "
                      f"prim {v['prims']} · defaultPrim {v['defaultPrim']} · mpu {v['metersPerUnit']}")
                continue

            # --scene 생략 + 씬별 레시피면, 이미 만들어 둔 것만 갱신한다 (실수로 90개 만들지 않게)
            if scene and not args.scene and not out_path.is_file():
                continue

            base = base_of(spec, scene)
            guard = PreservationGuard([base]).snapshot_before()
            kind = spec['kind']
            if kind == 'visibility':
                res = build_visibility(spec, out_path)
            elif kind == 'textures':
                res = build_textures(spec, out_path, scene)
            else:
                assert False, f'unreachable kind={kind!r}'
            preserved = guard.snapshot_after().all_unchanged

            v = verify(out_path)
            total += 1
            ok = v['sublayer_ok'] and preserved
            if not ok:
                bad.append(rel)
            print(f"[{SCRIPT_NAME}] {'OK  ' if ok else 'FAIL'} {rel}")
            print(f"[{SCRIPT_NAME}]        {res['detail']}")
            print(f"[{SCRIPT_NAME}]        합성 prim {v['prims']} · defaultPrim {v['defaultPrim']} · "
                  f"mpu {v['metersPerUnit']} · 원본 무변경 {'예' if preserved else '아니오'}")

    print(f'\n[{SCRIPT_NAME}] 처리 {total}개 · 실패 {len(bad)}개' + (f' -> {bad}' if bad else ''))
    if bad:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
