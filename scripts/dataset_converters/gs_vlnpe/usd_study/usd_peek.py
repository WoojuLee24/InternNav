"""USD 파일 안을 터미널에서 빠르게 확인하는 작은 도구.

**둘러보려면 이것 말고 GUI를 써라** — 이 컨테이너에는 Isaac Sim GUI가 이미 있고
`DISPLAY=:0`도 있다(`/isaac-sim/isaac-sim.sh`). Stage/Property/Layer 창이 다 있어서
탐색은 그쪽이 압도적으로 낫다. `usdview`를 따로 설치하지 말 것 — 바이너리도 PySide6도 없고,
이 컨테이너의 `pxr`은 Isaac이 쓰는 바로 그 빌드라 다른 USD를 얹으면 Isaac이 깨질 수 있다.

이 스크립트가 GUI보다 나은 경우는 좁다:
  - **빠르다** (~2초, GPU 불필요) — 값 하나만 확인할 때
  - **스크립트로 쓸 수 있다** — 자동 확인/회귀 검사
  - **깨진 sublayer를 크게 경고한다** — 값만 보면 못 알아채는 실수라서

Isaac Sim을 띄우지 않는다.

## 쓰는 법 (전부 repo 루트에서, 한 줄)

    # 1) 이 파일이 뭐라고 선언하나 (위쪽 축 · 단위 · 기본 물건)
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/usd_peek.py <파일>

    # 2) 안에 뭐가 들었나 (물건 계층)
    ... usd_peek.py <파일> --tree --depth 3

    # 3) 겹쳐진 파일 목록 (덧칠 파일을 열면 여기에 원본이 같이 보인다)
    ... usd_peek.py <파일> --layers

    # 4) **값 하나가 어느 파일에서 왔는지** — 이게 이 도구의 핵심이다
    ... usd_peek.py <파일> --prim /Root/... --attr visibility

4번이 중요하다. 덧칠 파일을 열어 값을 물어보면 **후보가 여러 줄로 나오고 맨 위가 이긴 값**이다.
덧칠이 실제로 이겼는지를 눈으로 확인하는 지점이다.
"""

import argparse
import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom


def check_sublayers(stage) -> list:
    """선언된 sublayer가 실제로 열리는지 확인한다.

    **왜 따로 확인하나**: 경로를 틀려도 USD는 경고 한 줄만 내고 그냥 넘어간다(`skipping`).
    게다가 내 덧칠 파일에 적어둔 값은 **자기 파일에서 오므로 그대로 읽힌다** — 그래서
    `--prim`으로 값만 물어보면 <b>경로가 틀린 걸 눈치챌 수 없다</b>. 실제로 겪은 함정이라
    도구가 대신 잡는다.
    """
    root = stage.GetRootLayer()
    out = []
    for declared in root.subLayerPaths:
        abs_path = root.ComputeAbsolutePath(declared)
        ok = Sdf.Layer.FindOrOpen(abs_path) is not None
        out.append((declared, abs_path, ok))
    return out


def warn_broken_sublayers(stage) -> bool:
    """깨진 sublayer가 있으면 크게 알린다. 있으면 True."""
    rows = check_sublayers(stage)
    broken = [r for r in rows if not r[2]]
    if not broken:
        return False
    print('!' * 70)
    print('경고: 아래에 깔려고 한 파일을 못 열었다 — 원본이 안 깔린 상태다.')
    for declared, abs_path, _ in broken:
        print(f'   적어둔 경로 : {declared}')
        print(f'   실제로 찾은 곳: {abs_path}   ← 여기에 파일이 없다')
    print('')
    print('   주의: 이 상태에서도 **내 파일에 적어둔 값은 그대로 읽힌다**(자기 파일에서 오므로).')
    print('   그래서 값만 물어보면 멀쩡해 보인다. 물건 수가 0이면 원본이 안 깔린 것이다.')
    print('!' * 70)
    print()
    return True


def show_stage(stage, path: Path):
    dp = stage.GetDefaultPrim()
    print(f'파일   {path}')
    print(f'크기   {path.stat().st_size:,} B')
    print(f'위쪽 축        {UsdGeom.GetStageUpAxis(stage)}')
    print(f'1단위 = 몇 m   {UsdGeom.GetStageMetersPerUnit(stage)}')
    print(f'기본 물건      {dp.GetPath() if dp else "(없음)"}')
    types = {}
    for p in stage.Traverse():
        t = str(p.GetTypeName()) or '(타입없음)'
        types[t] = types.get(t, 0) + 1
    total = sum(types.values())
    flag = '   ← 0이면 원본이 안 깔린 것이다' if total == 0 else ''
    print(f'물건 수        {total}{flag}')
    if types:
        print('  ' + ', '.join(f'{k}={v}' for k, v in sorted(types.items(), key=lambda x: -x[1])[:8]))


def show_layers(stage):
    rows = check_sublayers(stage)
    if rows:
        print('내가 아래에 깔겠다고 적은 파일:')
        for declared, abs_path, ok in rows:
            print(f'  {"OK  " if ok else "실패"} {declared}')
            if not ok:
                print(f'         -> {abs_path} 없음')
        print()
    print('실제로 겹쳐진 파일 (위에 있는 것이 이긴다):')
    for i, layer in enumerate(stage.GetLayerStack()):
        mark = '  ← 내가 쓴 것' if i == 0 and not layer.anonymous else ''
        name = layer.identifier
        if layer.anonymous:
            name += '   (임시 메모리 레이어 — 무시해도 된다)'
        print(f'  {i}. {name}{mark}')


def show_tree(stage, depth: int, limit: int):
    print(f'물건 계층 (깊이 {depth}까지, 최대 {limit}줄):')
    n = 0
    for p in stage.Traverse():
        path = str(p.GetPath())
        d = path.count('/')
        if d > depth:
            continue
        n += 1
        if n > limit:
            print('  … (더 있음, --limit 로 늘리세요)')
            break
        print(f'  {"  " * (d - 1)}{path}   [{p.GetTypeName() or "타입없음"}]')


def show_attr(stage, prim_path: str, attr_name: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        print(f'그런 물건이 없다: {prim_path}', file=sys.stderr)
        print('  --tree 로 먼저 경로를 확인하세요.', file=sys.stderr)
        return 2
    attr = prim.GetAttribute(attr_name)
    if not attr:
        names = [a.GetName() for a in prim.GetAttributes()][:20]
        print(f'그런 속성이 없다: {attr_name}', file=sys.stderr)
        print(f'  이 물건이 가진 속성: {", ".join(names)}', file=sys.stderr)
        return 2

    print(f'물건   {prim_path}')
    print(f'속성   {attr_name}')
    print(f'지금 값  {attr.Get()!r}')
    print()
    # 여기가 핵심 — 이 값을 주장하는 파일들을 강한 순서로 보여준다.
    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    if not stack:
        print('이 값을 적어둔 파일이 없다 (스키마 기본값을 쓰는 중).')
        return 0
    print('이 값을 주장하는 파일 (맨 위가 이긴 값):')
    for i, spec in enumerate(stack):
        win = '  ← 이긴 값' if i == 0 else ''
        print(f'  {i}. {spec.default!r}   from  {spec.layer.identifier}{win}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='USD 안을 들여다보는 손전등 (실습용)')
    ap.add_argument('usd', help='.usd / .usda / .usdz / 내가 만든 덧칠 파일')
    ap.add_argument('--tree', action='store_true', help='물건 계층 보기')
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--limit', type=int, default=60)
    ap.add_argument('--layers', action='store_true', help='겹쳐진 파일 목록')
    ap.add_argument('--prim', help='값을 볼 물건 경로 (예: /Root/geometry)')
    ap.add_argument('--attr', default='visibility', help='볼 속성 이름 (기본: visibility)')
    args = ap.parse_args()

    path = Path(args.usd)
    if not path.is_file():
        print(f'파일이 없다: {path}', file=sys.stderr)
        return 2
    stage = Usd.Stage.Open(str(path))

    # 값만 물어보는 경우에도 **먼저** sublayer 상태를 알린다 — 이게 없으면 경로를 틀려도
    # 값이 똑같이 나와서 눈치챌 수 없다.
    warn_broken_sublayers(stage)

    if args.prim:
        return show_attr(stage, args.prim, args.attr)
    show_stage(stage, path)
    print()
    show_layers(stage)
    if args.tree:
        print()
        show_tree(stage, args.depth, args.limit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
