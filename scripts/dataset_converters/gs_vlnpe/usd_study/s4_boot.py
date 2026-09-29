"""S4 — Isaac Sim 두 부팅 경로의 carb 설정 diff.

## 왜 이 스크립트가 있는가

`04_render_obs_isaac.py` docstring에 실측이 이렇게 적혀 있다:

  - `isaaclab.app.AppLauncher`를 거치면 렌더 한 프레임에 **30분 이상** 걸리거나 멈춘다
  - raw `isaacsim.SimulationApp({'headless': True})`으로 직접 띄우면 **수십 초**에 끝난다
  - **"정확히 어떤 kit 설정이 정지의 원인인지는 못 좁혔다"**

그리고 별개로 두 가지가 더 기록돼 있다:

  - `/rtx/sceneDb/ambientLightIntensity`가 0이었던 것이 조명 문제의 **근본 원인**이었다
    (raw 부팅이 IsaacLab kit을 우회하기 때문이라는 추론)
  - `isaaclab.sensors.camera.Camera`는 `/isaaclab/cameras_enabled`가 꺼져 있으면 즉시
    `RuntimeError` — `AppLauncher`를 거칠 때만 자동으로 켜진다

이 스크립트는 두 경로를 각각 띄워 **carb 설정 트리 전체를 JSON으로 덤프**하고 오프라인에서
diff해서, 위 추론들을 검증하고 30분 미스터리의 후보를 좁힌다. **GPU 렌더를 하지 않으므로
가장 싸다.**

한 프로세스에서 두 번 부팅할 수 없어 **3번 실행** 구조다(두 번 덤프 + 한 번 diff).

## 실행 (전부 한 줄, repo 루트에서)

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s4_boot.py --boot raw --log_dir logs/gs-vlnpe/usd_study

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s4_boot.py --boot applauncher --log_dir logs/gs-vlnpe/usd_study

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s4_boot.py --diff --log_dir logs/gs-vlnpe/usd_study

부팅이 매달릴 수 있으므로 `timeout --signal=KILL 1800`으로 감싸는 것을 권한다. 덤프 JSON은
`main()` 반환 전에 저장되므로 SIGKILL돼도 이미 파일은 남는다.

## 해석에 주의

설정 diff는 **상관**이지 인과가 아니다. 30분을 설명할 후보를 좁히는 것이 목표이고,
후보가 안 나오면 "이 범위에는 없었다"가 결과다 — 그 자체가 기록할 값어치가 있다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from usd_study_utils import (DEFAULT_LOG_DIR, TABLE_CSS, esc, glossary_html, glossary_note_html,
                             pill, stat_row_html, table_html)

import viz_utils  # noqa: E402

# 기록된 미스터리와 직접 연결된 키 — diff 리포트 맨 위에 따로 뽑는다.
WATCH_KEYS = [
    '/rtx/sceneDb/ambientLightIntensity',
    '/isaaclab/cameras_enabled',
    '/rtx/post/histogram/enabled',
    '/app/renderer/resolution/width',
    '/app/renderer/resolution/height',
    '/app/renderer/skipWhileMinimized',
    '/app/renderer/waitIdle',
    '/app/asyncRendering',
    '/app/asyncRenderingLowLatency',
    '/app/hydraEngine/waitIdle',
    '/app/updateOrder/checkForHydraRenderComplete',
    '/omni/replicator/asyncRendering',
    '/rtx/pathtracing/spp',
    '/rtx/pathtracing/totalSpp',
    '/rtx/rendermode',
    '/rtx/materialDb/syncLoads',
    '/omni/kit/loop/syncToPresent',
    '/physics/cooking/ueryCollisionCooking',
    '/physics/collisionCooking',
]

# 30분 미스터리의 후보로 특히 의심스러운 접두어 — diff에서 이 그룹만 따로 집계한다.
SUSPECT_PREFIXES = [
    '/app/renderer/',
    '/app/hydraEngine/',
    '/app/asyncRendering',
    '/app/updateOrder/',
    '/rtx/pathtracing/',
    '/rtx/materialDb/',
    '/rtx/hydra/',
    '/rtx/rendermode',
    '/omni/kit/loop/',
    '/omni/replicator/',
    '/physics/',
    '/isaaclab/',
    '/exts/omni.kit.material.library/',
]


# ---------------------------------------------------------------------------
# 부팅 + 덤프
# ---------------------------------------------------------------------------

def _flatten(node, prefix='', out=None):
    """carb 설정 dict를 `/a/b/c` -> 값 으로 평탄화."""
    if out is None:
        out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten(v, f'{prefix}/{k}', out)
    elif isinstance(node, (list, tuple)):
        # 리스트는 통째로 하나의 값으로 본다 (원소 인덱스까지 벌리면 diff가 산만해진다)
        out[prefix] = [str(x) for x in node]
    else:
        out[prefix] = node
    return out


def dump_settings():
    """carb 설정 트리 전체를 평탄한 dict로. 여러 API 이름을 순서대로 시도한다."""
    import carb
    import carb.settings

    s = carb.settings.get_settings()
    # `get_settings_dictionary('/')`는 python dict가 아니라 `carb.dictionary.Item`을 주고
    # (IDictionary에 get_dict가 없어) 평탄화가 안 된다 — 실측으로 확인. `get('/')`가
    # 진짜 dict(최상위 56키)를 주므로 이쪽을 먼저 쓴다.
    tree, how = None, None
    try:
        cand = s.get('/')
        if isinstance(cand, dict):
            tree, how = cand, 'get'
    except Exception as exc:  # noqa: BLE001
        print(f'[s4] get("/") 실패: {exc!r}', file=sys.stderr)
    if tree is None:
        fn = getattr(s, 'get_settings_dictionary', None)
        if fn is not None:
            try:
                tree, how = fn('/'), 'get_settings_dictionary'
            except Exception as exc:  # noqa: BLE001
                return {}, f'모든 API 실패: {exc!r}'
    if tree is None:
        return {}, '모든 API 실패'
    return _flatten(tree), how


def boot_and_dump(mode: str, out_dir: Path) -> dict:
    """`mode`로 Isaac Sim을 띄우고 설정을 덤프해 JSON으로 저장한다."""
    rec = {'mode': mode}
    t0 = time.time()
    if mode == 'raw':
        # `04_render_obs_isaac.py`가 실제로 쓰는 경로를 그대로 재현한다 (인자까지 동일).
        from isaacsim import SimulationApp

        app = SimulationApp({'headless': True})
        rec['boot_s'] = round(time.time() - t0, 2)
        rec['app_repr'] = type(app).__module__ + '.' + type(app).__name__
    elif mode == 'applauncher':
        # IsaacLab 경로 — 전용 kit 실험 파일을 거친다. 카메라를 쓰는 조건을 맞춰
        # enable_cameras=True로 띄운다(그게 `Camera` 센서를 쓰는 실제 조건이다).
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=True)
        rec['boot_s'] = round(time.time() - t0, 2)
        rec['app_repr'] = type(launcher.app).__module__ + '.' + type(launcher.app).__name__
    else:
        assert False, f'unreachable boot mode={mode!r}'

    settings, how = dump_settings()
    rec['dump_api'] = how
    rec['setting_count'] = len(settings)
    rec['settings'] = settings
    # kit 실험 파일 / 앱 정체를 알려주는 키를 따로 뽑는다.
    rec['identity'] = {k: v for k, v in settings.items()
                       if any(t in k.lower() for t in ('kitfile', 'experience', '/app/name', '/app/version', '/app/window/title'))}
    rec['argv'] = list(sys.argv)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'settings_{mode}.json'
    path.write_text(json.dumps(rec, indent=2, default=str, ensure_ascii=False), encoding='utf-8')
    print(f'[s4] {mode}: boot {rec["boot_s"]}s, 설정 {rec["setting_count"]}개 -> {path}')
    return rec


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def _norm(v):
    return json.dumps(v, sort_keys=True, default=str)


def build_diff(raw: dict, app: dict) -> dict:
    a, b = raw['settings'], app['settings']
    ka, kb = set(a), set(b)
    changed = sorted(k for k in ka & kb if _norm(a[k]) != _norm(b[k]))
    only_raw = sorted(ka - kb)
    only_app = sorted(kb - ka)
    watch = []
    for k in WATCH_KEYS:
        watch.append({'key': k, 'raw': a.get(k, '(없음)'), 'applauncher': b.get(k, '(없음)'),
                      'differs': _norm(a.get(k)) != _norm(b.get(k))})
    suspects = [k for k in changed + only_app + only_raw if any(k.startswith(p) for p in SUSPECT_PREFIXES)]
    return {
        'changed': changed, 'only_raw': only_raw, 'only_applauncher': only_app,
        'watch': watch, 'suspects': sorted(set(suspects)),
        'counts': {'raw_total': len(ka), 'app_total': len(kb), 'changed': len(changed),
                   'only_raw': len(only_raw), 'only_app': len(only_app), 'suspects': len(set(suspects))},
    }


def diff_report(raw: dict, app: dict, d: dict, out_dir: Path) -> Path:
    a, b = raw['settings'], app['settings']
    parts = [TABLE_CSS, glossary_note_html()]

    parts.append('<h2>부팅 자체</h2>')
    parts.append(table_html(
        ['', 'raw SimulationApp', 'AppLauncher'],
        [['부팅 시간(s)', raw.get('boot_s'), app.get('boot_s')],
         ['설정 개수', raw.get('setting_count'), app.get('setting_count')],
         ['덤프 API', raw.get('dump_api'), app.get('dump_api')],
         ['app 클래스', raw.get('app_repr'), app.get('app_repr')]],
        mono_cols={1, 2}))

    parts.append('<h3>어떤 kit/experience를 로드했나</h3>')
    ids = sorted(set(raw.get('identity', {})) | set(app.get('identity', {})))
    parts.append(table_html(['키', 'raw', 'AppLauncher'],
                            [[k, raw.get('identity', {}).get(k, '—'), app.get('identity', {}).get(k, '—')] for k in ids],
                            mono_cols={0, 1, 2}) if ids else '<p>identity 키를 못 찾음</p>')

    parts.append('<h2>기록된 미스터리와 직접 연결된 키</h2>')
    parts.append('<p>세 가지를 검증한다 — ① <code>ambientLightIntensity</code>가 raw에서 0인가 '
                 '② <code>cameras_enabled</code>가 AppLauncher에서만 켜지는가 '
                 '③ 30분을 설명할 렌더/동기화 설정이 있는가.</p>')
    parts.append(table_html(
        ['키', 'raw SimulationApp', 'AppLauncher', '비교'],
        [[w['key'], w['raw'], w['applauncher'], pill(not w['differs'], '같음', '다름')] for w in d['watch']],
        row_classes=['differ' if w['differs'] else '' for w in d['watch']],
        mono_cols={0, 1, 2}))

    c = d['counts']
    parts.append('<h2>전체 diff</h2>')
    parts.append(stat_row_html([('raw 키', f'{c["raw_total"]:,}'), ('AppLauncher 키', f'{c["app_total"]:,}'),
                                ('값이 다름', f'{c["changed"]:,}'), ('raw에만', f'{c["only_raw"]:,}'),
                                ('AppLauncher에만', f'{c["only_app"]:,}'), ('의심 그룹', f'{c["suspects"]:,}')]))

    parts.append('<h3>의심 그룹 — 30분 미스터리 후보</h3>')
    parts.append('<p>렌더·동기화·물리·머티리얼 로딩 관련 접두어만 골라낸 것. '
                 '<b>여기가 비면 "이 범위에는 원인이 없었다"가 결과다.</b></p>')
    if d['suspects']:
        rows = [[k, a.get(k, '(raw에 없음)'), b.get(k, '(AppLauncher에 없음)')] for k in d['suspects']]
        parts.append(table_html(['키', 'raw', 'AppLauncher'], rows, mono_cols={0, 1, 2}))
    else:
        parts.append('<p><span class="pill warn">의심 그룹 diff 없음</span></p>')

    for title, keys, show_both in (
        ('값이 다른 키 전체', d['changed'], True),
        ('AppLauncher에만 있는 키', d['only_applauncher'], False),
        ('raw에만 있는 키', d['only_raw'], False),
    ):
        parts.append(f'<h3>{esc(title)} ({len(keys)})</h3>')
        shown = keys[:400]
        if show_both:
            rows = [[k, a.get(k), b.get(k)] for k in shown]
            parts.append(table_html(['키', 'raw', 'AppLauncher'], rows, mono_cols={0, 1, 2}))
        else:
            src = b if 'AppLauncher에만' in title else a
            parts.append(table_html(['키', '값'], [[k, src.get(k)] for k in shown], mono_cols={0, 1}))
        if len(keys) > len(shown):
            parts.append(f'<p>… 그리고 {len(keys) - len(shown)}개 더 (전체는 <code>s4_diff.json</code>)</p>')

    parts.append(glossary_html())

    ambient = next(w for w in d['watch'] if w['key'].endswith('ambientLightIntensity'))
    cams = next(w for w in d['watch'] if w['key'].endswith('cameras_enabled'))
    summary = stat_row_html([
        ('raw 부팅', f'{raw.get("boot_s")}s'),
        ('AppLauncher 부팅', f'{app.get("boot_s")}s'),
        ('값 다른 키', f'{c["changed"]:,}'),
        ('의심 후보', f'{c["suspects"]:,}'),
        ('ambient raw', ambient['raw']),
        ('ambient AppLauncher', ambient['applauncher']),
        ('cameras_enabled raw', cams['raw']),
        ('cameras_enabled AppLauncher', cams['applauncher']),
    ])
    return viz_utils.save_gallery(out_dir, 'report.html', 'S4 · Isaac Sim 부팅 경로 carb 설정 diff',
                                  summary, ''.join(parts), eyebrow='gs_vlnpe usd_study')


def main():
    ap = argparse.ArgumentParser(description='S4 — 두 부팅 경로의 carb 설정 덤프/diff')
    ap.add_argument('--boot', choices=['raw', 'applauncher'], help='이 모드로 부팅해 설정을 덤프')
    ap.add_argument('--diff', action='store_true', help='이미 덤프된 JSON 2개를 비교해 리포트 생성')
    ap.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR))
    args = ap.parse_args()

    out_dir = Path(args.log_dir) / 's4_boot'
    if args.boot:
        boot_and_dump(args.boot, out_dir)
        # SimulationApp.close()가 수 분 걸릴 수 있고 JSON은 이미 저장됐다 — 바로 나간다.
        print('[s4] 덤프 저장 완료. (app close는 기다리지 않음)')
        return 0

    if args.diff:
        paths = {m: out_dir / f'settings_{m}.json' for m in ('raw', 'applauncher')}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            print(f'[fail] 먼저 --boot 로 덤프해야 한다. 없음: {missing}', file=sys.stderr)
            return 2
        raw = json.loads(paths['raw'].read_text(encoding='utf-8'))
        app = json.loads(paths['applauncher'].read_text(encoding='utf-8'))
        d = build_diff(raw, app)
        (out_dir / 's4_diff.json').write_text(json.dumps(d, indent=2, default=str, ensure_ascii=False), encoding='utf-8')
        path = diff_report(raw, app, d, out_dir)
        print(f'[s4] diff: 값 다른 키 {d["counts"]["changed"]}, 의심 후보 {d["counts"]["suspects"]}')
        print(f'[s4] report: {path}')
        return 0

    ap.error('--boot 또는 --diff 중 하나가 필요하다')


if __name__ == '__main__':
    sys.exit(main())
