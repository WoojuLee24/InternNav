"""R2 — augment 속도·병렬성 실측 (이전 G3 재측정).

## 이전 G3가 불충분했던 이유 (정정)
- **단일 프로세스**만 쟀다. 학습은 `dataloader_num_workers=4`(default_config.py:67)로 돈다.
- **T가 임의**(1/5/10)였다. 실제 학습 sample은 **T ≤ 12**(`internvla_n1_lerobot_dataset.py:1223` `max_len=12`).
- **빠진 비용**: `_snap_navigable`(replan당 full-grid ESDF **2회**), depth→BEV, 타일 파일 load,
  포즈 합성. 게다가 렌더를 **같은 pose로 20회** 반복해 GPU 캐시 최상 조건이었다.
- **버그**: `synthesize_action_poses` 인자 순서가 틀려 카메라가 z=7.0 m(천장 위)에 있었다 → 빈 화면을
  렌더해 시간이 낙관적으로 나왔다. (수정 완료; 이 스크립트는 올바른 pose로 잰다.)

## 이 스크립트가 재는 것
1. **realistic per-sample**: replan(snap 포함) + T=12 렌더(궤적 따라 pose가 매번 다름) + BEV.
2. **worker 스케일링**: `DataLoader(num_workers ∈ 0,2,4,8)` — Open3D `OffscreenRenderer`(EGL)를
   **worker 프로세스마다 lazy init**(`worker_init_fn` 대신 첫 접근 시 생성)해 fork 안전성 확인.
3. **batch 성능**: batch_size별 batch 준비시간 / samples-per-sec.

## 2026-08-17 정정 — "depth→BEV 165 ms/frame"은 잘못된 측정이었다
이전 판은 CPU BEV가 165 ms/frame이라 병목이라고 결론냈다. **원인은 BEV 계산이 아니라 torch 스레드
설정**이었다: 이 부분만 부모 프로세스에서 재는데(§3 단일 sample 분해), 부모의 torch 기본 스레드 수가
**코어 수와 같으면(24 == os.cpu_count())** OpenMP full-subscription spin-wait로 무너진다.

    threads 24 (= 코어 수): BEV만 2127 ms / 12프레임 = 177.3 ms/frame
    threads  1            : BEV만   30 ms / 12프레임 =   2.5 ms/frame   (71배)

23스레드만 돼도 정상이다(2dloader `local_map.limit_torch_threads` 참고 — 1~23 전부 1.0~1.7 ms).
`with_bev=False`는 여전히 학습 경로의 기본이지만, 이유는 **"느려서"가 아니라 모델이 GPU에서
`traj_depths`로 BEV를 다시 만들므로 중복이기 때문**이다. 이 스크립트는 두 설정을 모두 재서 보여준다.

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/08_bench_parallel.py --scene s8pcmisQ38h
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
_GS = _HERE.parents[0] / 'gs_vlnpe'
sys.path.insert(0, str(_GS)); sys.path.insert(0, str(_HERE))
from viz_utils import line_chart, save_gallery  # noqa: E402

T_FRAMES = 12          # internvla_n1_lerobot_dataset.py:1223 max_len


def limit_torch_threads(n: int = 1) -> None:
    """torch intra-op 스레드 제한. **코어 수와 같으면 BEV가 370배 느려진다**(모듈 docstring 참고).

    DataLoader worker는 원래 1스레드로 뜨므로 학습 경로에서는 no-op이고, 이 스크립트처럼 메인
    프로세스에서 재는 경우에만 효과가 있다.
    """
    if torch.get_num_threads() > n:
        torch.set_num_threads(n)
DEFAULTS = dict(data_root='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r',
                raw_root='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r',
                mesh_root='data/scene_data/mp3d_n1',
                esdf_dir='scripts/dataset_converters/gs_vlnpe/logs/esdf',
                geo_dir='data/embodiment_aug/scene_geo')


class AugmentDataset(Dataset):
    """augment 1회를 __getitem__으로 감싼 더미 Dataset — worker별 augmenter를 lazy init한다.

    Open3D 렌더러는 fork 후 자식에서 새로 만들어야 하므로 **절대 __init__에서 만들지 않는다**
    (부모가 만든 GL 컨텍스트를 fork로 물려받으면 깨진다). 첫 __getitem__에서 생성.
    """

    def __init__(self, scene, start_xy, goal_xy, floor_z, n, t_frames=T_FRAMES, paths=None):
        self.scene, self.start_xy, self.goal_xy = scene, start_xy, goal_xy
        self.floor_z, self.n, self.t = floor_z, n, t_frames
        self.paths = paths or DEFAULTS
        self._aug = None

    def _augmenter(self):
        if self._aug is None:
            # **worker당 torch 스레드 1개**로 제한. 안 하면 worker마다 코어 수(24)만큼 스레드를 띄워
            # 심하게 oversubscribe되고, torch 연산이 스레드 경합으로 수십~수백 배 느려진다
            # (실측: BEV 181 ms → 3 ms/frame, replan 49 ms → 24 ms). dataloader worker의 표준 관행.
            torch.set_num_threads(1)
            from embodiment_augment import EmbodimentAugmenter
            self._aug = EmbodimentAugmenter(self.paths['data_root'], self.paths['raw_root'],
                                            self.paths['mesh_root'], self.paths['esdf_dir'],
                                            self.paths['geo_dir'])
        return self._aug

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        aug = self._augmenter()
        rng = np.random.default_rng(i)
        r_b = float(rng.uniform(0.15, 0.35))
        traj = aug.replan(self.scene, self.start_xy, self.goal_xy, r_b, floor_z=self.floor_z)
        if traj is None:
            return torch.zeros(1)
        # 학습 경로와 동일: depth만 만든다(BEV는 모델이 GPU에서 traj_depths로부터 계산)
        _sub, d, _bev = aug.render_bev_along(self.scene, traj, self.t, floor_z=self.floor_z, with_bev=False)
        return torch.from_numpy(d[:1])            # 작은 텐서만 반환(IPC 비용 최소화)


def timed_loader(ds, num_workers, batch_size, _n_batches=None):
    """**epoch 전체**를 소비해 throughput을 잰다.

    주의: 몇 배치만 재면 prefetch 큐에서 꺼내는 시간만 재게 돼 말도 안 되는 수치가 나온다
    (실측: workers=8에서 53,747 samples/s). 전체 epoch을 돌려야 실제 처리량이 나온다.
    warm-up(worker 기동 + 렌더러 init)은 첫 배치로 흡수하고 나머지를 잰다.
    """
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                    collate_fn=lambda b: b, persistent_workers=False)
    it = iter(dl)
    first = next(it)                                # warm-up (worker 기동/렌더러 init) 제외
    n_seen = len(first)
    t0 = time.perf_counter()
    for b in it:
        n_seen += len(b)
    dt = time.perf_counter() - t0
    n_rest = n_seen - len(first)
    return (n_rest / dt if dt > 0 else float('nan')), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='s8pcmisQ38h')
    ap.add_argument('--workers', default='0,2,4,8')
    ap.add_argument('--batches', default='4,8,16')
    # 32은 너무 작다 — worker 8개면 prefetch가 epoch 전체를 삼켜 187 samples/s 같은 가짜 수치가 나온다
    # (이 스크립트 docstring이 경고하는 그 함정). 최소 200.
    ap.add_argument('--n_samples', type=int, default=200, help='epoch 크기(전체를 소비해 throughput 측정)')
    ap.add_argument('--start', default='3.01,-0.06', help='정합된 start xy (offline 캐시 모사)')
    ap.add_argument('--goal', default='7.32,1.6')
    ap.add_argument('--floor_z', type=float, default=2.84)
    ap.add_argument('--out_dir', default='logs/embodiment_augment/perf')
    ap.add_argument('--one', default=None,
                    help='내부용: "W,B" 한 조합만 측정해 결과 한 줄 출력(자식 프로세스 모드)')
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ⚠️ **부모 프로세스는 Open3D 렌더러를 절대 만들면 안 된다.** 렌더러(EGL/Filament 컨텍스트)가 살아있는
    # 상태에서 fork하면 자식이 **데드락**한다(실측: workers>=2에서 무한 대기). 그래서 정합값은
    # offline 캐시를 읽는 상황을 모사해 인자로 받고, 부모는 augmenter를 만들지 않는다.
    start_xy = np.array([float(x) for x in args.start.split(',')])
    goal_xy = np.array([float(x) for x in args.goal.split(',')])
    floor_z = args.floor_z

    # 자식 모드: 이 프로세스는 딱 한 조합만 재고 끝난다(부모 오염 문제 원천 차단)
    if args.one:
        limit_torch_threads()
        w, b = (int(x) for x in args.one.split(','))
        ds = AugmentDataset(args.scene, start_xy, goal_xy, floor_z, n=args.n_samples)
        sps, _dt = timed_loader(ds, w, b)
        print(f'RESULT {w} {b} {sps:.4f}', flush=True)
        return

    # --- 1) worker 스케일링 — **각 조합을 별도 subprocess로** 실행한다.
    # 이유: num_workers=0은 부모에서 렌더러를 만들고, 그 뒤 fork하는 조합이 데드락한다(실측).
    ds = AugmentDataset(args.scene, start_xy, goal_xy, floor_z, n=args.n_samples)
    import subprocess
    base_cmd = [sys.executable, str(Path(__file__).resolve()), '--scene', args.scene,
                '--n_samples', str(args.n_samples), '--start', args.start, '--goal', args.goal,
                '--floor_z', str(args.floor_z)]

    def run_one(w, b, timeout=240):
        try:
            out = subprocess.run(base_cmd + ['--one', f'{w},{b}'], capture_output=True,
                                 text=True, timeout=timeout).stdout
            for ln in out.splitlines():
                if ln.startswith('RESULT'):
                    return float(ln.split()[3])
        except subprocess.TimeoutExpired:
            return float('nan')
        return float('nan')

    wk_rows = []
    for w in [int(x) for x in args.workers.split(',')]:
        sps = run_one(w, 4)
        wk_rows.append((w, sps))
        print(f'[R2] num_workers={w}: ' + ('시간초과/실패' if not np.isfinite(sps) else f'{sps:.2f} samples/sec'))

    # --- 3) batch 스케일링 (worker 4) ---
    bs_rows = []
    for b in [int(x) for x in args.batches.split(',')]:
        sps = run_one(4, b)
        bs_rows.append((b, sps))
        print(f'[R2] batch_size={b} (workers=4): ' + ('시간초과/실패' if not np.isfinite(sps) else f'{sps:.2f} samples/sec'))

    # --- 3) 단일 sample 분해 (worker 스윕 이후에! 여기서 부모에 렌더러가 생긴다) ---
    from embodiment_augment import EmbodimentAugmenter
    aug0 = EmbodimentAugmenter(**DEFAULTS)
    t0 = time.perf_counter()
    traj = aug0.replan(args.scene, start_xy, goal_xy, 0.25, floor_z=floor_z)
    replan_ms = (time.perf_counter() - t0) * 1000
    assert traj is not None, 'replan 실패 — start/goal 확인'
    aug0.render_bev_along(args.scene, traj, T_FRAMES, floor_z=floor_z, with_bev=False)   # warm
    t0 = time.perf_counter()
    aug0.render_bev_along(args.scene, traj, T_FRAMES, floor_z=floor_z, with_bev=False)
    render_ms = (time.perf_counter() - t0) * 1000
    # CPU BEV 비용은 **torch 스레드 수에 따라 두 자릿수 배 달라진다** — 둘 다 잰다(2026-08-17 정정).
    # 1스레드에서는 BEV가 렌더 시간(~350 ms)의 노이즈보다 작으므로 **같은 스레드 설정에서 짝을 지어
    # 여러 번 재고 median을 뺀다**(한 번만 재면 음수가 나온다 — 실측).
    import os as _os
    n_cpu = _os.cpu_count()

    def _med(with_bev, reps=3):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            aug0.render_bev_along(args.scene, traj, T_FRAMES, floor_z=floor_z, with_bev=with_bev)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))

    bev_by_threads = {}
    for nt in (n_cpu, 1):
        torch.set_num_threads(nt)
        bev_by_threads[nt] = max(_med(True) - _med(False), 0.0)
    limit_torch_threads()                      # 이후 측정은 항상 1스레드로
    bev_ms = bev_by_threads[1]
    single_ms = replan_ms + render_ms
    align_ms = float('nan')
    print(f'[R2] single-sample: replan={replan_ms:.0f}ms  depth렌더 x{T_FRAMES}={render_ms:.0f}ms  '
          f'합 {single_ms:.0f}ms')
    for nt, v in bev_by_threads.items():
        print(f'[R2] CPU BEV x{T_FRAMES}: torch threads={nt:2d} -> {v:.0f}ms '
              f'({v/T_FRAMES:.1f} ms/frame)')

    ok_w = [(w, s) for w, s in wk_rows if np.isfinite(s)]
    best = max(ok_w, key=lambda z: z[1]) if ok_w else (0, float('nan'))
    base = dict(ok_w).get(0, float('nan'))
    charts = ''
    if len(ok_w) >= 2:
        line_chart([('samples/sec', (0, 255, 90), [w for w, _ in ok_w], [s for _, s in ok_w])],
                   out_dir / 'perf_workers.jpg', x_label='num_workers', y_label='samples/sec')
        charts += '<h3>worker 스케일링</h3><img src="perf_workers.jpg" style="width:80%">'
    ok_b = [(b, s) for b, s in bs_rows if np.isfinite(s)]
    if len(ok_b) >= 2:
        line_chart([('samples/sec', (255, 210, 60), [b for b, _ in ok_b], [s for _, s in ok_b])],
                   out_dir / 'perf_batch.jpg', x_label='batch_size', y_label='samples/sec')
        charts += '<h3>batch 스케일링 (workers=4)</h3><img src="perf_batch.jpg" style="width:80%">'

    summary = (
        f'<p><b>{args.scene}</b> · T={T_FRAMES}프레임/sample(학습 max_len과 동일) · CPU {n_cpu}코어 · '
        f'epoch {args.n_samples} samples · torch 1스레드</p>'
        f'<table border=1 cellpadding=6><tr><th>구성요소</th><th>ms</th><th>비고</th></tr>'
        f'<tr><td>replan (snap+A*+refine+spline)</td><td>{replan_ms:.1f}</td><td>full-grid ESDF 2회 포함</td></tr>'
        f'<tr><td>depth 렌더 ×{T_FRAMES}</td><td>{render_ms:.1f}</td><td>궤적 따라 pose 매번 다름 (BEV는 모델이 GPU에서)</td></tr>'
        f'<tr><td>CPU BEV ×{T_FRAMES} (참고, 학습경로 미사용)</td><td>{bev_ms:.1f}</td>'
        f'<td>torch 1스레드 기준. 아래 정정 절 참고</td></tr>'
        f'<tr><td><b>sample 1개 합</b></td><td><b>{single_ms:.1f}</b></td><td>단일 프로세스</td></tr>'
        f'<tr><td>align (에피소드당 1회)</td><td>{align_ms:.0f}</td><td>캐시하면 sample당 0</td></tr></table>'
        f'<h3 style="color:#c62828">정정 (2026-08-17) — "depth→BEV 165 ms/frame"은 측정 오류였다</h3>'
        f'<p>이전 판은 CPU BEV가 165 ms/frame이라 <b>진짜 병목</b>이라고 결론냈다. '
        f'원인은 BEV 계산이 아니라 <b>torch intra-op 스레드 수</b>였다 — 이 항목만 부모 프로세스에서 '
        f'재는데, 부모의 기본 스레드 수가 <b>코어 수와 같으면({n_cpu} == os.cpu_count())</b> '
        f'OpenMP full-subscription spin-wait로 무너진다. 같은 코드·같은 입력으로 재측정:</p>'
        f'<table border=1 cellpadding=6><tr><th>torch threads</th><th>CPU BEV ×{T_FRAMES}</th>'
        f'<th>frame당</th><th>배수</th></tr>'
        + ''.join(f'<tr><td>{nt}{" (= 코어 수)" if nt == n_cpu else ""}</td><td>{v:.0f} ms</td>'
                  f'<td><b>{v/T_FRAMES:.1f} ms</b></td>'
                  f'<td>{v/max(bev_by_threads[1],1e-9):.0f}×</td></tr>'
                  for nt, v in bev_by_threads.items())
        + '</table>'
        f'<p>2dloader에서 스레드 수를 훑으면 <b>코어 수에서만</b> 절벽이 생긴다 — '
        f'1→1.41 ms, 8→1.14 ms, 16→1.05 ms, 23→1.09 ms, <b>24→402 ms</b>. '
        f'즉 "작은 텐서에 멀티스레드가 오버헤드"가 아니라 full subscription 문제다.</p>'
        f'<p><b>결론이 어떻게 바뀌나</b>: <code>render_bev_along(..., with_bev=False)</code>는 '
        f'여전히 학습 경로의 기본이다. 다만 이유가 <b>"CPU BEV가 병목이라서"가 아니라</b>, 모델이 '
        f'GPU에서 <code>traj_depths</code>로 BEV를 다시 만들므로 <b>중복이기 때문</b>이다. '
        f'스레드만 1로 제한하면 CPU BEV는 sample당 {bev_by_threads[1]:.0f} ms로, 켜도 무방한 수준이다. '
        f'worker throughput 표는 영향이 없다 — <code>__getitem__</code>은 numpy/scipy와 Open3D만 쓰고 '
        f'torch를 쓰지 않으며, DataLoader worker는 원래 1스레드로 뜬다.</p>'
        f'<table border=1 cellpadding=6><tr><th>num_workers</th><th>samples/sec</th></tr>'
        + ''.join(f'<tr><td>{w}</td><td>{"실패" if not np.isfinite(s) else f"{s:.2f}"}</td></tr>' for w, s in wk_rows)
        + '</table>'
        f'<table border=1 cellpadding=6><tr><th>batch_size (workers=4)</th><th>samples/sec</th></tr>'
        + ''.join(f'<tr><td>{b}</td><td>{"실패" if not np.isfinite(s) else f"{s:.2f}"}</td></tr>' for b, s in bs_rows)
        + '</table>'
        f'<p><b>정정</b>: 예전 판에서 "CPU BEV 165 ms/frame이 최대 병목"이라고 했는데 <b>오측이었다</b>. '
        f'torch 스레드가 코어 수와 같아(24=24) 생긴 경합이고, 스레드 1개면 '
        f'{bev_by_thr.get(1, float("nan"))/T_FRAMES:.1f} ms/frame이다. 게다가 이 loader에는 BEV가 '
        f'<b>필요한 곳이 없다</b> — S1 BEV는 모델이 GPU에서 계산하고, 경로 계획은 캐시된 3D occupancy를 쓴다. '
        f'(BEV를 planning 격자로 쓰는 것은 2dloader 설계다.) '
        f'실제 지배 비용은 <b>Open3D depth 렌더</b>({render_ms/T_FRAMES:.0f} ms/frame × T)이고, 이건 torch와 '
        f'무관해 스레드 설정에 영향받지 않는다.</p>'
        f'<p><b>병렬화 (핵심)</b>: Open3D 렌더러는 worker에서 <b>lazy init</b>해야 하고, 무엇보다 '
        f'<b>부모 프로세스가 렌더러를 만든 적이 없어야</b> 한다 — 살아있는 EGL/Filament 컨텍스트를 fork하면 '
        f'자식이 <b>데드락</b>한다(실측: workers≥2 무한 대기). 부모는 offline 정합 캐시만 읽고 렌더는 전부 '
        f'worker에서. 최고 {best[1]:.2f} samples/sec @ num_workers={best[0]}'
        + (f' (단일 프로세스 {base:.2f} 대비 {best[1]/base:.1f}배)' if np.isfinite(base) and base > 0 else '')
        + '</p>')
    (out_dir / 'summary.html').write_text(summary, encoding='utf-8')
    save_gallery(out_dir, 'report.html', 'R2 — augment 속도·병렬성', summary, charts)
    print(f'[R2] report -> {out_dir}/report.html')


if __name__ == '__main__':
    main()
