"""nav3d 경로 다듬기 — `find_path` 최단경로를 바닥 / 계단 구간으로 나눠 따로 다듬는다.

    find_path (벽에 r_b 로 붙어 도는 꺾은선, 높이는 꺾인 점에만)
      -> 5 cm 로 촘촘히 + navmesh 표면에 수직 투영 -> 점마다 "바닥 / 계단" 라벨
      -> 바닥 구간: 탄성 밴드 — 매 반복 (a) 이웃 중점 쪽으로 당기고(곧게) (b) navmesh 경계에서 `push_margin` 까지 밀어냄.
                    움직인 점과 이웃 선분이 navmesh 안 + 경계 여유 ≥ `min_margin` 일 때만 받아들인다.
                    끝으로 Chaikin 모서리 깎기 2회(같은 검사, 실패 시 그 구간은 깎지 않음).
      -> 계단 구간: 곡선으로 다듬지 않는다. 진행 방향에 수직으로 navmesh 폭을 재서 **폭 가운데**로 옮기고,
                    가운데까지 거리(오프셋)를 2 m 중앙값 + 1 m 평균으로 다듬고, navmesh 구멍(난간 기둥 등)을 가로지르는
                    곳만 오프셋을 반씩 줄여 원본 쪽으로 되돌린다.
      -> 이어 붙이고 5 cm 로 다시 찍어 높이(z)를 navmesh 표면에서 읽는다.

용어
- **경계 여유(margin)**: navmesh 경계까지 거리. navmesh 는 이미 벽에서 `agent_radius` 만큼 깎였으므로
  벽까지 실제 거리 ≈ margin + agent_radius. find_path 꺾인 점은 margin 0 (경계 위).
- **계단 같은 곳(stairs-like)**: 층에 안 든 삼각형(넓은 평평한 조각 ≥ 4 m² 는 뺌)을 이은 덩어리 중 높이 폭 ≥ `stairs_min_zspan`(0.3 m),
  면적 ≥ `stairs_min_area`(1 m²), 평균 경사 ≥ `stairs_min_slope_deg`(10°) 인 곳
  (층 사이 계단 + 집 안 반 층 계단 + 경사로). 01 의 "연결부"(서로 다른 층을 잇는 것)보다 넓은 정의다 —
  짧은 계단도 곡선으로 깎지 않으려고.

좌표는 전부 우리 좌표(Z-up). habitat 호출 직전에만 `nu.ours_to_hab`.
"""

from dataclasses import dataclass, replace

import numpy as np

import nav3d_utils as nu


@dataclass
class RefineCfg:
    spacing_m: float = 0.5          # 바닥 탄성 밴드 점 간격
    push_margin_m: float = 0.10     # 경계에서 이만큼 떨어지도록 민다 (목표)
    min_margin_m: float = 0.02      # 움직인 점·선분이 지켜야 할 최소 경계 여유 (하드)
    band_iters: int = 15
    band_lambda: float = 0.3        # 이웃 중점 쪽으로 당기는 비율
    chaikin_iters: int = 2
    stairs_step_m: float = 0.25     # 계단 재표본 간격
    stairs_scan_m: float = 2.0      # 폭 재기 최대 거리 (한쪽)
    stairs_avg_m: float = 1.0       # 계단 오프셋 이동 평균 창
    stairs_median_m: float = 2.0    # 계단 오프셋 중앙값 창 (구멍·기둥에 튀는 값 제거)
    min_run_m: float = 0.3          # 이보다 짧은 계단/바닥 조각은 이웃에 합친다
    check_step_m: float = 0.10


class NavScene:
    """navmesh 하나 + 표면 투영 + 평면·층·계단 같은 곳 라벨."""

    def __init__(self, pf, min_level_area=10.0, level_tol=0.3, storey_gap=1.5, stairs_min_zspan=0.3,
                 stairs_min_area=1.0, stairs_min_slope_deg=10.0, stairs_group='sloped', stairs_flat_keep_m2=4.0):
        self.pf = pf
        self.tris, self.isl = nu.navmesh_triangles_islands(pf)
        self.surf = nu.NavSurface(self.tris)
        a, n = nu.tri_area_normal(self.tris)
        cz = self.tris[:, :, 2].mean(1)
        flat = n[:, 2] > np.cos(np.deg2rad(nu.UP_COS_DEG))
        self.levels = nu.find_levels(cz[flat], a[flat], min_area_m2=min_level_area)
        lv = nu.assign_level(cz, self.levels, level_tol)
        lv[~flat] = -1
        self.lv = lv
        self.lv2st, self.storeys = nu.group_storeys(self.levels, storey_gap) if self.levels else (np.zeros(0, int), [])
        # 기울어진 삼각형만으로 덩어리를 만든다. lv < 0 전체로 묶으면 어느 층에도 안 든 평평한 바닥(층 사이 높이)까지
        # 경사 조각과 한 덩어리가 돼 "계단" 이 됐다(VFuaQ6m2Qom 102 m² 덩어리 중 평평한 면이 87 %).
        # stairs_group='unleveled' 는 수정 전 방식(비교 그림용, 03s): 층에 안 든 삼각형 전부로 덩어리.
        # 'sloped': 층에 안 든 평평한 조각 중 넓은 것(≥ stairs_flat_keep_m2, 층 사이 높이의 바닥)만 뺀다.
        # 평평한 삼각형을 전부 빼면 디딤판·참까지 빠져 계단이 줄무늬로 끊겼다(노은역 실측).
        if stairs_group == 'sloped':
            fu = flat & (lv < 0)
            fc = nu.tri_components(self.tris, fu)
            big = np.zeros(len(self.tris), bool)
            for c in np.unique(fc[fc >= 0]):
                m = fc == c
                if a[m].sum() >= stairs_flat_keep_m2:
                    big |= m
            comp = nu.tri_components(self.tris, (lv < 0) & ~big)
        elif stairs_group == 'unleveled':
            comp = nu.tri_components(self.tris, lv < 0)
        elif stairs_group == 'none':     # 계단 구별 없음 — 전부 바닥처럼 다듬는 비교용 (03 --stairs_mode none)
            comp = np.full(len(self.tris), -1)
        else:
            assert False, f'unreachable stairs_group={stairs_group!r}'
        self.stairs_tri = np.zeros(len(self.tris), bool)
        for c in np.unique(comp[comp >= 0]):
            m = comp == c
            t = self.tris[m]
            if a[m].sum() <= 0:          # 넓이 0 삼각형만 모인 덩어리(1pXnu·dhjEz·2n8kA 에서 터짐)
                continue
            slope = np.degrees(np.arccos(np.clip(np.average(n[m, 2], weights=a[m]), -1, 1)))
            # 문턱·작은 경사 조각은 빼야 한다 — 그것까지 "계단" 으로 보면 탁 트인 바닥에서 폭 가운데 맞추기가 돌아
            # 경로가 옆으로 최대 2 m 밀렸다(vln_ce kEZ7cmS4wCh 실측: 길이 14 -> 18 m, 같은 루트 8 -> 4)
            if (t[:, :, 2].max() - t[:, :, 2].min() >= stairs_min_zspan and a[m].sum() >= stairs_min_area
                    and slope >= stairs_min_slope_deg):
                self.stairs_tri |= m
        self.normals = n
        self.agent_radius = float(pf.nav_mesh_settings.agent_radius)
        self.agent_height = float(pf.nav_mesh_settings.agent_height)

    # ---- navmesh 질의 (우리 좌표) ----
    def margin(self, p, r=1.0) -> float:
        return float(self.pf.distance_to_closest_obstacle(nu.ours_to_hab(p).astype(np.float32), r))

    def surface_z(self, xy, z_near, tol=0.3):
        c = self.surf.candidates(np.asarray(xy, float))
        if not c:
            return None
        z = min(c, key=lambda v: abs(v - z_near))
        return z if abs(z - z_near) <= tol else None

    def point_ok(self, p, min_margin) -> bool:
        z = self.surface_z(p[:2], p[2])
        if z is None:
            return False
        return self.margin(np.array([p[0], p[1], z]), min_margin + 0.05) >= min_margin

    def segment_ok(self, a, b, min_margin, step) -> bool:
        L = float(np.linalg.norm(b[:2] - a[:2]))
        k = max(1, int(np.ceil(L / step)))
        for t in np.arange(1, k) / k:
            if not self.point_ok(a + t * (b - a), min_margin):
                return False
        return True

    def segment_min_margin(self, a, b, step, cap=0.5) -> float:
        """선분 a-b (양 끝 제외) 표본의 최소 경계 여유. navmesh 밖 점이 있으면 -1."""
        L = float(np.linalg.norm(b[:2] - a[:2]))
        k = max(1, int(np.ceil(L / step)))
        m = cap
        for t in np.arange(1, k) / k:
            p = a + t * (b - a)
            z = self.surface_z(p[:2], p[2])
            if z is None:
                return -1.0
            m = min(m, self.margin(np.array([p[0], p[1], z]), cap))
        return m

    def find_path(self, a, b):
        import habitat_sim
        sp = habitat_sim.ShortestPath()
        sp.requested_start = nu.ours_to_hab(a).astype(np.float32)
        sp.requested_end = nu.ours_to_hab(b).astype(np.float32)
        if not self.pf.find_path(sp):
            return None
        return nu.hab_to_ours(np.asarray(sp.points, dtype=np.float64))

    def project(self, P):
        """5 cm 로 찍은 꺾은선 -> (점 (N,3) z 는 표면, 계단 라벨 (N,), 표면 투영 성공 (N,))."""
        D = nu.densify(P, 0.05)
        z, idx = self.surf.track_idx(D[:, :2], z0=D[0, 2])
        ok = np.isfinite(z)
        D[ok, 2] = z[ok]
        lab = np.where(idx >= 0, self.stairs_tri[np.maximum(idx, 0)], False)
        return D, lab, ok


# ---------------------------------------------------------------------------
def _runs(lab: np.ndarray, D: np.ndarray, min_run_m: float):
    """라벨 열 -> [(i0, i1, is_stairs)] (i1 포함, 이웃 run 과 끝점 공유). 짧은 조각은 이웃에 합친다."""
    lab = lab.copy()
    for _ in range(3):
        runs, i = [], 0
        while i < len(lab):
            j = i
            while j + 1 < len(lab) and lab[j + 1] == lab[i]:
                j += 1
            runs.append([i, j, bool(lab[i])])
            i = j + 1
        changed = False
        for r in runs:
            L = np.linalg.norm(np.diff(D[r[0]:r[1] + 1, :2], axis=0), axis=1).sum()
            if L < min_run_m and len(runs) > 1:
                lab[r[0]:r[1] + 1] = not r[2]
                changed = True
        if not changed:
            break
    out = []
    for k, (i0, i1, s) in enumerate(runs):
        out.append((i0 if k == 0 else i0 - 1, i1, s))
    return out


def _resample(X, step):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]
    if L[-1] < 1e-9:
        return X[[0, -1]]
    n = max(1, int(np.ceil(L[-1] / step)))
    s = np.linspace(0, L[-1], n + 1)
    return np.column_stack([np.interp(s, L, X[:, k]) for k in range(3)])


def _subsample_keep_corners(seg, spacing):
    """간격 `spacing` 으로 솎되 **꺾인 점은 반드시 남긴다** — 그냥 재표본하면 find_path 꼭짓점이 빠져
    문틀(얇은 벽) 같은 곳을 선분이 가로질러 navmesh 밖으로 나간다(노은역 문에서 실측)."""
    v = np.diff(seg[:, :2], axis=0)
    h = np.arctan2(v[:, 1], v[:, 0])
    dh = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)
    corner = np.zeros(len(seg), bool)
    corner[1:-1] = dh > np.deg2rad(1.0)
    keep, last = [0], 0.0
    L = np.r_[0, np.cumsum(np.linalg.norm(v, axis=1))]
    for i in range(1, len(seg) - 1):
        if corner[i] or L[i] - L[keep[-1]] >= spacing:
            keep.append(i)
    keep.append(len(seg) - 1)
    return seg[keep].copy()


def _push(sc: NavScene, p, cfg: RefineCfg, away_from=None):
    """경계에서 push_margin 까지 민다. 방향 = 주변 8방향 중 경계 여유가 가장 커지는 쪽(경계 위 점도 방향이 생긴다)."""
    q = p.copy()
    d = sc.margin(q, cfg.push_margin_m + 0.3)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    dirs = np.column_stack([np.cos(ang), np.sin(ang)])
    for _ in range(6):
        if d >= cfg.push_margin_m:
            break
        step = max(0.02, cfg.push_margin_m - d)
        best, best_d = None, d
        for v in dirs:
            c = q.copy()
            c[:2] = q[:2] + v * step
            z = sc.surface_z(c[:2], q[2])
            if z is None:
                continue
            c[2] = z
            dc = sc.margin(c, cfg.push_margin_m + 0.3)
            if dc > best_d + 1e-4:
                best, best_d = c, dc
        if best is None:
            break
        q, d = best, best_d
    return q


def refine_floor(sc: NavScene, seg, cfg: RefineCfg):
    W = _subsample_keep_corners(seg, cfg.spacing_m)
    n = len(W)
    if n <= 2:
        return W
    for _ in range(cfg.band_iters):
        moved = 0
        for i in range(1, n - 1):
            mid = 0.5 * (W[i - 1] + W[i + 1])
            cand = W[i] + cfg.band_lambda * (mid - W[i])
            z = sc.surface_z(cand[:2], W[i][2])
            if z is None:
                cand = W[i].copy()
            else:
                cand[2] = z
            bis = (W[i - 1][:2] - W[i][:2]) / max(np.linalg.norm(W[i - 1][:2] - W[i][:2]), 1e-9) + \
                  (W[i + 1][:2] - W[i][:2]) / max(np.linalg.norm(W[i + 1][:2] - W[i][:2]), 1e-9)
            cand = _push(sc, cand, cfg, away_from=-bis if np.linalg.norm(bis) > 1e-6 else None)
            if np.linalg.norm(cand[:2] - W[i][:2]) < 1e-4:
                continue
            # 받아들이는 조건: 점·양옆 선분이 navmesh 안이고, 최소 여유가 (min_margin 또는 지금 값) 보다 나빠지지 않음.
            # 이웃이 아직 경계 위(여유 0)일 수 있으니 '절대 기준'이 아니라 '나빠지지 않음' 으로 본다.
            old = min(sc.margin(W[i], 0.5), sc.segment_min_margin(W[i - 1], W[i], cfg.check_step_m),
                      sc.segment_min_margin(W[i], W[i + 1], cfg.check_step_m))
            new = min(sc.margin(cand, 0.5), sc.segment_min_margin(W[i - 1], cand, cfg.check_step_m),
                      sc.segment_min_margin(cand, W[i + 1], cfg.check_step_m))
            if new >= 0 and new >= min(old, cfg.min_margin_m) - 1e-3:
                W[i] = cand
                moved += 1
        if moved == 0:
            break
    for _ in range(cfg.chaikin_iters):
        C = [W[0]]
        for i in range(len(W) - 1):
            q, r = 0.75 * W[i] + 0.25 * W[i + 1], 0.25 * W[i] + 0.75 * W[i + 1]
            C += [q, r]
        C.append(W[-1])
        C = np.array(C)
        C[1], C[-2] = W[0] * 0.5 + C[1] * 0.5, W[-1] * 0.5 + C[-2] * 0.5   # 끝점 근처는 덜 깎는다
        before = min(sc.segment_min_margin(W[i], W[i + 1], cfg.check_step_m) for i in range(len(W) - 1))
        after = min(sc.segment_min_margin(C[i], C[i + 1], cfg.check_step_m) for i in range(len(C) - 1))
        good = after >= 0 and after >= min(before, cfg.min_margin_m) - 1e-3
        if not good:
            break
        W = C
    return W


def refine_stairs(sc: NavScene, seg, cfg: RefineCfg):
    S = _subsample_keep_corners(seg, cfg.stairs_step_m)
    offs, es = np.zeros(len(S)), np.zeros((len(S), 2))
    for i in range(len(S)):
        a, b = S[max(0, i - 2)], S[min(len(S) - 1, i + 2)]
        t = b[:2] - a[:2]
        if np.linalg.norm(t) < 1e-6:
            continue
        t /= np.linalg.norm(t)
        e = np.array([-t[1], t[0]])
        ext, opened = [], False
        for sgn in (1, -1):
            k, zprev, last = 0, S[i][2], 0.0
            while True:
                k += 1
                if k * 0.05 > cfg.stairs_scan_m:      # 이쪽은 막힘 없이 트여 있다 -> 가운데 맞추기 안 함
                    opened = True
                    break
                q = S[i][:2] + sgn * e * k * 0.05
                c = [(z, j) for z, j in sc.surf.candidates_idx(q) if abs(z - zprev) <= 0.15]
                if not c:
                    break
                z, j = min(c, key=lambda v: abs(v[0] - zprev))
                if not sc.stairs_tri[j] and k * 0.05 > 0.3:   # 계단 밖(참·바닥)으로 0.3 m 넘게 나가면 폭 끝으로 본다
                    break
                zprev, last = z, k * 0.05
            ext.append(last)
        if opened:
            offs[i], es[i] = 0.0, e
            continue
        offs[i] = 0.5 * (ext[0] - ext[1])
        es[i] = e
    # 오프셋(가운데까지 거리)을 넓은 중앙값(구멍·난간 기둥에 튀는 값 제거) -> 평균으로 매끈하게
    hm = max(1, int(round(cfg.stairs_median_m / cfg.stairs_step_m / 2)))
    ha = max(1, int(round(cfg.stairs_avg_m / cfg.stairs_step_m / 2)))
    med = np.array([np.median(offs[max(0, i - hm):i + hm + 1]) for i in range(len(offs))])
    o = np.array([med[max(0, i - ha):i + ha + 1].mean() for i in range(len(med))])

    def place(o):
        X = S.copy()
        for i in range(len(S)):
            q = S[i].copy()
            q[:2] = S[i][:2] + es[i] * o[i]
            z = sc.surface_z(q[:2], S[i][2], tol=0.3)
            if z is not None:
                q[2] = z
                X[i] = q
        return X

    # navmesh 구멍을 가로지르는 선분이 있으면 그 양 끝 점의 오프셋만 반씩 줄인다 (최대 5번 -> 원본 쪽으로)
    base = min(sc.segment_min_margin(S[i], S[i + 1], cfg.check_step_m) for i in range(len(S) - 1)) if len(S) > 1 else 0.0
    floor_m = min(base, cfg.min_margin_m)
    X = place(o)
    for _ in range(5):
        bad = [i for i in range(len(X) - 1) if sc.segment_min_margin(X[i], X[i + 1], cfg.check_step_m) < floor_m - 1e-3]
        if not bad:
            return X
        for i in bad:
            o[i] *= 0.5
            o[i + 1] *= 0.5
        X = place(o)
    return S


def _on_mesh(sc: NavScene, X) -> bool:
    """5 cm 로 찍은 모든 점이 navmesh 표면 위인가 (평가의 on 과 같은 자: track_idx 성공)."""
    D = nu.densify(X, 0.05)
    z, _ = sc.surf.track_idx(D[:, :2], z0=D[0, 2])
    return bool(np.isfinite(z).all())


def refine(sc: NavScene, P, cfg: RefineCfg = RefineCfg()):
    """find_path 꺾은선 P -> dict(traj (N,3), stairs (N,) bool, raw (M,3), raw_stairs).

    구간마다 다듬은 결과를 5 cm 로 다시 찍어 navmesh 위인지 확인하고, 아니면 덜 다듬은 것으로 물러선다
    (바닥: Chaikin 1회 -> 0회 -> 원본, 계단: 원본). 다듬는 중 검사는 10 cm 간격이라 그 사이에서 모서리를
    1 cm 안팎 넘는 경우가 vln_ce 경로 44개(약 2 %)에서 나왔다."""
    D, lab, ok = sc.project(P)
    runs = _runs(lab, D, cfg.min_run_m)
    force_raw = set()          # 원본으로 되돌릴 계단 구간 — 옮긴 계단 끝점과 바닥 원본을 이은 선이 구멍을 지나면 (VLzqgDo317F ep11)
    for _ in range(3):
        new, n_fallback, again = {}, 0, False
        for k, (i0, i1, st) in enumerate(runs):       # 계단 먼저 — 그 끝점이 바닥 구간의 고정 끝점이 된다
            if st:
                X = D[i0:i1 + 1].copy() if k in force_raw else refine_stairs(sc, D[i0:i1 + 1], cfg)
                if not _on_mesh(sc, X):
                    X, n_fallback = D[i0:i1 + 1].copy(), n_fallback + 1
                new[k] = X
        for k, (i0, i1, st) in enumerate(runs):
            if not st:
                seg = D[i0:i1 + 1].copy()
                if k - 1 in new:
                    seg[0] = new[k - 1][-1]
                if k + 1 in new:
                    seg[-1] = new[k + 1][0]
                X = refine_floor(sc, seg, cfg)
                for ch in range(cfg.chaikin_iters - 1, -2, -1):
                    if _on_mesh(sc, X):
                        break
                    n_fallback += 1
                    if ch >= 0:
                        X = refine_floor(sc, seg, replace(cfg, chaikin_iters=ch))
                    else:
                        X = seg
                if not _on_mesh(sc, X):
                    nb = {j for j in (k - 1, k + 1) if j in new and j not in force_raw}
                    force_raw |= nb
                    again = again or bool(nb)
                new[k] = X
        if not again:
            break
    parts, labs = [], []
    for k, (i0, i1, st) in enumerate(runs):
        X = nu.densify(new[k], 0.05)
        if parts:
            X = X[1:]
        parts.append(X)
        labs.append(np.full(len(X), st))
    T = np.vstack(parts)
    z, _ = sc.surf.track_idx(T[:, :2], z0=T[0, 2])
    good = np.isfinite(z)
    T[good, 2] = z[good]
    return {'traj': T, 'stairs': np.concatenate(labs), 'raw': D, 'raw_stairs': lab, 'raw_on': ok, 'traj_on': good,
            'n_runs': len(runs), 'n_stairs_runs': sum(r[2] for r in runs), 'n_fallback': n_fallback + len(force_raw)}


# ---------------------------------------------------------------------------
def turn_per_m(X, step=0.25):
    R = _resample(X, step)
    if len(R) < 3:
        return 0.0
    v = np.diff(R[:, :2], axis=0)
    h = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    L = np.linalg.norm(v, axis=1).sum()
    return float(np.degrees(np.abs(np.diff(h)).sum()) / max(L, 1e-6))


def evaluate(sc: NavScene, X, stairs=None, on=None, md=None, r_b=None, skip_end_m=0.3):
    """경로 품질 — navmesh 위 비율, 경계 여유(양 끝 skip_end_m 제외), mesh 3D 여유, 회전량, 계단 정면 각도."""
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]
    inner = (L > skip_end_m) & (L < L[-1] - skip_end_m)
    sub = np.arange(0, len(X), 2)                     # 10 cm 간격으로 질의
    mg = np.array([sc.margin(X[i]) for i in sub])
    mi = inner[sub]
    out = {'length_m': float(L[-1]), 'on_frac': float(np.mean(on)) if on is not None else None,
           'margin_min_m': float(mg[mi].min()) if mi.any() else float(mg.min()),
           'margin_p5_m': float(np.percentile(mg[mi], 5)) if mi.any() else float(mg.min()),
           'turn_per_m_deg': turn_per_m(X)}
    if md is not None:
        H = [0.5, sc.agent_height - (r_b if r_b is not None else 0.05)]   # 위 구(반지름 r_b) 꼭대기 = 키 (01h 참고)
        c = np.min([md(X[inner] + [0, 0, h]) for h in H], axis=0) if inner.any() else np.array([np.inf])
        out['mesh_clear_min_m'] = float(c.min())
        if r_b is not None:
            out['mesh_clear_lt_rb_frac'] = float((c < r_b).mean())
    if stairs is not None and stairs.any():
        z, idx = sc.surf.track_idx(X[:, :2], z0=X[0, 2])
        ang = []
        for i in np.where(stairs & (idx >= 0))[0][::2]:
            nn = sc.normals[idx[i]]
            g = nn[:2]
            if np.linalg.norm(g) < 0.2:              # 참(수평)은 건너뜀
                continue
            j0, j1 = max(0, i - 3), min(len(X) - 1, i + 3)
            t = X[j1, :2] - X[j0, :2]
            if np.linalg.norm(t) < 1e-6:
                continue
            c = abs(np.dot(t / np.linalg.norm(t), g / np.linalg.norm(g)))
            ang.append(np.degrees(np.arccos(np.clip(c, 0, 1))))
        out['stairs_frac'] = float(stairs.mean())
        out['stairs_headon_med_deg'] = float(np.median(ang)) if ang else None
        out['stairs_headon_p90_deg'] = float(np.percentile(ang, 90)) if ang else None
    return out


if __name__ == '__main__':
    # self-check (habitat 불필요한 부분): run 나누기, 재표본, 회전량
    D = np.column_stack([np.linspace(0, 10, 201), np.zeros(201), np.zeros(201)])
    lab = np.zeros(201, bool)
    lab[80:120] = True
    lab[150:153] = True                    # 0.1 m 짜리 계단 조각 -> 바닥에 합쳐져야 함
    r = _runs(lab, D, 0.3)
    assert [(s) for _, _, s in r] == [False, True, False], r
    assert r[1][0] == 79 and r[2][0] == 119, r
    assert abs(turn_per_m(D)) < 1e-9
    zig = np.array([[0, 0, 0], [0.5, 0, 0], [0.52, 0.3, 0], [3, 0.3, 0]], float)
    ks = _subsample_keep_corners(nu.densify(zig, 0.05), 1.0)
    assert any(np.allclose(k[:2], [0.5, 0]) for k in ks) and any(np.allclose(k[:2], [0.52, 0.3]) for k in ks), ks
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float)
    assert 80 < turn_per_m(sq, 0.1) * 2 < 100, turn_per_m(sq, 0.1)
    print('[nav3d_path] self-check OK')
