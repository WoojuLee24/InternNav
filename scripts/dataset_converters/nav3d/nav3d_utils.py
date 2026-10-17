"""nav3d 공용 — 좌표 변환, navmesh 빌드·로드, navmesh 삼각형, 층·연결부 찾기, mesh 3D 거리.

## 좌표 규약 (실측으로 정함 — 01 의 V1 이 매번 다시 검증한다)
우리 = **Z-up** (x, y, z). habitat = **Y-up**. trimesh 로 Z-up mesh 를 glb 로 내보내 habitat 에 올리면
habitat 좌표가 아래처럼 된다(미리 회전하면 오히려 틀린다 — 2026-10-13 noeun 에서 확인):

    habitat (x, y, z) = 우리 (x, z, -y)

## 층·연결부
- **층(level)**: navmesh 의 거의 수평인 삼각형 면적을 높이(z)별로 쌓아, 면적이 `min_level_area_m2` 이상인 높이.
- **연결부(connector)**: 어느 층에도 속하지 않는 navmesh 삼각형(기울었거나 층 사이 높이)의 연결 성분 중
  **서로 다른 두 층에 닿는 것** — 계단·에스컬레이터·경사로. recast 는 계단 디딤판을 경사면으로 근사하므로
  navmesh 위에서는 셋이 똑같이 "기운 띠"로 보인다. 계단인지는 **원본 mesh** 를 아래로 쏴서 표면이
  톱니(디딤판·챌판)인지로 가른다(`stepness_rms_m` — 계단 ≈ 챌판/√12 ≈ 0.05 m, 매끈한 경사 < 0.02 m).
"""

import numpy as np
import trimesh

UP_COS_DEG = 15.0          # "거의 수평" 판정 각


# ---------------------------------------------------------------------------
# 좌표
def ours_to_hab(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def hab_to_ours(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.stack([q[..., 0], -q[..., 2], q[..., 1]], axis=-1)


# ---------------------------------------------------------------------------
# mesh / navmesh
def load_mesh_zup(path: str) -> trimesh.Trimesh:
    """지오메트리만(텍스처 없이) — gs_vlnpe 10-12 RAM 사고(텍스처 아틀라스) 재발 방지."""
    m = trimesh.load(path, force='mesh', process=False, skip_materials=True)
    assert isinstance(m, trimesh.Trimesh) and len(m.faces), path
    return m


def add_box_blockers(mesh: trimesh.Trimesh, blockers: list) -> trimesh.Trimesh:
    """mesh 에 축정렬 상자 벽을 더한다 — 3DGS 가 못 살린 투명 유리 등. blockers=[{'name','min':[x,y,z],'max':[x,y,z]}]."""
    boxes = []
    for b in blockers:
        lo, hi = np.asarray(b['min'], float), np.asarray(b['max'], float)
        assert (hi > lo).all(), b
        bx = trimesh.creation.box(extents=hi - lo)
        bx.apply_translation((lo + hi) / 2)
        boxes.append(bx)
    return trimesh.util.concatenate([mesh] + boxes)


def export_glb(mesh: trimesh.Trimesh, out_glb: str) -> None:
    mesh.export(out_glb)        # 회전하지 않는다 (모듈 docstring 의 좌표 규약)


def navmesh_settings(cfg: dict):
    from habitat_sim.nav import NavMeshSettings
    s = NavMeshSettings()
    s.set_defaults()
    for k, v in cfg.items():
        setattr(s, k, v)
    return s


def build_navmesh(glb: str, cfg: dict, out_navmesh: str):
    """glb -> recompute_navmesh -> .navmesh 저장. 저장본을 다시 읽은 PathFinder 를 돌려준다."""
    import habitat_sim
    bc = habitat_sim.SimulatorConfiguration()
    bc.scene_id = glb
    bc.enable_physics = False
    bc.create_renderer = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bc, [habitat_sim.AgentConfiguration()]))
    try:
        ok = bool(sim.recompute_navmesh(sim.pathfinder, navmesh_settings(cfg)))
        assert ok, 'recompute_navmesh 실패'
        sim.pathfinder.save_nav_mesh(out_navmesh)
    finally:
        sim.close()
    return load_pathfinder(out_navmesh)


def load_pathfinder(navmesh: str):
    import habitat_sim
    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(navmesh)
    assert pf.is_loaded, navmesh
    return pf


def navmesh_triangles(pf, island: int = -1) -> np.ndarray:
    """(T,3,3) 우리 좌표 삼각형."""
    v = np.asarray(pf.build_navmesh_vertices(island), dtype=np.float64).reshape(-1, 3)
    idx = np.asarray(pf.build_navmesh_vertex_indices(island), dtype=np.int64).reshape(-1, 3)
    return hab_to_ours(v)[idx]


def tri_area_normal(tris: np.ndarray):
    c = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    a = 0.5 * np.linalg.norm(c, axis=1)
    n = c / np.maximum(2 * a, 1e-12)[:, None]
    n[n[:, 2] < 0] *= -1                       # navmesh 는 위를 향한다 — 감김 방향 무시
    return a, n


def navmesh_triangles_islands(pf):
    """섬별로 뽑아 이어 붙인 (T,3,3) 삼각형 + 같은 순서의 섬 번호.

    ⚠️ `build_navmesh_vertex_indices(i)` 를 섬 순서로 이어 붙인 것은 `(-1)` 전체 목록과 **순서가 다르다**
    (개수만 같다 — 2026-10-13 noeun 에서 섬 라벨이 어긋나 연결된 계단이 끊긴 것처럼 보였다).
    그래서 삼각형도 섬별로 뽑아 순서를 맞춘다. 01 V4 가 `get_island` 로 라벨을 표본 검사한다.
    """
    T, I = [], []
    for i in range(pf.num_islands):
        t = navmesh_triangles(pf, i)
        T.append(t)
        I.append(np.full(len(t), i))
    return np.concatenate(T), np.concatenate(I)


def check_island_labels(pf, tris, isl, n=300, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    k = rng.choice(len(tris), size=min(n, len(tris)), replace=False)
    got = np.array([pf.get_island(np.asarray(pf.snap_point(ours_to_hab(tris[i].mean(0))))) for i in k])
    agree = float((got == isl[k]).mean())
    return {'n': int(len(k)), 'agree': agree, 'passed': agree >= 0.98}


# ---------------------------------------------------------------------------
# 층 · 연결부
def find_levels(z: np.ndarray, w: np.ndarray, bin_m: float = 0.1, min_area_m2: float = 10.0,
                merge_m: float = 0.5) -> list:
    """면적 가중 높이 히스토그램의 봉우리 -> [{'z', 'area_m2'}] (높이순)."""
    if len(z) == 0:
        return []
    edges = np.arange(np.floor(z.min() / bin_m) * bin_m, z.max() + 2 * bin_m, bin_m)
    h, _ = np.histogram(z, bins=edges, weights=w)
    order = np.argsort(-h)
    peaks = []
    for i in order:
        if h[i] < min_area_m2 * 0.25:
            break
        zc = 0.5 * (edges[i] + edges[i + 1])
        if all(abs(zc - p) > merge_m for p in peaks):
            peaks.append(zc)
    levels = []
    for p in sorted(peaks):
        m = np.abs(z - p) <= merge_m / 2
        area = float(w[m].sum())
        if area >= min_area_m2:
            levels.append({'z': float(np.average(z[m], weights=w[m])), 'area_m2': area})
    return levels


def group_storeys(levels, gap):
    """높이순 평면을 gap 미만 간격끼리 묶는다 -> 평면별 층 번호 (L,), 층 목록."""
    st, out = [], []
    for k, L in enumerate(levels):
        if out and L['z'] - levels[out[-1][-1]]['z'] < gap:
            out[-1].append(k)
        else:
            out.append([k])
    for s, ks in enumerate(out):
        for k in ks:
            st.append(s)
    storeys = [{'levels': ks, 'z': float(np.average([levels[k]['z'] for k in ks], weights=[levels[k]['area_m2'] for k in ks])),
                'z_range': [levels[ks[0]]['z'], levels[ks[-1]]['z']],
                'area_m2': float(sum(levels[k]['area_m2'] for k in ks))} for ks in out]
    return np.array(st, dtype=int), storeys


def assign_level(z: np.ndarray, levels: list, tol_m: float) -> np.ndarray:
    """가장 가까운 층 번호, tol 밖이면 -1."""
    if not levels:
        return np.full(len(z), -1)
    L = np.array([l['z'] for l in levels])
    d = np.abs(z[:, None] - L[None])
    k = d.argmin(1)
    k[d.min(1) > tol_m] = -1
    return k


def tri_components(tris: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """sel 삼각형들의 꼭짓점 공유 연결 성분 번호(sel 밖 = -1)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ids = np.where(sel)[0]
    comp = np.full(len(tris), -1)
    if len(ids) == 0:
        return comp
    key = np.round(tris[ids].reshape(-1, 3) / 0.01).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    n_t, n_v = len(ids), vid.max() + 1
    rows = np.repeat(np.arange(n_t), 3)
    g = coo_matrix((np.ones(3 * n_t), (rows, vid.ravel())), shape=(n_t, n_v)).tocsr()
    adj = g @ g.T
    _, lab = connected_components(adj, directed=False)
    comp[ids] = lab
    return comp


def surface_z_down(mesh: trimesh.Trimesh, xy: np.ndarray, z_from: np.ndarray, max_drop: float = 1.5):
    """(x,y,z_from) 에서 아래로 쏴서 처음 맞는 mesh 높이. 못 맞으면 nan."""
    o = np.column_stack([xy, z_from])
    d = np.tile([0.0, 0.0, -1.0], (len(o), 1))
    loc, ray, _ = mesh.ray.intersects_location(o, d, multiple_hits=True)
    out = np.full(len(o), np.nan)
    for r in np.unique(ray):
        zz = loc[ray == r, 2]
        zz = zz[(zz <= z_from[r] + 1e-6) & (zz >= z_from[r] - max_drop)]
        if len(zz):
            out[r] = zz.max()
    return out


def stepness(mesh: trimesh.Trimesh, tris: np.ndarray, n_pts: int = 400, seed: int = 0) -> dict:
    """연결부 위 점들의 실제 mesh 표면 높이를 평면 맞춤한 잔차 RMS — 계단(톱니) vs 매끈한 경사."""
    rng = np.random.default_rng(seed)
    a, _ = tri_area_normal(tris)
    k = rng.choice(len(tris), size=n_pts, p=a / a.sum())
    u, v = rng.random(n_pts), rng.random(n_pts)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tris[k]
    p = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    zs = surface_z_down(mesh, p[:, :2], p[:, 2] + 0.5)
    ok = np.isfinite(zs)
    if ok.sum() < 20:
        return {'n_hit': int(ok.sum()), 'stepness_rms_m': None, 'nav_minus_mesh_med_m': None}
    A = np.column_stack([p[ok, 0], p[ok, 1], np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, zs[ok], rcond=None)
    res = zs[ok] - A @ coef
    return {'n_hit': int(ok.sum()), 'stepness_rms_m': float(np.sqrt(np.mean(res ** 2))),
            'nav_minus_mesh_med_m': float(np.median(p[ok, 2] - zs[ok]))}


class NavSurface:
    """navmesh 표면에 **수직 투영** — xy 를 품은 삼각형 중 z 가 가장 가까운 것의 높이.

    `pf.snap_point` 는 3D 최근접점이라 계단 발치에서 살짝 뜬 점을 옆 계단 경사로 끌어간다(수평 0.5 m+).
    경로 점의 바닥 높이·"수평으로 navmesh 위인가" 판정은 이것으로 한다.
    """

    def __init__(self, tris: np.ndarray):
        from scipy.spatial import cKDTree
        self.t = tris
        self.c = tris[:, :, :2].mean(1)
        self.r = np.linalg.norm(tris[:, :, :2] - self.c[:, None], axis=2).max(1)
        self.tree = cKDTree(self.c)
        self.rmax = float(self.r.max())

    def candidates(self, xy, eps: float = 2e-3) -> list:
        """xy 를 품은 모든 삼각형의 표면 z (여러 층이 겹치면 여러 개).

        eps = 무게중심 좌표 허용(삼각형 크기 대비 0.2 %) — find_path 경로는 navmesh **경계선 위**를 지나므로
        1e-6 이면 부동소수 오차로 경계 위 점이 들쭉날쭉 '밖' 이 된다(노은역 출구 계단에서 실측).
        """
        zs = []
        for i in self.tree.query_ball_point(xy, self.rmax):
            if np.linalg.norm(xy - self.c[i]) > self.r[i] + eps:
                continue
            a, b, c = self.t[i]
            v0, v1, v2 = b[:2] - a[:2], c[:2] - a[:2], xy - a[:2]
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-12:
                continue
            u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
            w = (v0[0] * v2[1] - v2[0] * v0[1]) / den
            if u < -eps or w < -eps or u + w > 1 + eps:
                continue
            zs.append(a[2] + u * (b[2] - a[2]) + w * (c[2] - a[2]))
        return zs

    def candidates_idx(self, xy, eps: float = 2e-3) -> list:
        """[(z, 삼각형 번호)] — `candidates` 와 같은 판정."""
        out = []
        for i in self.tree.query_ball_point(xy, self.rmax):
            if np.linalg.norm(xy - self.c[i]) > self.r[i] + eps:
                continue
            a, b, c = self.t[i]
            v0, v1, v2 = b[:2] - a[:2], c[:2] - a[:2], xy - a[:2]
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-12:
                continue
            u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
            w = (v0[0] * v2[1] - v2[0] * v0[1]) / den
            if u < -eps or w < -eps or u + w > 1 + eps:
                continue
            out.append((a[2] + u * (b[2] - a[2]) + w * (c[2] - a[2]), i))
        return out

    def track_idx(self, xy: np.ndarray, z0: float, max_step_dz: float = 0.5):
        """`track` 과 같고 삼각형 번호(못 찾으면 -1)도 돌려준다."""
        z_out, i_out = np.full(len(xy), np.nan), np.full(len(xy), -1)
        prev = float(z0)
        for n, p in enumerate(xy):
            c = self.candidates_idx(p)
            if c:
                z, i = min(c, key=lambda v: abs(v[0] - prev))
                if abs(z - prev) <= max_step_dz:
                    z_out[n], i_out[n], prev = z, i, z
        return z_out, i_out

    def track(self, xy: np.ndarray, z0: float, max_step_dz: float = 0.5) -> np.ndarray:
        """경로를 따라가며 투영 — 매 점에서 **직전 점 높이에 가장 가까운** 표면을 고른다.

        Detour `find_path` 는 2D 로 당긴 꺾은선이라 높이가 꺾인 점에만 있다. 대합실→계단→승강장이 2D 로
        일직선이면 꼭짓점 2개뿐이고 그 사이 직선 z 는 허공(최대 1~4 m)을 지난다. 그래서 점별 최근접 z 가 아니라
        연속성으로 층을 고른다(걸을 수 있는 표면은 5 cm 마다 max_climb 이하로만 변한다).
        반환: (N,) 표면 z, 못 찾은 점은 nan.
        """
        out = np.full(len(xy), np.nan)
        prev = float(z0)
        for n, p in enumerate(xy):
            zs = self.candidates(p)
            if zs:
                z = min(zs, key=lambda v: abs(v - prev))
                if abs(z - prev) <= max_step_dz:
                    out[n] = prev = z
        return out

    def project(self, P: np.ndarray, max_dz: float = 1.0, eps: float = 1e-6) -> np.ndarray:
        """(N,3) -> (N,) 표면 z. xy 를 품은 삼각형이 없거나 |dz| > max_dz 면 nan."""
        out = np.full(len(P), np.nan)
        for n, p in enumerate(P):
            for i in self.tree.query_ball_point(p[:2], self.rmax):
                if np.linalg.norm(p[:2] - self.c[i]) > self.r[i] + eps:
                    continue
                a, b, c = self.t[i]
                v0, v1, v2 = b[:2] - a[:2], c[:2] - a[:2], p[:2] - a[:2]
                den = v0[0] * v1[1] - v1[0] * v0[1]
                if abs(den) < 1e-12:
                    continue
                u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
                w = (v0[0] * v2[1] - v2[0] * v0[1]) / den
                if u < -eps or w < -eps or u + w > 1 + eps:
                    continue
                z = a[2] + u * (b[2] - a[2]) + w * (c[2] - a[2])
                if abs(z - p[2]) <= max_dz and (np.isnan(out[n]) or abs(z - p[2]) < abs(out[n] - p[2])):
                    out[n] = z
        return out


# ---------------------------------------------------------------------------
# mesh 3D 거리
class MeshDistance:
    """mesh 까지 3D 거리. 표면 점 KD-tree 로 근사 후, 근사값 < `exact_below` 인 점만 정확히 다시 잰다."""

    def __init__(self, mesh: trimesh.Trimesh, spacing_m: float = 0.05, seed: int = 0):
        from scipy.spatial import cKDTree
        self.mesh = mesh
        n = int(mesh.area / spacing_m ** 2)
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
        self.tree = cKDTree(np.vstack([pts, mesh.vertices]))

    def __call__(self, p: np.ndarray, exact_below: float = 0.4) -> np.ndarray:
        d, _ = self.tree.query(p)
        m = d < exact_below
        if m.any():
            _, de, _ = trimesh.proximity.closest_point(self.mesh, p[m])
            d[m] = np.minimum(d[m], de)
        return d


def densify(poly: np.ndarray, step_m: float) -> np.ndarray:
    seg = np.diff(poly, axis=0)
    L = np.linalg.norm(seg, axis=1)
    out = [poly[:1]]
    for p0, s, l in zip(poly[:-1], seg, L):
        k = max(1, int(np.ceil(l / step_m)))
        out.append(p0 + s * (np.arange(1, k + 1) / k)[:, None])
    return np.vstack(out)


def rasterize_tris_xy(tris: np.ndarray, origin_xy, cell_m: float, shape_yx) -> np.ndarray:
    """삼각형 xy 투영을 격자에 채운다. 셀 (iy,ix) 는 [o + i*c, o + (i+1)*c) — gs_vlnpe world_to_cell 과 같다."""
    from PIL import Image, ImageDraw
    img = Image.new('1', (shape_yx[1], shape_yx[0]), 0)
    dr = ImageDraw.Draw(img)
    o = np.asarray(origin_xy, dtype=np.float64)
    for t in tris:
        q = (t[:, :2] - o) / cell_m - 0.5          # PIL 픽셀 중심 규약
        dr.polygon([tuple(x) for x in q], fill=1, outline=1)
    return np.asarray(img, dtype=bool)


if __name__ == '__main__':
    # self-check: 좌표 왕복, 층 찾기, 연결 성분, 래스터
    rng = np.random.default_rng(0)
    p = rng.normal(size=(100, 3))
    assert np.allclose(hab_to_ours(ours_to_hab(p)), p)
    z = np.r_[np.full(500, 0.0), np.full(300, 4.0), np.linspace(0, 4, 50)]
    lv = find_levels(z + rng.normal(0, 0.01, len(z)), np.ones(len(z)) * 0.1, min_area_m2=5)
    assert [round(l['z']) for l in lv] == [0, 4], lv
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
                    [[5, 5, 0], [6, 5, 0], [5, 6, 0]]], float)
    c = tri_components(tri, np.ones(3, bool))
    assert c[0] == c[1] != c[2], c
    ns = NavSurface(np.array([[[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 0, 3], [2, 0, 3], [0, 2, 5]]], float))
    zz = ns.project(np.array([[0.5, 0.5, 0.2], [0.5, 0.5, 3.4], [5, 5, 0]]))
    assert np.isclose(zz[0], 0) and np.isclose(zz[1], 3.5) and np.isnan(zz[2]), zz
    tz = ns.track(np.array([[0.2, 0.2], [0.5, 0.5], [1.5, 1.5]]), z0=3.2)
    assert np.allclose(tz[:2], [3.0 + 0.2 * 0 + 2 * 0.2 / 2, 3.5]) and np.isnan(tz[2]), tz
    r = rasterize_tris_xy(tri[:2], (0, 0), 0.1, (12, 12))
    assert r[:10, :10].mean() > 0.95 and not r[11, 11], r.sum()
    print('[nav3d_utils] self-check OK')
