import sys; sys.path.append('.')
import numpy as np, importlib.util as iu
sp=iu.spec_from_file_location('ns','scripts/eval/noeun_navmesh_server.py')
NS=iu.module_from_spec(sp); sp.loader.exec_module(NS)
import habitat_sim
def spots(navmesh, grid_m):
    pf=habitat_sim.PathFinder(); assert pf.load_nav_mesh(navmesh)
    lo,hi=pf.get_bounds(); out=[]; seen=set()
    for x in np.arange(lo[0],hi[0]+grid_m,grid_m):
        for z in np.arange(lo[2],hi[2]+grid_m,grid_m):
            for y in np.arange(lo[1],hi[1]+1.0,1.0):
                p=pf.snap_point(np.array([x,y,z],dtype=np.float32))
                if not np.all(np.isfinite(p)): continue
                w=NS.h2w(np.asarray(p,float))
                key=(round(w[0]/grid_m), round(w[1]/grid_m), round(w[2]/1.5))
                if key in seen: continue
                seen.add(key); out.append([float(t) for t in w])
    return out
for g in (3.0,5.0,8.0):
    s=spots('data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh',g)
    zs=sorted(set(round(p[2]) for p in s))
    print(f'  격자 {g} m -> 설 자리 {len(s):4d} 개 · 사진 {len(s)*4:5d} 장 · 높이대 {zs[:8]}')
