"""층별 걸을 수 있는 영역 윤곽 — 평면도 배경용."""
import json, importlib.util as iu
import numpy as np
sp=iu.spec_from_file_location('ns','scripts/eval/noeun_navmesh_server.py')
NS=iu.module_from_spec(sp); sp.loader.exec_module(NS)
import habitat_sim
pf=habitat_sim.PathFinder(); assert pf.load_nav_mesh('data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh')
V=json.load(open('data/data/noeun_eval/navmesh/noeun_station/verify.json'))
storeys=V['storeys']
lo,hi=pf.get_bounds()
step=0.5
out=[]
for i,st in enumerate(storeys):
    pts=[]
    for x in np.arange(lo[0],hi[0],step):
        for z in np.arange(lo[2],hi[2],step):
            y=st['z']  # ours z == habitat y
            p=pf.snap_point(np.array([x,y,z],dtype=np.float32))
            if not np.all(np.isfinite(p)): continue
            w=NS.h2w(np.asarray(p,float))
            if abs(w[2]-st['z'])>1.2: continue
            if abs(w[0]-x)>step or abs(w[1]-(-z))>step: continue
            pts.append([round(float(w[0]),2), round(float(w[1]),2)])
    out.append(pts)
    print(f'  층 {i}: 점 {len(pts)}')
json.dump(out, open('logs/landmark_tool/floor.json','w'))
print('저장 logs/landmark_tool/floor.json')
