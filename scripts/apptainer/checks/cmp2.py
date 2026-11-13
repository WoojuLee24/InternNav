"""goal_snap_d 의 각 프레임에서 메시광선 vs Isaac depth 를 같은 픽셀로 대조."""
import json, math, gzip
from pathlib import Path
import numpy as np, trimesh
W,H,HFOV=640,480,68.7
fx=(W/2)/math.tan(math.radians(HFOV)/2)
D=Path('logs/goal_snap_d')
mesh=trimesh.load('data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb',force='mesh',process=False,skip_materials=True)
RI=trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
recs=[json.loads(l) for l in (D/'snaps.jsonl').open() if 'depth' in json.loads(l)]
print(f'프레임 {len(recs)}장')
# 프레임마다 화면 여러 점을 찍어 비교 (중앙 + 격자)
pts=[(320,240),(160,180),(480,180),(220,330),(420,330),(320,120),(100,240),(540,240)]
rows=[]
for r in recs[::2]:
    dep=np.load(D/r['depth'])['depth'].astype(np.float32)
    if dep.ndim==3: dep=dep[...,0]
    c2w=np.asarray(r['c2w']); org=c2w[:3,3]
    dirs=[]; ds_isaac=[]
    for (u,v) in pts:
        p=dep[max(0,v-2):v+3, max(0,u-2):u+3]; p=p[np.isfinite(p)&(p>0.005)]
        if p.size==0: continue
        d_cam=np.array([(u-(W-1)/2)/fx,(v-(H-1)/2)/fx,1.0]); d=c2w[:3,:3]@d_cam; d/=np.linalg.norm(d)
        dirs.append(d); ds_isaac.append(float(np.median(p))*10.0)  # obs depth 는 max_depth=10 으로 정규화돼 있다
    if not dirs: continue
    loc,idx_r,_=RI.intersects_location(np.repeat([org],len(dirs),axis=0),np.array(dirs),multiple_hits=False)
    got={}
    for L,ir in zip(loc,idx_r): got.setdefault(int(ir), float(np.linalg.norm(L-org)))
    for i,di in enumerate(ds_isaac):
        if i in got: rows.append((got[i], di))
a=np.array(rows)
if len(a):
    diff=a[:,0]-a[:,1]
    print(f'\n비교점 {len(a)}개')
    print(f'  메시광선 거리  중앙 {np.median(a[:,0]):6.2f} m')
    print(f'  Isaac depth   중앙 {np.median(a[:,1]):6.2f} m')
    print(f'  차이(메시-isaac) 중앙 {np.median(diff):+.3f} m · 절대중앙 {np.median(np.abs(diff)):.3f} m')
    print(f'  |차이| < 0.1 m : {(np.abs(diff)<0.1).mean()*100:.1f}%   < 0.3 m : {(np.abs(diff)<0.3).mean()*100:.1f}%')
    print(f'  상관계수 {np.corrcoef(a[:,0],a[:,1])[0,1]:.4f}')
    big=a[np.abs(diff)>1.0]
    print(f'  1 m 넘게 어긋난 점 {len(big)}개' + (f' — 예: 메시 {big[0,0]:.2f} vs isaac {big[0,1]:.2f}' if len(big) else ''))
