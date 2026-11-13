"""같은 ep·step 에서 메시 광선 거리 vs Isaac depth 를 대조한다."""
import json, math, gzip
from pathlib import Path
import numpy as np, trimesh
W,H,HFOV=640,480,68.7
fx=(W/2)/math.tan(math.radians(HFOV)/2)
D=Path('logs/goal_snap_d'); S=Path('logs/goal_snap')
mesh=trimesh.load('data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb',force='mesh',process=False,skip_materials=True)
RI=trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
ld={r['ep']:r for r in json.load(open('logs/landmark/mesh.json'))}

drec={}
for l in (D/'snaps.jsonl').open():
    r=json.loads(l); drec[(str(r['ep']),r['step'])]=r

print(f"{'ep':>4s} {'step':>5s} {'픽셀':>12s} {'메시광선':>9s} {'Isaac depth':>12s} {'차이':>8s}")
diffs=[]
for ep,r in sorted(ld.items(), key=lambda x:int(x[0])):
    if 'box' not in r: continue
    step=r['step']; key=(ep,step)
    if key not in drec: continue
    dr=drec[key]
    if 'depth' not in dr: continue
    x0,y0,x1,y1=r['box']; u,v=(x0+x1)/2,(y0+y1)/2
    dep=np.load(D/dr['depth'])['depth'].astype(np.float32)
    if dep.ndim==3: dep=dep[...,0]
    ui,vi=int(round(u)),int(round(v))
    if not(0<=ui<W and 0<=vi<H): continue
    # 박스 안 depth 중앙값 (광선은 중심 1 점이므로 중심 주변 5x5 중앙값으로 비교)
    patch=dep[max(0,vi-2):vi+3, max(0,ui-2):ui+3]
    patch=patch[np.isfinite(patch)&(patch>0.05)]
    if patch.size==0: continue
    d_isaac=float(np.median(patch))
    # 같은 step 의 c2w 로 광선 재계산 (goal_snap_d 의 c2w 사용)
    c2w=np.asarray(dr['c2w']); org=c2w[:3,3]
    d_cam=np.array([(u-(W-1)/2)/fx,(v-(H-1)/2)/fx,1.0]); d=c2w[:3,:3]@d_cam; d/=np.linalg.norm(d)
    loc,_,_=RI.intersects_location(np.array([org]),np.array([d]),multiple_hits=False)
    d_mesh=float(np.linalg.norm(loc[0]-org)) if len(loc) else float('nan')
    diff=d_mesh-d_isaac
    diffs.append(diff)
    print(f"{ep:>4s} {step:5d} ({ui:3d},{vi:3d}) {d_mesh:9.2f} {d_isaac:12.2f} {diff:+8.2f}")
if diffs:
    a=np.array([d for d in diffs if np.isfinite(d)])
    print(f"\n비교 {len(a)}개 | 차이 중앙 {np.median(a):+.3f} m · 절대 중앙 {np.median(np.abs(a)):.3f} m · 최대 {np.max(np.abs(a)):.2f} m")
