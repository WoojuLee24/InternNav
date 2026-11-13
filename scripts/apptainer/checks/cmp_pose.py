import json, math, gzip, collections
import numpy as np
old=collections.defaultdict(list)
for l in open('logs/goal_snap/snaps.jsonl'):
    r=json.loads(l); old[str(r['ep'])].append(r)
new={str(json.loads(l)['ep']):json.loads(l) for l in open('logs/goal_fast22/snaps.jsonl')}
VS='data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r/val_seen/val_seen.json.gz'
eps={str(e['episode_id']):e for e in json.load(gzip.open(VS,'rt'))['episodes']}
h2o=lambda q:(q[0],-q[2],q[1])
print(f"{'ep':>4s} {'기존 목적지거리':>14s} {'고속 목적지거리':>14s} {'카메라 위치차':>12s} {'방향차':>8s}")
rows=[]
for e in sorted(new, key=int):
    gp=h2o(eps[e]['goals'][0]['position'])
    o=min(old[e], key=lambda r: math.hypot(r['robot_pos'][0]-gp[0], r['robot_pos'][1]-gp[1]))
    n=new[e]
    co=np.array(o['c2w']); cn=np.array(n['c2w'])
    dpos=float(np.linalg.norm(co[:3,3]-cn[:3,3]))
    fo=co[:3,2]; fn=cn[:3,2]
    ang=math.degrees(math.acos(max(-1.0,min(1.0,float(np.dot(fo,fn))))))
    do=math.hypot(o['robot_pos'][0]-gp[0], o['robot_pos'][1]-gp[1])
    dn=math.hypot(n['robot_pos'][0]-gp[0], n['robot_pos'][1]-gp[1])
    rows.append((e,do,dn,dpos,ang))
    print(f"{e:>4s} {do:14.2f} {dn:14.2f} {dpos:12.2f} {ang:7.1f}°")
a=np.array([[r[3],r[4]] for r in rows])
print(f"\n카메라 위치차 중앙 {np.median(a[:,0]):.2f} m · 최대 {a[:,0].max():.2f} m")
print(f"방향차      중앙 {np.median(a[:,1]):.1f}° · 최대 {a[:,1].max():.1f}°")
