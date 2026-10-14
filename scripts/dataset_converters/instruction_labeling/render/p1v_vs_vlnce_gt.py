"""P1-V2: 경로 3종(우리 greedy_wp / InternData train 릴리스 / VLN-CE 공식 GT 궤적 *_gt.json.gz locations) 쌍별 거리.

    /usr/bin/python scripts/dataset_converters/instruction_labeling/render/p1v_vs_vlnce_gt.py val_unseen   (또는 train)
"""
import sys, gzip, json; sys.path.insert(0,'scripts/dataset_converters/instruction_labeling/render')
import numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from release_io import *
from habitat_rig_sim import RigSim, ACTION_NAMES
from action_gen import generate_actions
cfg=load_cfg('scripts/dataset_converters/instruction_labeling/configs/render.yaml'); cfg['camera']['rigs']={'125cm_0deg':0.0}
sp=sys.argv[1]; G='/home/irteam/data-vol1/vlnverse_emr/datasets/R2R_VLNCE_v1-3_preprocessed'
gt=json.load(gzip.open(f'{G}/{sp}/{sp}_gt.json.gz'))
raw=json.load(gzip.open(cfg['paths']['raw_root']/sp/f'{sp}.json.gz'))['episodes']
by=defaultdict(dict)
for e in raw: by[Path(e['scene_id']).stem].setdefault(e['trajectory_id'],e)
if sp=='train':
    rs_=set(list_release_scenes(cfg)); by={s:t for s,t in by.items() if s in rs_}
def walk(rs,ep,acts):
    rs.set_start(ep['start_position'],ep['start_rotation']); P=[rs.agent_position()]
    for a in acts[1:]: rs.agent.act(ACTION_NAMES[int(a)]); P.append(rs.agent_position())
    return np.array(P)
def pd_(A,B):
    d=np.linalg.norm(A[:,None,[0,2]]-B[None,:,[0,2]],axis=-1); a,b=d.min(1),d.min(0); return (a.mean()+b.mean())/2, max(a.max(),b.max())
rows=[]
for scene,trajs in sorted(by.items()):
    rs=RigSim(cfg,scene_glb(cfg,scene),gpu=3); rel={}
    if sp=='train':
        for ep in release_episodes(cfg,scene):
            r=match_raw(cfg,scene,ep['task'])
            if r and r['trajectory_id'] not in rel: rel[r['trajectory_id']]=list(release_episode_data(cfg,scene,ep['episode_index'],[])['action'])
    for tid,ep in trajs.items():
        if sp=='train' and tid not in rel: continue
        V=np.array(gt[str(ep['episode_id'])]['locations'])
        O=walk(rs,ep,generate_actions(rs,ep,'greedy_wp',0.25)['actions'])
        m,x=pd_(O,V); row={'scene':scene,'tid':tid,'ours_vs_vlnce_mean':m,'ours_vs_vlnce_max':x,'n_vlnce':len(V),'n_ours':len(O)}
        if tid in rel:
            R=walk(rs,ep,rel[tid]); m,x=pd_(R,V); row.update(rel_vs_vlnce_mean=m,rel_vs_vlnce_max=x,n_rel=len(R))
            m,x=pd_(O,R); row.update(ours_vs_rel_mean=m,ours_vs_rel_max=x)
        rows.append(row)
    rs.close()
df=pd.DataFrame(rows); df.to_csv(f'logs/instruction_labeling/p1v/{sp}_vs_vlnce_gt.csv',index=False)
for c in [c for c in df.columns if c.endswith('_mean') or c.endswith('_max')]:
    print(f'{sp} {c:22s} median {df[c].median():.3f}  mean {df[c].mean():.3f}  p95 {df[c].quantile(.95):.3f}  >1m {(df[c]>1).mean():.3f}')
for c in ('n_vlnce','n_ours','n_rel'):
    if c in df: print(sp,c,'mean %.1f'%df[c].mean())
