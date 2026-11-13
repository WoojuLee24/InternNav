"""목적지 후보 시각화 — 층별 평면도에 랜드마크와 그 옆 목적지."""
import json, math, collections
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
F='.venv_label/fonts/NanumGothic-Regular.ttf'
font_manager.fontManager.addfont(F)
plt.rcParams['font.family']=font_manager.FontProperties(fname=F).get_name()
plt.rcParams['axes.unicode_minus']=False
BIG,MID,SML=17,14,12
import sys
G=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'logs/goal_candidates/goals.json'))
lms=G['landmarks']
floors=json.load(open('logs/landmark_tool/floor.json'))
names=['지하 2층 (승강장)','지하 1층 (대합실)','지상 (출구)']
out=Path('logs/report_landmark/img'); out.mkdir(parents=True, exist_ok=True)

fig,axs=plt.subplots(3,1,figsize=(13.5,15.2))
for i,ax in enumerate(axs):
    if i<len(floors) and floors[i]:
        P=np.array(floors[i]); ax.scatter(P[:,0],P[:,1],s=5,c='0.86',marker='s',linewidths=0)
    sub=[l for l in lms if l['storey']==i]
    # 목적지 후보 (서는 자리) + 바라보는 방향
    for l in sub:
        for s in l['spots']:
            ax.scatter(s['xyz'][0],s['xyz'][1],s=15,c='#2E7355',alpha=.75,linewidths=0,zorder=3)
            ax.plot([s['xyz'][0], s['xyz'][0]+math.cos(s['yaw'])*0.9],
                    [s['xyz'][1], s['xyz'][1]+math.sin(s['yaw'])*0.9],
                    c='#2E7355',lw=.8,alpha=.55,zorder=2)
    L=np.array([l['xyz'] for l in sub]) if sub else np.zeros((0,3))
    if len(L): ax.scatter(L[:,0],L[:,1],s=42,c='#2C5F8A',edgecolors='white',linewidths=.6,zorder=4)
    ng=sum(len(l['spots']) for l in sub)
    ax.set_title(f'{names[i]}  ·  랜드마크 {len(sub)}개 · 목적지 후보 {ng}개',fontsize=BIG)
    ax.set_aspect('equal'); ax.grid(alpha=.22); ax.tick_params(labelsize=SML)
    ax.set_xlabel('x [m]  (역 평면 좌표)',fontsize=MID); ax.set_ylabel('y [m]',fontsize=MID)
h=[Line2D([],[],marker='o',ls='',ms=10,color='#2C5F8A',markeredgecolor='white',label='랜드마크 (멈출 대상)'),
   Line2D([],[],marker='o',ls='',ms=7,color='#2E7355',label='목적지 후보 (서는 자리)'),
   Line2D([],[],ls='-',c='#2E7355',lw=1.6,label='바라보는 방향 — 랜드마크 쪽'),
   Line2D([],[],marker='s',ls='',ms=9,color='0.86',label='걸을 수 있는 바닥')]
fig.legend(handles=h,fontsize=MID,loc='lower center',ncol=4,framealpha=.96,bbox_to_anchor=(0.5,0.003))
fig.suptitle('④ 목적지 후보 — 랜드마크마다 그 옆 0.6~2 m 에 서는 자리를 둔다',fontsize=BIG+3)
fig.tight_layout(rect=[0,0.035,1,0.975])
fig.savefig(out/'goal_map.png',dpi=112); plt.close(fig)
print('저장', out/'goal_map.png')

# 거리 분포
d=[s['dist_m'] for l in lms for s in l['spots']]
fig,ax=plt.subplots(figsize=(10,4.6))
ax.hist(d,bins=28,color='#2E7355',alpha=.85,edgecolor='white')
ax.axvline(2.0,ls='--',c='#AC4529',lw=2)
ax.text(2.03,ax.get_ylim()[1]*.92,'상한 2 m',fontsize=MID,color='#AC4529')
ax.set_xlabel('목적지에서 랜드마크까지 거리 [m]',fontsize=MID)
ax.set_ylabel('목적지 후보 수',fontsize=MID)
ax.set_title(f'목적지–랜드마크 거리 — 전부 2 m 안 (중앙 {np.median(d):.2f} m)',fontsize=BIG)
ax.tick_params(labelsize=SML); ax.grid(alpha=.22,axis='y')
fig.tight_layout(); fig.savefig(out/'goal_dist.png',dpi=112); plt.close(fig)
print('저장', out/'goal_dist.png', f'· 중앙 {np.median(d):.2f} m · 최대 {max(d):.2f} m')
