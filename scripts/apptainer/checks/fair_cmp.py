"""공정 비교 — Gemma 점 vs DINO+SAM 마스크, 둘 다 **같은 메시 광선**으로 거리를 잰다.

차이가 나는 것은 '어디를 가리켰나'(위치 지정) 뿐이다. 거리 계산은 동일.
"""
import json, base64, urllib.request, math, gzip, re
import numpy as np, torch, trimesh
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

W,H,HFOV=640,480,68.7; FX=(W/2)/math.tan(math.radians(HFOV)/2)
h2o=lambda q:np.array([q[0],-q[2],q[1]])
VS='data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r/val_seen/val_seen.json.gz'
eps={str(e['episode_id']):e for e in json.load(gzip.open(VS,'rt'))['episodes']}
snaps={str(json.loads(l)['ep']):json.loads(l) for l in open('logs/goal_fast22/snaps.jsonl')}
mesh=trimesh.load('data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb',force='mesh',process=False,skip_materials=True)
RI=mesh.ray
MID='IDEA-Research/grounding-dino-base'
dp=AutoProcessor.from_pretrained(MID); det=AutoModelForZeroShotObjectDetection.from_pretrained(MID).to('cuda').eval()
spn=SamProcessor.from_pretrained('facebook/sam-vit-base'); sam=SamModel.from_pretrained('facebook/sam-vit-base').to('cuda').eval()

def ask(path,prompt,mx=60):
    img=base64.b64encode(open(path,'rb').read()).decode()
    body={"model":"gemma-4-31B-it","max_tokens":mx,"temperature":0.0,
      "messages":[{"role":"user","content":[{"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img}"}}]}]}
    r=urllib.request.Request("http://127.0.0.1:8200/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=300).read().decode())['choices'][0]['message']['content'].strip()

def raycast(us,vs,c2w):
    c2w=np.asarray(c2w); org=c2w[:3,3]
    us=np.atleast_1d(np.asarray(us,float)); vs=np.atleast_1d(np.asarray(vs,float))
    d=np.stack([(us-(W-1)/2)/FX,(vs-(H-1)/2)/FX,np.ones(len(us))],axis=1)@c2w[:3,:3].T
    d/=np.linalg.norm(d,axis=1,keepdims=True)
    loc,_,_=RI.intersects_location(np.repeat([org],len(d),axis=0),d,multiple_hits=False)
    if len(loc)==0: return None,None
    p=np.median(loc,axis=0)
    return p, float(np.linalg.norm(p-org))

Q_NAME=("You are a robot standing at the destination of a navigation route in a subway station, looking straight ahead. "
        "Name ONE clearly visible object next to you that a person could use as a landmark to know where to stop. "
        "Answer with ONLY a short noun phrase. No explanation.")
Q_PT=("Point at the center of {x}. Answer with ONLY two decimals between 0 and 1 "
      "(fraction of image width and height, origin top-left): x y")   # Gemma 는 1000 기준으로 답한다 — 비율로 받아야 안전

print(f"{'ep':>4s} {'Gemma 가 고른 대상':26s} {'Gemma점':>14s} {'카메라거리':>9s} | {'DINO+SAM':>9s} {'차이':>7s} {'목적지까지(G/D)':>16s}")
rows=[]
for ep in sorted(snaps,key=int):
    s=snaps[ep]; gp=h2o(eps[ep]['goals'][0]['position']); path=f"logs/goal_fast22/{s['rgb']}"
    try:
        name=ask(path,Q_NAME,40).strip().strip('."')
        pt=ask(path,Q_PT.format(x=name),16)
        m=re.findall(r'[01]?\.\d+|[01]\b', pt)
        if len(m)<2: print(f"{ep:>4s} {name[:26]:26s} 좌표실패 {pt!r}"); continue
        fx_, fy_ = float(m[0]), float(m[1])
        u,v=int(round(fx_*(W-1))), int(round(fy_*(H-1)))
        if not (0<=u<W and 0<=v<H): print(f"{ep:>4s} {name[:26]:26s} 범위밖 ({u},{v}) from {pt!r}"); continue
        pg,dg=raycast(u,v,s['c2w'])
        # DINO + SAM
        img=Image.open(path).convert('RGB')
        inp=dp(images=img,text=name.rstrip('.')+'.',return_tensors='pt').to('cuda')
        with torch.no_grad(): o=det(**inp)
        r=dp.post_process_grounded_object_detection(o,inp.input_ids,threshold=0.25,text_threshold=0.25,target_sizes=[img.size[::-1]])[0]
        dd=pd_=None
        if len(r['scores']):
            i=int(torch.argmax(r['scores'])); box=[float(x) for x in r['boxes'][i]]
            si=spn(img,input_boxes=[[box]],return_tensors='pt').to('cuda')
            with torch.no_grad(): so=sam(**si,multimask_output=False)
            mk=spn.image_processor.post_process_masks(so.pred_masks.cpu(),si['original_sizes'].cpu(),si['reshaped_input_sizes'].cpu())[0].numpy()[0,0]
            vv,uu=np.nonzero(mk)
            if len(uu):
                if len(uu)>600:
                    k=np.random.RandomState(0).choice(len(uu),600,replace=False); uu,vv=uu[k],vv[k]
                pd_,dd=raycast(uu,vv,s['c2w'])
        if pg is None: print(f"{ep:>4s} {name[:26]:26s} 광선실패"); continue
        gg=math.hypot(pg[0]-gp[0],pg[1]-gp[1])
        if dd is None:
            print(f"{ep:>4s} {name[:26]:26s} ({u:3d},{v:3d}) {dg:9.2f} | {'검출X':>9s} {'-':>7s} {gg:7.2f}/{'-':>7s}")
            rows.append((ep,name,dg,None,gg,None)); continue
        gd_=math.hypot(pd_[0]-gp[0],pd_[1]-gp[1])
        print(f"{ep:>4s} {name[:26]:26s} ({u:3d},{v:3d}) {dg:9.2f} | {dd:9.2f} {dg-dd:+7.2f} {gg:7.2f}/{gd_:7.2f}")
        rows.append((ep,name,dg,dd,gg,gd_))
    except Exception as e:
        print(f"{ep:>4s} 실패 {type(e).__name__} {str(e)[:50]}")
a=[(r[2],r[3]) for r in rows if r[3] is not None]
if a:
    A=np.array(a); e=A[:,0]-A[:,1]
    print(f"\n같은 대상·같은 거리계산 — {len(A)}건")
    print(f"  카메라거리 차: 절대 중앙 {np.median(np.abs(e)):.2f} m · 최대 {np.abs(e).max():.2f} m · 상관 {np.corrcoef(A[:,0],A[:,1])[0,1]:.3f}")
    b=np.array([(r[4],r[5]) for r in rows if r[5] is not None]); e2=b[:,0]-b[:,1]
    print(f"  목적지거리 차: 절대 중앙 {np.median(np.abs(e2)):.2f} m · 최대 {np.abs(e2).max():.2f} m")
    agree=sum(1 for x in e2 if abs(x)<0.5)
    print(f"  0.5 m 안에서 일치: {agree}/{len(b)}")
json.dump([{'ep':r[0],'name':r[1],'gemma_cam':r[2],'dino_cam':r[3],'gemma_goal':r[4],'dino_goal':r[5]} for r in rows],
          open('logs/fair_cmp.json','w'),ensure_ascii=False,indent=1)
