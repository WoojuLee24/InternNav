"""Gemma 가 거리를 얼마나 맞히나 — 메시 광선 실측과 대조."""
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

dp=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
det=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base').to('cuda').eval()
spn=SamProcessor.from_pretrained('facebook/sam-vit-base'); sam=SamModel.from_pretrained('facebook/sam-vit-base').to('cuda').eval()

def ask(img_path, prompt, max_tokens=120):
    img=base64.b64encode(open(img_path,'rb').read()).decode()
    body={"model":"gemma-4-31B-it","max_tokens":max_tokens,"temperature":0.0,
      "messages":[{"role":"user","content":[{"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img}"}}]}]}
    r=urllib.request.Request("http://127.0.0.1:8200/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=300).read().decode())['choices'][0]['message']['content'].strip()

def measure(img_path, phrase, c2w, gp):
    img=Image.open(img_path).convert('RGB')
    inp=dp(images=img,text=phrase.rstrip('.')+'.',return_tensors='pt').to('cuda')
    with torch.no_grad(): o=det(**inp)
    r=dp.post_process_grounded_object_detection(o,inp.input_ids,threshold=0.25,text_threshold=0.25,target_sizes=[img.size[::-1]])[0]
    if len(r['scores'])==0: return None
    i=int(torch.argmax(r['scores'])); box=[float(v) for v in r['boxes'][i]]
    si=spn(img,input_boxes=[[box]],return_tensors='pt').to('cuda')
    with torch.no_grad(): so=sam(**si,multimask_output=False)
    m=spn.image_processor.post_process_masks(so.pred_masks.cpu(),si['original_sizes'].cpu(),si['reshaped_input_sizes'].cpu())[0].numpy()[0,0]
    vs,us=np.nonzero(m)
    if len(us)==0: return None
    if len(us)>600:
        k=np.random.RandomState(0).choice(len(us),600,replace=False); us,vs=us[k],vs[k]
    c2w=np.asarray(c2w); org=c2w[:3,3]
    d=np.stack([(us-(W-1)/2)/FX,(vs-(H-1)/2)/FX,np.ones(len(us))],axis=1)@c2w[:3,:3].T
    d/=np.linalg.norm(d,axis=1,keepdims=True)
    loc,_,_=RI.intersects_location(np.repeat([org],len(d),axis=0),d,multiple_hits=False)
    if len(loc)==0: return None
    p=np.median(loc,axis=0)
    return dict(cam_dist=float(np.linalg.norm(p-org)), goal_dist=float(math.hypot(p[0]-gp[0],p[1]-gp[1])), score=float(r['scores'][i]))

Q1=("You are a robot standing at the destination of a navigation route in a subway station, "
    "looking straight ahead. Name ONE clearly visible object next to you that a person could use as a landmark "
    "to know where to stop. Answer with ONLY a short noun phrase (e.g. 'the white pillar'). No explanation.")
Q2="How far away is {x} from the camera, in meters? Answer with ONLY a number."

print(f"{'ep':>4s} {'Gemma 가 고른 대상':28s} {'Gemma 거리':>10s} {'실측 거리':>9s} {'차이':>7s}")
rows=[]
for ep in sorted(snaps,key=int):
    s=snaps[ep]; gp=h2o(eps[ep]['goals'][0]['position'])
    path=f"logs/goal_fast22/{s['rgb']}"
    try:
        name=ask(path,Q1,40).strip().strip('.').strip('"')
        if len(name)>40: name=name[:40]
        dtxt=ask(path,Q2.format(x=name),16)
        mnum=re.search(r'[\d]+\.?\d*', dtxt)
        gd=float(mnum.group()) if mnum else float('nan')
        mm=measure(path,name,s['c2w'],gp)
        if mm is None:
            print(f"{ep:>4s} {name[:28]:28s} {gd:10.2f} {'검출실패':>9s} {'-':>7s}"); continue
        rows.append((ep,name,gd,mm['cam_dist'],mm['goal_dist']))
        print(f"{ep:>4s} {name[:28]:28s} {gd:10.2f} {mm['cam_dist']:9.2f} {gd-mm['cam_dist']:+7.2f}")
    except Exception as e:
        print(f"{ep:>4s} 실패 {type(e).__name__} {str(e)[:40]}")
if rows:
    a=np.array([[r[2],r[3]] for r in rows if r[2]==r[2]])
    if len(a):
        err=a[:,0]-a[:,1]
        print(f"\nGemma 거리 추정 vs 메시 실측 — {len(a)}건")
        print(f"  평균 오차 {err.mean():+.2f} m · 절대 중앙 {np.median(np.abs(err)):.2f} m · 최대 {np.abs(err).max():.2f} m")
        print(f"  상관계수 {np.corrcoef(a[:,0],a[:,1])[0,1]:.3f}")
json.dump([{'ep':r[0],'name':r[1],'gemma_m':r[2],'mesh_cam_m':r[3],'mesh_goal_m':r[4]} for r in rows],
          open('logs/gemma_dist_test.json','w'), ensure_ascii=False, indent=1)
