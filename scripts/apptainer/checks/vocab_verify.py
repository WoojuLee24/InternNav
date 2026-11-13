"""Gemma 가 뽑은 이름을 GroundingDINO 가 실제로 찾아내는지 — 이름별 검출률."""
import json, collections
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
MID='IDEA-Research/grounding-dino-base'
dp=AutoProcessor.from_pretrained(MID); det=AutoModelForZeroShotObjectDetection.from_pretrained(MID).to('cuda').eval()
D=json.load(open('/tmp/vocab_from_gemma.json'))
cnt=collections.Counter(D['count']); per=D['per_img']

# Gemma 가 "이 사진에 있다" 고 한 이름만, 그 사진에서 검출되는지 본다 (2 회 이상 등장한 이름)
names=[n for n,c in cnt.items() if c>=1]
res={n:{'said':0,'found':0,'scores':[]} for n in names}
for path, lst in per.items():
    img=Image.open(path).convert('RGB')
    for n in set(lst):
        if n not in res: continue
        res[n]['said']+=1
        inp=dp(images=img,text=n.rstrip('.')+'.',return_tensors='pt').to('cuda')
        with torch.no_grad(): o=det(**inp)
        r=dp.post_process_grounded_object_detection(o,inp.input_ids,threshold=0.25,text_threshold=0.25,target_sizes=[img.size[::-1]])[0]
        if len(r['scores']):
            res[n]['found']+=1; res[n]['scores'].append(float(max(r['scores'])))
rows=[]
for n,v in res.items():
    if v['said']==0: continue
    rate=v['found']/v['said']
    sc=float(np.mean(v['scores'])) if v['scores'] else 0.0
    rows.append((n,v['said'],v['found'],rate,sc))
rows.sort(key=lambda r:(-r[1],-r[3]))
print(f"{'이름':26s} {'Gemma가말함':>10s} {'DINO검출':>8s} {'검출률':>7s} {'평균score':>9s}")
for n,s,f,rate,sc in rows[:40]:
    mark='OK ' if (rate>=0.8 and sc>=0.35) else ('약함' if rate>=0.5 else '안됨')
    print(f"{n[:26]:26s} {s:10d} {f:8d} {rate:6.0%} {sc:9.2f}  {mark}")
ok=[r for r in rows if r[3]>=0.8 and r[4]>=0.35]
weak=[r for r in rows if 0.5<=r[3]<0.8 or (r[3]>=0.8 and r[4]<0.35)]
bad=[r for r in rows if r[3]<0.5]
print(f"\n이름 {len(rows)} 종류 중  확실 {len(ok)} · 약함 {len(weak)} · 안됨 {len(bad)}")
print("안되는 이름 예:", ', '.join(r[0] for r in bad[:12]))
json.dump({'rows':rows}, open('/tmp/vocab_verify.json','w'), ensure_ascii=False, indent=1)
