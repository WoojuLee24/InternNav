"""① Gemma 가 사진에서 물체 이름을 뽑고  ② GroundingDINO 가 그 이름을 아는지 시험한다."""
import json, base64, urllib.request, re, collections, glob
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def ask(path, prompt, mx=200):
    img=base64.b64encode(open(path,'rb').read()).decode()
    body={"model":"gemma-4-31B-it","max_tokens":mx,"temperature":0.0,
      "messages":[{"role":"user","content":[{"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img}"}}]}]}
    r=urllib.request.Request("http://127.0.0.1:8200/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=300).read().decode())['choices'][0]['message']['content'].strip()

Q=("List the distinct physical objects visible in this subway station image that could serve as a landmark "
   "for telling someone where to stop. Use short common English noun phrases (1-3 words). "
   "Answer with ONLY a comma-separated list, at most 10 items. No explanation.")

imgs=sorted(glob.glob('logs/goal_fast22/rgb/*.jpg'))
print(f'사진 {len(imgs)} 장에서 Gemma 가 이름 뽑기\n')
cnt=collections.Counter(); per_img={}
for p in imgs:
    try:
        t=ask(p,Q)
        names=[re.sub(r'^(a|an|the)\s+','',x.strip().strip('.').lower()) for x in t.split(',')]
        names=[n for n in names if 0<len(n)<=28]
        per_img[p]=names
        for n in names: cnt[n]+=1
    except Exception as e:
        print(f'  {p} 실패 {type(e).__name__}')
print(f'뽑힌 이름 {len(cnt)} 종류 · 상위:')
for n,c in cnt.most_common(28): print(f'   {c:3d}  {n}')
json.dump({'count':dict(cnt),'per_img':per_img}, open('/tmp/vocab_from_gemma.json','w'), ensure_ascii=False, indent=1)
