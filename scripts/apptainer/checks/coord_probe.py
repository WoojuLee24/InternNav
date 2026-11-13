"""Gemma 가 어떤 좌표계로 답하는지 — 화면 밖으로 나간 ep 들로 기준 해상도를 추정한다."""
import json, base64, urllib.request, re
def ask(path,prompt,mx=24):
    img=base64.b64encode(open(path,'rb').read()).decode()
    body={"model":"gemma-4-31B-it","max_tokens":mx,"temperature":0.0,
      "messages":[{"role":"user","content":[{"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img}"}}]}]}
    r=urllib.request.Request("http://127.0.0.1:8200/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=300).read().decode())['choices'][0]['message']['content'].strip()

p='logs/goal_fast22/rgb/ep38_goal.jpg'   # 640x480, 기둥이 화면 왼쪽 절반 (u 약 0~330)
qs=[
 ("A 640x480 명시",      "The image is 640x480 pixels. Point at the center of the large white and brown pillar. Answer with ONLY: x y"),
 ("B 해상도 안 알려줌",   "Point at the center of the large white and brown pillar. Answer with ONLY two integers: x y"),
 ("C 0-1000 정규화",     "Point at the center of the large white and brown pillar. Answer with ONLY two integers in 0-1000 normalized image coordinates: x y"),
 ("D 비율(0-1)",         "Point at the center of the large white and brown pillar. Answer with ONLY two decimals between 0 and 1 (fraction of width and height): x y"),
 ("E 네 좌표계는?",       "Point at the center of the large white and brown pillar, and state the coordinate system you used. Format: x y (range: WxH)"),
]
for n,q in qs:
    try: print(f"[{n}] {ask(p,q,48)!r}")
    except Exception as e: print(f"[{n}] 실패 {e}")
