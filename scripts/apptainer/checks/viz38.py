import json, math
import numpy as np, cv2, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
VOCAB=['pillar','column','sign','signboard','vending machine','bench','seat','stairs','escalator',
 'ticket gate','turnstile','door','glass door','station map','information board','trash can','elevator',
 'railing','kiosk','ticket machine','fire extinguisher','planter','advertisement screen']
img=Image.open('logs/goal_fast22/rgb/ep38_goal.jpg').convert('RGB')
dp=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
det=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base').to('cuda').eval()
inp=dp(images=img,text='. '.join(VOCAB)+'.',return_tensors='pt').to('cuda')
with torch.no_grad(): o=det(**inp)
r=dp.post_process_grounded_object_detection(o,inp.input_ids,threshold=0.30,text_threshold=0.30,target_sizes=[img.size[::-1]])[0]
im=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
cols=[(0,0,255),(0,165,255),(0,255,255),(0,255,0),(255,0,0),(255,0,255)]
for i,(s,l,b) in enumerate(zip(r['scores'], r.get('text_labels',r.get('labels')), r['boxes'])):
    b=[int(v) for v in b]; c=cols[i%len(cols)]
    cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),c,2)
    cv2.putText(im,f"{l} {float(s):.2f}",(b[0]+3,max(14,b[1]+16)),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,2)
    print(f"  {str(l):18s} score {float(s):.2f} box {b}")
cv2.imwrite('logs/stop_cues_fast22c/ep38_boxes_base.jpg',im)
print('저장 logs/stop_cues_fast22c/ep38_boxes_base.jpg')
