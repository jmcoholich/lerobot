from pathlib import Path
import cv2, pickle, json
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path('/data3/extracted_data')
OUT=Path(__file__).resolve().parent
sets={'demo': sorted((ROOT/'both_in_bin_interleaved').glob('demonstration_*')), 'original': sorted((ROOT/'base_rollouts_original').glob('demonstration_*')), 'recent': sorted(ROOT.glob('demonstration_dracorex_base_*'))}
def getframe(path,idx):
 cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,img=cap.read();cap.release()
 if not ok: raise ValueError((path,idx))
 return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
def tile(frames, labels, dest,cols=3,w=480):
 h=w*360//640
 out=Image.new('RGB',(cols*w,((len(frames)+cols-1)//cols)*(h+30)),(245,245,245));draw=ImageDraw.Draw(out)
 for i,(im,label) in enumerate(zip(frames,labels)):
  x=i%cols*w;y=i//cols*(h+30)
  out.paste(im.resize((w,h)),(x,y+30));draw.text((x+5,y+8),label,fill=(0,0,0))
 out.save(dest)
# Common views; independent video timestamps, early stationary scene.
ps=[sets['demo'][0],sets['original'][1],sets['recent'][0]]
frames=[];labels=[]
for p in ps:
 for c in range(3):
  frames.append(getframe(p/f'cam_{c}_rgb_video.avi',0));labels.append(p.name.replace('demonstration_','')+f' / cam{c}')
tile(frames,labels,OUT/'initial_comparison.jpg')
# Every old/new rollout, sample elapsed camera timestamp percentiles; timestamp is acquisition time.
for group in ['original','recent']:
 for p in sets[group]:
  frames=[]; labels=[]
  for frac in np.linspace(0,1,9):
   for c in [0,1]:
    with open(p/f'cam_{c}_rgb_video.metadata','rb') as f:m=pickle.load(f)
    ts=np.asarray(m['timestamps']);elapsed=(ts-ts[0])/1000;target=frac*elapsed[-1];idx=np.abs(elapsed-target).argmin()
    frames.append(getframe(p/f'cam_{c}_rgb_video.avi',idx));labels.append(f'{p.name.replace("demonstration_","")} cam{c} t={elapsed[idx]:.1f}s')
  tile(frames,labels,OUT/(p.name+'.jpg'),cols=6,w=400)
# First and last wrist/front views of all demonstrations.
for start in range(0,50,10):
 frames=[];labels=[]
 for p in sets['demo'][start:start+10]:
  for c in [0,1]:
   cap=cv2.VideoCapture(str(p/f'cam_{c}_rgb_video.avi'));n=cap.get(cv2.CAP_PROP_FRAME_COUNT);cap.release()
   for idx in [0,int(n)-1]:
    frames.append(getframe(p/f'cam_{c}_rgb_video.avi',idx));labels.append(p.name.replace('demonstration_both_in_bin_interleaved_','')+f' cam{c} '+('start' if idx==0 else 'end'))
 tile(frames,labels,OUT/f'demos_{start:02}.jpg',cols=4,w=320)
