from pathlib import Path
import cv2,json
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/data3/extracted_data');OUT=Path(__file__).resolve().parent
sets={'demo':sorted((ROOT/'both_in_bin_interleaved').glob('demonstration_*')),'original':sorted((ROOT/'base_rollouts_original').glob('demonstration_*')),'recent':sorted(ROOT.glob('demonstration_dracorex_base_*'))}
frames={}; rows=[]
for group,ps in sets.items():
 for p in ps:
  ims=[]
  for c in range(3):
   cap=cv2.VideoCapture(str(p/f'cam_{c}_rgb_video.avi'));ok,im=cap.read();cap.release()
   if not ok:raise ValueError(p)
   ims.append(im)
  frames[p.name]=ims
  hsv=cv2.cvtColor(ims[1],cv2.COLOR_BGR2HSV)
  r={'group':group,'run':p.name}
  for color,lo,hi in [('blue',(85,70,35),(125,255,255)),('pink',(130,50,35),(179,255,255))]:
   mask=cv2.inRange(hsv,np.array(lo),np.array(hi));mask[230:]=0
   n,l,stats,cent=cv2.connectedComponentsWithStats(mask)
   ix=1+np.argmax(stats[1:,4]);r[color+'_center']=cent[ix].tolist();r[color+'_area']=int(stats[ix,4]);r[color+'_bbox']=stats[ix,:4].tolist()
  r['blue_left']=r['blue_center'][0]<r['pink_center'][0]
  rows.append(r)
np.savez_compressed(OUT/'initial_frames.npz',**{k:np.array(v) for k,v in frames.items()})
(OUT/'initial_scene_metrics.json').write_text(json.dumps(rows,indent=2))
# All initial wrist views to evaluate task layout coverage; sorting naturally by path preserves reproducibility.
for group,ps in sets.items():
 w,h=320,180;cols=5;canvas=Image.new('RGB',(w*cols,((len(ps)+cols-1)//cols)*(h+25)),'white');draw=ImageDraw.Draw(canvas)
 for i,p in enumerate(ps):
  x=i%cols*w;y=i//cols*(h+25);im=Image.fromarray(cv2.cvtColor(frames[p.name][1],cv2.COLOR_BGR2RGB));canvas.paste(im.resize((w,h)),(x,y+25));draw.text((x+4,y+6),p.name.replace('demonstration_','').replace('both_in_bin_interleaved_','demo_'),fill='black')
 canvas.save(OUT/f'{group}_initial_wrist.jpg')
for group in sets:
 rr=[r for r in rows if r['group']==group]
 print(group,'n',len(rr),'blue_left',sum(r['blue_left'] for r in rr))
 for col in ['blue','pink']:
  pts=np.array([r[col+'_center'] for r in rr]);print(col,'median',np.median(pts,axis=0),'range',pts.min(axis=0),pts.max(axis=0))
