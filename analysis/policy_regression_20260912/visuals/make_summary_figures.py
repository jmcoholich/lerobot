from pathlib import Path
import numpy as np,cv2,h5py,pickle
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent; ROOT=Path('/data3/extracted_data'); data=np.load(OUT/'initial_frames.npz')
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
f=ImageFont.truetype(font,22);sm=ImageFont.truetype(font,18)
rows=[('demonstration_both_in_bin_interleaved_0_blue','March 23 demos (50/50)','Pink left / blue right'),('demonstration_base_prompt_no_intervention_1','May 9 original runs (10/10)','Pink left / blue right'),('demonstration_dracorex_base_0','September 11 recent runs (7/7)','Blue left / pink right')]
out=Image.new('RGB',(1920,440),'white');d=ImageDraw.Draw(out)
for i,(name,label,sub) in enumerate(rows):
 d.text((i*640+12,10),label,font=f,fill='black');d.text((i*640+12,43),sub,font=sm,fill='black');out.paste(Image.fromarray(cv2.cvtColor(data[name][1],cv2.COLOR_BGR2RGB)),(i*640,80))
out.save(OUT/'layout_comparison.png')
# Show commanded and measured site to distinguish policy target errors from tracking.
fig,ax=plt.subplots(2,1,figsize=(11,6),sharey=True,layout='constrained')
paths=[ROOT/'base_rollouts_original/demonstration_base_prompt_no_intervention_1',ROOT/'demonstration_dracorex_base_0']
for a,p,label in zip(ax,paths,['Original run 1: blue first, then pink','Recent run 0: pink first, then repeated empty grasps']):
 with h5py.File(next(p.glob('deoxys*.h5*'))) as h:
  ts=h['timestamp'][:];t=ts-ts[0];target=h['cartesian_pose_cmd'][:];eef=h['eef_pos'][:].reshape(-1,3);g=h['gripper_action'][:];w=h['gripper_state'][:]
 a.plot(t,target[:,1]*1000,label='Commanded y',alpha=.85);a.plot(t,eef[:,1]*1000,label='Measured y',lw=1)
 a.fill_between(t,-160,160,where=g>0,color='gray',alpha=.15,label='Close commanded')
 a.set(title=label,xlabel='Seconds since first logged command',ylabel='Robot y (mm)',ylim=(-160,160));a.grid(alpha=.2)
ax[0].legend(loc='upper right',ncol=3,fontsize=9)
fig.savefig(OUT/'commanded_vs_measured_sites.png',dpi=150)
plt.close(fig)
