import h5py,numpy as np,pathlib,pickle,cv2,json
root=pathlib.Path('/data3/extracted_data/demonstration_dracorex_base_0')
report=[]
with h5py.File(root/'trajectories_dracorex_base_0.h5') as f:
 for cam,name in [(0,'side'),(1,'wrist'),(2,'front')]:
  meta=pickle.loads((root/f'cam_{cam}_rgb_video.metadata').read_bytes());ts=np.array(meta['timestamps'])/1000
  video=cv2.VideoCapture(str(root/f'cam_{cam}_rgb_video.avi'))
  for chunk in [0,3,6,10]:
   c=f[f'chunks/{chunk:06d}'];start=float(c['timing/started_at_unix'][()]);image=c['observations/observation.images.camera_'+name][()][0].transpose(1,2,0)*255
   if name=='front':image=image[:,140:500]
   center=np.argmin(abs(ts-start));diffs=[]
   for idx in range(max(0,center-8),min(len(ts),center+8)):
    video.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,raw=video.read()
    if not ok:continue
    raw=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
    if name=='front':raw=raw[:,140:500]
    diffs.append((float(np.abs(raw.astype(float)-image).mean()),idx))
   mae,idx=min(diffs);row=dict(camera=name,chunk=chunk,mean_abs_pixel_diff=mae,closest_frame=idx,frame_timestamp_s=float(ts[idx]),step_start_timestamp_s=start,image_minus_step_start_ms=float((ts[idx]-start)*1000));report.append(row);print(row,flush=True)
  video.release()
(pathlib.Path(__file__).resolve().parent/'camera_identity.json').write_text(json.dumps(report,indent=2))
