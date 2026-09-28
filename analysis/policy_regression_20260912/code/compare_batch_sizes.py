import os,sys,json,time,pathlib,tempfile,struct
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path[:0]=['/home/jeremiah/lerobot/src','/home/jeremiah/openteach']
import h5py,numpy as np,torch,draccus
from scipy.spatial.transform import Rotation
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modelling_pi05_taco import PI05PolicyTaco
from safetensors.numpy import load_file
paths=sorted(pathlib.Path('/data3/extracted_data').glob('demonstration_dracorex_base_*/trajectories*.h5'))
with h5py.File(paths[0]) as f:m=json.loads(f.attrs['metadata_json'])
torch.backends.cuda.matmul.allow_tf32=m['matmul_allow_tf32']
torch.backends.cudnn.allow_tf32=m['cudnn_allow_tf32']
config=draccus.decode(PI05Config,m['policy_config'])
policy=PI05PolicyTaco.from_pretrained(str(pathlib.Path(m['checkpoint_file']).parent),config=config,local_files_only=True)
policy.eval()
stats=load_file(str(pathlib.Path(m['checkpoint_file']).parent/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
lo,hi=stats['action.min'],stats['action.max']
report=[]
@torch.inference_mode()
def run(c,num):
 x=c['sampling/000000/inputs'];t=lambda d:torch.from_numpy(d[()]).cuda()
 args=[[t(x['images'][k]) for k in sorted(x['images'],key=int)],[t(x['image_masks'][k]) for k in sorted(x['image_masks'],key=int)],t(x['tokens']),t(x['masks'])]
 torch.cuda.synchronize();start=time.perf_counter()
 y=policy.model.sample_actions(*args,noise=t(x['noise'])[:num],num_samples=num,guidance_actions=None,guidance_scale=None,consistency_guidance=None)
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 return y.cpu().numpy(),elapsed
for p in paths:
 with h5py.File(p) as f:
  for idx in [0,1,3,6,10]:
   c=f[f'chunks/{idx:06d}']
   b15,t15=run(c,15);b1,t1=run(c,1)
   orig=c['sampling/000000/full_output'][()]
   aa=(b15[0,:75,:8]+1)*.5*(hi-lo)+lo;bb=(b1[0,:75,:8]+1)*.5*(hi-lo)+lo
   xyz=np.linalg.norm(aa[:,:3]-bb[:,:3],axis=-1)
   ori=(Rotation.from_quat(aa[:,3:7]).inv()*Rotation.from_quat(bb[:,3:7])).magnitude()
   row=dict(gripper_threshold=0.0,demo=p.parent.name,chunk=idx,saved_replay_bitwise=bool(np.array_equal(b15,orig)),saved_replay_maxerr=float(np.abs(b15-orig).max()),batch15_s=t15,batch1_s=t1,xyz_mean_m=float(xyz.mean()),xyz_max_m=float(xyz.max()),quat_angle_mean_deg=float(np.rad2deg(ori).mean()),quat_angle_max_deg=float(np.rad2deg(ori).max()),gripper_threshold_disagreements=int(np.count_nonzero((aa[:,7]>=0)!=(bb[:,7]>=0))),gripper_max_abs=float(np.abs(aa[:,7]-bb[:,7]).max()))
   report.append(row);print(json.dumps(row),flush=True)
(pathlib.Path(__file__).resolve().parent/'batch_comparison.json').write_text(json.dumps(report,indent=2))
