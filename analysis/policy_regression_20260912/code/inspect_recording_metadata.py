"""Read-only trajectory provenance and timing audit. Run from the LeRobot checkout."""
from pathlib import Path
import hashlib,json,struct
import h5py,numpy as np
ROOT=Path('/home/jeremiah/lerobot')
DEST=Path(__file__).parent
DTYPES={'F32':'<f4','F64':'<f8','I64':'<i8','I32':'<i4'}
def tensors(data):
 size=struct.unpack('<Q',data[:8])[0]
 header=json.loads(data[8:8+size])
 return {name:np.frombuffer(data[8+size+x['data_offsets'][0]:8+size+x['data_offsets'][1]],DTYPES[x['dtype']]).reshape(x['shape']) for name,x in header.items() if name!='__metadata__'}
rows=[]
for path in sorted(Path('/data3/extracted_data').glob('demonstration_dracorex_base_*/trajectories*.h5')):
 with h5py.File(path) as f:
  meta=json.loads(f.attrs['metadata_json']);checkpoint=Path(meta['checkpoint_file'])
  row={'log':str(path),'checkpoint_sha256_recorded':meta['checkpoint_sha256'],'policy_config':meta['policy_config'],'interventions':meta['interventions'],'camera_input_changes':{},'processor_checks':{},'timings':[],'all_executed_actions_unperturbed':True,'source_matches_current':{}}
  for key,data in f['artifacts/source'].items():
   matches=[p for p in ROOT.rglob(key) if '/analysis/' not in str(p)]
   row['source_matches_current'][key]=any(p.read_bytes()==data[()].tobytes() for p in matches)
  for name in ['preprocessor','postprocessor']:
   artifacts=f['artifacts/'+name]
   key=next(k for k in artifacts if k.endswith('.safetensors'))
   a=tensors(artifacts[key][()].tobytes());b=tensors((checkpoint.parent/key).read_bytes())
   row['processor_checks'][name]={'json_semantic_equal':json.loads(artifacts['processor.json'][()].tobytes())==json.loads((checkpoint.parent/('policy_'+name+'.json')).read_bytes()),'state_action_values_and_shapes_equal':all(np.array_equal(a[k],b[k]) for k in a if k.startswith(('action.','observation.state.')) and not k.endswith('.count')),'all_tensor_values_equal_flattened':all(np.array_equal(a[k].reshape(-1),b[k].reshape(-1)) for k in a)}
  chunks=list(f['chunks'].values())
  for c in chunks:
   n=int(c.attrs['execution_horizon'])
   row['all_executed_actions_unperturbed'] &= c.attrs['execution_source']=='original_sample_0' and np.array_equal(c['queued_actions'][()],c['original_actions'][()][0:1,:n])
   row['timings'].append({k:float(v[()]) for k,v in c['timing'].items()})
  for key in chunks[0]['observations']:
   if 'images' not in key:continue
   x=[c['observations'][key][()] for c in chunks]
   row['camera_input_changes'][key]={'identical_consecutive_chunks':sum(np.array_equal(a,b) for a,b in zip(x,x[1:])),'mean_absolute_consecutive_differences':[float(np.mean(np.abs(a-b))) for a,b in zip(x,x[1:])]}
  rows.append(row)
sha=hashlib.sha256()
with checkpoint.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
result={'checkpoint_sha256_current':sha.hexdigest(),'recordings':rows}
(DEST/'recording_metadata.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'recordings':len(rows),'checkpoint_sha256_current':sha.hexdigest(),'all_checkpoint_hashes_match':all(r['checkpoint_sha256_recorded']==sha.hexdigest() for r in rows),'all_queues_unperturbed':all(r['all_executed_actions_unperturbed'] for r in rows)},indent=2))
