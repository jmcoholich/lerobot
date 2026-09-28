#!/usr/bin/env python3
"""Read-only comparison of timestamped robot logs; no hardware commands.

Run: MPLCONFIGDIR=/tmp/mpl-policy-regression /home/jeremiah/miniforge3/envs/openteach/bin/python analysis/policy_regression_20260912/dynamics/analyze_dynamics.py
Logs are sampled just before sending each command; tracking error is therefore
not a measurement of final accuracy after applying that command. Unmatched
trajectories and contact conditions prevent causal system identification.
"""
from pathlib import Path
import csv
import datetime
import json
import pickle
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data3/extracted_data')
OUT=Path(__file__).resolve().parent
PATTERNS={'demos':'both_in_bin_interleaved/*/deoxys*',
          'original':'base_rollouts_original/*/deoxys*',
          'recent':'demonstration_dracorex_base_*/deoxys*'}

def scalar(x):
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,bytes): return x.decode()
    return x

def summary(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    if len(a)==0:return {}
    return dict(zip(['min','p05','median','p95','p99','max'],map(float,np.quantile(a,[0,.05,.5,.95,.99,1]))),n=len(a),mean=float(a.mean()))

def put_stats(row,key,a):
    row.update({key+'_'+k:v for k,v in summary(a).items()})

def write_csv(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

class MetadataUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise pickle.UnpicklingError('Only primitive metadata is allowed')

runs=[];events=[];gaps=[];cameras=[];all_data={};attrs={};pooled={}
for group,pattern in PATTERNS.items():
    pooled[group]={}
    for p in sorted(ROOT.glob(pattern)):
        with h5py.File(p,'r') as f:
            d={k:v[:] for k,v in f.items() if isinstance(v,h5py.Dataset)}
            attr={k:scalar(v) for k,v in f.attrs.items()}
        name=p.parent.name
        all_data[name]=d;attrs[name]=attr
        t=d['timestamp'];dt=np.diff(t);pos=d['eef_pos'].reshape(-1,3);cmd=d['cartesian_pose_cmd'][:,:3]
        action=d['arm_action'];q=d['gripper_state'];ga=d['gripper_action'];close=ga>=0
        for meta in sorted(p.parent.glob('*.metadata')):
            with meta.open('rb') as mf:
                metadata=MetadataUnpickler(mf).load()
            ts=np.asarray(metadata['timestamps'],dtype=float)/1000.0
            cr={'group':group,'run':name,'camera':meta.name.split('_rgb')[0],
                'n_frames':len(ts),'camera_duration_s':float(ts[-1]-ts[0]),
                'camera_start_minus_first_command_s':float(ts[0]-t[0]),
                'camera_end_minus_last_command_s':float(ts[-1]-t[-1]),
                'camera_effective_hz':float((len(ts)-1)/(ts[-1]-ts[0])),
                'record_frequency_metadata':metadata['record_frequency'],
                'video_codec':metadata.get('video_codec','not_recorded')}
            put_stats(cr,'frame_dt_s',np.diff(ts))
            cameras.append(cr)
        err=cmd-pos;disp=np.diff(pos,axis=0);valid=(dt>.025)&(dt<.080)
        speed=np.linalg.norm(disp,axis=1)/dt
        cerr=np.linalg.norm(err,axis=1)
        step=np.linalg.norm(np.diff(cmd,axis=0),axis=1)
        actnorm=np.linalg.norm(action[:,:3],axis=1)
        # Error relative to simultaneous desired eef field: note OSC firmware may
        # leave this field at the reset pose, so report variation as a caveat.
        row={'group':group,'run':name,'file':str(p),'n':len(t),
             'start_utc':datetime.datetime.fromtimestamp(t[0],datetime.timezone.utc).isoformat(),
             'duration_s':float(t[-1]-t[0]),'effective_hz':float((len(t)-1)/(t[-1]-t[0])),
             'gap_gt_100ms':int((dt>.1).sum()),'gap_gt_300ms':int((dt>.3).sum()),
             'gap_gt_800ms':int((dt>.8).sum()),
             'extra_time_above_50ms_s':float(np.maximum(dt-.05,0).sum()),
             'start_x_m':pos[0,0],'start_y_m':pos[0,1],'start_z_m':pos[0,2],
             'eef_path_length_m':float(np.linalg.norm(disp,axis=1).sum()),
             'translation_clipping_fraction':float((actnorm>=.09999).mean()),
             'rotation_clipping_fraction':float((np.linalg.norm(action[:,3:6],axis=1)>=.19999).mean()),
             'grip_close_fraction':float(close.mean()),
             'grip_negative_gt_1_magnitude_fraction':float((ga< -1).mean()),
             'done_values':str(np.unique(d.get('done',[])).tolist())}
        if group in ('original','recent'):
            horizon=100 if group=='original' else 75
            ii=np.arange(1,len(t)); boundary=ii%horizon==0
            row['inferred_execution_horizon']=horizon
            put_stats(row,'chunk_boundary_dt_s',dt[boundary])
            put_stats(row,'within_chunk_dt_s',dt[(~boundary)&(ii>1)])
            pooled[group].setdefault('chunk_boundary_dt_s',[]).extend(dt[boundary].tolist())
            pooled[group].setdefault('within_chunk_dt_s',[]).extend(dt[(~boundary)&(ii>1)].tolist())
        for key,a in [('dt_s',dt),('normal_dt_s',dt[valid]),('eef_speed_m_s',speed[valid]),
                      ('command_error_m',cerr),('next_sample_command_error_m',np.linalg.norm(cmd[:-1]-pos[1:],axis=1)[valid]),
                      ('command_step_m',step),('gripper_width_m',q),('gripper_action',ga)]:
            put_stats(row,key,a);pooled[group].setdefault(key,[]).extend(np.asarray(a).tolist())
        rcmd=Rotation.from_quat(d['cartesian_pose_cmd'][:,3:7]).as_matrix()
        rstate=d['eef_pose'][:,:3,:3]
        angle=Rotation.from_matrix(rcmd@np.swapaxes(rstate,1,2)).magnitude()
        put_stats(row,'command_orientation_error_rad',angle)
        pooled[group].setdefault('command_orientation_error_rad',[]).extend(angle.tolist())
        if 'last_tau_ext_hat_filtered' in d:
            ext=np.linalg.norm(d['last_tau_ext_hat_filtered'],axis=1)
            put_stats(row,'tau_ext_norm',ext);pooled[group].setdefault('tau_ext_norm',[]).extend(ext.tolist())
            row['desired_eef_position_range_m']=float(np.max(np.ptp(d['last_eef_pose_d'][:,:3,3],axis=0)))
            row['tool_transform_unique']=len(np.unique(d['last_F_T_EE'].reshape(-1,16),axis=0))
        # Crude first-order response gain: projected displacement / command
        # error, normal intervals, away from tabletop, error5-50mm. This is a
        # descriptive diagnostic, not causal system ID.
        moving=valid&(cerr[:-1]>.005)&(cerr[:-1]<.05)&(pos[:-1,2]>.14)
        gain=np.sum(err[:-1]*disp,axis=1)/(np.sum(err[:-1]**2,axis=1)+1e-20)
        put_stats(row,'response_projection_away_from_table',gain[moving])
        pooled[group].setdefault('response_projection_away_from_table',[]).extend(gain[moving].tolist())
        for idx in np.flatnonzero(dt>.1):
            gaps.append({'group':group,'run':name,'before_command_index':int(idx+1),'time_s':float(t[idx]-t[0]),'dt_s':dt[idx],
                         'gripper_changed':bool(close[idx]!=close[idx+1]),'previous_gripper_changed':bool(idx>0 and close[idx]!=close[idx-1]),
                         'index_mod_75':int((idx+1)%75),'index_mod_100':int((idx+1)%100)})
        # Consecutive closed-command segments; plateau is all available records
        # >=0.5s after close. This distinguishes object-width grasps from fully
        # closed fingers but does not label task success.
        starts=np.flatnonzero(close&~np.r_[False,close[:-1]])
        stops=np.flatnonzero(close&~np.r_[close[1:],False])+1
        for ci,(a,b) in enumerate(zip(starts,stops)):
            settle=np.arange(a,b)[t[a:b]-t[a]>=.5]
            lower=np.arange(a,b)[q[a:b]<.06]
            width=float(np.median(q[settle])) if len(settle) else None
            outcome='insufficient_hold'
            if width is not None:
                outcome='object_width' if .03<width<.065 else 'fully_closed' if width<.01 else 'other_width'
            events.append({'group':group,'run':name,'event':ci,'index':int(a),'end_index':int(b),
                           'time_s':float(t[a]-t[0]),'duration_s':float(t[min(b,len(t)-1)]-t[a]),
                           'x_m':pos[a,0],'y_m':pos[a,1],'z_m':pos[a,2],
                           'target_x_m':cmd[a,0],'target_y_m':cmd[a,1],'target_z_m':cmd[a,2],
                           'time_to_width_under_60mm_s':float(t[lower[0]]-t[a]) if len(lower) else None,
                           'settled_width_m':width,'min_width_m':float(q[a:b].min()),'width_proxy':outcome,
                           'gripper_cmd_at_close':float(ga[a]),'closed_samples':int(b-a)})
        runs.append(row)

write_csv('runs.csv',runs);write_csv('grasp_events.csv',events);write_csv('command_gaps.csv',gaps);write_csv('camera_timing.csv',cameras)
pooled_stats={g:{k:summary(a) for k,a in vals.items()} for g,vals in pooled.items()}
aggregate={g:{'runs':sum(r['group']==g for r in runs),'samples':sum(r['n'] for r in runs if r['group']==g),
              'stats':pooled_stats[g],
              'grasp_event_counts':{kind:sum(e['group']==g and e['width_proxy']==kind for e in events)
                                    for kind in ['object_width','fully_closed','other_width','insufficient_hold']},
              'sustained_grasp_event_counts_minimum_2s':{kind:sum(e['group']==g and e['width_proxy']==kind and e['duration_s']>=2 for e in events)
                                    for kind in ['object_width','fully_closed','other_width','insufficient_hold']}}
           for g in PATTERNS}
(OUT/'aggregate.json').write_text(json.dumps(aggregate,indent=2,default=scalar))
(OUT/'controller_metadata.json').write_text(json.dumps(attrs,indent=2,default=scalar))
site_rows=[]
for r in runs:
    if r['group']=='demos':continue
    ev=[e for e in events if e['run']==r['run'] and e['duration_s']>=2]
    sr={'group':r['group'],'run':r['run']}
    for n,e in enumerate(ev,1):
        for k in ('time_s','y_m','target_y_m','z_m','target_z_m','settled_width_m','width_proxy'):
            sr[f'grasp_{n}_{k}']=e[k]
    site_rows.append(sr)
write_csv('sustained_grasp_sites.csv',site_rows)
print(json.dumps(aggregate,indent=2,default=scalar))

# Comparable per-run plots, all original and recent runs, common y scales.
for group in ('original','recent'):
    subset=[r for r in runs if r['group']==group]
    fig,axes=plt.subplots(len(subset),3,figsize=(15,2.0*len(subset)),squeeze=False)
    for row,ax in zip(subset,axes):
        d=all_data[row['run']];t=d['timestamp']-d['timestamp'][0];pos=d['eef_pos'].reshape(-1,3);cmd=d['cartesian_pose_cmd'][:,:3]
        ax[0].plot(t,pos[:,2],label='actual z');ax[0].plot(t,cmd[:,2],alpha=.65,label='command z');ax[0].set_ylim(.04,.36)
        ax[0].set_ylabel(row['run'].rsplit('_',1)[-1]+'\nz (m)');ax[0].legend(fontsize=7,loc='upper right')
        ax[1].plot(t,1000*np.linalg.norm(cmd-pos,axis=1));ax[1].set_ylim(0,100);ax[1].set_ylabel('cmd - state (mm)')
        ax[2].plot(t,d['gripper_state']*1000,label='width');ax[2].plot(t,(d['gripper_action']>=0)*80,alpha=.5,label='close cmd');ax[2].set_ylim(-5,85);ax[2].set_ylabel('width (mm)');ax[2].legend(fontsize=7)
        for a in ax:a.grid(alpha=.2);a.set_xlim(0,60)
    for a in axes[-1]:a.set_xlabel('seconds since first robot command')
    fig.suptitle(group+' rollouts: pre-command state and commanded target');fig.tight_layout();fig.savefig(OUT/(group+'_tracking.png'),dpi=140);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4))
groups=list(PATTERNS);colors=['#777777','#228833','#cc3311']
for g,c in zip(groups,colors):
    dt=np.array(pooled[g]['dt_s']);x=np.sort(dt*1000);y=np.arange(1,len(x)+1)/len(x)
    axes[0].plot(x,y,label=g,color=c)
    ev=[e for e in events if e['group']==g and e['settled_width_m'] is not None]
    axes[1].hist([e['settled_width_m']*1000 for e in ev],bins=np.arange(0,86,2),histtype='step',label=g,color=c)
    axes[2].hist(np.array(pooled[g]['command_error_m'])*1000,bins=np.arange(0,121,2),density=True,histtype='step',label=g,color=c)
axes[0].set_xscale('log');axes[0].set_xlabel('robot command interval (ms)');axes[0].set_ylabel('cumulative sample fraction');axes[0].set_xlim(20,1200)
axes[1].set_xlabel('median settled closed gripper width (mm)');axes[1].set_ylabel('closed segments')
axes[2].set_xlabel('simultaneous cmd - state error (mm)');axes[2].set_ylabel('density')
for a in axes:a.legend();a.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'group_comparison.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,group in zip(axes,['original','recent']):
    rr=[r for r in runs if r['group']==group]
    for ri,r in enumerate(rr):
        ev=[e for e in events if e['run']==r['run'] and e['duration_s']>=2]
        yy=[e['target_y_m']*1000 for e in ev]
        xx=np.arange(1,len(ev)+1)
        ax.plot(xx,yy,color='gray',alpha=.45)
        for x,y,e in zip(xx,yy,ev):
            ax.scatter(x,y,marker='o' if e['width_proxy']=='object_width' else 'x',
                       color='green' if e['width_proxy']=='object_width' else 'red',s=45)
        ax.annotate(r['run'].rsplit('_',1)[-1],(xx[-1]+.035,yy[-1]),fontsize=9)
    ax.set_title(group);ax.set_xticks([1,2,3,4]);ax.set_xlabel('sustained grasp number (hold >=2s)');ax.grid(alpha=.2)
    ax.axhline(0,color='black',alpha=.2)
axes[0].set_ylabel('commanded robot y at close onset (mm)')
fig.suptitle('Original runs switch pickup sides; recent runs repeatedly command the same side\ngreen circle: object-width gripper; red cross: fully closed fingers')
fig.tight_layout();fig.savefig(OUT/'grasp_site_comparison.png',dpi=170);plt.close(fig)
