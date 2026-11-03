"""Experiment 1: manually specified human-inspired swing, torque tracking, free dart.

No policy training, object attachment, launch impulse, or in-flight state edits.
The pelvis remains fixed to the existing visible support. Hip counter-motion is
joint motion only, not a demonstration of free-standing weight transfer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicHermiteSpline

from inspire_grip import InspireGrip
from inspire_model import NAMES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment1'
ARM = ['left_'+n+'_joint' for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw',
                                   'elbow','wrist_roll','wrist_pitch','wrist_yaw')]


def fit_arm(s, grip, pitch, seed, waist, elbow_height=1.25,up_weight=.2):
    """Offline joint editing by IK; nothing in this function runs during flight."""
    m,d=s.model,s.data
    ids=np.array([m.joint(n).id for n in ARM]);adr=m.jnt_qposadr[ids]
    d.qpos[m.joint('waist_yaw_joint').qposadr[0]]=waist[0]
    d.qpos[m.joint('waist_pitch_joint').qposadr[0]]=waist[1]
    aim=np.array([np.cos(pitch),-.08,np.sin(pitch)]);aim/=np.linalg.norm(aim)
    def residual(q):
        d.qpos[adr]=q;mujoco.mj_forward(m,d)
        r=d.xmat[s.hand].reshape(3,3)
        p=d.xpos[s.hand]+r@np.asarray(s.config['dart_pos_hand'])
        elbow=d.xpos[m.body('left_elbow_link').id]
        return np.r_[8*(p-grip), 1.8*(-r[:,2]-aim),
                     up_weight*(r[:,0]-[0,0,1]), .3*(elbow[2]-elbow_height), .025*(q-seed)]
    low=m.jnt_range[ids,0]+.04;high=m.jnt_range[ids,1]-.04
    low[4:]=[-1.2,-.95,-.95];high[4:]=[1.2,.95,.95]
    result=least_squares(residual,np.clip(seed,low,high),bounds=(low,high),max_nfev=300)
    return result.x


def make_plan(s, duration=.20, release_fraction=.42, legacy_first=False,human=False,wrist_snap=0.):
    seed=np.array([s.config['arm_pose'][n] for n in ARM])
    # Aim, draw the hand toward the cheek, accelerate through, then follow through.
    targets=[([.46,.19,1.35],.17,[0,0]),
             ([.28,.21,1.51],.35,[-.12,-.035]),
             ([.69,.17,1.53],-.07,[.13,.055])]
    if human:
        targets=[targets[0],([.27,.20,1.32],.35,[-.10,-.025]),
                 ([.73,.14,1.40],.25,[.10,.045])]
    poses=[]
    for p,pitch,w in targets:
        seed=fit_arm(s,p,pitch,seed,w,1.07 if human else 1.25,.6 if human else .2);poses.append(seed.copy())
    if not legacy_first:poses[0]=np.array([s.config['arm_pose'][n] for n in ARM])
    start=2.10;end=start+duration
    times=np.array([0.,.65,1.65,start,end,end+.40,end+1.5])
    q0=s.data.qpos.copy()
    # Restore the original reset before constructing the full references.
    s.reset();q0=s.data.qpos.copy();rows=np.tile(q0,(len(times),1))
    for i,pose in enumerate([poses[0],poses[0],poses[1],poses[1],poses[2],poses[2],poses[2]]):
        for n,val in zip(ARM,pose):rows[i,s.model.joint(n).qposadr[0]]=val
        w=targets[0 if i<2 else 1 if i<4 else 2][2]
        for name,val in [('waist_yaw_joint',w[0]),('waist_pitch_joint',w[1]),
                         ('left_hip_pitch_joint',-w[1]*.3),('right_hip_pitch_joint',-w[1]*.3),
                         ('left_hip_yaw_joint',-w[0]*.15),('right_hip_yaw_joint',-w[0]*.15)]:
            rows[i,s.model.joint(name).qposadr[0]]=val
    # Smooth start/stop, fastest near the center of the forward stroke.
    slopes=np.zeros_like(rows)
    if human:
        # Edit successive poses along a nearly straight dart stroke. Joint-space
        # endpoint interpolation alone produces a curved, twisting hand path.
        dense_times=list(times[:4]);dense_rows=list(rows[:4]);seed=poses[1].copy()
        for u in np.linspace(0,1,10)[1:]:
            blend=u*u*(3-2*u)
            p=(1-blend)*np.asarray(targets[1][0])+blend*np.asarray(targets[2][0])
            pitch=(1-blend)*targets[1][1]+blend*targets[2][1]
            w=(1-blend)*np.asarray(targets[1][2])+blend*np.asarray(targets[2][2])
            seed=fit_arm(s,p,pitch,seed,w,1.07,.6)
            row=rows[4].copy()
            for n,v in zip(ARM,seed):row[s.model.joint(n).qposadr[0]]=v
            for n,v in [('waist_yaw_joint',w[0]),('waist_pitch_joint',w[1]),
                        ('left_hip_pitch_joint',-w[1]*.3),('right_hip_pitch_joint',-w[1]*.3),
                        ('left_hip_yaw_joint',-w[0]*.15),('right_hip_yaw_joint',-w[0]*.15)]:
                row[s.model.joint(n).qposadr[0]]=v
            dense_times.append(start+u*duration);dense_rows.append(row)
        dense_times.extend(times[-2:]);dense_rows.extend([dense_rows[-1].copy(),dense_rows[-1].copy()])
        times=np.asarray(dense_times);rows=np.asarray(dense_rows);slopes=np.zeros_like(rows)
        for i in range(4,len(times)-3):slopes[i]=(rows[i+1]-rows[i-1])/(times[i+1]-times[i-1])
    if wrist_snap:
        # Deliberate human-like uncocking: elbow leads, then wrist flexion pushes
        # the barrel forward. Keep wrist roll/yaw steady instead of twirling it.
        pitch_adr=s.model.joint('left_wrist_pitch_joint').qposadr[0]
        roll_adr=s.model.joint('left_wrist_roll_joint').qposadr[0]
        yaw_adr=s.model.joint('left_wrist_yaw_joint').qposadr[0]
        for i,t in enumerate(times):
            if i<2:continue
            u=np.clip((t-(start+.025))/.105,0,1)
            rows[i,pitch_adr]=-.35+abs(wrist_snap)*u*u*(3-2*u)
            if wrist_snap>0:
                rows[i,roll_adr]=poses[1][4]
                rows[i,yaw_adr]=poses[1][6]
        slopes[:]=0
        for i in range(4,len(times)-3):slopes[i]=(rows[i+1]-rows[i-1])/(times[i+1]-times[i-1])
    # Cubic zero-velocity endpoints, acceleration and physical tracking measured below.
    spline=CubicHermiteSpline(times,rows,slopes)
    # Dense offline IK changes MjData; restore the actual initial grasp before stepping.
    s.reset()
    return dict(start=start,end=end,release_command=start+duration*release_fraction,
                duration=duration,release_fraction=release_fraction,times=times.tolist(),
                joint_names=ARM,arm_keyframes=[q.tolist() for q in poses],
                human_refinement=human,full_qpos_keyframes=rows.tolist()),spline


def make_simple_plan(s):
    """Direct joint editing: no IK branches, no pronation or wrist-yaw swing."""
    m=s.model;s.reset();initial=s.data.qpos.copy()
    arm0=np.array([s.config['arm_pose'][n] for n in ARM])
    back=arm0.copy();back[[0,3,5]]=[-.99,-.48,-.60]
    finish=arm0.copy();finish[[0,3,5]]=[-1.35,.45,-.25]
    start,end=2.1,2.32
    times=np.unique(np.r_[0.,.65,1.6,start,np.linspace(start,end,12),2.125,2.205,2.7,3.8])
    rows=np.tile(initial,(len(times),1));slopes=np.zeros_like(rows)
    def blend(t,a,b):
        u=np.clip((t-a)/(b-a),0,1)
        return u*u*(3-2*u),6*u*(1-u)/(b-a)
    for i,t in enumerate(times):
        if t<1.6:
            h,_=blend(t,.65,1.6);q=arm0+(back-arm0)*h;v=np.zeros(7)
        else:
            h,dh=blend(t,start,end);q=back+(finish-back)*h;v=(finish-back)*dh
            wh,dwh=blend(t,2.125,2.205);q[5]=back[5]+(finish[5]-back[5])*wh;v[5]=(finish[5]-back[5])*dwh
        for n,value,speed in zip(ARM,q,v):
            a=m.joint(n).qposadr[0];rows[i,a]=value;slopes[i,a]=speed
        # Small trunk and hip contributions; pelvis remains on its fixture.
        h,dh=blend(t,start,end)
        if t<1.6:load,_=blend(t,.65,1.6)
        else:load=1.
        for n,lo,hi in [('waist_yaw_joint',-.025,.035),('waist_pitch_joint',-.01,.025),
                        ('left_hip_pitch_joint',.003,-.0075),('right_hip_pitch_joint',.003,-.0075)]:
            a=m.joint(n).qposadr[0];rows[i,a]=lo*load+(hi-lo)*h;slopes[i,a]=(hi-lo)*dh
    curve=CubicHermiteSpline(times,rows,slopes);s.reset()
    return {'start':start,'end':end,'release_command':2.13,'duration':end-start,
            'release_fraction':(2.13-start)/(end-start),'times':times.tolist(),
            'joint_names':ARM,'arm_keyframes':[arm0.tolist(),back.tolist(),finish.tolist()],
            'full_qpos_keyframes':rows.tolist(),'direct_joint_edit':True,
            'wrist_roll_yaw_constant':True},curve


def run(duration=.20,release_fraction=.42,keep_states=False,legacy_first=False,human=False,wrist_snap=0.,arm_lead=0.,yaw_bias=0.,grip_scale=1.,simple=False):
    s=InspireGrip();m,d=s.model,s.data
    if grip_scale!=1.:
        s.config['force_targets']=(np.asarray(s.config['force_targets'])*grip_scale).tolist()
        s.config.update(force_gain=.08,kp=6.,kd=.15,squeeze=.035)
    if human:
        # Idle right-hand thumb at zero intersects its palm. Start from the
        # mirrored collision-tested grasp, rather than exciting that collision.
        for n,v in zip(('R_'+n[2:] for n in NAMES),s.config['q_grasp']):
            s.config['arm_pose'][n]=v
        s.reset()
    if yaw_bias:
        s.config['arm_pose']['waist_yaw_joint']=yaw_bias;s.reset()
    plan,curve=make_simple_plan(s) if simple else make_plan(s,duration,release_fraction,legacy_first,human,wrist_snap)
    plan['wrist_snap_amplitude_rad']=wrist_snap
    plan['arm_lead_s']=arm_lead;plan['waist_yaw_bias_rad']=yaw_bias
    plan['grip_controller']={k:s.config[k] for k in ('force_targets','force_gain','kp','kd','squeeze')}
    lead_q=np.array([m.joint(n).qposadr[0] for n in ARM[:4]+['waist_yaw_joint','waist_pitch_joint']])
    waist_q=int(m.joint('waist_yaw_joint').qposadr[0])
    # More responsive tracking while retaining the model's existing motor ceilings.
    body_names=[m.joint(int(m.actuator_trnid[a,0])).name for a in s.ra]
    for i,n in enumerate(body_names):
        if n in ARM:s.kp[i]=220 if 'wrist' not in n else 65;s.kd[i]=12 if 'wrist' not in n else 2.2
    qadr=m.jnt_qposadr;vadr=m.jnt_dofadr
    hinge=np.where(m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)[0]
    trace=[];states=[];release=None;impact=None;command=None
    jac=np.zeros((3,m.nv));rotjac=np.zeros_like(jac)
    max_torque=np.zeros(m.nu);max_speed=np.zeros(m.nv);max_limit=0.;pre_release_drift=0.
    lost_since=None;recontact=False;first_collision=None
    for tick in range(round(5./m.opt.timestep)):
        t=float(d.time);qt=curve(min(t,plan['times'][-1]));vt=curve(min(t,plan['times'][-1]),1);at=curve(min(t,plan['times'][-1]),2)
        if arm_lead:
            lead_time=min(t+arm_lead,plan['times'][-1])
            qt[lead_q]=curve(lead_time)[lead_q];vt[lead_q]=curve(lead_time,1)[lead_q];at[lead_q]=curve(lead_time,2)[lead_q]
        qt[waist_q]+=yaw_bias
        s.target[:]=qt;s.velocity_target=np.zeros(m.nv);s.acceleration_target=np.zeros(m.nv)
        s.velocity_target[vadr[hinge]]=vt[qadr[hinge]]
        s.acceleration_target[vadr[hinge]]=at[qadr[hinge]]
        if t>=plan['release_command'] and s.closed:
            s.closed=False;command={'time_s':t,'com':d.xipos[s.dart].tolist()}
        s.step()
        max_torque=np.maximum(max_torque,np.abs(d.ctrl));max_speed=np.maximum(max_speed,np.abs(d.qvel))
        limits=m.jnt_range[hinge];q=d.qpos[qadr[hinge]]
        max_limit=max(max_limit,float(np.max(np.maximum(limits[:,0]-q,q-limits[:,1]))))
        if tick%4==0:
            row=s.sample();mujoco.mj_jacBodyCom(m,d,jac,rotjac,s.dart)
            row['world_contacts']=[]
            for c in d.contact:
                b1,b2=m.geom_bodyid[[c.geom1,c.geom2]]
                if s.dart in (b1,b2) and 0 in (b1,b2):
                    row['world_contacts'].append(m.geom(c.geom2 if b1==s.dart else c.geom1).name)
            velocity=jac@d.qvel;row['velocity_m_s']=velocity.tolist()
            row['tip']=d.site_xpos[m.site('spare_dart_1_tip').id].tolist()
            row['dart_axis']=d.xmat[s.dart].reshape(3,3)[:,0].tolist()
            if s.closed:pre_release_drift=max(pre_release_drift,row['relative_drift_m'])
            contact=sum(row['normal_N'])>.01
            # Require sustained clearance, then report the first no-contact state.
            if not s.closed and not contact:
                if lost_since is None:lost_since=row.copy()
                if release is None and row['time_s']-lost_since['time_s']>=.015:release=lost_since
            else:
                if release is not None and contact:recontact=True
                lost_since=None
            if release is not None and row['world_contacts'] and impact is None:
                impact=row.copy()
            if s.closed and row['other_contacts'] and first_collision is None:first_collision=row.copy()
            trace.append(row)
        if keep_states and tick%round(1/(120*m.opt.timestep))==0:
            states.append((float(d.time),d.qpos.copy(),d.qvel.copy()))
    result={'manual_animation':True,'supported_pelvis':True,'plan':plan,'release_command':command,
            'actual_release':release,'first_impact':impact,'pre_release_unwanted_contact':first_collision,
            'finger_recontact_after_release':recontact,'max_closed_grip_drift_m':pre_release_drift,
            'max_joint_limit_violation_rad':max_limit,'warnings':[int(w.number) for w in d.warning],
            'finite':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            'final_com':d.xipos[s.dart].tolist(),
            'max_motor_torque_Nm':{m.actuator(i).name:float(v) for i,v in enumerate(max_torque)},
            'peak_joint_speed_rad_s':{m.joint(j).name:float(max_speed[vadr[j]]) for j in hinge}}
    if release:
        result['release_speed_m_s']=float(np.linalg.norm(release['velocity_m_s']))
        result['release_forward_speed_m_s']=release['velocity_m_s'][0]
    return s,result,trace,states


def render(s,result,trace,states,folder,slow=False):
    """Replay recorded physics states; the trail is a visual measurement overlay."""
    from PIL import Image,ImageDraw,ImageFont
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/'swing.mp4';m,d=s.model,s.data
    renderer=mujoco.Renderer(m,680,1100);camera=mujoco.MjvCamera()
    option=mujoco.MjvOption();option.geomgroup[3]=0
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',23)
    small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    pipe=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1440x820','-r','30','-i','-','-an','-c:v','libx264','-preset','veryfast',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    saved_times=np.array([v[0] for v in states]);row_times=np.array([v['time_s'] for v in trace])
    movie_times=list(np.arange(0,5,1/30))
    if slow:movie_times+=list(np.arange(1.7,3.6,1/120))
    try:
        for frame,t in enumerate(movie_times):
            i=int(np.argmin(abs(saved_times-t)));d.qpos[:]=states[i][1];d.qvel[:]=states[i][2];d.time=states[i][0]
            mujoco.mj_forward(m,d)
            camera.lookat[:]=[1.35,0,1.05];camera.distance=4.15;camera.azimuth=90;camera.elevation=-10
            if (frame>=150 or result['plan'].get('direct_joint_edit')) and t<2.5:
                camera.lookat[:]=[.40,.13,1.22];camera.distance=1.35;camera.azimuth=90;camera.elevation=-7
            renderer.update_scene(d,camera,scene_option=option)
            # A thin line traces actual COM history. It applies no force and hides no geometry.
            history=[row['com'] for row in trace if result['release_command']['time_s']<=row['time_s']<=t]
            if history:
                history=np.asarray(history)[::5]
                for a,b in zip(history[:-1],history[1:]):
                    if renderer.scene.ngeom>=renderer.scene.maxgeom:break
                    g=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1.,.48,.10,1.]))
                    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.002,a,b);renderer.scene.ngeom+=1
            image=Image.new('RGB',(1440,820),'#101923');image.paste(Image.fromarray(renderer.render()),(0,80))
            draw=ImageDraw.Draw(image);row=trace[int(np.argmin(abs(row_times-t)))]
            phase='AIM' if t<.65 else 'DRAW BACK' if t<1.65 else 'SET' if t<2.1 else 'FORWARD STROKE'
            if 13<=int(folder.name[:2])<=18:phase='DEBUG: INVALID INITIAL GRASP'
            if t>=result['release_command']['time_s']:phase='OPENING FINGERS'
            if result['actual_release'] and t>=result['actual_release']['time_s']:phase='FREE FLIGHT'
            if result['first_impact'] and t>=result['first_impact']['time_s']:phase='IMPACT / LANDING'
            draw.text((22,16),'EXPERIMENT 1 / LEFT-HAND DART THROW',font=font,fill='white')
            draw.text((22,48),folder.name+' | '+('SLOW REPLAY 0.25x' if frame>=150 else 'REAL TIME 1x'),font=small,fill='#91dccc')
            x=1120;draw.text((x,96),phase,font=small,fill='#ffbd78')
            wp=m.joint('left_wrist_pitch_joint')
            lines=[f"Simulation: {t:.3f} s",f"Stroke: {result['plan']['duration']:.2f} s",
                   f"Open command: {result['release_command']['time_s']:.3f} s",'',
                   f"Wrist pitch: {np.degrees(d.qpos[wp.qposadr[0]]):+.1f} deg",
                   f"Wrist speed: {d.qvel[wp.dofadr[0]]:+.1f} rad/s",'',
                   'Current dart velocity:',*[f'{a}: {v:+.2f} m/s' for a,v in zip('XYZ',row['velocity_m_s'])],
                   '',f"Dart height: {row['com'][2]:.3f} m",'',
                   'Contact normals / N',*[f'{a}: {v:.2f}' for a,v in zip(('Thumb','Index','Middle'),row['normal_N'])],
                   '', 'Fixed pelvis support', 'Torque-driven joints', 'No dart launch impulse']
            for k,line in enumerate(lines):draw.text((x,140+k*24),line,font=small,fill='#d6e0e8')
            if result.get('release_speed_m_s') is not None:
                draw.text((22,775),f"Measured release {result['release_speed_m_s']:.2f} m/s | orange trail = actual dart COM | target board at X = 3.0 m",font=small,fill='white')
            pipe.stdin.write(np.asarray(image).tobytes())
            if frame in (60,69,90,204,210):image.save(folder/f'frame_{frame:03d}.png')
        pipe.stdin.close();code=pipe.wait()
        if code:raise RuntimeError('FFmpeg encoding failed')
    finally:
        renderer.close()
        if pipe.poll() is None:pipe.terminate();pipe.wait()
    return output


def save_trial(s,r,trace,states,folder,slow=False):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'result.json').write_text(json.dumps(r,indent=2))
    (folder/'trace.json').write_text(json.dumps(trace,separators=(',',':')))
    np.savez_compressed(folder/'states.npz',time=[v[0] for v in states],qpos=[v[1] for v in states],qvel=[v[2] for v in states])
    video=render(s,r,trace,states,folder,slow)
    print('Saved '+str(video),flush=True)


def candidate(parameters):
    duration,fraction=parameters
    _,r,_,_=run(duration,fraction)
    rel=r.get('actual_release');impact=r.get('first_impact')
    return {'duration':duration,'release_fraction':fraction,
            'speed':r.get('release_speed_m_s'),
            'velocity':rel['velocity_m_s'] if rel else None,
            'impact':impact['com'] if impact else None,
            'drift':r['max_closed_grip_drift_m'],'limit':r['max_joint_limit_violation_rad'],
            'unwanted':r['pre_release_unwanted_contact'] is not None,
            'warnings':r['warnings']}


def record_candidate(parameters):
    index,duration,fraction,legacy,*variant=parameters
    human=bool(variant and variant[0])
    snap=float(variant[1]) if len(variant)>1 else 0.
    lead=float(variant[2]) if len(variant)>2 else 0.
    yaw=float(variant[3]) if len(variant)>3 else 0.
    grip_scale=float(variant[4]) if len(variant)>4 else 1.
    s,r,trace,states=run(duration,fraction,True,legacy,human,snap,lead,yaw,grip_scale)
    folder=OUT/'trials'/f'{index:02d}_{"snap_" if snap else "human_" if human else ""}stroke_{duration:.2f}_release_{fraction:+.2f}'
    if folder.exists():folder=folder.with_name(folder.name+'_rerun_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    save_trial(s,r,trace,states,folder)
    return {'id':folder.name,'video':str((folder/'swing.mp4').relative_to(ROOT)),
            'velocity':r['actual_release']['velocity_m_s'] if r['actual_release'] else None,
            'speed':r.get('release_speed_m_s'),'impact':r['first_impact']['com'] if r['first_impact'] else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=float,default=.20)
    parser.add_argument('--release-fraction',type=float,default=.42)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--sweep',action='store_true')
    parser.add_argument('--record-all',action='store_true')
    parser.add_argument('--human-sweep',action='store_true')
    parser.add_argument('--refined',action='store_true')
    parser.add_argument('--snap-sweep',action='store_true')
    parser.add_argument('--aligned-snap',action='store_true')
    parser.add_argument('--snap-timing',action='store_true')
    parser.add_argument('--sequence-sweep',action='store_true')
    parser.add_argument('--grip-sequence',action='store_true')
    parser.add_argument('--anticipate-release',action='store_true')
    parser.add_argument('--simple',action='store_true')
    parser.add_argument('--replay',type=Path,help='Render archived states without running another swing')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.simple:
        s,r,trace,states=run(keep_states=True,human=True,simple=True)
        folder=OUT/'trials/42_simple_elbow_wrist'
        if folder.exists():folder=folder.with_name(folder.name+'_rerun_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
        save_trial(s,r,trace,states,folder,True)
        print(json.dumps({k:r[k] for k in ('release_speed_m_s','actual_release','finger_recontact_after_release','first_impact') if k in r},indent=2));return
    if args.anticipate_release:
        params=[(39+i,.30,f,False,True,-.90,.03,.12,1.6) for i,f in enumerate((-.05,0.,.05))]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'anticipate_release.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));return
    if args.grip_sequence:
        params=[(35+i,.30,f,False,True,-.90,lead,.12,1.6) for i,(lead,f) in enumerate((lead,f) for lead in (.03,.05) for f in (.10,.15))]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'grip_sequence.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));return
    if args.sequence_sweep:
        params=[(31+i,.30,f,False,True,-.90,lead,.12) for i,(lead,f) in enumerate((lead,f) for lead in (.05,.09) for f in (.15,.20))]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'sequence_sweep.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));return
    if args.snap_timing:
        params=[(29,.30,.13,False,True,-.90),(30,.30,.15,False,True,-.90)]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'snap_timing.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));return
    if args.aligned_snap:
        params=[(27,.30,.10,False,True,-.90),(28,.30,.20,False,True,-.90)]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'aligned_snap.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));return
    if args.snap_sweep:
        params=[(23+i,.30,f,False,True,a) for i,(a,f) in enumerate((a,f) for a in (.90,1.15) for f in (.10,.18))]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'snap_sweep.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));return
    if args.replay:
        folder=args.replay;s=InspireGrip();r=json.loads((folder/'result.json').read_text());trace=json.loads((folder/'trace.json').read_text())
        saved=np.load(folder/'states.npz');states=list(zip(saved['time'],saved['qpos'],saved['qvel']))
        target=OUT/(folder.name[:2]+'_featured_'+('wrist_snap' if r['plan'].get('wrist_snap_amplitude_rad') else 'replay'))
        render(s,r,trace,states,target,True);return
    if args.refined:
        params=[(19+i,d,f,False,True) for i,(d,f) in enumerate(((.18,.05),(.24,.05),(.24,.20),(.32,.20)))]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'refined_sweep.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));return
    if args.human_sweep:
        params=[(13+i,d,f,False,True) for i,(d,f) in enumerate((d,f) for d in (.18,.24,.32) for f in (.0,.15))]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'human_sweep.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));return
    if args.record_all:
        params=[(0,.20,.42,True)]+[(i+1,d,f,False) for i,(d,f) in enumerate((d,f) for d in (.14,.20,.28) for f in (-.15,.05,.25,.45))]
        with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(record_candidate,params))
        (OUT/'trials/index.json').write_text(json.dumps(results,indent=2));return
    if args.sweep:
        parameters=[(i,d,f,False) for i,(d,f) in enumerate((d,f) for d in (.14,.20,.28) for f in (-.15,.05,.25,.45))]
        with ProcessPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(record_candidate,parameters))
        (OUT/'sweep.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2));return
    s,result,trace,states=run(args.duration,args.release_fraction,True)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    (OUT/'trace.json').write_text(json.dumps(trace,separators=(',',':')))
    print(json.dumps({k:result[k] for k in ('release_speed_m_s','release_forward_speed_m_s','max_closed_grip_drift_m','max_joint_limit_violation_rad','warnings','final_com') if k in result},indent=2),flush=True)
    folder=OUT/'trials'/('99_manual_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    save_trial(s,result,trace,states,folder,args.render)


if __name__=='__main__':main()
