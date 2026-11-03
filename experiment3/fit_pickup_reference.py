"""Uncertainty-aware hand/pen reference and full-mesh review; no robot control."""
import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from PIL import Image
from pickup_common import read_json,write_json,sha256,HERE
from pen_tracking import track_pen

PARENTS=np.array([0,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])


def fixed_bone_lengths(joints,lengths):
    result=np.zeros_like(joints)
    for k in range(1,21):
        direction=joints[k]-joints[PARENTS[k]]
        result[k]=result[PARENTS[k]]+direction/max(np.linalg.norm(direction),1e-10)*lengths[k]
    return result


def subject_shape(dataset,root,pass_name):
    lengths=[]
    for clip in dataset['clips']:
        if clip['role']=='holdout':continue
        report=read_json(root/clip['id']/pass_name/'report.json')
        for r in report['frames'][::10]:
            if r['valid'] and r['phase']!='acquisition' and r.get('usable_reference'):
                with np.load(root/clip['id']/pass_name/f'{r["frame_index"]:06d}.npz') as data:
                    j=data['joints'];lengths.append(np.linalg.norm(j-j[PARENTS],axis=1))
    if not lengths:raise ValueError('No accepted training/grip frames to estimate consistent hand shape')
    return np.median(lengths,axis=0)


def palm_frame(joints):
    x=joints[9]-joints[0];x/=max(np.linalg.norm(x),1e-10)
    z=joints[17]-joints[5];z-=x*np.dot(x,z);z/=max(np.linalg.norm(z),1e-10)
    y=np.cross(z,x)
    return np.column_stack([x,y,z])


def smooth_joints(times,joints,phases,valid):
    result=np.full_like(joints,np.nan)
    for i in np.flatnonzero(valid):
        # Never bridge an unobserved interval or smooth between semantic phases.
        ids=np.flatnonzero(valid&(abs(times-times[i])<=.1)&(phases==phases[i]))
        ids=ids[np.abs(ids-i)<=6]
        weights=np.exp(-.5*((times[ids]-times[i])/.04)**2)
        result[i]=np.sum(joints[ids]*weights[:,None,None],axis=0)/weights.sum()
    return result


def fit_camera(rows,width,height):
    hs=[np.array(r['table']['board_to_image']) for r in rows[::30] if r['table']['valid']]
    def objective(logf):
        f=np.exp(logf);inv=np.linalg.inv([[f,0,width/2],[0,f,height/2],[0,0,1]])
        residual=[]
        for H in hs:
            a,b=(inv@H)[:,:2].T
            residual.extend([np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),
                             (np.linalg.norm(a)-np.linalg.norm(b))/(np.linalg.norm(a)+np.linalg.norm(b))])
        return float(np.mean(np.square(residual)))
    bounds=np.log([width*.5,width*3]);fit=minimize_scalar(objective,bounds=bounds,method='bounded')
    focal=float(np.exp(fit.x))
    return dict(focal_px=focal,principal_point=[width/2,height/2],distortion=[0,0,0,0,0],
                objective=float(fit.fun),intrinsics_measured=False,metric_supervision_allowed=False,
                weakly_identified=True,assumptions='square pixels, centered principal point, zero lens distortion, planar board',
                sensitivity_factors=[.8,1,1.2],note='Effective camera only. A low image fit error does not verify depth.')


def approximate_root(joints,points,H,width,height,focal):
    K=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]])
    rays=(points-[width/2,height/2])/focal
    A=np.zeros((42,3));b=np.empty(42)
    A[::2,0]=1;A[1::2,1]=1;A[::2,2]=-rays[:,0];A[1::2,2]=-rays[:,1]
    b[::2]=rays[:,0]*joints[:,2]-joints[:,0];b[1::2]=rays[:,1]*joints[:,2]-joints[:,1]
    translation=np.linalg.lstsq(A,b,rcond=None)[0]
    B=np.linalg.inv(K)@H;scale=2/(np.linalg.norm(B[:,0])+np.linalg.norm(B[:,1]))
    a,b=B[:,0]*scale,B[:,1]*scale
    U,_,Vt=np.linalg.svd(np.column_stack([a,b,np.cross(a,b)]));R=U@Vt
    if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
    origin=B[:,2]*scale
    return R.T@(joints[0]+translation-origin),R


def mesh_overlay(image,raw,record,faces):
    canvas=cv2.resize(image,(540,960));layer=canvas.copy();verts=raw['vertices'];xyz=verts+raw['translation']
    uv=xyz[:,:2]/xyz[:,2:]*record['assumed_focal_px']+np.array([image.shape[1]/2,image.shape[0]/2])
    uv*=np.array([540/image.shape[1],960/image.shape[0]])
    normal=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    shade=.4+.6*abs(normal@np.array([.2,-.4,.8944]))
    for face in np.argsort(xyz[faces,2].mean(1))[::-1]:
        triangle=faces[face]
        if xyz[triangle,2].min()<=0:continue
        pixels=np.int32(np.clip(uv[triangle],-10000,10000))
        cv2.fillConvexPoly(layer,pixels,tuple(int(c*shade[face]) for c in [230,180,75]),cv2.LINE_AA)
    return cv2.addWeighted(layer,.7,canvas,.3,0)


def make_overlay(image,row,record,raw,pen,faces,clip_id):
    left=cv2.resize(image,(540,960));right=left.copy()
    if record and record['valid']:right=mesh_overlay(image,raw,record,faces)
    if pen['valid']:
        pts=np.int32(np.array(pen['endpoints_px'])*.5)
        color=(0,180,255) if pen['hidden_span_inferred'] else (40,240,40)
        cv2.line(right,tuple(pts[0]),tuple(pts[1]),color,3)
        for point in pts:cv2.circle(right,tuple(point),5,color,-1)
    frame=np.hstack([left,right]);cv2.rectangle(frame,(0,0),(1080,98),(20,20,20),-1)
    texts=[f'{clip_id} | source {row["time_s"]:.3f}s | {row["phase"]} | ORIGINAL (left), ESTIMATE (right)',
           'HaMeR LEFT surface; hidden fingers and contact remain inferred; depth is uncalibrated',
           ('Pen: '+pen['confidence']) if pen['valid'] else 'Pen: unresolved; no synthetic trajectory filled in',
           'Hand: '+('finite prediction' if record and record['valid'] else 'no accepted hand prediction')]
    for i,line in enumerate(texts):cv2.putText(frame,line,(10,21+i*23),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--pass-name',default='hamer_v2');p.add_argument('--clip');p.add_argument('--skip-video',action='store_true')
    a=p.parse_args();dataset=read_json(a.dataset/'manifest.json');faces=np.load(a.dataset/'mano_faces_left.npy')
    out=a.dataset/'milestone2';out.mkdir(exist_ok=True);summaries=[]
    bone_lengths=subject_shape(dataset,a.dataset,a.pass_name)
    write_json(out/'subject_hand_shape.json',dict(model_bone_lengths_m=bone_lengths.tolist(),
               source_roles=['train','grip_reference'],holdout_used=False,human_dimensions_measured=False))
    for entry in dataset['clips']:
        if a.clip and entry['id']!=a.clip:continue
        clip=read_json(a.dataset/entry['manifest']);rows=clip['frames'];times=np.array([r['time_s'] for r in rows]);phases=np.array([r['phase'] for r in rows])
        predictions=read_json(a.dataset/entry['id']/a.pass_name/'report.json');by_index={r['frame_index']:r for r in predictions['frames']}
        cd=out/entry['id'];cd.mkdir(exist_ok=True);joints=np.full((len(rows),21,3),np.nan);valid=np.zeros(len(rows),bool)
        usable=np.zeros(len(rows),bool);pen_records=[];raw_records={};previous=None;previous_time=None
        camera=fit_camera(rows,1080,1920);roots=np.full((len(rows),3),np.nan);height_sensitivity=np.full((len(rows),3),np.nan)
        palm_board=np.full((len(rows),3,3),np.nan);rest_lengths=[]
        for i,row in enumerate(rows):
            image=cv2.imread(str(a.dataset/row['file']));rec=by_index.get(row['frame_index']);raw=None
            if rec and rec['valid']:
                raw_path=a.dataset/entry['id']/a.pass_name/f'{row["frame_index"]:06d}.npz'
                with np.load(raw_path,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
                joints[i]=raw['joints'];valid[i]=True;usable[i]=rec.get('usable_reference',False)
                if row['table']['valid']:
                    for k,factor in enumerate(camera['sensitivity_factors']):
                        root,R=approximate_root(raw['joints'],raw['projected_joints'],np.array(row['table']['board_to_image']),1080,1920,camera['focal_px']*factor)
                        height_sensitivity[i,k]=root[2]
                        if k==1:roots[i]=root;palm_board[i]=R.T@palm_frame(raw['joints'])
                raw_records[i]=raw
            if previous_time is not None and row['time_s']-previous_time>.2:previous=None
            pen=track_pen(image,rec.get('bbox') if rec and rec['valid'] else None,previous,row['table'] if row['table']['valid'] else None)
            if pen['valid']:
                previous=np.array(pen['endpoints_px']);previous_time=row['time_s']
                if row['phase']=='rest' and not pen['hidden_span_inferred']:rest_lengths.append(pen['projected_table_length_m'])
            pen_records.append(dict(frame_index=row['frame_index'],time_s=row['time_s'],phase=row['phase'],**pen))
        # Keep separate arrays for model outputs, filtered data and uncertain roots.
        relative=joints-joints[:,0:1,:];consistent=np.full_like(relative,np.nan)
        for i in np.flatnonzero(valid):consistent[i]=fixed_bone_lengths(relative[i],bone_lengths)
        smoothed=smooth_joints(times,consistent,phases,valid)
        for i in np.flatnonzero(valid):smoothed[i]=fixed_bone_lengths(smoothed[i],bone_lengths)
        palm=np.full((len(rows),3,3),np.nan);pinch=np.full((len(rows),4),np.nan)
        for i in np.flatnonzero(valid):
            palm[i]=palm_frame(smoothed[i]);pinch[i]=np.linalg.norm(smoothed[i,[8,12,16,20]]-smoothed[i,4],axis=1)
        confidence=np.where(valid,np.where(phases=='acquisition',.2,np.where(usable,.7,.25)),0.)
        temporal_jump=np.zeros(len(rows),bool)
        for i in range(1,len(rows)):
            if valid[i] and valid[i-1] and times[i]-times[i-1]<.1:
                temporal_jump[i]=bool(np.max(np.linalg.norm(consistent[i]-consistent[i-1],axis=1))>.03)
        confidence[temporal_jump]=np.minimum(confidence[temporal_jump],.1)
        np.savez_compressed(cd/'reference.npz',timestamps_s=times,source_frame_indices=[r['frame_index'] for r in rows],
              phases=phases,hand_model_valid=valid,usable_2d_reference=usable,joints_camera_raw=joints,
              joints_wrist_relative_raw=relative,joints_wrist_relative_smoothed=smoothed,palm_rotation_camera=palm,
              thumb_to_fingertip_distances_model_m=pinch,reference_weights=confidence,
              fixed_model_bone_lengths_m=bone_lengths,temporal_jump_flags=temporal_jump,
              wrist_board_approximate_m=roots,palm_rotation_board_approximate=palm_board,
              wrist_height_focal_sensitivity_m=height_sensitivity,metric_depth_valid=np.zeros(len(rows),bool),
              pen_endpoints_px=np.array([r.get('endpoints_px',[[np.nan,np.nan]]*2) for r in pen_records]),
              pen_line_valid=[r['valid'] for r in pen_records])
        write_json(cd/'pen_tracks.json',pen_records);write_json(cd/'effective_camera.json',camera)
        # Save per-phase contact sheets for qualitative comparison, never a metric accuracy claim.
        selected=[]
        for phase in ['approach','acquisition','lift','hold','presentation']:
            ids=np.flatnonzero(phases==phase)
            if len(ids):selected.extend(ids[np.unique(np.linspace(0,len(ids)-1,3,dtype=int))].tolist())
        contact=Image.new('RGB',(1080,480*((len(selected)+1)//2)),'#202020')
        for k,i in enumerate(selected):
            row=rows[i];im=cv2.imread(str(a.dataset/row['file']));rec=by_index.get(row['frame_index'])
            panel=make_overlay(im,row,rec,raw_records.get(i),pen_records[i],faces,entry['id'])
            panel=cv2.resize(panel,(540,480));contact.paste(Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)),((k%2)*540,(k//2)*480))
        contact.save(cd/'contact_sheet.jpg',quality=92)
        if not a.skip_video:
            video=cd/'reconstruction.mp4'
            encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1080x960','-r','15','-i','-',
                  '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
            last=None
            for t in np.arange(times[0],times[-1]+1/30,1/15):
                i=int(np.argmin(abs(times-t)))
                if i!=last:
                    row=rows[i];im=cv2.imread(str(a.dataset/row['file']));rec=by_index.get(row['frame_index'])
                    panel=make_overlay(im,row,rec,raw_records.get(i),pen_records[i],faces,entry['id']);last=i
                encoder.stdin.write(panel.tobytes())
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Video rendering failed')
        phase_stats={phase:dict(frames=int(sum(phases==phase)),hand_predictions=int(sum(valid&(phases==phase))),
                    pen_lines=sum(r['valid'] for r in pen_records if r['phase']==phase)) for phase in np.unique(phases)}
        summary=dict(clip=entry['id'],role=entry['role'],source_sha256=clip['source_sha256'],frames=len(rows),
                     hand_predictions=int(valid.sum()),usable_2d=int(usable.sum()),pen_lines=sum(r['valid'] for r in pen_records),
                     all_selected_frames_inferred=predictions['complete'],phase_statistics=phase_stats,
                     resting_pen_length_m=float(np.median(rest_lengths)) if rest_lengths else None,
                     camera_calibrated=False,contact_ground_truth=False,robot_ready=False,temporal_jumps=int(temporal_jump.sum()),
                     root_height_range_m=[float(np.nanmin(roots[:,2])),float(np.nanmax(roots[:,2]))] if valid.any() else None)
        write_json(cd/'report.json',summary);summaries.append(summary)
        print(entry['id'],summary,flush=True)
    write_json(out/'report.json',dict(clips=summaries,status='awaiting_visual_review',
                provenance=dict(hand='HaMeR left mesh, no extra reflection',pen='image-supported blue-white line fragments; occluded spans inferred',
                                depth='unmeasured effective camera; no metric depth supervision',smoothing='40ms Gaussian inside same phase; missing frames remain NaN',
                                weights='heuristic reference weights, not calibrated probabilities'),
                code_hashes={n:sha256(HERE/n) for n in ['fit_pickup_reference.py','pen_tracking.py','reconstruct_pickup.py']},
                next_gate='Review complete acquisition/lift/grip meshes and failure intervals before any robot retargeting'))


if __name__=='__main__':main()
