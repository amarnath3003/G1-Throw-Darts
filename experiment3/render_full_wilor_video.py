"""Run WiLoR on each original frame and render its complete MANO surface."""
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SRC=Path(r'C:\Users\Amarnath\Downloads\IMG_4238.mov')

def main():
    from reconstruct_hand import EDGES
    # Use the exact verified model-loading and legacy compatibility setup.
    import reconstruct_hand as runner
    import hashlib,os
    assets=json.loads((HERE/'configs/wilor_assets.json').read_text())
    original_load=torch.load
    def verified_load(path,*a,**kw):
        if Path(path).resolve()==(HERE/'models/pretrained_models/wilor_final.ckpt').resolve():
            kw.setdefault('weights_only',False);kw.setdefault('map_location','cpu');kw.setdefault('mmap',True)
        return original_load(path,*a,**kw)
    torch.load=verified_load
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    torch.set_num_threads(2)
    free,_=torch.cuda.mem_get_info()
    if free<4*1024**3:raise RuntimeError('Need 4 GiB free CUDA memory or change device to CPU.')
    print('Loading pinned WiLoR...',flush=True)
    pipe=WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'),dtype=torch.float32,wilor_pretrained_dir=str(HERE/'models'),verbose=False)
    import pickle
    with (HERE/'models/pretrained_models/MANO_RIGHT.pkl').open('rb') as f:
        faces=np.asarray(pickle.load(f,encoding='latin1')['f'],dtype=int)
    out=HERE/'data/full_wilor_video'; frames=out/'frames';frames.mkdir(parents=True,exist_ok=True)
    video_out=HERE.parent/'progress_videos';video_out.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(SRC));records=[];i=0;start=time.perf_counter()
    while True:
        ok,image=cap.read()
        if not ok:break
        preds=pipe.predict(image,hand_conf=.3)
        left=[p for p in preds if not bool(p['is_right'])]
        selected=max(left,key=lambda p:np.prod(np.asarray(p['hand_bbox'])[2:]-np.asarray(p['hand_bbox'])[:2])) if left else None
        h,w=image.shape[:2]
        if selected is not None:
            p=selected['wilor_preds'];verts=p['pred_vertices'][0];cam=p['pred_cam_t_full'][0]
            xyz=verts+cam;focal=float(p['scaled_focal_length'])
            xy=xyz[:,:2]/xyz[:,2:]*focal+[w/2,h/2]
            tris=faces; layer=image.copy()
            normals=np.cross(verts[tris[:,1]]-verts[tris[:,0]],verts[tris[:,2]]-verts[tris[:,0]])
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            shade=.35+.65*np.abs(normals@np.array([.2,-.4,.894427]))
            for fi in np.argsort(xyz[tris,2].mean(axis=1))[::-1]:
                tri=tris[fi]
                if np.min(xyz[tri,2])<=0:continue
                pts=np.rint(np.clip(xy[tri],-10000,10000)).astype(np.int32)
                cv2.fillConvexPoly(layer,pts,tuple(int(c*shade[fi]) for c in (230,205,115)),cv2.LINE_AA)
            image=cv2.addWeighted(layer,.78,image,.22,0)
            k2d=np.asarray(p['pred_keypoints_2d'])[0]
            for a,b in EDGES:
                if np.isfinite(k2d[[a,b]]).all():cv2.line(image,tuple(np.rint(k2d[a]).astype(int)),tuple(np.rint(k2d[b]).astype(int)),(30,245,60),1,cv2.LINE_AA)
            for x,y in k2d:
                if 0<=x<w and 0<=y<h:cv2.circle(image,(round(float(x)),round(float(y))),2,(10,30,255),-1)
            label='LEFT hand / full WiLoR mesh'
        else:label='LEFT hand not detected / no mesh'
        cv2.rectangle(image,(0,0),(w,32),(18,18,18),-1)
        cv2.putText(image,f'{i/59.005:.3f}s  {label}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(frames/f'frame_{i:04d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,94])
        records.append(dict(frame=i,time_s=i/59.005,detected=selected is not None))
        i+=1
        if i%30==0:print(f'{i} frames; {i/59.005:.2f}s video; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    cap.release()
    assert i==421,f'Expected 421 frames; got {i}'
    ffmpeg=subprocess.run(['ffmpeg','-y','-v','error','-framerate','84200/1427','-i',str(frames/'frame_%04d.jpg'),
        '-i',str(SRC),'-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
        '-c:a','copy','-t','7.135','-movflags','+faststart',str(video_out/'05_wilor_full_3d_hand.mp4')],capture_output=True,text=True)
    if ffmpeg.returncode:raise RuntimeError(ffmpeg.stderr)
    report=dict(source=str(SRC),model='WiLoR-mini',source_frames=i,source_duration_s=7.135,
        estimated_frame_rate=59.005,detected_frames=sum(r['detected'] for r in records),
        mesh_vertices=778,mesh_triangles=int(len(faces)),audio='copied from original',
        frames=records,elapsed_s=time.perf_counter()-start)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)

if __name__=='__main__':main()
