"""Render HaMeR's 778-vertex MANO surface on every source frame.

Uses WiLoR's hand boxes and explicitly marks them LEFT, following user-confirmed
anatomical handedness. The models are independent; boxes only seed the crop.
"""
import os
from pathlib import Path
import sys
import json
import subprocess
import time
import cv2
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
HAMER=HERE/'vendor/hamer'
sys.path.insert(0,str(HAMER))
DATA=HERE/'models/hamer/_DATA'
os.environ['YOLO_CONFIG_DIR']=str(HERE/'data/ultralytics')
os.environ['PYOPENGL_PLATFORM']='egl'
SRC=Path(r'C:\Users\Amarnath\Downloads\IMG_4238.mov')

def main():
    # Preserve NumPy 2 compatibility for MANO's legacy chumpy import.
    for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
        if name not in np.__dict__:setattr(np,name,value)
    import pytorch_lightning as pl
    from hamer.configs import get_config
    from hamer.models.hamer import HAMER
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.datasets.utils import expand_to_aspect_ratio
    from hamer.utils.renderer import cam_crop_to_full
    # Load only the official HaMeR checkpoint with CPU mmap (PyTorch 2.6).
    checkpoint=DATA/'hamer_ckpts/checkpoints/hamer.ckpt'
    original_load=torch.load
    def checkpoint_load(path,*a,**kw):
        if Path(path).resolve()==checkpoint.resolve():
            kw.setdefault('weights_only',False);kw.setdefault('map_location','cpu');kw.setdefault('mmap',True)
        return original_load(path,*a,**kw)
    torch.load=checkpoint_load
    cfg=get_config(str(DATA/'hamer_ckpts/model_config.yaml'),update_cachedir=False)
    cfg.defrost()
    cfg.MANO.MODEL_PATH=str(DATA/'data/mano')
    cfg.MANO.MEAN_PARAMS=str(DATA/'data/mano_mean_params.npz')
    if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS'):cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
    if cfg.MODEL.BACKBONE.TYPE=='vit':cfg.MODEL.BBOX_SHAPE=[192,256]
    cfg.freeze()
    print('Loading pinned HaMeR checkpoint...',flush=True)
    model=HAMER.load_from_checkpoint(str(checkpoint),strict=False,cfg=cfg,init_renderer=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda' and torch.cuda.mem_get_info()[0]<3*1024**3:raise RuntimeError('Need 3 GiB free GPU memory.')
    model=model.to(device).eval()
    faces=model.mano.faces.cpu().numpy()
    det=json.loads((HERE/'data/wilor_pilot/report.json').read_text())
    proposals=[]
    for r in det['frames']:
        if r['detections']:
            d=max(r['detections'],key=lambda d:np.prod(np.asarray(d['bbox'])[2:]-np.asarray(d['bbox'])[:2]))
            proposals.append((r['frame_index'],np.asarray(d['bbox'],dtype=np.float32)))
    def frame_box(index):
        before=[p for p in proposals if p[0]<=index];after=[p for p in proposals if p[0]>=index]
        if not before or not after:return None
        a,b=before[-1],after[0]
        if b[0]-a[0]>36:return None
        u=0 if b[0]==a[0] else (index-a[0])/(b[0]-a[0])
        return ((1-u)*a[1]+u*b[1]).reshape(1,4)
    out=HERE/'data/full_hamer_video';frames=out/'frames';frames.mkdir(parents=True,exist_ok=True)
    video_out=HERE.parent/'progress_videos';video_out.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(SRC));records=[];i=0;start=time.perf_counter()
    while True:
        ok,image=cap.read()
        if not ok:break
        h,w=image.shape[:2]; base=image.copy(); bbox=frame_box(i)
        if bbox is not None:
            batch=next(iter(torch.utils.data.DataLoader(ViTDetDataset(cfg,image,bbox,np.zeros(1,np.float32),rescale_factor=2.0),batch_size=1,num_workers=0)))
            batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            with torch.inference_mode():pred=model(batch)
            verts=pred['pred_vertices'][0].float().cpu().numpy().copy()
            verts[:,0]*=-1 # HaMeR internally flips left crops; restore camera handedness.
            cam=pred['pred_cam'].float().clone()
            cam[:,1]*=-1
            fullcam=cam_crop_to_full(cam,batch['box_center'].float(),batch['box_size'].float(),batch['img_size'].float(),
                       cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*max(w,h)).detach().cpu().numpy()[0]
            xyz=verts+fullcam
            focal=cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*max(w,h)
            xy=xyz[:,:2]/xyz[:,2:]*focal+[w/2,h/2]
            layer=base.copy();normals=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            shade=.35+.65*np.abs(normals@np.array([.2,-.4,.894427]))
            for fi in np.argsort(xyz[faces,2].mean(axis=1))[::-1]:
                tri=faces[fi]
                if xyz[tri,2].min()<=0:continue
                pts=np.rint(np.clip(xy[tri],-10000,10000)).astype(np.int32)
                cv2.fillConvexPoly(layer,pts,tuple(int(c*shade[fi]) for c in (230,205,115)),cv2.LINE_AA)
            image=cv2.addWeighted(layer,.78,base,.22,0)
            label='LEFT hand / full HaMeR mesh'
            detected=True
        else:label='hand box unavailable / no mesh';detected=False
        cv2.rectangle(image,(0,0),(w,32),(18,18,18),-1)
        cv2.putText(image,f'{i/59.005:.3f}s  {label}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(frames/f'frame_{i:04d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,94])
        records.append(dict(frame=i,time_s=i/59.005,mesh_rendered=detected));i+=1
        if i%30==0:print(f'{i}/421 frames; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    cap.release();assert i==421,f'Expected 421 frames; got {i}'
    result=subprocess.run(['ffmpeg','-y','-v','error','-framerate','84200/1427','-i',str(frames/'frame_%04d.jpg'),
        '-i',str(SRC),'-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
        '-c:a','copy','-t','7.135','-movflags','+faststart',str(video_out/'06_hamer_full_3d_hand.mp4')],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    report=dict(source=str(SRC),model='HaMeR',source_frames=i,duration_s=7.135,fps=59.005,
      mesh_vertices=int(model.mano.v_template.shape[0]),mesh_triangles=int(len(faces)),
      box_source='WiLoR detector boxes on 30 sampled frames, linearly interpolated only across gaps <=36 source frames; side labels overridden to user-confirmed LEFT',
      rendered_frames=sum(r['mesh_rendered'] for r in records),audio='copied from original',elapsed_s=time.perf_counter()-start,frames=records)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)

if __name__=='__main__':main()
