"""Dense pinned HaMeR with per-frame detection and explicit occlusion flags."""
import os
import sys
sys.modules['pyarrow'] = None
os.environ.setdefault('PYOPENGL_PLATFORM', 'win32' if sys.platform=='win32' else 'egl')
import argparse
import contextlib
import io
import time
from pathlib import Path
import cv2
import numpy as np
from pickup_common import HERE, read_json, write_json, sha256


def load_models(device):
    for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
        if name not in np.__dict__:setattr(np,name,value)
    # chumpy's Python 3.11+ compatibility, scoped to this inference process.
    import inspect
    if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
    import torch
    from hamer.models import HAMER
    from hamer.configs import get_config
    from ultralytics import YOLO
    assets=read_json(HERE/'configs/hamer_assets.json')
    for item in assets['files']:
        path=HERE/Path(item['file'].replace('\\','/'))
        if sha256(path)!=item['sha256']:raise ValueError('HaMeR asset hash mismatch: '+path.name)
    detector_path=HERE/'models/pretrained_models/detector.pt'
    detector_asset=next(i for i in read_json(HERE/'configs/wilor_assets.json')['files'] if i['file'].endswith('detector.pt'))
    if sha256(detector_path)!=detector_asset['sha256']:raise ValueError('Detector hash mismatch')
    cfg=get_config(str(HERE/'models/hamer/_DATA/hamer_ckpts/model_config.yaml'))
    cfg.defrost();cfg.MANO.MODEL_PATH=str(HERE/'models/hamer/_DATA/data/mano')
    cfg.MANO.MEAN_PARAMS=str(HERE/'models/hamer/_DATA/data/mano_mean_params.npz')
    cfg.MODEL.BBOX_SHAPE=[192,256];cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None);cfg.freeze()
    torch.set_num_threads(2)
    model=HAMER(cfg,init_renderer=False)
    weights=torch.load(HERE/'models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt',map_location='cpu',weights_only=False,mmap=True)
    status=model.load_state_dict(weights['state_dict'],strict=False)
    if set(dict(model.named_parameters())) & set(status.missing_keys):raise ValueError('Missing learned weights')
    model.eval().to(device)
    return model,YOLO(str(detector_path)),cfg,assets


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int);p.add_argument('--clip')
    p.add_argument('--pass-name',default='hamer_v2',help='Separate corrected pass; preserve earlier raw predictions')
    args=p.parse_args()
    import torch
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils.renderer import cam_crop_to_full
    model,detector,cfg,assets=load_models(args.device)
    dataset=read_json(args.dataset/'manifest.json')
    base_manifest=read_json(args.dataset/dataset['clips'][0]['manifest'])
    background_row=base_manifest['frames'][0]
    background=cv2.imread(str(args.dataset/background_row['file']))
    background_H=np.array(background_row['table']['board_to_image'])
    np.save(args.dataset/'mano_faces_left.npy',np.asarray(model.mano.faces)[:,::-1])
    write_json(args.dataset/'inference_runtime.json',dict(torch=torch.__version__,opencv=cv2.__version__,numpy=np.__version__,
               python=sys.version,device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
               script_sha256=sha256(__file__),purpose='inference only; no training'))
    for entry in dataset['clips']:
        if args.clip and entry['id']!=args.clip:continue
        clip=read_json(args.dataset/entry['manifest']);out=args.dataset/entry['id']/args.pass_name;out.mkdir(exist_ok=True)
        records=[];previous_box=None;previous_time=None
        frames=[clip['frames'][j] for j in np.unique(np.linspace(0,len(clip['frames'])-1,args.limit,dtype=int))] if args.limit else clip['frames']
        for i,r in enumerate(frames):
            dest=out/f'{r["frame_index"]:06d}.npz';side=dest.with_suffix('.json')
            if dest.exists() and side.exists():
                rec=read_json(side);records.append(rec)
                if rec.get('bbox') is not None:previous_box=np.array(rec['bbox']);previous_time=r['time_s']
                continue
            im=cv2.imread(str(args.dataset/r['file']));start=time.monotonic()
            detections=detector.predict(im,conf=.25,device=args.device,verbose=False)[0].boxes
            boxes=detections.xyxy.cpu().numpy();scores=detections.conf.cpu().numpy();labels=detections.cls.cpu().numpy()
            flags=[];box=None;original_label=None
            # The static speaker was a false-positive hand in the initial pilot.
            # Reject annotated empty scenes and intersect detections with actual
            # moving foreground after board-based background alignment.
            foreground=None
            if r['phase']!='rest' and r['table']['valid']:
                transform=np.array(r['table']['board_to_image'])@np.linalg.inv(background_H)
                aligned=cv2.warpPerspective(background,transform,(im.shape[1],im.shape[0]))
                difference=np.max(cv2.absdiff(im,aligned),axis=2)
                mask=(difference>32).astype(np.uint8)*255;mask[int(im.shape[0]*.55):]=0
                mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
                mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((17,17),np.uint8))
                count,_,stats,_=cv2.connectedComponentsWithStats(mask)
                components=[s for s in stats[1:] if s[4]>1800 and s[0]+s[2]>im.shape[1]*.5]
                if components:
                    x,y,w,h,_=max(components,key=lambda s:s[4]);foreground=np.array([x,y,x+w,y+h],float)
            eligible=[]
            for j,b in enumerate(boxes):
                if foreground is not None:
                    inter=np.maximum(0,np.minimum(b[2:],foreground[2:])-np.maximum(b[:2],foreground[:2]))
                    if np.prod(inter)>1800:eligible.append(j)
            if eligible and r['phase']!='rest':
                # One demonstrated hand; choose spatial continuity, not class alone.
                score=scores[eligible].copy();eligible_boxes=boxes[eligible]
                if previous_box is not None:score-=np.linalg.norm((eligible_boxes[:,:2]+eligible_boxes[:,2:])/2-(previous_box[:2]+previous_box[2:])/2,axis=1)/max(im.shape[:2])
                chosen=eligible[int(np.argmax(score))];box=boxes[chosen].copy();original_label=int(labels[chosen])
                box[:2]=np.maximum(box[:2],foreground[:2]-20);box[2:]=np.minimum(box[2:],foreground[2:]+20)
                flags.append('box_refined_with_aligned_static_background')
            elif foreground is not None and r['phase']!='rest':
                box=foreground;flags.append('foreground_box_fallback_no_detector')
            elif r['phase']!='rest' and previous_box is not None and r['time_s']-previous_time<=.1:
                box=previous_box;flags.append('short_box_carry_forward_not_detection')
            rec=dict(frame_index=r['frame_index'],time_s=r['time_s'],phase=r['phase'],valid=False,flags=flags,
                     detector_boxes=boxes.tolist(),detector_scores=scores.tolist(),detector_labels=labels.tolist(),
                     anatomical_hand='left',mirrored=False,side_source='reviewed_left_hand_demonstration',original_label=original_label)
            if box is not None:
                crop=ViTDetDataset(cfg,im,box[None],np.array([0]),rescale_factor=2.5)
                with contextlib.redirect_stdout(io.StringIO()):item=crop[0]
                batch={k:torch.as_tensor(v).unsqueeze(0).to(args.device) for k,v in item.items()}
                with torch.inference_mode():
                    pred=model(batch);cam=pred['pred_cam'].clone();cam[:,1]*=-1
                    focal=cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*batch['img_size'].max()
                    translation=cam_crop_to_full(cam,batch['box_center'].float(),batch['box_size'].float(),batch['img_size'].float(),focal).cpu().numpy()[0]
                joints=pred['pred_keypoints_3d'].cpu().numpy()[0];joints[:,0]*=-1
                verts=pred['pred_vertices'].cpu().numpy()[0];verts[:,0]*=-1
                points=(joints+translation)[:,:2]/(joints+translation)[:,2:]*float(focal)+np.array([im.shape[1]/2,im.shape[0]/2])
                valid=bool(np.isfinite(joints).all() and np.isfinite(verts).all() and np.isfinite(points).all())
                inside=(points[:,0]>=0)&(points[:,0]<im.shape[1])&(points[:,1]>=0)&(points[:,1]<im.shape[0])
                truncated=bool(box[0]<2 or box[1]<2 or box[2]>im.shape[1]-2 or box[3]>im.shape[0]-2)
                rec.update(valid=valid,bbox=box.tolist(),assumed_focal_px=float(focal),latency_s=time.monotonic()-start,
                           visible_reprojection_only=True,depth_calibrated=False,joints_in_frame=int(inside.sum()),
                           truncated_box=truncated,usable_reference=bool(valid and inside.all()))
                if r['phase']=='acquisition':flags.append('hidden_contacts_inferred_not_observed')
                if original_label==1:flags.append('detector_side_corrected_using_reviewed_left_hand')
                np.savez_compressed(dest,joints=joints,vertices=verts,projected_joints=points,translation=translation,
                                    **{f'mano_{k}':v.cpu().numpy()[0] for k,v in pred['pred_mano_params'].items()})
                previous_box=box;previous_time=r['time_s']
            else:
                flags.append('no_hand_box');np.savez_compressed(dest,joints=np.full((21,3),np.nan))
            write_json(side,rec);records.append(rec)
            if i%30==0:print(entry['id'],i+1,'/',len(frames),flush=True)
        write_json(out/'report.json',dict(source_sha256=clip['source_sha256'],assets=assets,frames=records,script_sha256=sha256(__file__),
                   valid_frames=sum(r['valid'] for r in records),complete=len(frames)==len(clip['frames']),
                   conventions='camera-oriented left joints/vertices; canonical MANO is right; no second reflection; focal assumed'))


if __name__=='__main__':main()
