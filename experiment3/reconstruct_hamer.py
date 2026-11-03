"""Official HaMeR inference on pilot boxes, with user-confirmed left side."""
import os
import sys
# This machine's optional pyarrow DLL import hangs via pandas/Lightning.
# Inference does not use Arrow; disable only this optional backend in this process.
sys.modules['pyarrow']=None
os.environ['PYOPENGL_PLATFORM']='win32'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch

HERE=Path(__file__).resolve().parent
for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
    if name not in np.__dict__:setattr(np,name,value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=30)
    args=parser.parse_args()
    from hamer.models import HAMER
    from hamer.configs import get_config
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils.renderer import cam_crop_to_full
    assets=json.loads((HERE/'configs/hamer_assets.json').read_text())
    for item in assets['files']:
        h=hashlib.sha256()
        with (HERE/item['file']).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==item['sha256']
    cfg=get_config(str(HERE/'models/hamer/_DATA/hamer_ckpts/model_config.yaml'))
    cfg.defrost()
    cfg.MANO.MODEL_PATH=str(HERE/'models/hamer/_DATA/data/mano')
    cfg.MANO.MEAN_PARAMS=str(HERE/'models/hamer/_DATA/data/mano_mean_params.npz')
    cfg.MODEL.BBOX_SHAPE=[192,256]
    cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None)
    cfg.freeze()
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        if torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('Need 4 GiB free CUDA memory')
        torch.cuda.reset_peak_memory_stats()
    print('Loading official HaMeR checkpoint...',flush=True)
    model=HAMER(cfg,init_renderer=False)
    weights=torch.load(HERE/'models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt',map_location='cpu',weights_only=False,mmap=True)
    status=model.load_state_dict(weights['state_dict'],strict=False)
    learned=set(dict(model.named_parameters()))
    assert not (learned & set(status.missing_keys)),f'Missing learned weights: {status.missing_keys}'
    print('Checkpoint loaded:',status,flush=True)
    del weights
    model.eval().to(device)
    baseline=json.loads((HERE/'data/wilor_pilot/report.json').read_text())
    frames=[];joints_all=[]
    out=HERE/'data/hamer_pilot';out.mkdir(exist_ok=True)
    for r in baseline['frames'][:args.limit]:
        image=cv2.imread(str(HERE/'data'/r['file']))
        record={k:r[k] for k in ['pilot_index','frame_index','pts_s','relative_time_s','file','phase']}
        record.update(detections=[],selected_left_detection=None,flags=[])
        raw={}
        if r['detections']:
            chosen=max(r['detections'],key=lambda p:np.prod(np.array(p['bbox'])[2:]-np.array(p['bbox'])[:2]))
            record['detections']=[dict(index=0,is_right=False,bbox=chosen['bbox'],original_detector_is_right=chosen['is_right'])]
            if chosen['is_right']:record['flags'].append('detector_side_corrected_using_user_confirmed_left_hand')
            dataset=ViTDetDataset(cfg,image,np.array([chosen['bbox']]),np.array([0]),rescale_factor=2.5)
            batch={k:torch.as_tensor(v).unsqueeze(0).to(device) for k,v in dataset[0].items()}
            start=time.perf_counter()
            with torch.no_grad():
                pred=model(batch)
                cam=pred['pred_cam'].clone();cam[:,1]*=-1
                focal=cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*batch['img_size'].max()
                trans=cam_crop_to_full(cam,batch['box_center'].float(),batch['box_size'].float(),batch['img_size'].float(),focal).cpu().numpy()
            record['latency_s']=time.perf_counter()-start
            verts=pred['pred_vertices'].cpu().numpy();verts[:,:,0]*=-1
            joints=pred['pred_keypoints_3d'].cpu().numpy();joints[:,:,0]*=-1
            points=(joints+trans[:,None,:])[:,:,:2]/(joints+trans[:,None,:])[:,:,2:]*float(focal)+np.array([image.shape[1]/2,image.shape[0]/2])
            raw=dict(hand0_pred_vertices=verts,hand0_pred_keypoints_3d=joints,hand0_pred_cam_t_full=trans,
                     hand0_scaled_focal_length=np.array(float(focal)),hand0_pred_keypoints_2d=points,
                     hand0_pred_cam=cam.cpu().numpy())
            raw.update({f'canonical_mano_{k}':v.cpu().numpy() for k,v in pred['pred_mano_params'].items()})
            assert verts.shape==(1,778,3) and np.isfinite(verts).all()
            joints_all.append(joints[0]-joints[0,0])
            record['selected_left_detection']=0
            if record['phase']=='acquisition_occluded':record['flags'].append('occluded_contact_is_estimated')
        else:
            joints_all.append(np.full((21,3),np.nan));record['flags'].append('no_hand_box')
        prefix=f"frame_{record['frame_index']:04d}"
        np.savez_compressed(out/(prefix+'.npz'),**raw)
        record['overlay']=prefix+'.png'
        frames.append(record)
        print(f"t={record['relative_time_s']:.3f} mesh={bool(raw)} flags={record['flags']}",flush=True)
    np.savez_compressed(out/'hand_raw.npz',joints_wrist_relative=np.array(joints_all),timestamps_s=np.array([r['relative_time_s'] for r in frames]),source_frame_indices=np.array([r['frame_index'] for r in frames]))
    report=dict(backend='HaMeR official model and official crop preprocessing; reused detector boxes',
                implementation_commit=assets['implementation_commit'],asset_provenance=assets['files'],
                anatomical_hand='left',image_mirrored=False,device=device,dtype='float32',
                total_frames=len(frames),selected_frames=sum(r['selected_left_detection'] is not None for r in frames),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                conventions=dict(mesh='Camera-oriented left mesh; X mirrored from canonical right prediction as official demo',
                                 canonical_mano='Raw right MANO rotation matrices; not robot-ready left wrist rotations',
                                 world='No world/camera-motion recovery. Assumed focal length; no metric ground truth'),
                limitations='Left label is user supervision. Hidden fingers remain inferred; this is not a controlled accuracy comparison with baseline WiLoR.',frames=frames)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {report["selected_frames"]}/{len(frames)} full hand meshes.',flush=True)


if __name__=='__main__':main()
