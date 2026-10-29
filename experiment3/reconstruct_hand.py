"""Local WiLoR-mini pilot: PTS-linked raw predictions, overlays and quality flags."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
(HERE/'data/ultralytics').mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(HERE/'data/ultralytics')
import numpy as np
import cv2
import torch

# Legacy MANO pickle may import chumpy's removed NumPy aliases. Process-local only.
for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
    if name not in np.__dict__:
        setattr(np,name,value)

EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
       (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=30)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--start',type=int,default=0)
    args=parser.parse_args()
    manifest=json.loads((HERE/'data/source_manifest.json').read_text())
    annotations=json.loads((HERE/'configs/video_annotations.json').read_text())
    assert manifest['sha256']==annotations['source_sha256']
    assets=json.loads((HERE/'configs/wilor_assets.json').read_text())
    for asset in assets['files']:
        p=HERE/asset['file']
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                h.update(block)
        assert h.hexdigest()==asset['sha256'],f'Asset hash mismatch: {p}'
    # Compatibility with PyTorch 2.6: this specific verified WiLoR checkpoint
    # includes non-tensor training metadata. Keep loading on CPU before .to(cuda).
    original_load=torch.load
    def checkpoint_load(path,*a,**kw):
        if Path(path).resolve()==(HERE/'models/pretrained_models/wilor_final.ckpt').resolve():
            kw.setdefault('weights_only',False)
            kw.setdefault('map_location','cpu')
            kw.setdefault('mmap',True)
        return original_load(path,*a,**kw)
    torch.load=checkpoint_load
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    torch.set_num_threads(2)
    if args.device=='cuda':
        free,total=torch.cuda.mem_get_info()
        if free<4*1024**3:
            raise RuntimeError('Less than 4 GiB free GPU memory; run --device cpu or retry after other work completes.')
        torch.cuda.reset_peak_memory_stats()
    print('Loading pinned WiLoR-mini float32 baseline...',flush=True)
    pipe=WiLorHandPose3dEstimationPipeline(device=torch.device(args.device),dtype=torch.float32,
                                         wilor_pretrained_dir=str(HERE/'models'),verbose=False)
    out=HERE/'data/wilor_pilot'
    out.mkdir(exist_ok=True)
    entries=manifest['pilot_frames'][args.start:args.start+args.limit]
    results=[]
    all_joints=[]
    for n,entry in enumerate(entries):
        image=cv2.imread(str(HERE/'data'/entry['file']))
        start=time.perf_counter()
        preds=pipe.predict(image,hand_conf=.3)
        elapsed=time.perf_counter()-start
        record=dict(**entry,latency_s=elapsed,detections=[])
        arrays={}
        for j,pred in enumerate(preds):
            values=pred['wilor_preds']
            arrays.update({f'hand{j}_{key}':np.asarray(value) for key,value in values.items()})
            record['detections'].append(dict(index=j,is_right=bool(pred['is_right']),bbox=pred['hand_bbox']))
        # Preserve all detections; only confirmed-side predictions enter a candidate track.
        candidates=[j for j,p in enumerate(preds) if not bool(p['is_right'])]
        chosen=max(candidates,key=lambda j:np.prod(np.asarray(preds[j]['hand_bbox'])[2:]-np.asarray(preds[j]['hand_bbox'])[:2])) if candidates else None
        record['selected_left_detection']=chosen
        record['phase']=next((p['name'] for p in annotations['phases'] if p['start_s']<=entry['relative_time_s']<p['end_s']),'unknown')
        record['flags']=[]
        if chosen is None:
            record['flags'].append('no_left_hand_detection')
            all_joints.append(np.full((21,3),np.nan))
        else:
            pred=preds[chosen]
            points=np.asarray(pred['wilor_preds']['pred_keypoints_2d'])[0]
            joints=np.asarray(pred['wilor_preds']['pred_keypoints_3d'])[0]
            all_joints.append(joints-joints[0])
            height,width=image.shape[:2]
            inside=np.isfinite(points).all(axis=1)&(points[:,0]>=0)&(points[:,0]<width)&(points[:,1]>=0)&(points[:,1]<height)
            record['joints_in_frame']=int(inside.sum())
            if not inside.all():record['flags'].append('projected_landmarks_outside_image')
            if not np.isfinite(joints).all():record['flags'].append('nonfinite_3d')
            if record['phase']=='acquisition_occluded':record['flags'].append('contact_occluded_do_not_treat_as_contact_labels')
            for a,b in EDGES:
                if inside[a] and inside[b]:
                    cv2.line(image,tuple(np.rint(points[a]).astype(int)),tuple(np.rint(points[b]).astype(int)),(30,240,40),2,cv2.LINE_AA)
            for j,point in enumerate(points):
                if inside[j]:cv2.circle(image,tuple(np.rint(point).astype(int)),3,(20,40,255),-1)
        label=f"{entry['relative_time_s']:.3f}s  LEFT / {record['phase']}"
        cv2.rectangle(image,(0,0),(image.shape[1],60),(15,15,15),-1)
        cv2.putText(image,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
        if record['flags']:cv2.putText(image,record['flags'][0][:56],(8,46),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,200,255),1,cv2.LINE_AA)
        prefix=f"frame_{entry['frame_index']:04d}"
        np.savez_compressed(out/(prefix+'.npz'),**arrays)
        cv2.imwrite(str(out/(prefix+'.png')),image)
        record['overlay']=prefix+'.png'
        results.append(record)
        print(f"{n+1}/{len(entries)} t={entry['relative_time_s']:.3f} hands={len(preds)} left={chosen} {elapsed:.2f}s",flush=True)
    np.savez_compressed(out/'hand_raw.npz',joints_wrist_relative=np.array(all_joints),
                        timestamps_s=np.array([e['relative_time_s'] for e in entries]),
                        source_frame_indices=np.array([e['frame_index'] for e in entries]))
    report=dict(backend='WiLoR-mini demo, unmodified upstream source',implementation_commit=assets['implementation_commit'],
        model_revision=next(a['revision'] for a in assets['files'] if a['file'].endswith('wilor_final.ckpt')),
        asset_provenance=assets['files'],device=args.device,dtype='float32',torch_version=torch.__version__,
        source_sha256=manifest['sha256'],anatomical_hand='left',image_mirrored=False,
        selected_frames=sum(r['selected_left_detection'] is not None for r in results),total_frames=len(results),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else None,
        conventions={'joint_order':'OpenPose 21: wrist, thumb, index, middle, ring, little (4 each)',
            'image':'pixels x-right y-down; original 480x854',
            'raw_3d':'Upstream MANO scale; camera-oriented, camera translation separate; left X reflected by upstream pipeline',
            'hand_raw':'Raw joints minus predicted wrist each frame, no smoothing or robot retargeting',
            'camera':'Uncalibrated assumed focal length = 5000/256 * max(image width,height); translation not metric ground truth',
            'orientation':'Upstream left axis-angle outputs saved verbatim; not yet validated for robot use'},
        quality='Detection and projected overlays are a qualitative pilot, not 3D/contact ground truth.',frames=results)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Variable-duration slideshow honors selected source PTS intervals.
    concat=[]
    for i,r in enumerate(results):
        concat.append(f"file '{r['overlay']}'")
        concat.append('option framerate 600')
        dt=results[i+1]['relative_time_s']-r['relative_time_s'] if i+1<len(results) else 1/59
        concat.append(f'duration {dt:.9f}')
    concat.append(f"file '{results[-1]['overlay']}'")
    concat.append('option framerate 600')
    (out/'frames.txt').write_text('\n'.join(concat)+'\n')
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(out/'frames.txt'),
                    '-fps_mode','vfr','-enc_time_base','1:600','-video_track_timescale','600','-c:v','libx264','-pix_fmt','yuv420p',str(HERE/'research/wilor_pilot.mp4')],check=True)
    print('Pilot report:',out/'report.json',flush=True)


if __name__=='__main__':main()
