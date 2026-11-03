"""Validate the extracted dataset and render timestamp-correct review artifacts."""
import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pickup_common import read_json, write_json, sha256, HERE


def annotate(frame, row, clip_id):
    frame=cv2.resize(frame,(540,960));scale=np.array([540,960])/np.array([1080,1920])
    table=row['table']
    if table['valid']:
        points=np.asarray(table['corners_px']).reshape(-1,4,2)*scale
        for marker in points:
            cv2.polylines(frame,[np.int32(marker)],True,(0,210,0),2)
    if row['pen']['visible']:
        points=np.int32(np.array(row['pen']['endpoints_px'])*scale)
        cv2.line(frame,tuple(points[0]),tuple(points[1]),(0,0,255),2)
    lines=[f'{clip_id} | source {row["time_s"]:.3f}s | {row["phase"]}',
           'TABLE: accepted' if table['valid'] else 'TABLE: rejected',
           'PEN: visible component' if row['pen']['visible'] else 'PEN: unresolved / occluded',
           'Planar tracking only; no hand reconstruction yet']
    cv2.rectangle(frame,(0,0),(540,104),(20,20,20),-1)
    for i,line in enumerate(lines):cv2.putText(frame,line,(8,22+i*24),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    return frame


def validate(dataset):
    root=read_json(dataset/'manifest.json');errors=[];results=[]
    assert [c['id'] for c in root['clips']]==['4276','4277','4279','4280']
    for entry in root['clips']:
        m=read_json(dataset/entry['manifest']);frames=m['frames'];pts=np.array(m['all_pts_s'])
        if not np.all(np.diff(pts)>0):errors.append(entry['id']+': PTS not monotonic')
        indices=[r['frame_index'] for r in frames]
        if indices!=sorted(set(indices)):errors.append(entry['id']+': duplicate/unordered frames')
        for r in frames:
            if abs(r['pts_s']-pts[r['frame_index']])>1e-8:errors.append(entry['id']+': source PTS mismatch')
            if not (dataset/r['file']).is_file():errors.append(entry['id']+': missing frame')
            if r['phase']=='put_down':errors.append(entry['id']+': put-down in extraction')
        lengths=[r['pen']['projected_length_m'] for r in frames if r['phase']=='rest' and r['pen']['visible'] and 'projected_length_m' in r['pen']]
        measured=float(np.median(lengths)) if lengths else None
        results.append(dict(clip=entry['id'],role=entry['role'],**m['summary'],
                       resting_length_relative_error=(measured/.145-1) if measured else None,
                       pen_tracking_complete=False))
    return dict(schema_version=1,integrity_pass=not errors,errors=errors,clips=results,
                total_frames=sum(r['selected_frames'] for r in results),
                table_accepted=sum(r['accepted_table_frames'] for r in results),
                metric_3d_validated=False,hand_reconstruction_started=False,
                limitations=['Pen tracking is a conservative visible-component baseline, not a complete hand-object tracker.',
                             'Resting pen length differs from measured 145mm; preserve 48mm marker scale and investigate before metric acceptance.',
                             'Phase boundaries are approximate; cap identity is not automatically validated.',
                             'No intrinsics, airborne-depth validation, finger reconstruction, contact labels or robot training.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    args=p.parse_args();root=read_json(args.dataset/'manifest.json')
    report=validate(args.dataset);review=args.dataset/'review';review.mkdir(exist_ok=True)
    write_json(review/'validation.json',report)
    sheet=Image.new('RGB',(4*270,3*480),'#202020')
    for col,entry in enumerate(root['clips']):
        clip=read_json(args.dataset/entry['manifest']);frames=clip['frames'];times=np.array([r['time_s'] for r in frames])
        cd=args.dataset/entry['id'];destination=review/f'{entry["id"]}_tracking.mp4'
        ff=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','540x960','-r','15','-i','-',
                             '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE)
        last=None;image=None
        for t in np.arange(times[0],times[-1]+1/30,1/15):
            idx=int(np.argmin(abs(times-t)))
            if idx!=last:
                row=frames[idx];image=annotate(cv2.imread(str(args.dataset/row['file'])),row,entry['id']);last=idx
            ff.stdin.write(image.tobytes())
        ff.stdin.close()
        if ff.wait()!=0:raise RuntimeError('Video encoder failed')
        phases=['approach','acquisition','hold'] if entry['id']!='4280' else ['presentation']*3
        for y,phase in enumerate(phases):
            candidates=[r for r in frames if r['phase']==phase]
            row=candidates[len(candidates)//2] if candidates else frames[len(frames)//2]
            f=annotate(cv2.imread(str(args.dataset/row['file'])),row,entry['id'])
            sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(f,(270,480)),cv2.COLOR_BGR2RGB)),(col*270,y*480))
        print('Reviewed',entry['id'],flush=True)
    sheet.save(review/'tracking_contact_sheet.jpg',quality=93)
    root['review_status']='milestone_1_ready_for_user_review; reconstruction_not_started'
    root['review']=dict(color='baseline RGB selected after visual comparison; SDR alternative retained',
                        planar_geometry='accepted with metric scale discrepancy warning',pen='incomplete; acquisition and held tracking require hand-aware refinement')
    write_json(args.dataset/'manifest.json',root)
    write_json(review/'source_hashes.json',{str(p.relative_to(HERE)):sha256(p) for p in [HERE/'prepare_pickup.py',HERE/'pickup_common.py',HERE/'review_pickup.py',HERE/'configs/pickup_v1.json']})
    print(report['total_frames'],'frames; integrity',report['integrity_pass'],flush=True)


if __name__=='__main__':main()
