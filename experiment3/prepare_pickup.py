"""PTS-preserving multi-clip extraction, ArUco table and visible pen tracking."""
import argparse
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pickup_common import HERE, ROOT, read_json, write_json, sha256, fresh_directory, board_corners, phase_at


def select_frames(times, clip):
    selected = set()
    # Sparse stationary calibration, 30Hz motion, native-rate acquisition.
    for target in np.r_[np.arange(0, min(clip['start'], 3), .5), np.arange(clip['start'], clip['end'], 1/30)]:
        selected.add(int(np.argmin(abs(times-target))))
    if clip['dense']:
        selected.update(np.flatnonzero((times >= clip['dense'][0]) & (times <= clip['dense'][1])).tolist())
    return sorted(selected)


def table_track(frame, detector, board):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return dict(valid=False, ids=[], reason='no_markers'), None
    pairs = [(board[int(i)], c[0]) for c, i in zip(corners, ids.flatten()) if int(i) in board]
    if len(pairs) < 2:
        return dict(valid=False, ids=ids.flatten().tolist(), reason='insufficient_spread'), None
    obj = np.concatenate([p[0] for p in pairs]); pixels = np.concatenate([p[1] for p in pairs])
    H, mask = cv2.findHomography(obj, pixels, cv2.RANSAC, 3.)
    if H is None:
        return dict(valid=False, ids=ids.flatten().tolist(), reason='homography_failed'), None
    error = np.linalg.norm(cv2.perspectiveTransform(obj[None], H)[0]-pixels, axis=1)
    med, p95 = np.percentile(error, [50,95])
    valid = len(pairs) >= 3 and med <= 2 and p95 <= 5
    return dict(valid=bool(valid), ids=ids.flatten().tolist(), median_px=float(med), p95_px=float(p95),
                board_to_image=H.tolist(), image_to_board=np.linalg.inv(H).tolist(),
                corners_px=pixels.tolist(), corners_board_m=obj.tolist(),
                inliers=mask.flatten().tolist()), H


def pen_track(frame, H, previous=None):
    """Only visible low-saturation elongated components; no occlusion filling."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,135]), np.array([179,85,255]))
    # The top edge of the white marker page is itself a long white component.
    # Exclude it explicitly instead of calling it a 30cm pen.
    mask[int(h*.48):] = 0
    mask[:int(h*.08)] = 0
    mask[:,:int(w*.22)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        rect = cv2.minAreaRect(c); sides = sorted(rect[1])
        if sides[0] < 4 or sides[1]/max(sides[0],1) < 7 or not .12*h < sides[1] < .42*h:
            continue
        pts = c[:,0,:].astype(float); center = pts.mean(0)
        _, _, vt = np.linalg.svd(pts-center, full_matrices=False); axis = vt[0]
        proj = (pts-center)@axis
        endpoints = center + np.array([proj.min(),proj.max()])[:,None]*axis
        if previous is not None and np.linalg.norm(endpoints-previous[::-1]) < np.linalg.norm(endpoints-previous):
            endpoints = endpoints[::-1]
        elif previous is None and (endpoints[0,1] > endpoints[1,1]):
            endpoints = endpoints[::-1]
        if H is not None:
            plane=cv2.perspectiveTransform(endpoints.astype(np.float32)[None],np.linalg.inv(H))[0]
            if not .075 < np.linalg.norm(plane[1]-plane[0]) < .22:continue
        # A component touching the ROI boundary is clipped, not a complete pen.
        if np.max(endpoints[:,1])>=int(h*.48)-2:continue
        score = cv2.contourArea(c) - (0 if previous is None else 2*np.linalg.norm(center-previous.mean(0)))
        candidates.append((score,endpoints))
    if not candidates:
        return dict(visible=False, reason='occluded_or_no_complete_component'), previous
    endpoints = max(candidates,key=lambda v:v[0])[1]
    result = dict(visible=True, endpoints_px=endpoints.tolist(), endpoint_identity='continuity_only_cap_not_verified',
                  confidence='visible_component_not_contact_ground_truth')
    if H is not None:
        plane = cv2.perspectiveTransform(endpoints.astype(np.float32)[None], np.linalg.inv(H))[0]
        result.update(projected_table_endpoints_m=plane.tolist(), projected_length_m=float(np.linalg.norm(plane[1]-plane[0])),
                      warning='Plane projection is metric only while resting; airborne height unknown')
    return result, endpoints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=str,default=str(HERE/'configs/pickup_v1.json'))
    p.add_argument('--out',type=str,required=True)
    p.add_argument('--resume',action='store_true',help='Recompute derived tracks using verified existing extraction')
    args=p.parse_args(); config=read_json(args.manifest)
    from pathlib import Path
    out=Path(args.out) if args.resume else fresh_directory(args.out)
    if args.resume and read_json(out/'manifest.json')['config_sha256']!=sha256(args.manifest):raise ValueError('Configuration changed')
    params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),params)
    dataset=dict(config=config,config_sha256=sha256(args.manifest),clips=[],review_status='pending_visual_review',
                 coordinate_frame='board metres: origin ID0 TL, x right, y page up, z out of page; images x right y down',
                 calibration='planar metric only; no measured camera intrinsics',color=config['color'])
    for clip in config['clips']:
        source=ROOT/config['source_directory']/clip['file']; cd=out/clip['id'];(cd/'frames').mkdir(parents=True,exist_ok=args.resume)
        if args.resume and read_json(cd/'manifest.json')['source_sha256']!=sha256(source):raise ValueError('Source changed')
        meta=__import__('json').loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_streams','-of','json',str(source)]))
        pts=np.array([float(f['best_effort_timestamp_time']) for f in meta['frames']]);times=pts-pts[0]
        if not np.all(np.diff(pts)>0):raise ValueError('Nonmonotonic PTS')
        chosen=select_frames(times,clip)
        # Compact arithmetic runs avoid FFmpeg's expression recursion limit.
        runs=[]; start=0
        while start < len(chosen):
            stop=start+1; step=chosen[stop]-chosen[start] if stop<len(chosen) else 1
            while stop+1<len(chosen) and chosen[stop+1]-chosen[stop]==step:stop+=1
            if stop>=len(chosen):stop=start
            runs.append(f'between(n\\,{chosen[start]}\\,{chosen[stop]})*not(mod(n-{chosen[start]}\\,{step}))')
            start=stop+1
        expression='+'.join(runs)
        if not args.resume:
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(source),'-vf','select='+expression,
                            '-fps_mode','passthrough','-q:v','2',str(cd/'frames/%06d.jpg')],check=True)
        if len(list((cd/'frames').glob('*.jpg')))!=len(chosen):raise ValueError('Extraction frame count mismatch')
        stream=meta['streams'][0]
        # No location metadata copied. Preserve technical timing/rotation/color only.
        technical={k:stream.get(k) for k in ['width','height','codec_name','pix_fmt','color_transfer','color_primaries','avg_frame_rate','time_base','side_data_list']}
        records=[];previous=None;medians=[];errors=[];lengths=[]
        for j,index in enumerate(chosen):
            filename=f'{clip["id"]}/frames/{j+1:06d}.jpg';frame=cv2.imread(str(out/filename))
            track,H=table_track(frame,detector,board_corners(config['marker_edge_m']))
            pen,previous=pen_track(frame,H if track['valid'] else None,previous)
            phase=phase_at(clip,float(times[index]))
            rec=dict(frame_index=index,pts_s=float(pts[index]),time_s=float(times[index]),file=filename,phase=phase,table=track,pen=pen)
            records.append(rec)
            if track['valid']:
                medians.append(track['median_px']);errors.append(track['p95_px'])
                for pt in track['corners_px']:cv2.circle(frame,tuple(np.int32(pt)),6,(0,220,0),2)
            if pen['visible']:
                ep=np.int32(pen['endpoints_px']);cv2.line(frame,tuple(ep[0]),tuple(ep[1]),(0,0,255),3)
                if phase=='rest' and 'projected_length_m' in pen:lengths.append(pen['projected_length_m'])
            cv2.putText(frame,f'{clip["id"]} {times[index]:.3f}s {phase} | planar only',(25,55),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),3)
        # Review movies are rendered separately at original timestamp speed by
        # review_pickup.py. Dense acquisition sampling must not speed up playback.
        # Comparison is deliberately separate; never change inference color mid-clip.
        sample=clip['start']+2
        filt='zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=full,format=rgb24,scale=360:-1'
        for name,vf in [('baseline','scale=360:-1'),('sdr',filt)]:
            subprocess.run(['ffmpeg','-y','-v','error','-ss',str(sample),'-i',str(source),'-vf',vf,'-frames:v','1',str(cd/f'color_{name}.png')],check=True)
        sheet=Image.new('RGB',(720,680),'#202020');draw=ImageDraw.Draw(sheet)
        for col,name in enumerate(['baseline','sdr']):
            im=Image.open(cd/f'color_{name}.png');sheet.paste(im,(col*360,30));draw.text((col*360+10,8),name,fill='white')
        sheet.save(cd/'color_comparison.jpg')
        result=dict(**clip,source_file=clip['file'],source_sha256=sha256(source),video=technical,all_pts_s=pts.tolist(),frames=records,
                    summary=dict(selected_frames=len(records),accepted_table_frames=len(medians),median_error_px=float(np.median(medians)),
                                 p95_corner_error_px=float(np.percentile(errors,95)),resting_pen_length_m=float(np.median(lengths)) if lengths else None,
                                 pen_visible_frames=sum(r['pen']['visible'] for r in records)))
        write_json(cd/'manifest.json',result)
        dataset['clips'].append(dict(id=clip['id'],role=clip['role'],manifest=f'{clip["id"]}/manifest.json',summary=result['summary']))
        print(clip['id'],result['summary'],flush=True)
    write_json(out/'manifest.json',dataset)


if __name__=='__main__':main()
