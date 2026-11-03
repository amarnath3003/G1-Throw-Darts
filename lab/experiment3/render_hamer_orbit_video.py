"""Render HaMeR's saved 3D HTML mesh sequence as a pan/orbit movie."""
import json
import subprocess
from pathlib import Path
import numpy as np
import cv2

HERE=Path(__file__).resolve().parent

def read_obj(path):
    vertices=[];faces=[]
    for line in path.read_text().splitlines():
        if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    return np.asarray(vertices,dtype=np.float32),np.asarray(faces,dtype=np.int32)

def main():
    report=json.loads((HERE/'data/hamer_pilot/report.json').read_text())
    mesh_dir=HERE/'data/hamer_mesh'
    valid=[]
    for row in report['frames']:
        path=mesh_dir/f"hand_{row['frame_index']:04d}.obj"
        if row.get('selected_left_detection') is not None and path.exists():valid.append((row['relative_time_s'],*read_obj(path)))
    assert valid,'No HaMeR mesh frames found'
    times=np.asarray([x[0] for x in valid]);vertices=[x[1] for x in valid];faces=valid[0][2]
    center=np.mean(vertices[0],axis=0)
    radius=max(np.linalg.norm(v-center,axis=1).max() for v in vertices)
    width,height=960,720; fps=30;duration=7.135;count=round(fps*duration)
    frame_dir=HERE/'data/hamer_orbit_frames';frame_dir.mkdir(exist_ok=True)
    light=np.array([.3,-.5,.8124]);light/=np.linalg.norm(light)
    for i in range(count):
        t=i/fps
        ix=int(np.argmin(np.abs(times-t)))
        v=vertices[ix]-center
        # Full camera orbit with a gentle independent pan and elevation sweep.
        az=2*np.pi*t/duration+np.deg2rad(25)
        el=np.deg2rad(20+8*np.sin(2*np.pi*t/duration))
        direction=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        target=np.array([.025*np.sin(2*np.pi*t/duration),.018*np.sin(4*np.pi*t/duration),.008*np.cos(2*np.pi*t/duration)])
        forward=-direction;right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
        up=np.cross(right,forward);cam=v-target
        depth=4*radius-cam@forward
        px=cam@right;py=cam@up
        focal=.40*min(width,height)/radius
        xy=np.stack([width/2+focal*px/depth,height/2-focal*py/depth],axis=1)
        canvas=np.zeros((height,width,3),np.uint8)
        for y in range(height):
            shade=13+int(8*y/height);canvas[y,:]=(shade,shade+3,shade+8)
        xyz=v
        tri_xyz=xyz[faces];norm=np.cross(tri_xyz[:,1]-tri_xyz[:,0],tri_xyz[:,2]-tri_xyz[:,0])
        norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-9)
        intensity=.32+.68*np.abs(norm@light)
        order=np.argsort(depth[faces].mean(axis=1))[::-1]
        for j in order:
            f=faces[j]
            if np.min(depth[f])<=.05:continue
            pts=np.rint(np.clip(xy[f],-2000,2000)).astype(np.int32)
            if np.any(pts[:,0]<-100) or np.any(pts[:,0]>width+100) or np.any(pts[:,1]<-100) or np.any(pts[:,1]>height+100):continue
            c=int(95+130*intensity[j]);cv2.fillConvexPoly(canvas,pts,(int(c*.75),int(c*.9),c),cv2.LINE_AA)
        cv2.putText(canvas,'HaMeR  /  LEFT HAND  /  MESH CAMERA ORBIT',(24,36),cv2.FONT_HERSHEY_SIMPLEX,.68,(235,241,248),1,cv2.LINE_AA)
        cv2.putText(canvas,f'{t:05.2f}s  |  view {i/count*360:03.0f} deg  |  frame {valid[ix][0]:.3f}s',(24,height-24),cv2.FONT_HERSHEY_SIMPLEX,.57,(195,207,222),1,cv2.LINE_AA)
        cv2.imwrite(str(frame_dir/f'{i:04d}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
        if (i+1)%60==0:print(f'{i+1}/{count}',flush=True)
    target=HERE.parent/'progress_videos'/'07_hamer_3d_camera_orbit.mp4'
    result=subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(fps),'-i',str(frame_dir/'%04d.jpg'),'-t',str(duration),
        '-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    print(f'Saved {target} ({count} frames at {fps} fps).')

if __name__=='__main__':main()
