"""Display existing WiLoR triangle meshes; does not rerun or change inference."""
import json
import pickle
from pathlib import Path
import numpy as np
import cv2
import plotly.graph_objects as go
import argparse
import subprocess
for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
    if name not in np.__dict__:setattr(np,name,value)

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['wilor','hamer'],default='wilor')
    args=parser.parse_args()
    backend=args.backend
    label='WiLoR' if backend=='wilor' else 'HaMeR'
    with (HERE/'models/pretrained_models/MANO_RIGHT.pkl').open('rb') as f:
        faces=np.asarray(pickle.load(f,encoding='latin1')['f'],dtype=int)
    # Both exporters mirror canonical-right X for this left hand; reverse winding.
    faces=faces[:,::-1].copy()
    root=HERE/'data'/f'{backend}_pilot'
    report=json.loads((root/'report.json').read_text())
    out=HERE/'data'/f'{backend}_mesh'
    out.mkdir(exist_ok=True)
    animation=[];video_records=[]
    for record in report['frames']:
        chosen=record['selected_left_detection']
        if chosen is None:
            image=cv2.imread(str(HERE/'data'/record['file']))
            cv2.putText(image,'No valid left-hand reconstruction',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
            cv2.imwrite(str(out/f"mesh_{record['frame_index']:04d}.png"),image)
            video_records.append(record)
            continue
        path=root/f"frame_{record['frame_index']:04d}.npz"
        raw=np.load(path,allow_pickle=False)
        vertices=raw[f'hand{chosen}_pred_vertices'][0]
        cam=raw[f'hand{chosen}_pred_cam_t_full'][0]
        focal=float(raw[f'hand{chosen}_scaled_focal_length'])
        image=cv2.imread(str(HERE/'data'/record['file']))
        h,w=image.shape[:2]
        xyz=vertices+cam
        xy=xyz[:,:2]/xyz[:,2:]*focal+[w/2,h/2]
        layer=image.copy()
        normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        shade=.35+.65*np.abs(normals@np.array([.2,-.4,.894427]))
        for face in np.argsort(xyz[faces,2].mean(axis=1))[::-1]:
            triangle=faces[face]
            if np.min(xyz[triangle,2])<=0:continue
            points=np.rint(np.clip(xy[triangle],-10000,10000)).astype(np.int32)
            color=tuple(int(c*shade[face]) for c in (230,205,115))
            cv2.fillConvexPoly(layer,points,color,cv2.LINE_AA)
        overlay=cv2.addWeighted(layer,.78,image,.22,0)
        cv2.rectangle(overlay,(0,0),(w,48),(20,20,20),-1)
        cv2.putText(overlay,f"{label} full mesh / LEFT / {record['relative_time_s']:.3f}s",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(overlay,'Estimated surface; hidden contacts remain uncertain',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(out/f"mesh_{record['frame_index']:04d}.png"),overlay)
        video_records.append(record)
        # OBJ is in the upstream camera-oriented hand frame, not calibrated world space.
        obj=''.join(f'v {x:.8f} {y:.8f} {z:.8f}\n' for x,y,z in vertices)
        obj+=''.join(f'f {a+1} {b+1} {c+1}\n' for a,b,c in faces)
        (out/f"hand_{record['frame_index']:04d}.obj").write_text(obj)
        centered=vertices-raw[f'hand{chosen}_pred_keypoints_3d'][0,0]
        mesh=go.Mesh3d(x=centered[:,0],y=-centered[:,1],z=-centered[:,2],
                       i=faces[:,0],j=faces[:,1],k=faces[:,2],color='#83bada',flatshading=False,
                       lighting=dict(ambient=.5,diffuse=.8),name=f'{label} left hand')
        animation.append(go.Frame(name=f"{record['relative_time_s']:.3f}s",data=[mesh]))
    fig=go.Figure(data=animation[0].data,frames=animation)
    fig.update_layout(title=f'{label} full 3D surface — drag to rotate; slider selects valid frames',
        scene=dict(aspectmode='data',xaxis_title='X',yaxis_title='-Y',zaxis_title='-Z'),
        sliders=[dict(steps=[dict(method='animate',args=[[f.name],dict(mode='immediate',frame=dict(duration=0,redraw=True))],label=f.name) for f in animation])],
        annotations=[dict(text='Wrist-centered estimated mesh. Missing left detections omitted. Camera motion and metric world scale are not recovered.',showarrow=False,x=.5,y=-.08,xref='paper',yref='paper')])
    fig.write_html(HERE/'research'/f'{backend}_mesh_3d.html',include_plotlyjs=True)
    import shutil
    shutil.copyfile(out/'mesh_0118.png',HERE/'research'/f'{backend}_mesh_preview.png')
    lines=[]
    for i,r in enumerate(video_records):
        lines.extend([f"file 'mesh_{r['frame_index']:04d}.png'",'option framerate 600'])
        dt=video_records[i+1]['relative_time_s']-r['relative_time_s'] if i+1<len(video_records) else 1/59
        lines.append(f'duration {dt:.9f}')
    lines.extend([f"file 'mesh_{video_records[-1]['frame_index']:04d}.png'",'option framerate 600'])
    (out/'frames.txt').write_text('\n'.join(lines)+'\n')
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(out/'frames.txt'),'-fps_mode','vfr','-enc_time_base','1:600','-video_track_timescale','600','-c:v','libx264','-pix_fmt','yuv420p',str(HERE/'research'/f'{backend}_mesh.mp4')],check=True)
    print(f'Exported {len(animation)} meshes, {len(vertices)} vertices / {len(faces)} triangles each.')


if __name__=='__main__':main()
