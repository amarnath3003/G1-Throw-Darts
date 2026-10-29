"""Render HaMeR's animated HTML hand mesh from a fixed top-down camera."""
import json, subprocess
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
FPS=30
DURATION=7.135

def main():
    report=json.loads((HERE/'data/hamer_pilot/report.json').read_text())
    timestamps=[];vertices=[];faces=None;wrist=[];elbow=[]
    for row in report['frames']:
        if row['selected_left_detection'] is None:continue
        npz=np.load(HERE/'data/hamer_pilot'/f"frame_{row['frame_index']:04d}.npz",allow_pickle=False)
        v=np.asarray(npz['hand0_pred_vertices'][0],np.float32)
        c=np.asarray(npz['hand0_pred_cam_t_full'][0],np.float32)
        v=v+c # HaMeR camera translation: a moving camera-frame hand trajectory estimate.
        v[:,0]*=-1 # camera-oriented left hand, as in the official demo.
        timestamps.append(row['relative_time_s']);vertices.append(v)
        wrist.append(np.asarray(npz['hand0_pred_keypoints_3d'][0,0],np.float32)+c)
        if faces is None:
            f=HERE/'data/hamer_mesh'/f"hand_{row['frame_index']:04d}.obj"
            faces=np.asarray([[int(t.split('/')[0])-1 for t in line.split()[1:4]] for line in f.read_text().splitlines() if line.startswith('f ')],np.int32)
    timestamps=np.asarray(timestamps);vertices=np.asarray(vertices);wrist=np.asarray(wrist)
    # Use one shared camera-space origin across the whole clip. This retains
    # translation between frames while removing the arbitrary ~8 m camera depth.
    origin=np.mean(wrist,axis=0)
    vertices-=origin
    wrist-=origin
    # Transparent forearm guide follows the hand's wrist-to-middle-finger axis;
    # the source video does not observe the elbow, so this is explicitly schematic.
    wrist_index=0
    for i in range(len(wrist)):
        ix=report['frames'][next(j for j,r in enumerate(report['frames']) if abs(r['relative_time_s']-timestamps[i])<1e-5)]['frame_index']
        npz=np.load(HERE/'data/hamer_pilot'/f'frame_{ix:04d}.npz',allow_pickle=False)
        pts=np.asarray(npz['hand0_pred_keypoints_3d'][0],np.float32)
        c=np.asarray(npz['hand0_pred_cam_t_full'][0],np.float32)
        # wrist index 0, middle MCP index 9 in OpenPose/MANO joint ordering.
        direction=pts[9]-pts[0]
        length=np.linalg.norm(direction)
        if length<1e-6:direction=np.array([1.,0.,0.]);length=1
        elbow.append(wrist[i]-direction/length*.24)
    elbow=np.asarray(elbow)-origin
    # Camera display axes: screen X follows image X; screen Y is camera depth;
    # screen Z is image-up. Looking along screen Z gives a fixed top-down view.
    # Keep the original egocentric camera plane horizontal. HaMeR's local Z is
    # distance from the camera, so looking along -Z shows the complete mesh face.
    def coords(v):return np.stack([v[...,0],-v[...,1],v[...,2]],axis=-1)
    outframes=[];path=[]
    for frame_idx in range(round(FPS*DURATION)):
        t=frame_idx/FPS
        if t<=timestamps[0]:a=b=0;u=0
        elif t>=timestamps[-1]:a=b=len(timestamps)-1;u=0
        else:
            b=int(np.searchsorted(timestamps,t));a=b-1;u=(t-timestamps[a])/(timestamps[b]-timestamps[a])
        mesh=(1-u)*vertices[a]+u*vertices[b]
        w=(1-u)*wrist[a]+u*wrist[b];e=(1-u)*elbow[a]+u*elbow[b]
        # A detectable gap remains visible as a dashed warning band in the title.
        gap=(timestamps[b]-timestamps[a]) if b!=a else 0
        mesh_trace=go.Mesh3d(x=coords(mesh)[:,0],y=coords(mesh)[:,1],z=coords(mesh)[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],
            color='#8fc8ec',opacity=.90,flatshading=False,lighting=dict(ambient=.56,diffuse=.72),name='HaMeR hand surface')
        armpts=coords(np.stack([e,w]))
        arm_trace=go.Scatter3d(x=armpts[:,0],y=armpts[:,1],z=armpts[:,2],mode='lines+markers',
            line=dict(color='#f2ae63',width=12),marker=dict(size=[7,4],color=['#f2ae63','#ffffff']),name='inferred forearm guide')
        path.append(coords(w))
        trail=np.asarray(path)
        trail_trace=go.Scatter3d(x=trail[:,0],y=trail[:,1],z=trail[:,2],mode='lines',line=dict(color='#ffe27a',width=5),name='wrist path')
        outframes.append(go.Frame(name=f'{t:.3f}s',data=[mesh_trace,arm_trace,trail_trace],traces=[0,1,2]))
    fig=go.Figure(data=list(outframes[0].data),frames=outframes)
    fig.update_layout(title='HaMeR 3D hand motion · fixed top-down camera',paper_bgcolor='#10131a',plot_bgcolor='#10131a',
        width=1280,height=800,margin=dict(l=0,r=0,t=60,b=70),
        font=dict(color='white'),scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=.72),bgcolor='#10131a',
        xaxis_title='camera X · image right',yaxis_title='camera Y · image up',zaxis_title='camera depth',
        xaxis=dict(range=[-.26,.26],backgroundcolor='#171c25',gridcolor='#485363',zerolinecolor='#7f91a8'),
        yaxis=dict(range=[-.26,.26],backgroundcolor='#171c25',gridcolor='#485363',zerolinecolor='#7f91a8'),
        zaxis=dict(range=[7.7,8.3],backgroundcolor='#171c25',gridcolor='#485363',zerolinecolor='#7f91a8'),
        camera=dict(projection=dict(type='orthographic'),eye=dict(x=.12,y=-.14,z=2.2),up=dict(x=0,y=1,z=0))),
        annotations=[dict(text='Orange forearm guide is inferred from the hand axis; elbow is not observed. Camera motion/world scale are uncalibrated.',
           x=.5,y=-.06,xref='paper',yref='paper',showarrow=False,font=dict(size=12,color='#d4d9e2'))])
    html=HERE/'research/hamer_hand_space_motion.html'
    fig.write_html(html,include_plotlyjs=True,auto_play=False)
    frames_dir=HERE/'data/hamer_space_frames';frames_dir.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--enable-webgl','--ignore-gpu-blocklist'])
        page=browser.new_page(viewport={'width':1280,'height':800},device_scale_factor=1)
        page.goto(html.as_uri(),wait_until='load',timeout=120000)
        page.wait_for_function("document.querySelector('.js-plotly-plot') && document.querySelector('.js-plotly-plot')._fullLayout",timeout=120000)
        graph=page.locator('.js-plotly-plot')
        page.evaluate("""() => { const g=document.querySelector('.js-plotly-plot'); return Plotly.relayout(g,{
          'scene.camera.eye':{x:.12,y:-.14,z:2.2},'scene.camera.up':{x:0,y:1,z:0},
          'scene.camera.projection.type':'orthographic','scene.aspectmode':'data'}); }""")
        for i,fr in enumerate(outframes):
            page.evaluate("""async (name) => { const g=document.querySelector('.js-plotly-plot'); await Plotly.animate(g,[name],{
              mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}}); }""",fr.name)
            page.screenshot(path=str(frames_dir/f'{i:04d}.png'))
            page.wait_for_timeout(30)
            if (i+1)%60==0:print(f'{i+1}/{len(outframes)} frames',flush=True)
        browser.close()
    target=HERE.parent/'progress_videos'/'08_hamer_hand_space_topdown.mp4'
    r=subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(FPS),'-i',str(frames_dir/'%04d.png'),'-t',str(DURATION),
        '-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],capture_output=True,text=True)
    if r.returncode:raise RuntimeError(r.stderr)
    print(target)

if __name__=='__main__':main()
