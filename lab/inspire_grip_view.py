"""Interactive controls and a deterministic video of real MuJoCo states."""
import json
import queue
import subprocess
import time

import mujoco
import numpy as np

from inspire_grip import InspireGrip, OUT


def aim_camera(camera,s,mode='hand'):
    d=s.data
    center=d.xpos[s.hand]+d.xmat[s.hand].reshape(3,3)@np.asarray(s.config['dart_pos_hand'])
    if mode=='room':
        camera.lookat[:]=[.7,.1,.95];camera.distance=2.8;camera.azimuth=210;camera.elevation=-15
    elif mode=='drop':
        camera.lookat[:]=[center[0],center[1],.72];camera.distance=1.8;camera.azimuth=250;camera.elevation=-12
    else:
        camera.lookat[:]=center+[0,0,-.045];camera.distance=.38
        camera.azimuth=100 if mode=='hand' else 260;camera.elevation=-12


def viewer(checkpoint):
    import tkinter as tk
    from tkinter import ttk
    import mujoco.viewer
    s=InspireGrip(checkpoint)
    actions=queue.Queue();paused=False
    window=tk.Tk();window.title('G1 | LEFT five-finger grip');window.geometry('430x530')
    window.configure(bg='#101923')
    style=ttk.Style(window);style.theme_use('clam')
    style.configure('TLabel',background='#101923',foreground='#edf4f7',font=('Segoe UI',11))
    frame=tk.Frame(window,bg='#101923');frame.pack(fill='both',expand=True,padx=22,pady=20)
    ttk.Label(frame,text='LEFT HAND  /  FIVE FINGERS',font=('Segoe UI',17,'bold')).pack(anchor='w')
    ttk.Label(frame,text='Inspire DFQ · thumb + index + middle').pack(anchor='w',pady=(4,18))
    state=tk.StringVar();ttk.Label(frame,textvariable=state,font=('Segoe UI',14,'bold')).pack(anchor='w')
    telemetry=tk.StringVar();ttk.Label(frame,textvariable=telemetry,justify='left').pack(anchor='w',pady=14)
    for label,action in [('Release dart  [O]','open'),('Close grip  [C]','close'),
                         ('Reload dart and reset pose  [R]','reset'),('Pause / resume  [Space]','pause'),
                         ('Hand view  [1]','hand'),('Opposite view  [2]','back'),('Robot view  [3]','room')]:
        ttk.Button(frame,text=label,command=lambda a=action:actions.put(a)).pack(fill='x',pady=3)
    ttk.Label(frame,text='Pelvis supported. Reload places the dart.\nOpening releases it through contact physics.',
              font=('Segoe UI',9)).pack(anchor='w',pady=14)
    key_actions={79:'open',67:'close',82:'reset',32:'pause',49:'hand',50:'back',51:'room'}
    def key(k):
        if k in key_actions:actions.put(key_actions[k])
    with mujoco.viewer.launch_passive(s.model,s.data,key_callback=key) as v:
        v.opt.geomgroup[3]=0;aim_camera(v.cam,s)
        running=True
        def close():
            nonlocal running
            running=False
        window.protocol('WM_DELETE_WINDOW',close)
        while running and v.is_running():
            start=time.perf_counter();window.update()
            while not actions.empty():
                action=actions.get_nowait()
                if action=='open':s.closed=False
                elif action=='close':s.closed=True
                elif action=='reset':s.reset()
                elif action=='pause':paused=not paused
                else:aim_camera(v.cam,s,action)
            if not paused:
                for _ in range(20):s.step()
            row=s.sample();normals=row['normal_N']
            good=s.closed and min(normals)>.03 and row['relative_drift_m']<.008
            state.set('PAUSED' if paused else 'HOLDING — three contacts' if good else
                      'RELEASED' if not s.closed else 'CONTACT NOT ESTABLISHED')
            telemetry.set(f"Sim time: {row['time_s']:.1f} s\n"
                          f"Thumb  {normals[0]:.2f} N   Index  {normals[1]:.2f} N   Middle  {normals[2]:.2f} N\n"
                          f"Dart height: {row['com'][2]:.3f} m\n"
                          f"Grip drift: {row['relative_drift_m']*1000:.2f} mm")
            v.sync();time.sleep(max(0,.01-(time.perf_counter()-start)))
        window.destroy()


def record(checkpoint):
    from PIL import Image,ImageDraw,ImageFont
    s=InspireGrip(checkpoint);m,d=s.model,s.data
    r=mujoco.Renderer(m,600,800);o=mujoco.MjvOption();o.geomgroup[3]=0
    camera=mujoco.MjvCamera()
    font_path='C:/Windows/Fonts/segoeui.ttf'
    font=ImageFont.truetype(font_path,24);small=ImageFont.truetype(font_path,18)
    output=OUT/'left_hand_hold_release.mp4'
    command=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x760','-r','30',
             '-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(output)]
    pipe=subprocess.Popen(command,stdin=subprocess.PIPE)
    states=[]
    # Warmup is simulated, never teleport the dart into a successful state.
    for _ in range(2000):s.step()
    for frame in range(360):
        # Video 0..8 seconds is a hold; then a genuine command opens the hand.
        if frame==240:s.closed=False
        target_time=1.+(frame+1)/30
        while d.time<target_time-1e-9:s.step()
        row=s.sample();states.append(row)
        image=Image.new('RGB',(1600,760),'#101923')
        for column,mode in enumerate(('hand','back' if frame<240 else 'drop')):
            aim_camera(camera,s,mode);r.update_scene(d,camera,scene_option=o)
            image.paste(Image.fromarray(r.render()),(column*800,70))
        draw=ImageDraw.Draw(image)
        phase='HOLD · THUMB + INDEX + MIDDLE' if s.closed else 'RELEASE · OPEN HAND, FREE FALL'
        draw.text((24,17),'G1 / LEFT FIVE-FINGER INSPIRE HAND',font=font,fill='#f0f6f9')
        draw.text((820,17),phase,font=font,fill='#7fddc4' if s.closed else '#ffc477')
        n=row['normal_N'];draw.text((24,687),f"Contact normals   Thumb {n[0]:.2f} N   Index {n[1]:.2f} N   Middle {n[2]:.2f} N",font=small,fill='white')
        draw.text((820,687),f"Simulation {d.time:.2f} s  |  Dart height {row['com'][2]:.3f} m",font=small,fill='white')
        draw.text((24,720),'26.7 g dart · six control channels · physical contact · supported pelvis',font=small,fill='#9fafbc')
        pipe.stdin.write(np.asarray(image).tobytes())
        if frame==150:image.save(OUT/'left_hand_grip.png')
    pipe.stdin.close();code=pipe.wait();r.close()
    if code:raise RuntimeError('FFmpeg failed')
    (OUT/'video_trace.json').write_text(json.dumps(states,indent=2))
    print(output,flush=True)
