"""Record the live five-finger frontend window: room, grasp, release, reload."""
import json
from pathlib import Path
import subprocess
import time
import tkinter as tk
from tkinter import ttk

import mujoco
from PIL import Image, ImageTk

from inspire_grip import InspireGrip, OUT
from inspire_grip_view import aim_camera

TITLE = 'G1 Darts - Left Hand Progress'
OUTPUT = Path(__file__).resolve().parent / 'progress_videos/04_left_five_finger_progress.mp4'


def main():
    s=InspireGrip();m,d=s.model,s.data
    root=tk.Tk();root.title(TITLE);root.geometry('1440x880+40+40');root.resizable(False,False)
    root.configure(bg='#101923')
    style=ttk.Style(root);style.theme_use('clam')
    style.configure('TLabel',background='#101923',foreground='#edf4f7',font=('Segoe UI',11))
    ttk.Label(root,text='G1 DARTS / LEFT FIVE-FINGER HAND',font=('Segoe UI',20,'bold')).place(x=24,y=16)
    phase=tk.StringVar();ttk.Label(root,textvariable=phase,font=('Segoe UI',13)).place(x=24,y=55)
    picture=tk.Label(root,bg='#101923',bd=0);picture.place(x=0,y=98,width=1100,height=720)
    panel=tk.Frame(root,bg='#101923');panel.place(x=1120,y=112,width=300,height=690)
    ttk.Label(panel,text='INSPIRE DFQ',font=('Segoe UI',18,'bold')).pack(anchor='w')
    ttk.Label(panel,text='Left arm · inward wrist\nThumb + index + middle\nRing and little fingers tucked\n\n26.7 g free-body dart\nSupported pelvis',justify='left').pack(anchor='w',pady=16)
    state=tk.StringVar();telemetry=tk.StringVar()
    ttk.Label(panel,textvariable=state,font=('Segoe UI',13,'bold')).pack(anchor='w',pady=8)
    ttk.Label(panel,textvariable=telemetry,justify='left').pack(anchor='w',pady=12)
    def release():s.closed=False
    def close():s.closed=True
    for label,fn in [('Release dart',release),('Close grip',close),('Reload dart',s.reset)]:
        ttk.Button(panel,text=label,command=fn).pack(fill='x',pady=5)
    ttk.Label(panel,text='Force controller trained in simulation.\nHardware transfer unvalidated.',wraplength=285).pack(anchor='w',pady=20)
    ttk.Label(root,text='Live MuJoCo frontend capture · backend-operated controls · contact-based hold and gravity release').place(x=24,y=840)
    renderer=mujoco.Renderer(m,720,1100);camera=mujoco.MjvCamera();option=mujoco.MjvOption();option.geomgroup[3]=0
    for _ in range(1000):s.step()
    def draw(mode):
        aim_camera(camera,s,mode)
        renderer.update_scene(d,camera,scene_option=option)
        photo=ImageTk.PhotoImage(Image.fromarray(renderer.render()));picture.configure(image=photo);picture.image=photo
        row=s.sample();n=row['normal_N']
        state.set('HOLDING / three contacts' if s.closed and min(n)>.03 else 'RELEASED' if not s.closed else 'CLOSING')
        telemetry.set(f"Thumb:  {n[0]:.2f} N\nIndex:    {n[1]:.2f} N\nMiddle:  {n[2]:.2f} N\n\nDart height: {row['com'][2]:.3f} m\nGrip drift: {1000*row['relative_drift_m']:.2f} mm\nSim time: {d.time:.2f} s")
        root.update();return row
    phase.set('ENVIRONMENT / robot and supported setup');draw('room')
    OUTPUT.parent.mkdir(exist_ok=True);log=(OUTPUT.parent/'_qa/04_left_five_finger_progress.log').open('w')
    recorder=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','gdigrab','-framerate','25',
        '-draw_mouse','0','-i','title='+TITLE,'-t','25','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-an',
        '-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(OUTPUT)],stdout=log,stderr=log)
    start=time.perf_counter();last=0.;target=d.time;released=False;reloaded=False;rows=[]
    try:
        while recorder.poll() is None:
            elapsed=time.perf_counter()-start
            if elapsed>27:break
            target+=min(elapsed-last,.1)*.7;last=elapsed
            if elapsed>=17 and not released:release();released=True
            if elapsed>=22 and not reloaded:s.reset();target=d.time;reloaded=True
            while d.time<target:s.step()
            if elapsed<5:mode='room';text='ENVIRONMENT / robot and supported setup'
            elif elapsed<12:mode='hand';text='GRIP / inward wrist, thumb opposing index and middle'
            elif elapsed<17:mode='back';text='OPPOSITE VIEW / stable physical grasp'
            elif elapsed<22:mode='drop';text='RELEASE / open fingers and let gravity drop the dart'
            else:mode='hand';text='RELOAD / reset for the next attempt'
            phase.set(text);row=draw(mode);row['recording_time_s']=elapsed;rows.append(row)
        if recorder.poll() is None:recorder.communicate(input=None,timeout=10)
        if recorder.returncode:raise RuntimeError('Frontend recording failed')
    finally:
        if recorder.poll() is None:recorder.terminate();recorder.wait()
        renderer.close();root.destroy();log.close()
    (OUTPUT.parent/'_qa/04_left_five_finger_progress_trace.json').write_text(json.dumps(rows,indent=2))
    print(OUTPUT)


if __name__=='__main__':main()
