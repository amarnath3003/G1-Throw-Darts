"""Live MuJoCo playback with hot-switchable PPO checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import torch
from stable_baselines3 import PPO
from env import DartThrowEnv

HERE=Path(__file__).resolve().parent
DEFAULT_BEST=HERE/'runs'/'20261029T131905437183'/'policy.zip'


def checkpoints():
    found=[]
    if DEFAULT_BEST.exists():
        found.append((f'Best evaluated · smoke policy (256 steps; 0/1 bullseyes)',DEFAULT_BEST,
                      DEFAULT_BEST.parent/'normalize.pkl'))
    for run in sorted((HERE/'runs').iterdir(),key=lambda x:x.stat().st_mtime,reverse=True):
        cp=run/'checkpoints'
        if not cp.exists():continue
        for model in sorted(cp.glob('rl_model_*_steps.zip'),key=lambda x:x.stat().st_mtime,reverse=True):
            suffix=model.stem.removeprefix('rl_model_')
            norm=cp/f'rl_model_vecnormalize_{suffix}.pkl'
            if norm.exists():
                config=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
                variant=config.get('variant') or 'legacy'
                found.append((f'{run.name} · {suffix.replace("_steps", " steps")} · {variant}',model,norm))
    return found


class Viewer:
    def __init__(self):
        torch.set_num_threads(1)
        self.env=DartThrowEnv(randomize=False,record=False)
        self.root=tk.Tk();self.root.title('Experiment 2 | PPO checkpoint viewer');self.root.geometry('1120x760')
        self.root.configure(bg='#101923');style=ttk.Style(self.root);style.theme_use('clam')
        style.configure('TLabel',background='#101923',foreground='#edf4f7',font=('Segoe UI',11))
        style.configure('TButton',font=('Segoe UI',10))
        top=ttk.Frame(self.root);top.pack(fill='x',padx=18,pady=14)
        ttk.Label(top,text='G1 DART THROW / PPO',font=('Segoe UI',18,'bold')).pack(side='left')
        self.pick=tk.StringVar();self.choices=[]
        self.combo=ttk.Combobox(top,textvariable=self.pick,state='readonly',width=60)
        self.combo.pack(side='left',padx=18);ttk.Button(top,text='Refresh checkpoints',command=self.refresh).pack(side='left',padx=5)
        ttk.Button(top,text='Load / switch policy',command=self.load).pack(side='left',padx=5)
        self.canvas=tk.Label(self.root,bg='#101923');self.canvas.pack(padx=18,pady=4)
        self.status=tk.StringVar(value='Choose a checkpoint.');ttk.Label(self.root,textvariable=self.status,font=('Segoe UI',12,'bold')).pack(anchor='w',padx=20,pady=8)
        bar=ttk.Frame(self.root);bar.pack(fill='x',padx=18,pady=10)
        ttk.Button(bar,text='Run / resume',command=self.play).pack(side='left',padx=4)
        ttk.Button(bar,text='Pause',command=self.pause).pack(side='left',padx=4)
        ttk.Button(bar,text='Reset throw',command=self.reset).pack(side='left',padx=4)
        ttk.Label(bar,text='Deterministic PPO · fixed dart setup · supported pelvis').pack(side='right')
        self.obs=None;self.policy=None;self.norm_mean=None;self.norm_var=None;self.running=False;self.terminal=False
        self.frame=None;self.steps=0;self.last_info={};self.lock=threading.Lock();self.refresh()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.draw();self.root.after(20,self.tick)
        # Start with the best evaluated checkpoint and display a real rollout.
        self.root.after(300,self.load);self.root.after(900,self.play)

    def refresh(self):
        self.choices=checkpoints();self.combo['values']=[x[0] for x in self.choices]
        if self.choices and self.combo.current()<0:self.combo.current(0)
        self.status.set(f'{len(self.choices)} loadable policies. Checkpoints appear here after PPO saves them.')

    def load(self):
        i=self.combo.current()
        if i<0:return
        label,model_path,norm_path=self.choices[i]
        try:
            # A copied VecNormalize checkpoint is paired with its exact PPO weights.
            import cloudpickle
            with norm_path.open('rb') as f:normalizer=cloudpickle.load(f)
            mean=np.asarray(normalizer.obs_rms.mean,dtype=np.float32)
            var=np.asarray(normalizer.obs_rms.var,dtype=np.float32)
            policy=PPO.load(str(model_path),device='cpu')
            if mean.shape!=(73,) or var.shape!=(73,):raise ValueError('Checkpoint observation statistics do not match this environment')
            run=model_path.parent.parent if model_path.parent.name=='checkpoints' else model_path.parent
            config=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
            variant=config.get('variant') or 'legacy'
            if self.env.variant!=variant:
                replacement=DartThrowEnv(randomize=False,record=False,variant=variant)
                self.env.close();self.env=replacement
        except Exception as exc:
            messagebox.showerror('Checkpoint could not load',str(exc));return
        self.policy=policy;self.norm_mean=mean;self.norm_var=var;self.steps=0;self.terminal=False
        self.obs,_=self.env.reset(seed=73129)
        self.status.set(f'Loaded: {label}   |   reset into the real three-finger grasp')
        self.draw()

    def play(self):
        if self.policy is None:self.load()
        if self.policy is not None and not self.terminal:self.running=True
    def pause(self):self.running=False
    def reset(self):
        self.pause();self.obs,_=self.env.reset(seed=73129);self.terminal=False;self.steps=0;self.last_info={}
        self.status.set('Reset to the ready pose.');self.draw()

    def tick(self):
        if self.running and self.policy is not None and not self.terminal:
            raw=np.asarray(self.obs,dtype=np.float32)
            norm=np.clip((raw-self.norm_mean)/np.sqrt(self.norm_var+1e-8),-10,10)
            action,_=self.policy.predict(norm,deterministic=True)
            self.obs,reward,terminated,truncated,info=self.env.step(action)
            self.steps+=1;self.last_info=info;self.terminal=bool(terminated or truncated)
            row=self.env.sim.sample();
            if self.steps%2==0 or self.terminal:
                state='THROW COMPLETE · '+str(info.get('outcome')) if self.terminal else 'RUNNING'
                self.status.set(f'{state}   ·   contacts T/I/M {row["normal_N"][0]:.1f}/{row["normal_N"][1]:.1f}/{row["normal_N"][2]:.1f} N   ·   dart {row["com"][0]:.2f}, {row["com"][1]:.2f}, {row["com"][2]:.2f} m')
                self.draw()
        self.root.after(20,self.tick)

    def draw(self):
        from PIL import Image,ImageTk,ImageDraw
        frame=Image.fromarray(self.env.render()).resize((960,576));draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,960,34),fill='#101923');draw.text((14,9),'LIVE MuJoCo STATE · PPO controls arm / waist / release · original finger physics',fill='white')
        self.frame=ImageTk.PhotoImage(frame);self.canvas.configure(image=self.frame)

    def close(self):
        self.running=False;self.env.close();self.root.destroy()

    def run(self):self.root.mainloop()


if __name__=='__main__':Viewer().run()
