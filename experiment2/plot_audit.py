"""Plot measured trajectories, not idealized replacements for failed throws."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from env import DartThrowEnv


def main():
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args()
    root=args.audit.parent;report=json.loads(args.audit.read_text())
    env=DartThrowEnv();qadr=env.sim.dart_q;local=env.m.body_ipos[env.sim.dart].copy();env.close()
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    colors=['#7f8c8d','#e67e22','#2874a6','#239b56']
    labels=['100k legacy','200k legacy','200k, wider controls (untrained)','arc_v2 after adaptation']
    for i,(key,value) in enumerate(report['matched_evaluation'].items()):
        for j,folder in enumerate(sorted((root/key).iterdir())):
            if not (folder/'states.npz').exists():continue
            states=np.load(folder/'states.npz');row=json.loads((folder/'result.json').read_text())
            if not row.get('release'):continue
            keep=states['time']>=row['release']['time_s'];q=states['qpos'][keep,qadr:qadr+7]
            if not len(q):continue
            com=q[:,:3]+Rotation.from_quat(q[:,[4,5,6,3]]).apply(np.tile(local,(len(q),1)))
            axes[0].plot(com[:,0],com[:,2],color=colors[i],alpha=.65,label=labels[i] if j==0 else None)
            v=row['release']['velocity'];axes[1].scatter(v[0],v[2],color=colors[i],alpha=.65,label=labels[i] if j==0 else None)
    axes[0].axvline(3,color='#333333',ls='--',lw=1);axes[0].scatter([3],[1.73],marker='x',s=90,color='#b03a2e',label='Bullseye')
    axes[0].axhline(0,color='#333333',lw=1);axes[0].set(xlabel='Forward position (m)',ylabel='Dart COM height (m)',title='Actual MuJoCo free-flight paths',xlim=(.3,3.2),ylim=(0,2.3))
    vx=np.linspace(1.7,6.5,200);flight=2.5/vx;vz=.31/flight+4.905*flight
    axes[1].plot(vx,vz,'--',color='#b03a2e',label='Required arc, representative release')
    axes[1].set(xlabel='Forward release velocity (m/s)',ylabel='Upward release velocity (m/s)',title='Release velocity falls below required arc',ylim=(-.5,8.))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    fig.savefig(root/'trajectory_diagnosis.png',dpi=170)
    print(root/'trajectory_diagnosis.png')


if __name__=='__main__':main()
