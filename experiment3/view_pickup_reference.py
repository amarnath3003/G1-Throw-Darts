"""Offline wrist-centered 3D viewers and reference-quality plots."""
import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pickup_common import read_json,write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);args=p.parse_args()
    root=args.dataset;faces=np.load(root/'mano_faces_left.npy');created=[]
    for clip in read_json(root/'manifest.json')['clips']:
        directory=root/'milestone2'/clip['id'];report=read_json(root/clip['id']/'hamer_v2/report.json')
        valid=[r for r in report['frames'] if r['valid']];chosen=np.unique(np.linspace(0,len(valid)-1,min(48,len(valid)),dtype=int))
        animations=[]
        for index in chosen:
            row=valid[index]
            with np.load(root/clip['id']/'hamer_v2'/f'{row["frame_index"]:06d}.npz') as data:
                vertices=(data['vertices']-data['joints'][0])*np.array([1,-1,-1])*1000
            mesh=go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],
                          color='#78afcf',flatshading=False,lighting=dict(ambient=.55,diffuse=.7),name='Estimated left hand',hoverinfo='skip')
            animations.append(go.Frame(name=f'{row["time_s"]:.3f}s {row["phase"]}',data=[mesh]))
        fig=go.Figure(data=animations[0].data,frames=animations)
        fig.update_layout(title=f'{clip["id"]}: HaMeR estimated LEFT hand — drag to rotate',
             scene=dict(aspectmode='cube',xaxis=dict(title='Model X (mm)',range=[-160,160]),
                        yaxis=dict(title='Model −Y (mm)',range=[-160,160]),zaxis=dict(title='Model −Z (mm)',range=[-160,160])),
             margin=dict(l=10,r=10,t=60,b=110),
             sliders=[dict(currentvalue=dict(prefix='Source: '),steps=[dict(method='animate',args=[[f.name],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))],label=f.name) for f in animations])],
             annotations=[dict(x=.5,y=-.15,xref='paper',yref='paper',showarrow=False,text='Wrist-centered model units. Hidden fingers estimated; no verified contact, pen mesh or calibrated world trajectory.')])
        html=directory/'hand_3d.html';fig.write_html(html,include_plotlyjs=True);created.append(html)
        with np.load(directory/'reference.npz') as data:
            t=data['timestamps_s'];phase=data['phases'];weights=data['reference_weights'];pinch=data['thumb_to_fingertip_distances_model_m']*1000
            heights=data['wrist_height_focal_sensitivity_m']*1000;pen=data['pen_line_valid']
            fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
            axes[0].plot(t,weights,label='Heuristic hand reference weight');axes[0].plot(t,pen.astype(float)*.9,alpha=.5,label='Pen line available (0/0.9)')
            axes[0].set_ylabel('Weight / availability');axes[0].legend(loc='upper right');axes[0].set_ylim(-.05,1.05)
            axes[1].plot(t,pinch[:,0],label='Thumb–index');axes[1].plot(t,pinch[:,1],label='Thumb–middle')
            axes[1].set_ylabel('Model distance (mm)');axes[1].legend(loc='upper right')
            axes[2].plot(t,heights[:,1],label='Approximate effective-camera height')
            axes[2].fill_between(t,np.min(heights,axis=1),np.max(heights,axis=1),alpha=.25,label='±20% focal sensitivity')
            axes[2].axhline(0,color='black',linewidth=.8);axes[2].set_ylabel('Estimated wrist height (mm)');axes[2].set_xlabel('Original video time (s)');axes[2].legend(loc='upper right')
            for ax in axes:
                ax.grid(alpha=.2)
                ids=np.flatnonzero(phase=='acquisition')
                if len(ids):ax.axvspan(t[ids[0]],t[ids[-1]],color='orange',alpha=.12)
            fig.suptitle(f'{clip["id"]}: reference diagnostics — depth is NOT calibrated; orange = acquisition')
            plot=directory/'diagnostics.png';fig.savefig(plot,dpi=150);plt.close(fig);created.append(plot)
        print('Viewer and diagnostics',clip['id'],flush=True)
    write_json(root/'milestone2/local_viewer_hashes.json',{f.relative_to(root).as_posix():sha256(f) for f in created})


if __name__=='__main__':main()
