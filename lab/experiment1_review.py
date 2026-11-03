"""Build the local video gallery and audit every archived attempt, including failures."""
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import html
import numpy as np

ROOT=Path(__file__).resolve().parent/'experiment1'


def audit(folder):
    r=json.loads((folder/'result.json').read_text());trace=json.loads((folder/'trace.json').read_text())
    rows=[v for v in trace if v['time_s']<r['release_command']['time_s']]
    loss=None;start=None;established=False
    for row in rows:
        touching=sum(row['normal_N'])>.01 or any(n.startswith(('L_','left_wrist')) for n in row['other_contacts'])
        if min(row['normal_N'])>.03:established=True
        if touching:start=None
        elif established:
            if start is None:start=row['time_s']
            if row['time_s']-start>.015 and loss is None:loss=start
    number=int(folder.name[:2]);invalid=13<=number<=18
    status=('Invalid reset / retained debug run' if invalid else 'Slipped before open command' if loss is not None
            else 'Finger recontact after opening' if r['finger_recontact_after_release'] else 'Released without finger recontact')
    impact=r.get('first_impact')
    speed=r.get('release_speed_m_s') if not invalid and loss is None else None
    video=folder/'swing.mp4'
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    if decode.returncode or decode.stderr.strip():raise RuntimeError(str(video)+': '+decode.stderr)
    data=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,codec_name','-of','json',str(video)],capture_output=True,text=True,check=True)
    meta=json.loads(data.stdout)
    return {'id':folder.name,'number':number,'status':status,'invalid_initialization':invalid,
            'premature_contact_loss_s':loss,'three_contacts_established':established,
            'finger_recontact':r['finger_recontact_after_release'],'release_speed_m_s':speed,
            'release_command_s':r['release_command']['time_s'],
            'measured_clearance_s':r['actual_release']['time_s'] if r['actual_release'] and speed is not None else None,
            'first_impact_com':impact['com'] if impact and not invalid else None,
            'first_impact_geoms':sorted(set(impact.get('world_contacts',[]))) if impact and not invalid else [],
            'final_com':r['final_com'],'peak_grip_drift_mm':1000*r['max_closed_grip_drift_m'],
            'max_joint_limit_violation_rad':r['max_joint_limit_violation_rad'],
            'peak_left_finger_speed_rad_s':max(v for k,v in r['peak_joint_speed_rad_s'].items() if k.startswith('L_')),
            'finite':r['finite'],'warnings':r['warnings'],'video_decode_passed':True,
            'video_metadata':meta,'wrist_snap_rad':r['plan'].get('wrist_snap_amplitude_rad',0)}


def main():
    folders=sorted(p.parent for p in (ROOT/'trials').glob('*/result.json'))
    with ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(audit,folders))
    (ROOT/'audit.json').write_text(json.dumps(reports,indent=2))
    cards=[]
    for r in reports:
        name=html.escape(r['id']);url='trials/'+r['id']
        speed='—' if r['release_speed_m_s'] is None else f"{r['release_speed_m_s']:.2f} m/s"
        pos=r['first_impact_com'];impact='No ground impact in rollout' if pos is None else 'First impact X/Y/Z: '+', '.join(f'{v:.2f}' for v in pos)+' m'
        cards.append(f'''<article><h2>#{r['number']:02d} · {'Wrist snap' if r['wrist_snap_rad'] else 'Face-level swing' if r['number']>=13 else 'Initial swing'}</h2>
<video controls preload="none" poster="{url}/frame_060.png" src="{url}/swing.mp4"></video>
<p class="status">{html.escape(r['status'])}</p><p>Release: {speed} · open command {r['release_command_s']:.3f}s</p>
<p>{impact}</p><p class="muted">{name}</p><a href="{url}/result.json">Raw result</a> · <a href="{url}/trace.json">Physics trace</a></article>''')
    featured=''
    for p in sorted(ROOT.glob('*featured*/swing.mp4')):
        featured+=f'<article class="featured"><h2>{html.escape(p.parent.name)} · real time + slow replay</h2><video controls preload="metadata" src="{p.relative_to(ROOT).as_posix()}"></video></article>'
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment 1 · Every swing</title>
<style>body{background:#101923;color:#eaf0f6;font:16px system-ui;margin:0;padding:30px}main{max-width:1500px;margin:auto}h1{font-size:34px}h2{font-size:20px}p{line-height:1.5}a{color:#8fddcb}.muted{color:#9badbd;font-size:13px}.status{color:#ffd09b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:22px}article{background:#192633;padding:18px;border-radius:14px}video{width:100%;border-radius:8px}.featured{grid-column:1/-1}header{max-width:1000px;margin-bottom:28px}</style></head><body><main><header><h1>Experiment 1 / every swing</h1>'''
    page+=f'<p>{len(reports)} attempts, each with its own video and physics trace. Orange trails show the simulated dart COM.</p>'
    page+='<p>Manual joint animation; torque-driven original left Inspire hand; 26.7 g free dart; fixed pelvis support. No launch impulse. Hip joints and waist move, but this is not a balance or weight-transfer demonstration.</p><p>Failures remain visible. Runs 13–18 retain an invalid reset bug, corrected in run 19. Motor torque ceilings are enforced; measured finger speeds can exceed the source velocity specification during the swing. No hardware readiness or successful board hit is claimed.</p><a href="audit.json">All diagnostics and video checks</a></header><div class="grid">'+featured+''.join(reversed(cards))+'</div></main></body></html>'
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(reports[-4:],indent=2))


if __name__=='__main__':main()
