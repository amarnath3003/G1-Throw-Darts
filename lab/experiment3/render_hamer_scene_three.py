"""Render the HaMeR mesh animation as a real Three.js 3D scene and MP4."""
import json
import math
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FPS = 30
DURATION = 7.135
WIDTH, HEIGHT = 1280, 800


def build_payload():
    report = json.loads((HERE / 'data/hamer_pilot/report.json').read_text())
    times, raw_meshes, raw_joints, faces = [], [], [], None
    for row in report['frames']:
        if row['selected_left_detection'] is None:
            continue
        frame_id = row['frame_index']
        npz = np.load(HERE / 'data/hamer_pilot' / f'frame_{frame_id:04d}.npz', allow_pickle=False)
        camera_t = np.asarray(npz['hand0_pred_cam_t_full'][0], dtype=np.float32)
        verts = np.asarray(npz['hand0_pred_vertices'][0], dtype=np.float32) + camera_t
        joints = np.asarray(npz['hand0_pred_keypoints_3d'][0], dtype=np.float32) + camera_t
        times.append(float(row['relative_time_s']))
        raw_meshes.append(verts)
        raw_joints.append(joints)
        if faces is None:
            obj = HERE / 'data/hamer_mesh' / f'hand_{frame_id:04d}.obj'
            faces = np.asarray([
                [int(v.split('/')[0]) - 1 for v in line.split()[1:4]]
                for line in obj.read_text().splitlines() if line.startswith('f ')
            ], dtype=np.uint16)

    times = np.asarray(times, dtype=np.float32)
    raw_meshes = np.asarray(raw_meshes, dtype=np.float32)
    raw_joints = np.asarray(raw_joints, dtype=np.float32)
    # Use HaMeR's full camera-space wrist trajectory plus each articulated mesh.
    # Median-filter the noisy camera-depth estimates and compress that axis only
    # for display; the trajectory remains a 3D motion instead of a 2D tabletop path.
    raw_meshes[..., 0] *= -1  # camera-oriented left hand, matching official HaMeR demo
    raw_joints[..., 0] *= -1
    raw_wrists = raw_joints[:, 0]
    depth = np.pad(raw_wrists[:, 2], (2, 2), mode='edge')
    depth = np.asarray([np.median(depth[i:i+5]) for i in range(len(raw_wrists))], dtype=np.float32)
    root_x = raw_wrists[:, 0] - np.mean(raw_wrists[:, 0])
    root_y = raw_wrists[:, 1] - np.mean(raw_wrists[:, 1])
    root_z = (depth - np.median(depth)) * .035
    meshes = []
    for index, (verts, wrist) in enumerate(zip(raw_meshes, raw_wrists)):
        local = verts - wrist
        local_world = np.stack([local[:, 0], local[:, 2], -local[:, 1]], axis=-1)
        local_world[:, 0] += root_x[index]
        local_world[:, 1] += root_z[index]
        local_world[:, 2] -= root_y[index]
        meshes.append(local_world)
    meshes = np.asarray(meshes, dtype=np.float32)
    bounds_x = [float(meshes[..., 0].min()), float(meshes[..., 0].max())]
    bounds_z = [float(meshes[..., 2].min()), float(meshes[..., 2].max())]
    center_x = (bounds_x[0] + bounds_x[1]) / 2
    center_z = (bounds_z[0] + bounds_z[1]) / 2
    center_y = float((meshes[..., 1].min() + meshes[..., 1].max()) / 2)
    extent = float(max(np.ptp(meshes[..., axis]) for axis in range(3)))
    return {
        'times': times.tolist(), 'meshes': meshes.tolist(), 'faces': faces.tolist(),
        'center': [center_x, center_y, center_z],
        'extent': max(extent, .25),
        'frames': round(FPS * DURATION), 'fps': FPS,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title></title><style>*{box-sizing:border-box}html,body,#canvas{margin:0;width:100%;height:100%;overflow:hidden;background:#171a20}canvas{display:block}</style>
<body><div id="canvas"></div>
<script id="payload" type="application/json">__DATA__</script>
<script type="module">
import * as THREE from './vendor/three.module.js';
const D=JSON.parse(document.querySelector('#payload').textContent), fps=D.fps;
const host=document.querySelector('#canvas');
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});
renderer.setPixelRatio(1);renderer.setSize(innerWidth,innerHeight);renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;host.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#171a20');
const target=new THREE.Vector3(...D.center),span=D.extent;
const halfY=span*1.35,halfX=halfY*innerWidth/innerHeight;
const camera=new THREE.OrthographicCamera(-halfX,halfX,halfY,-halfY,.01,100);
camera.position.copy(target).add(new THREE.Vector3(.08,span*2.4,span*1.25));camera.up.set(0,0,-1);camera.lookAt(target);
scene.add(new THREE.HemisphereLight('#ffffff','#526071',2.3));
const key=new THREE.DirectionalLight('#e7f5ff',2.8);key.position.copy(target).add(new THREE.Vector3(-1,3,1.8));scene.add(key);
const fill=new THREE.DirectionalLight('#6cbaff',1.6);fill.position.copy(target).add(new THREE.Vector3(1,1,-1.5));scene.add(fill);
const positions=new Float32Array(D.meshes[0].flat());const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setIndex(D.faces.flat());geometry.computeVertexNormals();
const handMat=new THREE.MeshStandardMaterial({color:'#2388d5',roughness:.66,metalness:0,side:THREE.DoubleSide});
const hand=new THREE.Mesh(geometry,handMat);scene.add(hand);
let frame=0,last=performance.now(),acc=0;
let playing=true;
function sample(arr,t){if(t<=D.times[0])return [arr[0],arr[0],0];if(t>=D.times.at(-1))return [arr.at(-1),arr.at(-1),0];let b=1;while(D.times[b]<t)b++;const a=b-1;return [arr[a],arr[b],(t-D.times[a])/(D.times[b]-D.times[a])];}
function interp(a,b,u){const o=new Float32Array(a.length);for(let i=0;i<a.length;i++)o[i]=a[i]+(b[i]-a[i])*u;return o}
function setFrame(n){frame=Math.max(0,Math.min(D.frames-1,n));const t=frame/fps;
 const [va,vb,u]=sample(D.meshes,t),verts=interp(va.flat(),vb.flat(),u);geometry.attributes.position.array.set(verts);geometry.attributes.position.needsUpdate=true;geometry.computeVertexNormals();geometry.computeBoundingSphere();
 renderer.render(scene,camera);
}
function tick(now){requestAnimationFrame(tick);const dt=Math.min((now-last)/1000,.1);last=now;if(playing){acc+=dt;if(acc>=1/fps){const steps=Math.floor(acc*fps);acc-=steps/fps;setFrame((frame+steps)%D.frames)}}else renderer.render(scene,camera)}
addEventListener('resize',()=>{const hx=halfY*innerWidth/innerHeight;camera.left=-hx;camera.right=hx;camera.top=halfY;camera.bottom=-halfY;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);setFrame(frame)});
window.setFrameIndex=setFrame;setFrame(0);requestAnimationFrame(tick);
</script></body></html>'''


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    research = HERE / 'research'
    target = HERE.parent / 'progress_videos' / '09_hamer_real_3d_scene.mp4'
    frames_dir = HERE / 'data/hamer_three_frames'
    frames_dir.mkdir(exist_ok=True)
    payload = json.dumps(build_payload(), separators=(',', ':'))
    html_path = research / 'hamer_real_3d_scene.html'
    html_path.write_text(HTML_TEMPLATE.replace('__DATA__', payload), encoding='utf-8')

    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(research), **kwargs)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--enable-webgl', '--ignore-gpu-blocklist'])
            page = browser.new_page(viewport={'width': WIDTH, 'height': HEIGHT}, device_scale_factor=1)
            page.on('pageerror', lambda err: print(f'PAGE ERROR: {err}', flush=True))
            page.on('console', lambda msg: print(f'CONSOLE {msg.type}: {msg.text}', flush=True) if msg.type == 'error' else None)
            page.goto(f'http://127.0.0.1:{server.server_port}/hamer_real_3d_scene.html', wait_until='load', timeout=120000)
            page.wait_for_function('window.setFrameIndex !== undefined', timeout=120000)
            page.wait_for_timeout(3000)
            for i in range(round(FPS * DURATION)):
                page.evaluate('(n) => window.setFrameIndex(n)', i)
                page.screenshot(path=str(frames_dir / f'{i:04d}.png'))
                if (i + 1) % 60 == 0:
                    print(f'{i+1}/{round(FPS*DURATION)} frames', flush=True)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()

    r = subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-framerate', str(FPS), '-i', str(frames_dir / '%04d.png'),
        '-t', str(DURATION), '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target),
    ], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    print(target)
    print(html_path)


if __name__ == '__main__':
    main()
