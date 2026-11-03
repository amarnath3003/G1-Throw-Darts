"""Build an isolated, self-contained MuJoCo snapshot for pickup experiments."""
from pathlib import Path
import hashlib
import json
import sys
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inspire_model import build_model

OUT = Path(__file__).resolve().parent


def main():
    folder = OUT / 'configs'
    folder.mkdir(exist_ok=True)
    if (folder / 'scene.mjb').exists():
        raise SystemExit('Snapshot exists; preserve it or explicitly move it before rebuilding.')
    m = build_model()
    # Relocate the complete equipment table; remove its tray and decorative darts.
    center = np.array([.62, .10, .77])
    old = m.geom_pos[m.geom('table_top').id].copy()
    delta = center - old
    for i in range(m.ngeom):
        name = m.geom(i).name or ''
        if name.startswith('table_'):
            m.geom_pos[i] += delta
        if name in ('dart_tray', 'hand_drop_mat') or name.startswith(('spare_dart_2_', 'spare_dart_3_')):
            m.geom_contype[i] = m.geom_conaffinity[i] = 0
            m.geom_rgba[i, 3] = 0
    # Narrow table fits beside the supported torso; legs follow its new corners.
    m.geom_size[m.geom('table_top').id] = [.25, .20, .025]
    legs = [i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('table_leg_')]
    for i, (x, y) in zip(legs, [(-1,-1),(-1,1),(1,-1),(1,1)]):
        m.geom_pos[i, :2] = center[:2] + [x*.21, y*.16]
    d = mujoco.MjData(m)
    mujoco.mj_setConst(m, d)
    # Recompile to rebuild static collision BVHs after moving world geoms.
    mujoco.mj_saveLastXML(str(folder / 'scene.xml'), m)
    tree = ET.parse(folder / 'scene.xml')
    for geom in tree.findall('.//geom'):
        name = geom.get('name', '')
        if name in ('dart_tray', 'hand_drop_mat') or name.startswith(('spare_dart_2_', 'spare_dart_3_')):
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
            geom.set('rgba', '0 0 0 0')
    tree.write(folder / 'scene.xml')
    m = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
    mujoco.mj_saveModel(m, str(folder / 'scene.mjb'), None)
    shutil.copyfile(ROOT / 'inspire_grip/policy.json', folder / 'grasp_target.json')
    sources = ['inspire_model.py', 'inspire_grip.py', 'dart_model.py', 'dart_profile.json',
               'scene.xml', 'inspire_grip/policy.json', 'experiment3/setup_scene.py']
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'assets').rglob('*') if p.is_file()]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    config = dict(schema_version=1, mujoco_version=mujoco.__version__, supported_base=True,
                  table_center=center.tolist(), table_half_size=[.25,.20,.025],
                  table_top_z=.795, dart_xy=[.52,.28], dart_quat_wxyz=[0,0,1,0],
                  settle_seconds=2., physics_timestep=.0005, source_sha256=hashes,
                  model_sha256=hashlib.sha256((folder/'scene.mjb').read_bytes()).hexdigest())
    (folder/'scene.json').write_text(json.dumps(config, indent=2)+'\n')
    print('Saved isolated model, source hashes and grasp target to', folder)


if __name__ == '__main__':
    main()
