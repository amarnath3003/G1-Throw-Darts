"""Regression checks for the arm trial's convention and clearance failure modes."""
import hashlib
import json

import mujoco
import numpy as np

from retarget_arm import ArmIK, HERE, OUT, PickupScene, palm_frame


def main():
    s = PickupScene()
    ik = ArmIK(s)
    raw_path = HERE/'data/hamer_pilot/hand_raw.npz'
    raw = np.load(raw_path)
    valid = raw['joints_wrist_relative'][np.isfinite(raw['joints_wrist_relative']).all(axis=(1, 2))]
    for j in valid:
        r = palm_frame(j)
        assert np.allclose(r.T@r, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(r), 1, atol=1e-6)

    # Reproduce a genuinely colliding candidate from exploration. AABB fallback
    # must not erase its wrist/table penetration or classify it as clear.
    ik.forward(np.array([.34701091, -.05412691, .74135698, .21082678,
                         -.96338636, .16226866, -1.48524825]))
    distance = float(ik.distances().min())
    assert distance < -.001
    assert any(c.dist < -.001 and s.model.geom('table_top').id in (c.geom1, c.geom2)
               for c in ik.d.contact)

    report = json.loads((OUT/'report.json').read_text())
    assert report['source']['sha256'] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert report['model_sha256'] == s.config['model_sha256']
    assert report['planning_pass'] and report['pass']
    assert {r['timestep_s'] for r in report['physics']} == {.0005, .00025}
    for r in report['physics']:
        assert r['pass'] and r['new_arm_penetration_count'] == 0
        assert r['min_environment_clearance_m'] > .005
        assert not r['pickup_success']
        history = np.load(OUT/f"rollout_{r['timestep_s']:g}.npz")
        assert np.isfinite(history['qpos']).all() and np.all(np.diff(history['time_s']) > 0)
        # No fake lift: the free object's translation stays at its settled reset.
        dart_path = history['qpos'][:, s.dq:s.dq+3]
        assert np.max(np.linalg.norm(dart_path-dart_path[0], axis=1)) < .002

    result = dict(pass_checks=True, valid_proper_palm_frames=len(valid),
                  known_collision_distance_m=distance, model_and_source_hashes_match=True,
                  timesteps=[r['timestep_s'] for r in report['physics']], no_artificial_dart_lift=True,
                  mujoco_version=mujoco.__version__)
    (OUT/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
