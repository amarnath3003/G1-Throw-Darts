"""Check room geometry, dart inertia, and a short unconstrained physics rollout."""
import json
from pathlib import Path

import mujoco
import numpy as np

from view_scene import load_scene
from dart_model import DartDynamics, load_profile, properties

root = Path(__file__).resolve().parent
model, data = load_scene()
bull = data.site_xpos[model.site("bullseye").id]
oche = data.site_xpos[model.site("oche_center").id]
assert abs(bull[2]-1.73) < 1e-8
assert abs((3.0-oche[0])-2.37) < 1e-8
assert model.nu == 29
assert model.nq == 43  # 36 robot + 7 free dart
dart = model.body("spare_dart_1").id
profile = load_profile()
prop = properties(profile)
assert abs(model.body_mass[dart]-profile["mass_kg"]) < 1e-10
assert abs(prop["tip_x_m"]-model.body_ipos[dart,0]-profile["com_behind_tip_m"]) < 1e-10
assert (model.body_inertia[dart] > 0).all()
assert model.body_inertia[dart].max() <= model.body_inertia[dart].sum()/2 + 1e-12
report = {"model_nq": model.nq, "model_nv": model.nv, "actuators": model.nu,
          "geoms": model.ngeom, "bull_world_m": bull.tolist(),
          "oche_to_board_face_m": 3.0-float(oche[0]),
          "free_dart_mass_kg": float(model.body_mass[dart]),
          "free_dart_com_local_m": model.body_ipos[dart].tolist(),
          "free_dart_inertia_kg_m2": model.body_inertia[dart].tolist()}
dynamics = DartDynamics(model, profile=profile)
for _ in range(2000):
    dynamics.step(data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
assert not any(w.number for w in data.warning)
report["physics_smoke_seconds"] = float(data.time)
report["status"] = "passed"
(root / "validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, indent=2))
