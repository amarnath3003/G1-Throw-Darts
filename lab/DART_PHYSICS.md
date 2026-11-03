# Dart physics and transfer readiness

## Reference and coordinates

Default: the experimental dart in James and Potts (2018),
[Experimental validation of dynamic stability analysis applied to dart flight](https://doi.org/10.1007/s12283-018-0279-9),
sections 2.1/3.2 and equations 31–35. This replaces the previous arbitrary 25 g
assembly; it is not a Target Bolide replica.

`dart_profile.json` is the source of truth, using SI units. The body origin is
the barrel midpoint; local +X points toward the tip. Named sites expose the tip,
grip, center of mass, and aerodynamic center. `assets/dart.xml` exports the same
free-body asset without the room or robot.

## Evidence audit

| Property | Default | Evidence |
|---|---:|---|
| Complete mass | 26.7 g, reported spread ±0.2 g | Measured in reference |
| Metal / tail assembly mass | 24.9 / 1.8 g | Reported aggregate measurements |
| Balance point behind tip | 48.5 mm | Knife-edge measurement |
| Barrel length / diameter | 57.4 / 7.08 mm | Measured; 80% tungsten composition reported |
| Nylon stem excluding thread | 44.7 mm | Measured |
| Flight span / root chord | 42.9 / 50.3 mm | Measured |
| Fin taper / quarter-chord sweep | 0.44 / 27.9° | Reference analytical geometry |
| Aerodynamic center behind tip | 126.5 mm | Reference analytical approximation |
| Normal-force slope | 1.69 per radian | Reference analytical estimate |
| Transverse inertia | 1.81446e-5 kg m² | Inferred from reported pitch derivative and assumed reference air density 1.225 kg/m³ |
| Axial inertia | 2.41183e-7 kg m² | Approximate geometry; assumed 65/35 stem/fin mass partition |
| Exposed point length / diameter | 26 / 2.3 mm | Unverified reconstruction assumptions |
| Flight thickness | 0.2 mm | Assumed rigid sheet |
| Drag, friction, contact compliance | Profile values | Unmeasured engineering assumptions |

The paper's 23.5 g barrel excludes the point; 24.9 g is the combined metal
assembly. Total measured mass and COM override density-based estimates. The
paper reports a discrepancy between geometric and measured COM. Internal
threading, bores, and exact material distribution are not reconstructed. Shallow
grip grooves are a geometric approximation; flights cannot bend or flutter.

Material colors are visual. Mechanics use explicit mass/inertia, collision,
friction, and solver compliance. A steel/tungsten material label does not supply
a measured contact response. The model uses sliding, torsional, and rolling
friction, with dart geometry contact priority so the configured values apply
against default-priority surfaces. These effective coefficients need calibration
for the actual fingertip material. Different material pairs can need different
parameters. Solver compliance is not a measured Young's modulus.

## Flight implementation

`DartDynamics` evaluates air-relative velocity at the aerodynamic center,
including motion caused by rotation. Rear-fin normal forces generate restoring
pitch/yaw torque and angular damping. Axial/crossflow drag oppose relative motion;
weak quadratic roll drag dissipates spin. Gravity acts at the explicit COM.

The small-angle slope and static margin reproduce the reference's analytical
pitch derivatives. Large-angle behavior uses a bounded tanh extension; it is not
validated stall data. Agreement with the reference equations is a consistency
test, not an independent comparison against experimental trajectories.

Air loads enter `data.xfrc_applied` as forces and torques about COM. The runtime
owns the rows for the named free darts; controllers must not overwrite them after
force evaluation. Built-in fluid loads remain disabled to avoid double counting.

Training integration:

```python
from dart_model import DartDynamics

dynamics = DartDynamics(model, body_names=("spare_dart_1",))
# Optional at reset only; each environment must own a separate model:
sample = dynamics.randomize(data, rng)
# Every MuJoCo substep, after actuator commands have been set:
dynamics.step(data)
```

Use `("dart",)` with the isolated XML. A bare `mj_step` omits these air loads.
The constructor rejects stale mass/inertia after profile edits. Rebuild first.
Randomization is non-cumulative and reproducible from a NumPy RNG. It covers
mass, COM, inertia, fin force, drag, friction, and wind. Only the mass range
follows reported spread; other ranges are robustness assumptions, not measured
confidence intervals. Geometry changes require rebuilding from a new profile.

## Checks and calibration boundary

`validate_dart.py` writes `dart_validation.json`. It checks mass/COM/inertia,
measured-inertia overrides, force/acceleration mapping, passive air-force power,
pitch/yaw restoring signs, reference stability derivatives, gravity-only motion,
timestep convergence, worker isolation, repeated randomization, and bad profiles.
These checks establish numerical consistency; hardware transfer is unverified.

Before claiming real-world accuracy:

1. Select the actual dart. Measure assembly/component mass, dimensions, balance,
   and flight shape; update profile values and evidence labels.
2. Where available, set `measured_inertia_kg_m2` to the three diagonal moments in
   the stated principal axes. Otherwise retain the inferred inertia uncertainty.
3. Record calibrated high-speed side/front flight. Estimate release state and
   fit drag/fin response on a training subset, keeping separate throws held out.
4. Measure grip/slip and opening/release with the actual fingertip material; fit
   contact parameters. Broad randomization cannot prove the match.
5. Compare held-out impact position, time of flight, and impact pitch/yaw with
   error bounds relative to the desired scoring target.
6. Evaluate the full hand/robot controller, latency, and actuator limits. Dart
   fidelity alone cannot establish whole-system transfer accuracy.

No calibrated board penetration or sticking is implemented. Before impact the
dart follows the explicit air model; board interaction is ordinary rigid contact.
