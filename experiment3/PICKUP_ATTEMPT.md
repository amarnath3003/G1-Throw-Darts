# Physical pickup attempts — 2026-10-31

**Result: no successful pickup.** The recorded attempts start with an empty hand and a free dart on the existing table. Contact was achieved, but the dart was pushed/rolled sideways and remained on the table during the attempted lift. None of the recorded physical trials raised its center of mass above the initial settled height.

## What came from HaMeR, and what was tuned

The earlier `retarget_arm.py` rehearsal uses softened relative HaMeR palm orientation. Its world-space reach path, posture preferences and timing are designed; its fingers use the existing open preshape. It is not full hand-motion retargeting.

The new **`try_pickup.py` controller is geometry-fitted/tuned**, not HaMeR finger retargeting and not a trained policy. It uses pad locations from the existing holding geometry, fits arm/finger configurations to the actual dart meshes, and executes approach/closure/lift through torque actuation. The later trials also use simulator ground-truth object position and axis for feedback; a camera-based object estimator is not implemented. The videos explicitly state this provenance.

Planning uses scratch `MjData` to evaluate candidate configurations, including a synthetic reference holding configuration for extracting pad points. That placement is never applied to the executing simulation. Execution does not assign object/robot positions, weld the dart, disable collisions, change friction or apply an assisting lift force. Only joint motor torques and the existing aerodynamic model act during the attempt. The supported-base scope and existing tabletop/dart reset are unchanged.

## Recorded experiments

| Trial | Change | Outcome |
| --- | --- | --- |
| 01 | Initial contact-fitted approach | Upper-arm/torso contact blocked descent; hand missed dart. |
| 02 | Add upper-arm/torso clearance to trajectory IK | Arm tracks closely; first touch rolls dart and the grasp misses. |
| 03 | Alternate approach branch | Rejected before physics: wrist trajectory failed reachability gate. |
| 04 | More thumb/finger separation and stronger squeeze | Brief index contact rolls dart; no lift. |
| 05 | Object-axis/position feedback plus finger-force adjustment | Contacts reached on all three intended fingers, but grasp lost before timed lift. |
| 06 | Begin lift after sustained thumb/finger contact; freeze closure | Triggered at 9.6025 s, then lost contact during lift. No dart lift. |

Trials are in `research/pickup_attempt/trial_*/`. Executed trials include commanded references, recorded physical states, contact logs, previews and MP4s. Trial 03 contains the rejected command path. Trials 04–06 also retain the exact candidate configuration. `contact_fit.json` contains the latest exploratory candidates, **not a validated pickup configuration**. Later contact-direction fitting did not yield a validated grasp and was not presented as a successful trial.

Trial 06 tracked arm joint targets within 0.01444 radian (0.83 degrees), with maximum measured arm/hand penetration about 0.0303 mm. The dart's final center of mass was approximately 2.0 mm below its original height after rolling. Its maximum height gain was zero. Force thresholds indicated contact, not force closure; the failed lift is the decisive result.

## Diagnosis

The initial dart rests on a point/flight configuration that rolls after a light touch, changing barrel height by about 2 mm. The final grasp has only roughly 0.4 mm planned hand/table clearance, so this change matters. Feedback recovers contact, but the thumb and finger normal directions do not provide enough opposing support: they push the dart laterally rather than sustaining a pinch. A joint-space grip that works when the dart is already placed in the hand is not automatically an acquisition grasp.

The next implementation should fit contact normals, grasp-wrench support and table clearance together, then verify a sustained physical hold before starting transport. HaMeR finger motion can provide a preshape/closure prior, but it cannot supply unseen contact geometry or prove force closure. Tightening the arm IK alone is insufficient. No additional video is required to investigate this simulation/controller failure.

## Reproduce the latest recorded attempt

From `hackathon`, using a fresh output name:

```powershell
python experiment3/try_pickup.py --run --render --feedback --contact-trigger --candidate-file experiment3/research/pickup_attempt/trial_06/candidate.json --name trial_06_repeat
```

This reproduces a **failed** acquisition attempt. `pickup_success` is explicitly false in the report. The success gate requires the dart to remain at least 30 mm above the table, without table contact, supported by thumb and finger contact throughout the final three seconds. A transient touch or commanded hand lift does not pass.

## Progress videos

- `progress_videos/10_g1_hamer_guided_arm_rehearsal.mp4`: prior empty-hand arm rehearsal.
- `progress_videos/11_g1_pickup_attempt_failed.mp4`: trial 06, with overview and contact close-up.

The original HaMeR and WiLoR results, the pinned scene snapshot, and Experiment 2 are preserved. No hardware commands or training runs were started.
