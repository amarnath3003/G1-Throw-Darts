# Physical pickup attempt — failed

Video: `11_g1_pickup_attempt_failed.mp4`, copied from Experiment 3 trial 06.

This controller is explicitly geometry-fitted/tuned, not HaMeR finger retargeting. It uses simulator object-pose feedback. The dart begins free on the table; execution uses motor torques with collision/gravity active and no attachment or object teleportation.

Thumb/finger contact triggered a lift attempt at 9.6025 seconds, but contact was lost and the dart stayed on the table. Maximum dart center-of-mass height gain was zero. This is not a successful pickup demonstration.

See `experiment3/PICKUP_ATTEMPT.md` and `experiment3/research/pickup_attempt/trial_06/report.json` for results and limitations.
