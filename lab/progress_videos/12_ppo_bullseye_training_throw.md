# PPO bullseye training throw

A recorded MuJoCo training rollout from remote run `20261101T135744254760`, worker 1, episode 816. The dart made tip-first board contact **1.17 mm from the bullseye center**, inside the simulator's 6.35 mm bullseye radius.

This is one successful stochastic training throw, not a held-out success-rate evaluation. PPO selected smooth joint-reference corrections for the existing frozen throwing controller; it did not learn the complete throw from scratch. The dart flight and board contact came from the regular MuJoCo physics and success check.
