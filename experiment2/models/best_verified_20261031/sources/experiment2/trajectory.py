"""Ballistic teaching signals; actual MuJoCo collisions alone decide success."""
import numpy as np


def guidance(position, velocity, goal):
    p, v, goal = (np.asarray(x, dtype=float) for x in (position, velocity, goal))
    delta = goal-p
    # A family of arcs, rather than one prescribed release speed/angle. The
    # closest member gives a useful velocity error even for floor-bound throws.
    times = np.linspace(.20, 1.20, 101)
    required = delta[None, :]/times[:, None]
    required[:, 2] += 4.905*times
    errors = np.linalg.norm(required-v, axis=1)
    index = int(np.argmin(errors))
    gap = float(errors[index])
    quality = float(1./(1.+(gap/2.)**2)) if delta[0] > 0 else 0.
    forward = v[0] > .3 and delta[0] > 0
    flight = float(delta[0]/v[0]) if forward else None
    crossing = p+v*flight+np.array([0., 0., -4.905*flight**2]) if forward else None
    min_speed = float(np.sqrt(max(0., 9.81*(delta[2]+np.linalg.norm(delta)))))
    return {
        'velocity_gap_m_s': gap, 'velocity_quality': quality,
        'desired_velocity_m_s': required[index].tolist(),
        'desired_flight_time_s': float(times[index]),
        'minimum_ballistic_speed_m_s': min_speed,
        'predicted_board_yz_m': crossing[1:].tolist() if forward else None,
        'predicted_board_error_m': float(np.linalg.norm(crossing[1:]-goal[1:])) if forward else None,
        'required_vz_at_current_vx_m_s': float(delta[2]/flight+4.905*flight) if forward else None,
        'vertical_velocity_deficit_m_s': float(delta[2]/flight+4.905*flight-v[2]) if forward else None,
    }


def release_score(metrics, axis, velocity):
    """Smooth non-vanishing arc reward plus nose/velocity alignment at detach."""
    axis, velocity = np.asarray(axis), np.asarray(velocity)
    cosine = float(axis@velocity/max(np.linalg.norm(velocity), 1e-9))
    alignment = max(0., cosine)**2
    return 24.*metrics['velocity_quality']+2.*alignment, cosine
