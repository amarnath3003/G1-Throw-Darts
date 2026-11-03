"""Versioned arm-prior edits; never modifies the dart's free-joint state."""
from copy import deepcopy
import numpy as np
from scipy.interpolate import CubicHermiteSpline


def deeper_elbow(plan,curve,model,delta=-.25,finish_delta=0.):
    """Add a larger, joint-limit-safe elbow windup with an exact derivative."""
    plan=deepcopy(plan);times=np.asarray(plan['times']);rows=curve(times);slopes=curve(times,1)
    address=model.joint('left_elbow_joint').qposadr[0]
    start,end=plan['start'],plan['end']
    for i,t in enumerate(times):
        if t<1.6:
            u=float(np.clip((t-.65)/.95,0,1));load=u*u*(3-2*u);rate=6*u*(1-u)/.95
        else:
            u=float(np.clip((t-start)/(end-start),0,1));load=1-u*u*(3-2*u);rate=-6*u*(1-u)/(end-start)
        rows[i,address]+=delta*load;slopes[i,address]+=delta*rate
        if t>=1.6:
            h=u*u*(3-2*u);dh=6*u*(1-u)/(end-start)
            rows[i,address]+=finish_delta*h;slopes[i,address]+=finish_delta*dh
    plan['arm_keyframes'][1][3]+=delta
    plan['arm_keyframes'][2][3]+=finish_delta
    plan['full_qpos_keyframes']=rows.tolist();plan['elbow_windup_delta_rad']=delta;plan['elbow_finish_delta_rad']=finish_delta
    return plan,CubicHermiteSpline(times,rows,slopes)
