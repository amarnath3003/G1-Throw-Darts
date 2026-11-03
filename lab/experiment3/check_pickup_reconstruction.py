"""Geometric regression checks and optional reconstruction artifact validation."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from fit_pickup_reference import fit_camera,approximate_root,palm_frame,smooth_joints
from pen_tracking import track_pen
from pickup_common import read_json,write_json


def geometry_checks():
    # Exact synthetic camera checks metre/pixel conventions and plane normal.
    f=1400.;K=np.array([[f,0,540],[0,f,960],[0,0,1]])
    R=Rotation.from_euler('xyz',[160,18,30],degrees=True).as_matrix();origin=np.array([.05,-.04,.8])
    H=K@np.column_stack([R[:,:2],origin]);H/=H[2,2]
    camera=fit_camera([dict(table=dict(valid=True,board_to_image=H.tolist()))]*31,1080,1920)
    assert abs(camera['focal_px']-f)<1.,camera
    rng=np.random.default_rng(7);joints=rng.normal(0,.04,(21,3));translation=np.array([.02,.03,.7])
    pts=(joints+translation)[:,:2]/(joints+translation)[:,2:]*f+[540,960]
    root,estimated_R=approximate_root(joints,pts,H,1080,1920,f)
    assert np.allclose(estimated_R,R,atol=1e-7)
    assert np.allclose(root,R.T@(joints[0]+translation-origin),atol=1e-7)
    palm=palm_frame(joints)
    assert np.allclose(palm.T@palm,np.eye(3),atol=1e-7) and abs(np.linalg.det(palm)-1)<1e-7
    # Missing observations remain missing; large gaps and phases stay isolated.
    times=np.array([0,.03,.06,.5,.53]);raw=np.zeros((5,21,3));raw[2]=np.nan;raw[3:]=1
    filtered=smooth_joints(times,raw,np.array(['reach']*3+['lift']*2),np.array([1,1,0,1,1],bool))
    assert np.isnan(filtered[2]).all() and np.allclose(filtered[:2],0) and np.allclose(filtered[3:],1)
    # A white reference page alone must not be returned as the pen.
    im=np.zeros((1920,1080,3),np.uint8);im[:]=[80,140,170]
    cv2.rectangle(im,(200,1000),(1000,1550),(240,240,240),-1)
    assert not track_pen(im)['valid']
    cv2.line(im,(640,400),(640,800),(230,235,200),8)
    pen=track_pen(im)
    assert pen['valid'] and abs(np.linalg.norm(np.diff(pen['endpoints_px'],axis=0))-400)<20
    return ['synthetic_camera_intrinsics','camera_board_roundtrip','proper_palm_rotation','missing_data_and_phase_smoothing','white_page_not_pen','visible_pen_line']


def validate(root):
    errors=[];stats=[];config=read_json(root/'manifest.json')
    for clip in config['clips']:
        m=read_json(root/clip['manifest']);report=read_json(root/clip['id']/'hamer_v2/report.json')
        rows=report['frames'];expected=[r['frame_index'] for r in m['frames']]
        if [r['frame_index'] for r in rows]!=expected:errors.append(clip['id']+': inference index mismatch')
        if report['source_sha256']!=m['source_sha256']:errors.append(clip['id']+': source hash mismatch')
        with np.load(root/'milestone2'/clip['id']/'reference.npz',allow_pickle=False) as ref:
            if not np.all(np.diff(ref['timestamps_s'])>0):errors.append(clip['id']+': timestamps')
            if ref['metric_depth_valid'].any():errors.append(clip['id']+': uncalibrated depth marked metric')
            valid=ref['hand_model_valid']
            if not np.isfinite(ref['joints_wrist_relative_smoothed'][valid]).all():errors.append(clip['id']+': nonfinite accepted joints')
            if not np.isnan(ref['joints_wrist_relative_smoothed'][~valid]).all():errors.append(clip['id']+': missing hand silently filled')
            rotations=ref['palm_rotation_camera'][valid]
            if not np.allclose(np.linalg.det(rotations),1,atol=1e-5):errors.append(clip['id']+': reflected palm')
            from fit_pickup_reference import PARENTS
            joints=ref['joints_wrist_relative_smoothed'][valid]
            lengths=np.linalg.norm(joints-joints[:,PARENTS],axis=2)
            if not np.allclose(lengths,ref['fixed_model_bone_lengths_m'],atol=1e-7):errors.append(clip['id']+': inconsistent hand shape')
        for r in rows:
            if r['phase']=='rest' and r['valid']:errors.append(clip['id']+': false positive in annotated empty scene')
            if r['valid']:
                with np.load(root/clip['id']/'hamer_v2'/f'{r["frame_index"]:06d}.npz',allow_pickle=False) as d:
                    if d['vertices'].shape!=(778,3) or not np.isfinite(d['vertices']).all():errors.append(clip['id']+': invalid mesh')
        stats.append(dict(clip=clip['id'],frames=len(rows),finite_hands=sum(r['valid'] for r in rows)))
    return dict(pass_integrity=not errors,errors=errors,stats=stats,metric_depth_verified=False,contacts_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path);args=p.parse_args();checks=geometry_checks()
    result=dict(geometry_checks=checks)
    if args.dataset:
        result.update(validate(args.dataset));write_json(args.dataset/'milestone2/integrity_checks.json',result)
        if result['errors']:raise AssertionError(result['errors'])
    print(result)
