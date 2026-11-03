"""Separate accuracy, reliability and timing instead of a conflated distance."""
import numpy as np

def endpoint_loss(row):
    if row.get('reward_terms',{}).get('board_impact',0)>0:
        return min(.5,float(row['impact_error_m']))
    crossing=row.get('board_plane_crossing')
    if row.get('release') and row['outcome'] in ('environment_hit','timeout') and crossing:
        return min(.5,float(crossing['error_m']))
    return .5

def diagnostics(rows):
    hits=[r for r in rows if r.get('reward_terms',{}).get('board_impact',0)>0]
    errors=[r['impact_error_m'] for r in hits]
    signed=[np.asarray(r['impact_position'])[1:]-np.array([0.,1.73]) for r in hits]
    release=[r['release']['time_s'] for r in rows if r.get('release')]
    commands=[r['command_time_s'] for r in rows if r.get('command_time_s') is not None]
    return dict(mean_capped_target_error_m=float(np.mean([endpoint_loss(r) for r in rows])),
        median_board_error_m=float(np.median(errors)) if errors else None,
        p90_board_error_m=float(np.percentile(errors,90)) if errors else None,
        mean_signed_impact_yz_m=np.mean(signed,axis=0).tolist() if signed else None,
        std_signed_impact_yz_m=np.std(signed,axis=0).tolist() if signed else None,
        within_2cm=sum(e<=.02 for e in errors),within_5cm=sum(e<=.05 for e in errors),
        collision_count=sum(r['outcome'] in ('robot_hit','recontact','bad_grip_contact') for r in rows),
        numerical_failure_count=sum(r['outcome']=='unstable_or_joint_limit' for r in rows),
        median_release_time_s=float(np.median(release)) if release else None,
        std_release_time_s=float(np.std(release)) if release else None,
        median_command_time_s=float(np.median(commands)) if commands else None,
        outcome_counts={k:sum(r['outcome']==k for r in rows) for k in sorted({r['outcome'] for r in rows})})

def paired_error_interval(previous,current):
    """Paired-seed bootstrap CI for candidate minus reference endpoint loss."""
    old={r['seed']:r for r in previous['episodes']}
    diffs=np.array([endpoint_loss(r)-endpoint_loss(old[r['seed']]) for r in current['episodes'] if r['seed'] in old])
    if not len(diffs):return None
    samples=np.random.default_rng(1701).choice(diffs,(2000,len(diffs)),replace=True).mean(axis=1)
    return dict(mean=float(diffs.mean()),lower=float(np.quantile(samples,.025)),upper=float(np.quantile(samples,.975)),n=len(diffs))
