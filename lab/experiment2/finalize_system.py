"""Select on existing validation seeds, then run an untouched final test."""
import runtime_bootstrap
import json
import shutil
import hashlib
from pathlib import Path
import numpy as np
import torch
from evaluate_diagnostic import evaluate
from physics_eval import physical_summary
from env import build_env
from train import normalizer_for


def main():
    torch.set_num_threads(1)
    exp=Path(__file__).resolve().parent
    run=exp/'runs/20261030T161744390253'
    output=exp/'analysis/final_20261031';output.mkdir(exist_ok=False)
    config=json.loads((run/'config.json').read_text())
    profile=config['throw_profile']
    latest=run/'checkpoints/rl_model_405100_steps.zip'
    result=evaluate(latest,'arc_v7',output/'latest_validation',list(range(83101,83113)),throw_profile=profile)
    env=build_env(variant='arc_v7',throw_profile=profile,randomize=True)
    distances=[]
    for seed in range(83101,83113):
        env.reset(seed=seed);distances.append(float(np.linalg.norm(env.goal-env.start_pos)))
    env.close()
    latest_score=physical_summary(result['episodes'],float(np.mean(distances)))
    previous=json.loads((run/'best_physics/evaluation.json').read_text())
    winner=latest if tuple(latest_score['selection_score'])>tuple(previous['summary']['selection_score']) else run/'best_physics/policy.zip'
    selected_config=config if winner==latest else json.loads((winner.parent/'config.json').read_text())
    selection=dict(selected=str(winner),latest_summary=latest_score,previous_summary=previous['summary'],
                   selection_seeds=list(range(83101,83113)),final_test_seeds=list(range(84101,84133)))
    (output/'selection.json').write_text(json.dumps(selection,indent=2))
    print('SELECTED '+str(winner),flush=True)
    bundle=exp/'models/best_verified_20261031';bundle.mkdir(exist_ok=False)
    shutil.copy2(winner,bundle/'policy.zip');shutil.copy2(normalizer_for(winner),bundle/'normalize.pkl')
    (bundle/'config.json').write_text(json.dumps(selected_config,indent=2))
    shutil.copytree(run/'sources',bundle/'sources')
    shutil.copy2(run/'source_manifest.json',bundle/'source_manifest.json')
    shutil.copy2(output/'selection.json',bundle/'selection.json')
    report=evaluate(bundle/'policy.zip','arc_v7',output/'heldout',list(range(84101,84133)),throw_profile=selected_config['throw_profile'])
    shutil.copy2(output/'heldout/evaluation.json',bundle/'evaluation.json')
    hits=[r['impact_error_m'] for r in report['episodes'] if r['reward_terms'].get('board_impact',0)>0]
    summary=dict(report['metrics'],valid_board_hits=len(hits),
        median_impact_error_among_hits_m=float(np.median(hits)) if hits else None,
        best_impact_error_among_hits_m=float(min(hits)) if hits else None,
        policy_sha256=hashlib.sha256((bundle/'policy.zip').read_bytes()).hexdigest(),bundle=str(bundle))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (bundle/'summary.json').write_text(json.dumps(summary,indent=2))
    print('FINAL '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
