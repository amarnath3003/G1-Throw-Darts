"""Finish an existing run, then continue its saved policy on CUDA without resetting learning."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-run',type=Path,required=True)
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();previous=args.after_run.resolve();runs=previous.parent
    status=runs/'gpu_handoff.json'
    def write(state,**extra):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps({'state':state,'previous_run':str(previous),
             'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2));temp.replace(status)
    write('waiting_for_final_checkpoint')
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        policy=previous/'policy.zip';norm=previous/'normalize.pkl';summary=previous/'training_summary.json'
        if policy.exists() and norm.exists() and summary.exists():
            try:
                with zipfile.ZipFile(policy) as archive:
                    if archive.testzip() is not None:raise ValueError('Corrupt checkpoint')
                json.loads(summary.read_text());break
            except (ValueError,OSError,zipfile.BadZipFile):pass
        time.sleep(5)
    else:
        write('failed',error='Previous run did not finish within 30 minutes');raise RuntimeError('Checkpoint wait expired')
    command=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--resume',str(policy),
             '--device','cuda','--steps',str(args.steps),'--workers',str(args.workers),
             '--rollout','512','--checkpoint-every','2048']
    old_config=json.loads((previous/'config.json').read_text())
    if old_config.get('randomize'):command.append('--randomize')
    with (runs/'gpu_training.out.log').open('w') as out,(runs/'gpu_training.err.log').open('w') as err:
        child=subprocess.Popen(command,stdout=out,stderr=err,cwd=Path(__file__).resolve().parents[1],
                               creationflags=subprocess.CREATE_NO_WINDOW)
    write('gpu_trainer_launched',pid=child.pid,command=command,starting_timesteps=json.loads(summary.read_text())['timesteps'])


if __name__=='__main__':main()
