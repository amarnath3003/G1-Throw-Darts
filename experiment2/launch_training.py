"""Launch a persistent, hidden Windows trainer with durable logs and PID data."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('trainer_args',nargs=argparse.REMAINDER)
    args=parser.parse_args();tail=args.trainer_args
    if tail and tail[0]=='--':tail=tail[1:]
    here=Path(__file__).resolve().parent
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');logs=here/'runs'
    out=logs/f'training_{stamp}.out.log';err=logs/f'training_{stamp}.err.log'
    command=[sys.executable,'-u',str(here/'train.py'),*tail]
    with out.open('w') as stdout,err.open('w') as stderr:
        process=subprocess.Popen(command,cwd=here.parent,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    status={'pid':process.pid,'command':command,'stdout':str(out),'stderr':str(err),
        'created_utc':datetime.now(timezone.utc).isoformat()}
    (logs/f'launch_{stamp}.json').write_text(json.dumps(status,indent=2))
    temp=logs/'active_training.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(logs/'active_training.json')
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
