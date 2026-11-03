"""Fetch ONLY the already pinned public reconstruction weights, verify SHA256."""
import requests
from pickup_common import HERE,read_json,sha256


def fetch(url,path,expected):
    if path.exists() and sha256(path)==expected:
        print('Verified existing',path.name,flush=True);return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.download')
    with requests.get(url,stream=True,timeout=(30,90)) as r:
        r.raise_for_status();size=0;mark=0
        with temporary.open('wb') as f:
            for chunk in r.iter_content(8*1024**2):
                f.write(chunk);size+=len(chunk)
                if size-mark>=128*1024**2:print(path.name,round(size/1e6),'MB',flush=True);mark=size
    if sha256(temporary)!=expected:raise ValueError('Downloaded hash mismatch: '+path.name)
    temporary.replace(path);print('Verified',path.name,flush=True)


if __name__=='__main__':
    asset=read_json(HERE/'configs/hamer_assets.json')['files'][0]
    fetch(f'https://huggingface.co/spaces/{asset["repo"]}/resolve/{asset["revision"]}/_DATA/hamer_ckpts/checkpoints/hamer.ckpt',
          HERE/'models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt',asset['sha256'])
    asset=next(a for a in read_json(HERE/'configs/wilor_assets.json')['files'] if a['file'].endswith('detector.pt'))
    fetch(f'https://huggingface.co/spaces/{asset["repo"]}/resolve/{asset["revision"]}/pretrained_models/detector.pt',
          HERE/'models/pretrained_models/detector.pt',asset['sha256'])
