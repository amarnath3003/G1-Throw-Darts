"""Pin HaMeR assets from the authors' public Space; resumable HTTP ranges."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from huggingface_hub import HfApi, hf_hub_download

HERE=Path(__file__).resolve().parent
OUT=HERE/'models/hamer'
OUT.mkdir(parents=True,exist_ok=True)
repo='geopavlakos/HaMeR'
info=HfApi().space_info(repo,files_metadata=True)
name='_DATA/hamer_ckpts/checkpoints/hamer.ckpt'
asset=next(s for s in info.siblings if s.rfilename==name)
target=OUT/name
target.parent.mkdir(parents=True,exist_ok=True)
if not target.exists():
    partial=target.with_suffix('.partial')
    progress=target.with_suffix('.ranges.json')
    completed=set(json.loads(progress.read_text())) if progress.exists() and partial.exists() else set()
    block=16*1024*1024
    count=(asset.size+block-1)//block
    def fetch(index):
        start=index*block;end=min(start+block,asset.size)-1
        url=f'https://huggingface.co/spaces/{repo}/resolve/{info.sha}/{name}?range_start={start}'
        for attempt in range(5):
            try:
                response=requests.get(url+f'&retry={attempt}',headers={'Range':f'bytes={start}-{end}'},timeout=(10,20))
                response.raise_for_status()
                assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{asset.size}'
                assert len(response.content)==end-start+1
                return index,response.content
            except Exception as error:
                print(f'Retry range {index}: {type(error).__name__}',flush=True)
                if attempt==4:raise
                time.sleep(1)
    with partial.open('r+b' if partial.exists() else 'w+b') as f, ThreadPoolExecutor(max_workers=4) as pool:
        # Only four outstanding buffers, so responses cannot accumulate in RAM.
        pending=iter(i for i in range(count) if i not in completed)
        futures={pool.submit(fetch,i) for i in [next(pending,None) for _ in range(4)] if i is not None}
        while futures:
            future=next(as_completed(futures));futures.remove(future)
            index,data=future.result()
            f.seek(index*block);f.write(data);f.flush()
            completed.add(index);progress.write_text(json.dumps(sorted(completed)))
            print(f'Checkpoint ranges {len(completed)}/{count}',flush=True)
            nxt=next(pending,None)
            if nxt is not None:futures.add(pool.submit(fetch,nxt))
    partial.rename(target)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

assert target.stat().st_size==asset.size and digest(target)==asset.lfs['sha256'],'Checkpoint failed official SHA check'
records=[dict(file=str(target.relative_to(HERE)),sha256=digest(target),repo=repo,revision=info.sha)]
for file in ['_DATA/hamer_ckpts/model_config.yaml','_DATA/data/mano_mean_params.npz']:
    path=Path(hf_hub_download(repo_id=repo,repo_type='space',revision=info.sha,filename=file,local_dir=OUT))
    records.append(dict(file=str(path.relative_to(HERE)),sha256=digest(path),repo=repo,revision=info.sha))
mano=OUT/'_DATA/data/mano/MANO_RIGHT.pkl'
mano.parent.mkdir(parents=True,exist_ok=True)
source=HERE/'models/pretrained_models/MANO_RIGHT.pkl'
shutil.copyfile(source,mano)
records.append(dict(file=str(mano.relative_to(HERE)),sha256=digest(mano),reused_from=str(source.relative_to(HERE))))
commit=__import__('subprocess').check_output(['git','-C',str(HERE/'vendor/hamer'),'rev-parse','HEAD'],text=True).strip()
(HERE/'configs/hamer_assets.json').write_text(json.dumps(dict(implementation_commit=commit,files=records),indent=2)+'\n')
print('HaMeR assets verified.',flush=True)
