"""Cache public WiLoR demo assets locally with revision and file hashes."""
import hashlib
import json
import os
import time
import requests
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi, hf_hub_download

HERE=Path(__file__).resolve().parent
repo='warmshao/WiLoR-mini'
revision=HfApi().model_info(repo).sha
official_repo='rolpotamias/WiLoR'
official_revision=HfApi().space_info(official_repo).sha
records=[]
def ranged_checkpoint(revision):
    """Bounded HTTP ranges avoid CDN timeouts on one 2.56 GB response."""
    folder=HERE/'models/pretrained_models'
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'wilor_final.ckpt'
    expected=next(s for s in HfApi().space_info(official_repo,revision=revision,files_metadata=True).siblings if s.rfilename=='pretrained_models/wilor_final.ckpt')
    if target.exists() and target.stat().st_size==expected.size:
        return target
    partial=folder/'wilor_final.ckpt.partial'
    existing=list((HERE/'models/.cache/huggingface/download/pretrained_models').glob('*.incomplete'))
    if not partial.exists() and len(existing)==1:
        existing[0].rename(partial)
    url=f'https://huggingface.co/spaces/{official_repo}/resolve/{revision}/pretrained_models/wilor_final.ckpt'
    with partial.open('ab') as output:
        position=output.tell()
        while position<expected.size:
            end=min(position+16*1024*1024,expected.size)-1
            for attempt in range(5):
                try:
                    response=requests.get(url+f'?range_start={position}',headers={'Range':f'bytes={position}-{end}'},timeout=(15,30))
                    response.raise_for_status()
                    assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {position}-{end}/{expected.size}'
                    assert len(response.content)==end-position+1
                    output.write(response.content)
                    output.flush()
                    position=end+1
                    print(f'Checkpoint {position/expected.size:.1%} ({position} bytes)',flush=True)
                    break
                except Exception:
                    if attempt==4:raise
                    time.sleep(1)
    digest=hashlib.sha256()
    with partial.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==expected.lfs['sha256'],'Checkpoint upstream SHA mismatch'
    partial.rename(target)
    return target

for name in ['mano_mean_params.npz','MANO_RIGHT.pkl','wilor_final.ckpt','detector.pt']:
    print('Downloading',name,flush=True)
    official=name in ('wilor_final.ckpt','detector.pt')
    source_repo=official_repo if official else repo
    source_revision=official_revision if official else revision
    path=ranged_checkpoint(source_revision) if name=='wilor_final.ckpt' else Path(hf_hub_download(repo_id=source_repo,repo_type='space' if official else 'model',revision=source_revision,filename='pretrained_models/'+name,local_dir=HERE/'models'))
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(block)
    records.append(dict(file=str(path.relative_to(HERE)),bytes=path.stat().st_size,sha256=digest.hexdigest(),repo=source_repo,revision=source_revision))
(HERE/'configs/wilor_assets.json').write_text(json.dumps(dict(repo=repo,revision=revision,files=records,
    implementation_repo='https://github.com/warmshao/WiLoR-mini',
    implementation_commit='ebec42f94c389070cdd7dda6fd1bf0b4a659c960',
    purpose='Local demonstration pilot; not a redistribution of model assets',
    terms=['https://github.com/rolpotamias/WiLoR/blob/main/license.txt','https://mano.is.tue.mpg.de/license.html']),indent=2)+'\n')
print('Assets downloaded and hashed.',flush=True)
