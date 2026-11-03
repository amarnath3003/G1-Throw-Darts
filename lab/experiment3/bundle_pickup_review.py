"""Bundle reconstruction outputs only, without videos, weights or credentials."""
import argparse
import tarfile
from pathlib import Path
from pickup_common import read_json,sha256,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    files=list((a.dataset/'milestone2').rglob('*'))
    for clip in read_json(a.dataset/'manifest.json')['clips']:
        files+=list((a.dataset/clip['id']/'hamer_v2').glob('*'))
    files+=[a.dataset/'inference_runtime.json',a.dataset/'mano_faces_left.npy',a.dataset/'reconstruction_packages.txt']
    files=sorted(f for f in files if f.is_file())
    write_json(a.dataset/'milestone2_artifact_hashes.json',{f.relative_to(a.dataset).as_posix():sha256(f) for f in files})
    files.append(a.dataset/'milestone2_artifact_hashes.json')
    out=a.dataset/'milestone2_review.tar.gz'
    with tarfile.open(out,'w:gz',compresslevel=1) as archive:
        for f in files:archive.add(f,arcname=f.relative_to(a.dataset).as_posix())
    print('Review bundle',out.name,out.stat().st_size,'bytes',sha256(out),flush=True)
