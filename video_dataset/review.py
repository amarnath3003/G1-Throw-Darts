"""Local review gallery. Human button clicks approve clips into selected/."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
from urllib.parse import urlparse

import cv2

from collect import ROOT, probe, save

LOCK = threading.Lock()
TOKEN = secrets.token_urlsafe(32)


def load_candidates(root=ROOT):
    return json.loads((root/"catalog/candidates.json").read_text(encoding="utf-8"))


def decisions(root=ROOT):
    path=root/"catalog/decisions.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def export_30fps(clip,output):
    """Nearest source-frame resampling with an explicit frame/time map; no interpolation."""
    cap=cv2.VideoCapture(str(clip))
    fps=cap.get(cv2.CAP_PROP_FPS)
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pts_result=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
                               "frame=best_effort_timestamp_time","-of","json",str(clip)],capture_output=True,text=True,check=True)
    pts=[float(f["best_effort_timestamp_time"]) for f in json.loads(pts_result.stdout)["frames"]]
    if not fps>0 or len(pts)!=count or any(b<=a for a,b in zip(pts,pts[1:])):
        raise ValueError("Invalid or nonmonotonic video timing")
    duration=probe(clip)["duration_s"]
    mapping=[]
    process=subprocess.Popen(["ffmpeg","-v","error","-nostdin","-y","-f","rawvideo","-pix_fmt","bgr24",
                              "-s",f"{width}x{height}","-r","30","-i","pipe:0","-an","-c:v","libx264",
                              "-preset","fast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(output)],stdin=subprocess.PIPE)
    source_index=-1
    current=None
    try:
        for j in range(round(duration*30)):
            t=j/30
            k=bisect_left(pts,t)
            if k==len(pts):
                k-=1
            elif k>0 and abs(pts[k-1]-t)<=abs(pts[k]-t):
                k-=1
            while source_index<k:
                ok,current=cap.read(); source_index+=1
                if not ok:
                    raise ValueError("Decode failed during export")
            process.stdin.write(current.tobytes())
            mapping.append({"output_frame":j,"output_time_s":t,"clip_frame":k,"clip_encoded_time_s":pts[k],"resampling_error_s":pts[k]-t})
    finally:
        process.stdin.close(); cap.release()
        return_code=process.wait()
    if return_code:
        raise RuntimeError("30 FPS export failed")
    return mapping


def decide(cid,decision,root=ROOT,actor="human_gallery"):
    if decision not in {"approve","reject","pending"}:
        raise ValueError("Unknown decision")
    candidate=next((c for c in load_candidates(root) if c["id"]==cid),None)
    if candidate is None:
        raise ValueError("Unknown candidate")
    bundle=root/"selected"
    bundle.mkdir(exist_ok=True)
    manifest_path=bundle/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    record={"decision":decision,"actor":actor,"at":datetime.now(timezone.utc).isoformat(),
            "candidate_sha256":candidate["media"]["sha256"]}
    if decision=="approve":
        # Build off to the side; publish the manifest only after a complete export.
        staging=bundle/(cid+".staging")
        staging.mkdir(exist_ok=True)
        original=root/candidate["preview"]
        if probe(original)["sha256"]!=candidate["media"]["sha256"]:
            raise ValueError("Candidate changed; re-inspect before approval")
        shutil.copy2(original,staging/"native.mp4")
        meta=dict(candidate)
        meta["approval"]=record
        meta["bundle_role"]="gvhmr_body_reference" if candidate["category"]=="body_motion" else "qualitative_reference_only"
        if candidate["category"]=="body_motion":
            mapping=export_30fps(original,staging/"gvhmr_30fps.mp4")
            source_times=candidate["source_encoded_frame_times_s"]
            for row in mapping:
                row["original_source_frame"]=candidate["source_frame_range"][0]+row["clip_frame"]
                row["original_source_encoded_time_s"]=source_times[row["clip_frame"]]
            with (staging/"frame_map.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(mapping[0]));writer.writeheader();writer.writerows(mapping)
            meta["gvhmr_media"]=probe(staging/"gvhmr_30fps.mp4")
            meta["resampling"]="nearest original frame at 30 FPS; duplicated frames add no motion information; physical playback speed remains unverified"
        save(staging/"metadata.json",meta)
        destination=bundle/cid
        if destination.exists():
            # Repeat approval updates only this known candidate directory.
            if destination.is_symlink() or destination.resolve().parent!=bundle.resolve() or destination.name!=cid:
                raise ValueError("Unsafe bundle destination")
            shutil.rmtree(destination)
        staging.replace(destination)
        entry={"id":cid,"role":meta["bundle_role"],"path":f"{cid}/metadata.json","approval":record}
        manifest["clips"]=[c for c in manifest["clips"] if c["id"]!=cid]+[entry]
    else:
        # Revocation removes eligibility from the manifest; retained files are archival.
        manifest["clips"]=[c for c in manifest["clips"] if c["id"]!=cid]
    save(manifest_path,manifest)
    state=decisions(root);state[cid]=record;save(root/"catalog/decisions.json",state)
    return record


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(ROOT),**kwargs)

    def json_response(self,value,status=200):
        payload=json.dumps(value,ensure_ascii=False).encode("utf-8")
        self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(payload)))
        self.end_headers();self.wfile.write(payload)

    def do_GET(self):
        if self.path=="/api/state":
            return self.json_response({"candidates":load_candidates(),"decisions":decisions(),"token":TOKEN})
        if urlparse(self.path).path in {"/","/index.html"}:
            self.path="/review.html"
        # Serve only review artifacts; never enumerate directories or expose raw .info
        # JSON (which includes expiring CDN URLs), Python sources, or arbitrary paths.
        request_path=urlparse(self.path).path
        if request_path!="/review.html" and not request_path.startswith("/candidates/"):
            return self.send_error(404)
        resolved=Path(self.translate_path(request_path)).resolve()
        allowed=(ROOT/"candidates").resolve()
        if request_path!="/review.html" and (not resolved.is_relative_to(allowed) or resolved.suffix not in {".mp4",".jpg"}):
            return self.send_error(404)
        super().do_GET()

    def do_POST(self):
        expected=f"http://{self.headers.get('Host')}"
        if self.path!="/api/decision" or self.headers.get("Origin")!=expected or self.headers.get("X-Review-Token")!=TOKEN:
            return self.json_response({"error":"Open this local gallery to make a review decision."},403)
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<4096:
                raise ValueError("Invalid request size")
            payload=json.loads(self.rfile.read(length))
            with LOCK:
                record=decide(payload["id"],payload["decision"])
            self.json_response(record)
        except Exception as e:
            self.json_response({"error":str(e)},400)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--port",type=int,default=8766)
    args=parser.parse_args()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    print(f"Darts review: http://127.0.0.1:{args.port} (loopback only)",flush=True)
    server.serve_forever()
