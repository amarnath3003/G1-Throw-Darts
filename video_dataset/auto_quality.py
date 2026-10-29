"""Screen downloaded footage and recommend body clips for one batch decision.

Pose visibility is a triage heuristic, not proof of correct 3-D reconstruction,
physical timing, finger separation, or permission to use the footage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from collect import ROOT, save

WEIGHTS=ROOT.parents[1]/"video_to_g1/yolo11n-pose.pt"
RULES={"version":1,"keypoint_confidence":.35,"minimum_body_visible_fraction":.85,
       "minimum_wrist_visible_fraction":.85,"minimum_wrist_inside_fraction":.95,
       "maximum_multiple_people_fraction":.05,"minimum_median_person_height_pixels":250,
       "maximum_cut_count":0,"border_margin_fraction":.015}


def inspect(path,model,sample_every=1):
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames=[];rows=[];previous=None;cuts=0;index=0

    def infer():
        nonlocal frames
        if not frames:
            return
        results=model.predict(frames,classes=[0],conf=.3,imgsz=640,verbose=False,device=0)
        for result in results:
            row={"body":False,"heights":0.,"multiple":False,"xy":np.zeros((17,2)),"conf":np.zeros(17)}
            if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
                boxes=result.boxes.xyxy.cpu().numpy()
                areas=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
                chosen=int(np.argmax(areas))
                points=result.keypoints.xy[chosen].cpu().numpy()
                confidence=result.keypoints.conf[chosen].cpu().numpy()
                row.update(xy=points,conf=confidence,heights=float(boxes[chosen,3]-boxes[chosen,1]),
                           multiple=bool(np.count_nonzero(areas>.3*areas[chosen])>1))
                body=[5,6,11,12,15,16]
                row["body"]=bool(np.all(confidence[body]>=RULES["keypoint_confidence"]))
            rows.append(row)
        frames=[]

    while True:
        ok,frame=cap.read()
        if not ok:
            break
        if index%sample_every==0:
            thumb=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(64,64)).astype(float)/255
            if previous is not None and np.mean(np.abs(thumb-previous))>.18:
                cuts+=1
            previous=thumb
            frames.append(frame)
            if len(frames)==16:
                infer()
        index+=1
    infer();cap.release()
    if not rows:
        raise ValueError(f"No decoded frames: {path}")
    xy=np.array([r["xy"] for r in rows]);conf=np.array([r["conf"] for r in rows])
    excursions=[]
    for wrist in (9,10):
        visible=conf[:,wrist]>=RULES["keypoint_confidence"]
        points=xy[visible,wrist]
        excursions.append(float(np.linalg.norm(np.ptp(points,axis=0))) if len(points)>1 else 0.)
    wrist=(9,10)[int(np.argmax(excursions))]
    good=conf[:,wrist]>=RULES["keypoint_confidence"]
    margin=RULES["border_margin_fraction"]
    inside=good&(xy[:,wrist,0]>width*margin)&(xy[:,wrist,0]<width*(1-margin))&(xy[:,wrist,1]>height*margin)&(xy[:,wrist,1]<height*(1-margin))
    metrics={"sampled_frames":len(rows),"decoded_frames":index,"sampling_every_n_frames":sample_every,
             "body_visible_fraction":float(np.mean([r["body"] for r in rows])),
             "wrist_visible_fraction":float(np.mean(good)),"wrist_inside_fraction":float(np.mean(inside)),
             "multiple_people_fraction":float(np.mean([r["multiple"] for r in rows])),
             "median_person_height_pixels":float(np.median([r["heights"] for r in rows])),
             "estimated_throwing_wrist_coco_index":wrist,"cut_count":cuts,"encoded_fps":fps}
    reasons=[]
    for name,limit in [("body_visible_fraction",.85),("wrist_visible_fraction",.85),("wrist_inside_fraction",.95),
                       ("median_person_height_pixels",250)]:
        if metrics[name]<limit:
            reasons.append(name+"_low")
    if metrics["multiple_people_fraction"]>.05:
        reasons.append("multiple_people")
    if cuts:
        reasons.append("possible_cut_or_camera_change")
    return {"pass":not reasons,"reasons":reasons,"metrics":metrics}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--screen-archive",action="store_true")
    args=parser.parse_args()
    model=YOLO(str(WEIGHTS))
    model_hash=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    candidates=json.loads((ROOT/"catalog/candidates.json").read_text(encoding="utf-8"))
    records=[]
    for c in candidates:
        if c["category"]!="body_motion":
            result={"pass":False,"reasons":["qualitative_reference_only"],"metrics":{}}
        else:
            result=inspect(ROOT/c["preview"],model)
            if c["recommendation"]!="recommended_for_body_reference":
                result["pass"]=False;result["reasons"].append("manual_inspection_caution")
        result.update(id=c["id"],candidate_sha256=c["media"]["sha256"],group=c["group"])
        records.append(result)
        print(c["id"],"recommend" if result["pass"] else "hold",result["reasons"],flush=True)
    # Recommend one representative per appearance group, plus weakest passing and
    # one held example. The user can inspect these few rather than every clip.
    passing=[r for r in records if r["pass"]]
    spot=[]
    for group in dict.fromkeys(r["group"] for r in passing):
        choices=[r for r in passing if r["group"]==group]
        spot.append(max(choices,key=lambda r:r["metrics"]["wrist_visible_fraction"])["id"])
    if passing:
        spot.append(min(passing,key=lambda r:r["metrics"]["wrist_inside_fraction"])["id"])
    held=[r for r in records if not r["pass"] and r["group"]!="reference"]
    if held:
        spot.append(held[0]["id"])
    report={"rules":RULES,"model_sha256":model_hash,"clips":records,"spot_check_ids":list(dict.fromkeys(spot)),
            "recommended_ids":[r["id"] for r in passing],"scope":"existing curated candidates; archive screen does not segment new throws",
            "limitations":["2D pose confidence is heuristic","no capture-speed verification","no finger/release or 3D quality validation",
                           "wrist choice based on motion, not verified handedness","no automatic dataset approval"]}
    save(ROOT/"catalog/auto_quality.json",report)
    if args.screen_archive:
        sources=[json.loads(p.read_text(encoding="utf-8")) for p in (ROOT/"catalog/sources").glob("*.json")]
        archive=[]
        for i,s in enumerate(sources):
            cache=ROOT/"catalog/archive_quality"/(s["id"]+".json")
            if cache.exists():
                r=json.loads(cache.read_text(encoding="utf-8"))
                if r.get("sha256")!=s["media"]["sha256"] or r.get("model_sha256")!=model_hash:
                    r=None
            else:
                r=None
            if r is None:
                fps=float(s["media"]["avg_frame_rate"].split('/')[0])/float(s["media"]["avg_frame_rate"].split('/')[1])
                r=inspect(ROOT/s["path"],model,sample_every=max(1,round(fps)))
                r.update(id=s["id"],sha256=s["media"]["sha256"],model_sha256=model_hash,
                         note="Sparse 1 FPS source screen only. Full-frame throw segmentation/QA still needed.")
                save(cache,r)
            archive.append(r)
            print(f"archive {i+1}/{len(sources)}: {s['id']}",flush=True)
        save(ROOT/"catalog/archive_quality.json",{"sources":archive,"counts":dict(Counter("promising" if r["pass"] else "held" for r in archive)),
             "note":"Sparse source screening. Passing does not certify every throw or approve any source."})


if __name__=="__main__":
    main()
