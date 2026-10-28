"""Build the first manually inspected review batch; does not approve any clips."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
from PIL import Image

from collect import ROOT, probe, save, sheet


# Times are seconds in each original encoded video, NOT verified physical time.
# Release intervals bracket visible acceleration/follow-through on sparse frames;
# they are review cues, not measured dart separation labels.
BODY = [
    ("hf_vid_1_1_rotated", "A", .8, 2.8, 1.8, 2.25),
    ("hf_vid_1_1_rotated", "A", 3., 5.1, 4.05, 4.95),
    ("hf_vid_1_1_rotated", "A", 5.5, 7.25, 6.3, 6.75),
    ("hf_vid_1_1_rotated", "A", 7.3, 9.05, 8.1, 8.55),
    ("hf_vid_1_1_rotated", "A", 9.05, 10.8, 9.45, 10.35),
    ("hf_vid_1_1_rotated", "A", 10.8, 12.75, 11.7, 12.15),
    ("hf_vid_31_rotated", "B", .6, 2.15, 1.5, 1.8),
    ("hf_vid_31_rotated", "B", 2.3, 3.8, 3., 3.3),
    ("hf_vid_31_rotated", "B", 3.9, 5.45, 4.8, 5.1),
    ("hf_vid_31_rotated", "B", 5.5, 7.1, 6.3, 6.6),
    ("hf_vid_31_rotated", "B", 7.4, 9., 8.4, 8.7),
    ("hf_vid_31_rotated", "B", 9.15, 10.95, 10.2, 10.5),
    ("hf_vid_76_rotated", "C", .55, 1.95, 1.2, 1.5),
    ("hf_vid_76_rotated", "C", 2., 3.6, 2.7, 3.),
    ("hf_vid_76_rotated", "C", 3.7, 5.25, 4.5, 4.8),
    ("hf_vid_76_rotated", "C", 5.4, 7.1, 6., 6.3),
    ("hf_vid_76_rotated", "C", 7.2, 8.9, 7.8, 8.1),
    ("hf_vid_76_rotated", "C", 9., 10.8, 9.6, 9.9),
    ("hf_vid_16_1_rotated", "D", 38.2, 39.9, 39.15, 39.4),
    ("hf_vid_16_1_rotated", "D", 34.2, 37.1, 36., 36.6),
]


def frame_times(path):
    result = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
                             "frame=best_effort_timestamp_time","-of","json",str(path)],
                            capture_output=True,text=True,check=True)
    return [float(f["best_effort_timestamp_time"]) for f in json.loads(result.stdout)["frames"]]


def candidates():
    for i,(source,group,start,end,lo,hi) in enumerate(BODY,1):
        yield {"id":f"D{i:02}","source_id":source,"group":group,"start_s":start,"end_s":end,
               "release_window_source_s":[lo,hi],"title":f"Full-body throw · appearance group {group} · take {i}",
               "category":"body_motion", "recommendation":"review_carefully" if group=="D" else "recommended_for_body_reference",
               "notes":("Full body visible; check throwing hand near right frame boundary through release." if group=="D" else
                        "Feet, torso and throwing arm visible in inspected frames. Fixed front/oblique camera; arm foreshortening and release blur."),
               "viewpoint":"front_oblique_high_camera", "timing":"appears normal speed; capture rate and editing unverified"}
    details = [
        ("R01","yt_KV2XqzIJDgw",64.,85.,"Grip demonstration", "Hand close-up. Useful for qualitative grip design; not a throw or a GVHMR body reference."),
        ("R02","yt_KV2XqzIJDgw",320.,330.,"Side-on slow-motion throw", "Throwing shoulder/elbow/wrist visible; legs cropped. Slow-motion factor unknown: exclude from velocity supervision."),
        ("R03","yt_KdbdYeST2fE",3.,10.,"Gary Anderson · slow-motion throw", "Multiple people, legs cropped, background dartboards. Slow-motion factor unknown; qualitative arm/release reference only."),
        ("R04","yt_KdbdYeST2fE",41.5,47.,"Gary Anderson · follow-through reference", "Upper-body view; knees/feet missing and recording speed unverified. Reference-only, not GVHMR full-body input."),
    ]
    for cid,source,start,end,title,notes in details:
        yield {"id":cid,"source_id":source,"group":"reference","start_s":start,"end_s":end,
               "release_window_source_s":None,"title":title,"category":"release_detail",
               "recommendation":"reference_only","notes":notes,"viewpoint":"side_upper_body_or_hand",
               "timing":"unknown or slow motion; no physical velocity supervision"}


def main():
    clips = []
    for candidate in candidates():
        cid = candidate["id"]
        source = json.loads((ROOT/"catalog/sources"/(candidate["source_id"]+".json")).read_text(encoding="utf-8"))
        path = ROOT/source["path"]
        pts_path = ROOT/"catalog/frame_times"/(candidate["source_id"]+".json")
        if not pts_path.exists():
            save(pts_path,frame_times(path))
        pts = json.loads(pts_path.read_text(encoding="utf-8"))
        indices = [i for i,t in enumerate(pts) if candidate["start_s"] <= t < candidate["end_s"]]
        if not indices:
            raise ValueError(f"Empty trim {cid}")
        first,last = indices[0],indices[-1]+1
        out = ROOT/"candidates"/cid
        out.mkdir(parents=True,exist_ok=True)
        preview = out/"preview.mp4"
        old_meta = json.loads((out/"metadata.json").read_text(encoding="utf-8")) if (out/"metadata.json").exists() else {}
        if not preview.exists() or old_meta.get("source_frame_range") != [first,last] or old_meta.get("source_id") != candidate["source_id"]:
            subprocess.run(["ffmpeg","-v","error","-nostdin","-y","-i",str(path),"-vf",
                            f"trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS",
                            "-an","-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",
                            "-fps_mode","passthrough","-movflags","+faststart",str(preview)],check=True)
        media = probe(preview)
        cap = cv2.VideoCapture(str(preview))
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(indices):
            raise ValueError(f"Frame-preserving trim failed: {cid}")
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,len(indices)//2))
        ok,frame=cap.read(); cap.release()
        if not ok:
            raise ValueError(f"Cannot decode {cid}")
        Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(out/"poster.jpg")
        sheet(preview,out/"contact_sheet.jpg")
        candidate.update({"source_url":source["source_url"],"creator":source["creator"],"source_sha256":source["media"]["sha256"],
                          "origin":"internet","license":source["license"],"redistribution":source["redistribution"],
                          "source_revision":source.get("source_revision"),"preview":preview.relative_to(ROOT).as_posix(),
                          "poster":(out/"poster.jpg").relative_to(ROOT).as_posix(),"media":media,
                          "source_frame_range":[first,last],"source_time_range_s":[pts[first],pts[last-1]],
                          "source_encoded_frame_times_s":[pts[i] for i in indices],
                          "handedness":"unverified; mirror status unknown", "impact":"not visible/attributable; target unknown",
                          "thrower_identity":"unverified; appearance group only", "leakage_group":"hf_dlc_all" if cid.startswith("D") else candidate["source_id"],
                          "velocity_supervision_eligible":False,"release_annotation":"coarse visual cue; dart separation not confirmed",
                          "inspection":"manual contact-sheet review, not per-frame pose validation", "status":"pending_user_review"})
        save(out/"metadata.json",candidate)
        clips.append(candidate)
        print(f"{cid}: {len(indices)} frames, {media['duration_s']:.2f}s",flush=True)
    save(ROOT/"catalog/candidates.json",clips)
    by_hash = defaultdict(list)
    for path in (ROOT/"catalog/sources").glob("*.json"):
        source=json.loads(path.read_text(encoding="utf-8"))
        by_hash[source["media"]["sha256"]].append(source["id"])
    save(ROOT/"catalog/deduplication.json",{"method":"exact SHA256; perceptual/reupload and session deduplication remains necessary",
         "exact_duplicate_groups":[ids for ids in by_hash.values() if len(ids)>1]})
    selected = ROOT/"selected"
    selected.mkdir(exist_ok=True)
    if not (selected/"manifest.json").exists():
        save(selected/"manifest.json",{"schema_version":1,"clips":[],"note":"Only explicit human approvals enter this bundle."})


if __name__ == "__main__":
    main()
