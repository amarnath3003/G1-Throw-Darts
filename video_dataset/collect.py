"""Acquire public darts references and retain timing/provenance for review."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageOps
import requests

ROOT = Path(__file__).resolve().parent
REPO = "pratikshapai/human-motion-tracking-deeplabcut"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    raw = json.loads(result.stdout)
    video = next(s for s in raw["streams"] if s["codec_type"] == "video")
    return {"codec": video["codec_name"], "width": video["width"], "height": video["height"],
            "avg_frame_rate": video["avg_frame_rate"], "r_frame_rate": video["r_frame_rate"],
            "time_base": video["time_base"], "start_time": video.get("start_time"),
            "duration_s": float(raw["format"]["duration"]), "frames": video.get("nb_frames"),
            "side_data": video.get("side_data_list", []), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sheet(path, output, count=12):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    canvas = Image.new("RGB", (4 * 320, 3 * 264), "#131923")
    draw = ImageDraw.Draw(canvas)
    for i in range(count):
        frame_number = round((n - 1) * i / (count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Undecodable frame {frame_number} in {path}")
        picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        picture = ImageOps.contain(picture, (320, 236))
        x, y = (i % 4) * 320, (i // 4) * 264
        canvas.paste(picture, (x + (320-picture.width)//2, y + (236-picture.height)//2))
        draw.text((x+8,y+240), f"{path.stem}  {frame_number / fps:.3f}s", fill="white")
    cap.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def dense(path, start=0., step=.4):
    cap = cv2.VideoCapture(str(path))
    canvas = Image.new("RGB", (1536, 1260), "#131923")
    draw = ImageDraw.Draw(canvas)
    for i in range(36):
        t = start + i * step
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
        ok, frame = cap.read()
        if not ok:
            break
        picture = ImageOps.contain(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(256,185))
        x,y = (i%6)*256,(i//6)*210
        canvas.paste(picture,(x+(256-picture.width)//2,y))
        draw.text((x+8,y+190),f"{t:.3f}s",fill="white")
    cap.release()
    output = ROOT/"contact_sheets"/f"dense_{path.stem}_{start:g}.jpg"
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output)
    print(output)


def youtube_index():
    for path in (ROOT/"sources/youtube").glob("*.mp4"):
        info = json.loads(path.with_suffix(".info.json").read_text(encoding="utf-8"))
        record = {"id":"yt_"+path.stem,"source_url":info["webpage_url"],"creator":info.get("uploader"),
                  "title":info["title"], "origin":"internet", "retrieved_at":datetime.now(timezone.utc).isoformat(),
                  "path":path.relative_to(ROOT).as_posix(),"license":info.get("license") or "not declared in metadata",
                  "redistribution":"unverified; local reference only", "playback_speed":"mixed or unverified",
                  "capture_fps":None,"source_session":"yt_"+path.stem,"thrower":"unresolved",
                  "status":"awaiting_inspection","media":probe(path)}
        save(ROOT/"catalog/sources"/(record["id"]+".json"),record)
        sheet(path,ROOT/"contact_sheets"/(record["id"]+".jpg"))
        print(record["id"],record["title"],flush=True)


def hf(all_files=False):
    api = f"https://huggingface.co/api/datasets/{REPO}"
    meta = requests.get(api, timeout=30); meta.raise_for_status(); meta = meta.json()
    revision = meta["sha"]
    response = requests.get(f"{api}/tree/{revision}/videos?limit=1000", timeout=30)
    response.raise_for_status()
    listing = [x for x in response.json() if x["path"].endswith(".mp4") and not Path(x["path"]).name.startswith(".")]
    listing.sort(key=lambda x: int(Path(x["path"]).stem.split("_")[1]))
    save(ROOT/"catalog/hf_inventory.json", {"repo":REPO,"revision":revision,"listing":listing,
         "dataset_card":meta.get("cardData"), "retrieved_at":datetime.now(timezone.utc).isoformat(),
         "rights":"No license declared in dataset card. Permissions unverified; exclude media from public redistribution."})
    selected = listing if all_files else [listing[i] for i in range(0,len(listing),15)]

    def get(item):
        path = ROOT/"sources/hf_dlc"/Path(item["path"]).name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item['path']}"
        if not path.exists() or path.stat().st_size != item["size"]:
            response = requests.get(url,timeout=120); response.raise_for_status()
            if len(response.content) != item["size"]:
                raise ValueError(f"Wrong file size: {item['path']}")
            temporary = path.with_suffix(".part"); temporary.write_bytes(response.content); temporary.replace(path)
        record = {"id":f"hf_{path.stem}", "source_url":f"https://huggingface.co/datasets/{REPO}/blob/{revision}/{item['path']}",
                  "download_url":url, "creator":"pratikshapai (dataset uploader; camera operator/subjects unidentified)",
                  "retrieved_at":datetime.now(timezone.utc).isoformat(), "path":path.relative_to(ROOT).as_posix(),
                  "source_revision":revision,"origin":"internet", "license":"not declared", "redistribution":"unverified; local reference only",
                  "playback_speed":"unverified", "capture_fps":None, "source_session":"hf_dlc_unresolved",
                  "thrower":"unresolved", "status":"awaiting_inspection", "media":probe(path)}
        save(ROOT/"catalog/sources"/(record["id"]+".json"),record)
        sheet(path,ROOT/"contact_sheets"/(record["id"]+".jpg"))
        print(record["id"], record["media"]["duration_s"], flush=True)
        return record["id"]

    with ThreadPoolExecutor(4) as pool:
        records = list(pool.map(get,selected))
    print(f"Acquired/probed {len(records)} of {len(listing)} videos",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("hf"); p.add_argument("--all",action="store_true")
    sub.add_parser("youtube-index")
    p = sub.add_parser("dense"); p.add_argument("path",type=Path);p.add_argument("--start",type=float,default=0.);p.add_argument("--step",type=float,default=.4)
    args = parser.parse_args()
    if args.command == "hf":
        hf(args.all)
    elif args.command == "youtube-index":
        youtube_index()
    elif args.command == "dense":
        dense(args.path,args.start,args.step)
