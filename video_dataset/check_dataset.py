"""Check media decoding, timing maps, and approval gating without selecting real clips."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from collect import ROOT, probe, save
from review import decide, load_candidates


def main():
    sources=[json.loads(p.read_text(encoding="utf-8")) for p in (ROOT/"catalog/sources").glob("*.json")]
    candidates=load_candidates()
    paths=[ROOT/s["path"] for s in sources]+[ROOT/c["preview"] for c in candidates]
    def decode(path):
        result=subprocess.run(["ffmpeg","-v","error","-xerror","-nostdin","-i",str(path),"-map","0:v:0","-f","null","-"],capture_output=True,text=True)
        if result.returncode:
            raise ValueError(f"Decode failed: {path.name}: {result.stderr[:500]}")
        return path.name
    with ThreadPoolExecutor(4) as pool:
        checked=list(pool.map(decode,paths))
    for c in candidates:
        assert probe(ROOT/c["preview"])["sha256"]==c["media"]["sha256"]
        times=c["source_encoded_frame_times_s"]
        assert len(times)==c["source_frame_range"][1]-c["source_frame_range"][0]
        assert all(b>a for a,b in zip(times,times[1:]))
        assert not c["velocity_supervision_eligible"]
    # Exercise approvals only in an OS temporary directory, never the user's bundle.
    with tempfile.TemporaryDirectory(prefix="darts_review_check_") as directory:
        root=Path(directory)
        clips=[candidates[6],candidates[-1]]
        save(root/"catalog/candidates.json",clips)
        save(root/"selected/manifest.json",{"schema_version":1,"clips":[]})
        for c in clips:
            path=root/c["preview"];path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/c["preview"],path)
        assert not (root/"selected"/clips[0]["id"]).exists()
        decide(clips[0]["id"],"approve",root,actor="isolated_automated_check")
        out=root/"selected"/clips[0]["id"]
        meta=json.loads((out/"metadata.json").read_text(encoding="utf-8"))
        assert meta["gvhmr_media"]["avg_frame_rate"]=="30/1"
        rows=list(csv.DictReader((out/"frame_map.csv").open(encoding="utf-8")))
        assert len(rows)==round(clips[0]["media"]["duration_s"]*30)
        assert max(abs(float(row["resampling_error_s"])) for row in rows)<=.02501
        assert all(clips[0]["source_frame_range"][0]<=int(row["original_source_frame"])<clips[0]["source_frame_range"][1] for row in rows)
        decode(out/"gvhmr_30fps.mp4")
        decide(clips[-1]["id"],"approve",root,actor="isolated_automated_check")
        assert not (root/"selected"/clips[-1]["id"] /"gvhmr_30fps.mp4").exists()
        decide(clips[0]["id"],"reject",root,actor="isolated_automated_check")
        manifest=json.loads((root/"selected/manifest.json").read_text(encoding="utf-8"))
        assert [c["id"] for c in manifest["clips"]]==[clips[-1]["id"]]
        decide(clips[-1]["id"],"pending",root,actor="isolated_automated_check")
        assert not json.loads((root/"selected/manifest.json").read_text(encoding="utf-8"))["clips"]
        try:
            decide("unknown","approve",root)
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown clip accepted")
    report={"status":"passed","source_videos":len(sources),"candidate_clips":len(candidates),
            "fully_decoded_files":len(checked),"approval_tests":"isolated temporary directory; no actual selections made",
            "checks":["complete video decode","candidate hashes","monotonic source frame maps","30 FPS nearest-frame export",
                      "body/reference routing","approval/rejection/reset manifest eligibility","unknown-ID rejection"],
            "limitations":["no automatic full-body pose validation","no capture-speed verification","no confirmed dart separation labels",
                           "exact file hashes only; perceptual/session deduplication incomplete"]}
    save(ROOT/"catalog/validation.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
