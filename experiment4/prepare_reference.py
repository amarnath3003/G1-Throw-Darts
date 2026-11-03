"""Prepare the reviewed left-arm throw from IMG_4281.MOV."""
import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from bisect import bisect_left
from pathlib import Path

from PIL import Image, ImageDraw

SOURCE = Path(__file__).resolve().parents[1] / "G1 Darts Ref videos/IMG_4281.MOV"
FIRST, LAST = 100, 254
FIRST_WINDUP, FIRST_PEAK, SECOND_AIM = 115, 150, 172
WINDUP, ACCELERATION, RELEASE_FIRST, RELEASE_LAST = 179, 218, 224, 226
WIDTH, HEIGHT = 960, 540


def probe(path, entries):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", entries, "-of", "json", str(path),
    ]))


def write_video(path, size, fps, frames):
    args = [
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as process:
        try:
            for frame in frames:
                process.stdin.write(frame)
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError(f"Encoding failed: {path}")


def nearest_frame(times, target):
    right = min(bisect_left(times, target), len(times) - 1)
    left = max(0, right - 1)
    return min((left, right), key=lambda index: abs(times[index] - target))


def phase(index):
    if index < FIRST_WINDUP:
        return "Aim"
    if index < FIRST_PEAK:
        return "Swing back 1"
    if index < SECOND_AIM:
        return "Forward reset"
    if index < WINDUP:
        return "Aim"
    if index < ACCELERATION:
        return "Swing back 2"
    if index < RELEASE_FIRST:
        return "Acceleration"
    if index <= RELEASE_LAST:
        return "Estimated release window"
    return "Follow-through"


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    checksum = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    timings = probe(SOURCE, "frame=best_effort_timestamp_time")["frames"]
    times = [float(frame["best_effort_timestamp_time"]) for frame in timings]
    selected = times[FIRST:LAST + 1]
    if len(selected) != LAST - FIRST + 1 or any(a >= b for a, b in zip(selected, selected[1:])):
        raise ValueError("Invalid presentation timestamps in selected interval")
    metadata = probe(SOURCE, "stream=width,height:stream_side_data=rotation")["streams"][0]
    if not any(item.get("rotation") in (-180, 180) for item in metadata.get("side_data_list", [])):
        raise ValueError("Unexpected source display rotation")

    # FFmpeg applies the MOV display matrix and edit list during decode.
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(SOURCE), "-an",
        "-vf", f"select=between(n\\,{FIRST}\\,{LAST}),scale={WIDTH}:{HEIGHT},format=rgb24,settb=1/60,setpts=N",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-",
    ])
    frame_bytes = WIDTH * HEIGHT * 3
    if len(raw) != len(selected) * frame_bytes:
        raise ValueError("Decoded frame count disagrees with presentation timestamps")
    frames = [raw[offset:offset + frame_bytes] for offset in range(0, len(raw), frame_bytes)]
    duration = times[LAST + 1] - selected[0]

    def mapping(fps):
        return [
            (index, index / fps, nearest_frame(selected, selected[0] + index / fps))
            for index in range(round(duration * fps))
        ]

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        rows = mapping(30)
        write_video(stage / "gvhmr_30fps.mp4", (WIDTH, HEIGHT), 30, (frames[i] for _, _, i in rows))
        with (stage / "frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["output_frame", "output_time_s", "source_decoded_frame", "source_pts_s"])
            writer.writerows((n, f"{t:.6f}", FIRST + i, f"{selected[i]:.6f}") for n, t, i in rows)

        def review_frames():
            for _, _, i in mapping(60):
                index = FIRST + i
                canvas = Image.new("RGB", (WIDTH, HEIGHT + 48), "#151c26")
                canvas.paste(Image.frombytes("RGB", (WIDTH, HEIGHT), frames[i]), (0, 0))
                ImageDraw.Draw(canvas).text(
                    (16, HEIGHT + 14),
                    f"IMG_4281  |  frame {index}  |  source {selected[i]:.3f} s  |  {phase(index)}",
                    fill="white", font_size=22,
                )
                yield canvas.tobytes()

        write_video(stage / "review.mp4", (WIDTH, HEIGHT + 48), 60, review_frames())
        for name, indices in (("native_frame_224", [224]), ("native_frame_226", [226])):
            subprocess.run([
                "ffmpeg", "-v", "error", "-i", str(SOURCE), "-vf",
                f"select=eq(n\\,{indices[0]})", "-frames:v", "1", "-update", "1",
                str(stage / f"{name}.png"),
            ], check=True)

        for name, fps, count in (("gvhmr_30fps.mp4", 30, len(rows)), ("review.mp4", 60, len(mapping(60)))):
            video = stage / name
            info = probe(video, "stream=nb_frames,avg_frame_rate:stream_side_data=rotation")["streams"][0]
            if int(info["nb_frames"]) != count or info["avg_frame_rate"] != f"{fps}/1":
                raise ValueError(f"Unexpected output timing: {name}")
            if any(item.get("rotation", 0) for item in info.get("side_data_list", [])):
                raise ValueError(f"Unexpected output rotation: {name}")
            subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"], check=True)

        manifest = {
            "schema_version": 1,
            "source": SOURCE.name,
            "source_sha256": checksum,
            "source_decoded_frame_count": len(times),
            "source_display_rotation_degrees": -180,
            "hand": "left",
            "mirrored": False,
            "timing": "Normal speed confirmed by user; edit-list-aware decoded presentation timestamps.",
            "frame_indexing": "Zero-based decoded display order.",
            "trim": {"first_frame": FIRST, "last_frame_inclusive": LAST,
                     "start_pts_s": selected[0], "end_pts_exclusive_s": times[LAST + 1]},
            "phase_starts": {phase(i): {"frame": i, "pts_s": times[i]}
                             for i in (FIRST, FIRST_WINDUP, FIRST_PEAK, SECOND_AIM, WINDUP, ACCELERATION, RELEASE_FIRST, RELEASE_LAST + 1)},
            "release_window": {"frames_inclusive": [RELEASE_FIRST, RELEASE_LAST],
                               "pts_s": [times[RELEASE_FIRST], times[RELEASE_LAST]],
                               "note": "Visual estimate; blur and finger overlap obscure exact separation."},
            "resampling": "Nearest source frame by presentation time, no synthesized frames.",
            "reconstruction_frames": len(rows),
            "max_reconstruction_timestamp_error_s": max(
                abs(selected[i] - (selected[0] + t)) for _, t, i in rows),
            "review_status": "awaiting_user_review",
            "validation": "Both videos fully decoded; frame counts, rates, and output rotation checked.",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out.resolve())
