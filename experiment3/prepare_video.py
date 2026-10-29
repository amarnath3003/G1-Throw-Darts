"""Extract a small timestamp-preserving pickup pilot and an SDR comparison."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(r'C:\Users\Amarnath\Downloads\IMG_4238.mov'))
    args = parser.parse_args()
    root = HERE/'data'
    frames = root/'pilot_frames'
    frames.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                       '-show_frames','-show_streams','-show_format','-of','json',str(source)]))
    timestamps = [float(f['best_effort_timestamp_time']) for f in probe['frames']]
    assert all(b>a for a,b in zip(timestamps,timestamps[1:])), 'PTS must increase'
    times = np.array(timestamps)-timestamps[0]
    chosen = sorted(set([int(np.argmin(abs(times-t))) for t in np.arange(0,times[-1],.25)]+[len(times)-1]))
    select = '+'.join(f'eq(n\\,{i})' for i in chosen)
    base = ['ffmpeg','-y','-v','error','-i',str(source)]
    subprocess.run(base+['-vf',f'select={select}','-fps_mode','passthrough',str(frames/'frame_%04d.png')],check=True)
    records = [dict(pilot_index=j, frame_index=i, pts_s=timestamps[i], relative_time_s=float(times[i]),
                    file=f'pilot_frames/frame_{j+1:04d}.png') for j,i in enumerate(chosen)]
    assert len(list(frames.glob('frame_*.png')))==len(records)
    manifest = dict(schema_version=1, source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_size_bytes=source.stat().st_size, streams=probe['streams'], format=probe['format'],
                    all_frame_pts_s=timestamps, pilot_frames=records,
                    extraction='FFmpeg select original decoded frame indices; no interpolation or FPS conversion',
                    color='Baseline FFmpeg RGB decode, original HLG metadata; SDR comparison separate')
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    annotations=HERE/'configs/video_annotations.json'
    if annotations.exists():
        (root/'annotations.json').write_text(annotations.read_text())
    sheet = Image.new('RGB',(6*192,5*368),'#14171b')
    draw = ImageDraw.Draw(sheet)
    for j,record in enumerate(records):
        im = Image.open(root/record['file']).convert('RGB')
        im.thumbnail((192,342))
        x,y = (j%6)*192,(j//6)*368
        sheet.paste(im,(x,y+26))
        draw.text((x+5,y+5),f"{record['relative_time_s']:.3f}s  frame {record['frame_index']}",fill='white')
    sheet.save(HERE/'research/phase2_timestamps.jpg',quality=90)
    # Separate explicit HLG -> linear -> tone mapping -> BT.709 conversion for review.
    idx = chosen[len(chosen)//2]
    filt = f'select=eq(n\\,{idx}),zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=full,format=rgb24'
    subprocess.run(base+['-vf',filt,'-frames:v','1',str(root/'sdr_comparison.png')],check=True)
    comparison = Image.new('RGB',(960,894),'#14171b')
    comparison.paste(Image.open(root/records[len(chosen)//2]['file']),(0,40))
    comparison.paste(Image.open(root/'sdr_comparison.png'),(480,40))
    draw = ImageDraw.Draw(comparison)
    draw.text((10,10),'Original RGB decode',fill='white')
    draw.text((490,10),'Explicit HLG to SDR / Hable',fill='white')
    comparison.save(HERE/'research/phase2_color_comparison.jpg',quality=90)
    print(f'Prepared {len(records)} pilot frames from {len(timestamps)} source frames; PTS preserved.')


if __name__ == '__main__':
    main()
