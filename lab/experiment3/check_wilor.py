"""Validate exported pilot data and create a visual review sheet."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
out=HERE/'data/wilor_pilot'
report=json.loads((out/'report.json').read_text())
raw=np.load(out/'hand_raw.npz',allow_pickle=False)
frames=report['frames']
assert raw['joints_wrist_relative'].shape==(len(frames),21,3)
assert np.all(np.diff(raw['timestamps_s'])>0)
selected=np.array([r['selected_left_detection'] is not None for r in frames])
assert np.isfinite(raw['joints_wrist_relative'][selected]).all()
assert np.isnan(raw['joints_wrist_relative'][~selected]).all()
assert np.max(np.abs(raw['joints_wrist_relative'][selected,0]))<1e-6
for r in frames:
    if r['selected_left_detection'] is not None:
        assert not r['detections'][r['selected_left_detection']]['is_right']
    assert (out/r['overlay']).exists()
sheet=Image.new('RGB',(6*192,((len(frames)+5)//6)*342),'#151515')
for i,r in enumerate(frames):
    im=Image.open(out/r['overlay']).convert('RGB')
    im.thumbnail((192,342))
    sheet.paste(im,((i%6)*192,(i//6)*342))
sheet.save(HERE/'research/wilor_overlay_sheet.jpg',quality=90)
summary=dict(total_frames=len(frames),selected_left_frames=int(selected.sum()),
             missing_left_times_s=[r['relative_time_s'] for r in frames if r['selected_left_detection'] is None],
             all_21_projected_in_frame=sum(r.get('joints_in_frame')==21 for r in frames),
             occluded_acquisition_frames=sum(r['phase']=='acquisition_occluded' for r in frames),
             latency_median_s=float(np.median([r['latency_s'] for r in frames])),
             peak_cuda_allocated_gib=report['peak_cuda_allocated_bytes']/2**30 if report['peak_cuda_allocated_bytes'] else None,
             validation='timestamps, shapes, finite selected joints, missing-data NaNs, wrist-relative origin, handedness and overlay files passed')
(HERE/'research/wilor_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
