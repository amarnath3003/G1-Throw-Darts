"""Versioned pickup artifacts, transforms and physical success criteria (SI)."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fresh_directory(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


def phase_at(clip, t):
    return next((p for a, b, p in clip['phases'] if a <= t < b), 'excluded')


def board_corners(edge=.048):
    # Original unrotated PNG: 10 pixels/mm. Pixel centers span 499/500
    # of a nominal marker. Use ideal PRINT edges rather than raster centers.
    # Right-handed board frame: origin ID0 upper-left, +x across page,
    # +y toward page top, +z out of paper. Detection order TL,TR,BR,BL.
    result = {}
    for i, (x, y) in enumerate([(0, 0), (.120, 0), (0, .207), (.120, .207)]):
        x *= edge/.05; y *= edge/.05
        result[i] = np.array([[x,-y],[x+edge,-y],[x+edge,-y-edge],[x,-y-edge]], np.float32)
    return result


class SuccessTracker:
    """Accumulate continuous support at PHYSICS frequency, never video FPS."""
    def __init__(self):
        self.lift_seconds = 0.
        self.ready_seconds = 0.
        self.pickup_success = False
        self.success = False

    def update(self, sample, dt):
        support = sample['forces'][0] > .05 and max(sample['forces'][1:]) > .05
        lifted = sample['clearance'] > .03 and not sample['table_contact'] and support and not sample['unsafe']
        self.lift_seconds = self.lift_seconds + dt if lifted else 0.
        self.pickup_success |= self.lift_seconds >= 3. - 1e-8
        ready = (lifted and self.pickup_success and sample['wrist_position_error'] <= .01
                 and sample['wrist_rotation_error'] <= np.deg2rad(10)
                 and sample['object_position_error'] <= .01
                 and sample['object_rotation_error'] <= np.deg2rad(10)
                 and sample['linear_speed'] < .02 and sample['angular_speed'] < .2)
        self.ready_seconds = self.ready_seconds + dt if ready else 0.
        self.success |= self.ready_seconds >= 2. - 1e-8
        return bool(self.success)
