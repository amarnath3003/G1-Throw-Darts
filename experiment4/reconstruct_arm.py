"""Export the visible left-arm reconstruction and its source-aligned overlay."""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation

from prepare_reference import write_video

WORKSPACE = Path(__file__).resolve().parents[2]
MODEL = WORKSPACE / "video_to_g1/vendor/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz"


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def reconstruct(prediction, reference, out):
    raw = torch.load(prediction, map_location="cpu", weights_only=True)
    params = raw["smpl_params_incam"]
    count = len(params["body_pose"])
    rows = list(csv.DictReader((reference / "frame_map.csv").open(encoding="utf-8")))
    if len(rows) != count:
        raise ValueError("Reconstruction and reference frame counts differ")
    body = smplx.create(str(MODEL), "smplx", num_betas=params["betas"].shape[-1], use_pca=False, batch_size=count)
    with torch.no_grad():
        joints = body(**params).joints.numpy()
    shoulder, elbow, wrist = (joints[:, index] for index in (16, 18, 20))
    left = normalize(shoulder - joints[:, 17])
    up = normalize(joints[:, 12] - joints[:, 0])
    forward = normalize(np.cross(left, up))
    up = np.cross(forward, left)
    basis = np.stack([forward, left, up], axis=-1)
    upper = np.einsum("ni,nij->nj", normalize(elbow - shoulder), basis)
    lower = np.einsum("ni,nij->nj", normalize(wrist - elbow), basis)
    # Use the wrist's orientation relative to the reconstructed upper torso.
    rotations = Rotation.from_rotvec(params["body_pose"].numpy().reshape(-1, 3)).as_matrix().reshape(count, 21, 3, 3)
    wrist_rotation = Rotation.from_rotvec(params["global_orient"].numpy()).as_matrix()
    for joint in (3, 6, 9, 13, 16, 18, 20):
        wrist_rotation = wrist_rotation @ rotations[:, joint - 1]
    wrist_rotation = basis.transpose(0, 2, 1) @ wrist_rotation
    pixels = np.einsum("nij,nkj->nki", raw["K_fullimg"].numpy(), joints[:, [12, 17, 16, 18, 20]])
    pixels = pixels[..., :2] / pixels[..., 2:]
    if not all(np.isfinite(values).all() for values in (upper, lower, wrist_rotation, pixels)):
        raise ValueError("Nonfinite arm reconstruction")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(prediction, out / "hmr4d_results.pt")
    np.savez_compressed(out / "human_arm.npz", upper=upper, lower=lower, wrist_rotation=wrist_rotation,
                        pixels=pixels, time=np.array([float(r["output_time_s"]) for r in rows]),
                        source_frame=np.array([int(r["source_decoded_frame"]) for r in rows]),
                        prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),
                        smplx_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest())

    def overlay():
        video = cv2.VideoCapture(str(reference / "gvhmr_30fps.mp4"))
        try:
            for index, points in enumerate(pixels.round().astype(int)):
                ok, frame = video.read()
                if not ok:
                    raise ValueError(f"Missing source image {index}")
                for a, b in ((0, 1), (0, 2), (2, 3), (3, 4)):
                    cv2.line(frame, points[a], points[b], (55, 245, 230), 3)
                for point in points:
                    cv2.circle(frame, point, 5, (55, 245, 230), -1)
                cv2.putText(frame, f"GVHMR | decoded source frame {rows[index]['source_decoded_frame']}",
                            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .67, (55, 245, 230), 2)
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
        finally:
            video.release()

    write_video(out / "reconstruction.mp4", (960, 540), 30, overlay())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prediction", "reference", "out"):
        parser.add_argument(name, type=Path)
    reconstruct(**vars(parser.parse_args()))
