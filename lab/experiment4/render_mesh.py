"""Render GVHMR's native SMPL-X surface over the exact input frames."""
import argparse
from pathlib import Path

import cv2
import torch
from gvhmr.utils.smplx_utils import make_smplx
from gvhmr.utils.vis.renderer_gl import make_renderer

from prepare_reference import write_video


def render(prediction, source, out):
    if out.exists():
        raise FileExistsError(out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction = torch.load(prediction, map_location="cpu", weights_only=True)
    model = make_smplx("supermotion").to(device)
    with torch.no_grad():
        vertices = model(**{k: v.to(device) for k, v in prediction["smpl_params_incam"].items()}).vertices
    video = cv2.VideoCapture(str(source))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(vertices):
        video.release()
        raise ValueError("Mesh and source frame counts differ")
    renderer = make_renderer(width, height, device=device, faces=model.faces, K=prediction["K_fullimg"][0])

    def overlay():
        for index, mesh in enumerate(vertices):
            ok, frame = video.read()
            if not ok:
                raise ValueError(f"Could not decode source frame {index}")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield renderer.render_mesh(mesh, image, [0.42, 0.72, 0.87]).tobytes()

    try:
        write_video(out, (width, height), 30, overlay())
    finally:
        video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prediction", "source", "out"):
        parser.add_argument(name, type=Path)
    render(**vars(parser.parse_args()))
