import os
import time
import argparse
from typing import List

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from omegaconf import OmegaConf
import torch
from torch import Tensor
from jaxtyping import Float
from PIL import Image
from torchvision import transforms as TF
from einops import repeat
import matplotlib

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from dpm.model import VDPM
from util.transforms import transform_points


VIDEO_SAMPLE_HZ = 1.0


def assign_colours(pts3d, colour=[0, 0, 1]):
    num_points = pts3d.shape[0]
    colors = (np.tile(np.array([colour]), (num_points, 1)) * 255).astype(np.uint8)
    return colors


def compute_box_edges(corners):
    """
    Compute all edges of a 3D bounding box

    Args:
        corners: torch tensor of shape (8, 3) containing the coordinates of the 8 corners
                 of a 3D bounding box

    Returns:
        edges: torch tensor of shape (12, 2, 3) containing the 12 edges of the box,
               each represented as a pair of 3D coordinates [start_point, end_point]
    """
    # Define the 12 edges of a cube by specifying pairs of corner indices
    edge_indices = torch.tensor(
        [
            # Edges along x-axis
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            # Edges along y-axis
            [0, 2],
            [1, 3],
            [4, 6],
            [5, 7],
            # Edges along z-axis
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=torch.long,
    )

    # Initialize edges tensor
    edges = torch.zeros((12, 2, 3), dtype=corners.dtype, device=corners.device)

    # Extract the start and end points for each edge
    for i, (start_idx, end_idx) in enumerate(edge_indices):
        edges[i, 0] = corners[start_idx]  # Start point
        edges[i, 1] = corners[end_idx]  # End point

    colors = torch.tensor(
        [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [0, 255, 255],  # Cyan
            [255, 0, 255],  # Magenta
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [128, 255, 0],  # Lime
            [255, 0, 128],  # Pink
            [0, 128, 255],  # Teal
            [128, 0, 0],  # Maroon
        ],
        dtype=torch.uint8,
        device=corners.device,
    )

    return edges, colors


def remove_static_tracks(
    tracks: Float[Tensor, "t n 3"], threshold=0.025
) -> Float[Tensor, "t n 3"]:
    # delta = tracks[1:] - tracks[[0]]
    delta = tracks[None, ...] - tracks[:, None, ...]
    max_displ = torch.linalg.norm(delta.abs(), dim=-1).amax((0, 1))
    dynamic = max_displ > threshold
    tracks_filtered = tracks[:, dynamic, :]
    return tracks_filtered


def compute_predictions(model, images):
    print("model inference started")

    start = time.perf_counter()

    with torch.no_grad():
        result = model.inference(None, images=images.unsqueeze(0))
    print("model inference finished")
    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")

    pointmaps = result["pointmaps"]

    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    pose_enc = result["pose_enc"]
    HW = pointmaps[0]["pts3d"].shape[2:4]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, HW)
    extrinsic = extrinsic[0]
    S = extrinsic.shape[0]
    extrinsic_CW = torch.cat(
        [extrinsic.cpu(), repeat(torch.tensor([0, 0, 0, 1]), "c -> s 1 c", s=S)], dim=1
    )
    extrinsic_WC = torch.linalg.inv(extrinsic_CW)

    return pointmaps, extrinsic_WC, intrinsic


def extract_frames(input_video):
    torch.cuda.empty_cache()

    video_path = input_video
    vs = cv2.VideoCapture(video_path)

    fps = float(vs.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_interval = max(int(fps / max(VIDEO_SAMPLE_HZ, 1e-6)), 1)

    count = 0
    frame_num = 0
    images = []
    try:
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            if count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(frame)
                frame_num += 1
            count += 1
    finally:
        vs.release()

    return images


def preprocess_images(images_np, mode="crop"):
    # Check for empty list
    if len(images_np) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for img_np in images_np:

        # Open image
        img = Image.fromarray(img_np)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (new_width / width) / 14) * 14
                )  # Make divisible by 14
            else:
                new_height = target_size
                new_width = (
                    round(width * (new_height / height) / 14) * 14
                )  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(images_np) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_model(cfg, device) -> VDPM:
    model = VDPM(cfg).to(device)

    _URL = "https://huggingface.co/edgarsucar/vdpm/resolve/main/model.pt"
    sd = torch.hub.load_state_dict_from_url(
        _URL, file_name="vdpm_model.pt", progress=True
    )
    print(model.load_state_dict(sd, strict=True))

    model.eval()
    return model


def log_predictions(images, pointmaps, poses, intrinsic, threshold=0.9):
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    S = len(pointmaps)
    pts3d_all = [pr["pts3d"] for pr in pointmaps]
    conf_all = [pr["conf"] for pr in pointmaps]

    # all view points in view v0
    pts3d_v0 = torch.stack([pts3d_all[s][:, 0] for s in range(S)], dim=1)[0].detach()
    conf_v0 = torch.stack([conf_all[s][:, 0] for s in range(S)], dim=1)[0].detach()
    # confident points based on view v0
    conf = conf_v0[0].view(-1).cpu().numpy()
    thresh = conf[conf.argsort()][int(conf.size * (1 - threshold))]
    mask = conf_v0[0].cpu().numpy() >= thresh
    tracks_xyz = remove_static_tracks(pts3d_v0[:, mask].cpu()).numpy()
    num_tracks = min(500, tracks_xyz.shape[1])
    indices = np.random.choice(tracks_xyz.shape[1], num_tracks, replace=False)
    tracks_xyz = tracks_xyz[:, indices]
    sorted_indices = np.argsort(tracks_xyz[0, ..., 1])
    tracks_xyz = tracks_xyz[:, sorted_indices]
    color_map = matplotlib.colormaps.get_cmap("hsv")
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=tracks_xyz.shape[1] - 1)
    track_colors = np.zeros((num_tracks, 3), dtype=np.float32)
    for t_idx in range(num_tracks):
        color = color_map(cmap_norm(t_idx))[:3]
        track_colors[t_idx] = color
    track_colors = (track_colors * 255).astype(np.uint8)
    tracks_xyz = tracks_xyz.swapaxes(0, 1)  # (num_tracks, t, 3)

    for i in range(S):
        rr.set_time("time", duration=i)

        points = pts3d_v0[i].cpu().numpy()
        colors = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        rr.log(
            "world/points",
            rr.Points3D(
                points[mask],
                colors=colors[mask],
            ),
        )

        pose = poses[i].cpu().numpy()
        if i > 0:
            rr.log(
                f"world/tracks",
                rr.LineStrips3D(
                    tracks_xyz[:, : i + 1],
                    colors=track_colors,
                    radii=rr.Radius.ui_points(0.5),
                ),
            )

        rr.log(
            "world/camera",
            rr.Transform3D(
                mat3x3=pose[:3, :3],
                translation=pose[:3, 3],
            ),
        )
        rr.log(
            "world/camera",
            rr.Pinhole(
                image_from_camera=intrinsic[0][i].cpu().numpy(),
                resolution=(images.shape[3], images.shape[2]),
            ),
        )

        rr.log(
            "world/camera/rgb",
            rr.Image(
                (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ).compress(jpeg_quality=75),
        )


def main(args):
    cfg = OmegaConf.load(args.config)

    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = (
        True  # for gpu >= Ampere and pytorch >= 1.12
    )

    model = load_model(cfg, device)

    input_video = args.input_video
    frames = extract_frames(input_video)
    images = preprocess_images(frames).to(device)  # (N, 3, H, W)

    pointmaps, poses, intrinsic = compute_predictions(model, images)

    # rerun visualisation
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
            rrb.Spatial2DView(
                name="RGB",
                origin="world/camera",
                contents=["$origin/rgb", "/world/tracks"],
            ),
        ),
        collapse_panels=True,
    )
    rr.script_setup(args, "rerun_vdpm", default_blueprint=blueprint)

    log_predictions(images, pointmaps, poses, intrinsic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun visualization script")
    parser.add_argument(
        "--input_video", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    main(args)

    rr.script_teardown(args)
