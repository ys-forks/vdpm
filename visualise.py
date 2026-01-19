import time
import argparse

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
import torch.nn.functional as F

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from dpm.model import VDPM


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


def extract_frames(input_video, hz=1.0):
    torch.cuda.empty_cache()

    video_path = input_video
    vs = cv2.VideoCapture(video_path)

    fps = float(vs.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_interval = max(int(fps / max(hz, 1e-6)), 1)

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


def log_predictions(
    images, pointmaps, poses, intrinsic, use_global_pts=False, threshold=0.1
):
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    S = len(pointmaps)
    C = pointmaps[0]["pts3d"].shape[-1]
    pts3d_all = []
    conf_all = []
    scale_factor = 0.25 if use_global_pts else 1.0
    for i in range(S):
        pts = pointmaps[i]["pts3d"][0].permute(0, 3, 1, 2)
        conf = pointmaps[i]["conf"][0].unsqueeze(1)
        if scale_factor != 1.0:
            pts = F.interpolate(pts, scale_factor=scale_factor, mode="nearest")
            conf = F.interpolate(conf, scale_factor=scale_factor, mode="nearest")
        pts3d_all.append(pts.permute(0, 2, 3, 1))
        conf_all.append(conf.squeeze(1))

    if scale_factor != 1.0:
        pts_colors = F.interpolate(images, scale_factor=scale_factor, mode="nearest")
    else:
        pts_colors = images

    # t0 as reference
    conf = conf_all[0].detach().cpu()
    thresh = torch.quantile(conf.view(-1), threshold).item()
    mask = conf >= thresh
    if use_global_pts:
        mask = mask.view(-1)
        pts3d = torch.stack(
            [pts3d_all[s].view(-1, C) for s in range(S)], dim=0
        ).detach()
        colors = (
            pts_colors.permute(0, 2, 3, 1)
            .cpu()
            .contiguous()
            .view(-1, images.shape[1])
            .numpy()
            * 255
        ).astype(np.uint8)
    else:
        mask = mask[0].view(-1)
        pts3d = torch.stack(
            [pts3d_all[s][0].view(-1, C) for s in range(S)], dim=0
        ).detach()
        colors = (
            pts_colors.permute(0, 2, 3, 1)[0]
            .cpu()
            .contiguous()
            .view(-1, images.shape[1])
            .numpy()
            * 255
        ).astype(np.uint8)

    tracks_xyz = remove_static_tracks(pts3d[:, mask].cpu()).numpy()
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

        points = pts3d[i].cpu().numpy()
        rr.log(
            "world/points",
            rr.Points3D(
                points[mask],
                colors=colors[mask],
            ),
        )

        pose = poses[i].cpu().numpy()
        if i > 0:
            min_start = max(0, i - 8)
            rr.log(
                f"world/tracks",
                rr.LineStrips3D(
                    tracks_xyz[:, min_start : i + 1],
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

    input_video = args.input_video
    frames = extract_frames(input_video, hz=args.hz)
    images = preprocess_images(frames, mode="pad").to(device)  # (N, 3, H, W)
    print(f"Preprocessed frames shape: {images.shape}.")

    model = load_model(cfg, device)
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

    log_predictions(
        images,
        pointmaps,
        poses,
        intrinsic,
        use_global_pts=args.use_global_pts,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun visualization script")
    parser.add_argument(
        "--input_video", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--hz", type=float, default=3.0, help="Video sampling frequency"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Confidence threshold"
    )
    parser.add_argument(
        "--use_global_pts",
        default=False,
        action="store_true",
        help="Visualize all points cross all images through timesteps",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    main(args)

    rr.script_teardown(args)
