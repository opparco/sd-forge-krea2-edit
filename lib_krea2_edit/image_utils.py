from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def order_reference_images(subject, scene=None):
    """Return training-matched reference order from subject-first UI inputs."""
    return [subject] if scene is None else [scene, subject]


def pil_to_bhwc(image: Image.Image) -> torch.Tensor:
    """Convert a PIL reference to Forge/Qwen's float BHWC representation."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def pil_mask_to_bhw(image: Image.Image) -> torch.Tensor:
    """Convert a PIL boost mask to a normalized Forge-style BHW tensor."""
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def limit_long_side(image: torch.Tensor, longest_side: int) -> torch.Tensor:
    """Downscale a BHWC image for Qwen grounding without upscaling it."""
    if longest_side <= 0:
        return image
    height, width = image.shape[1:3]
    if max(height, width) <= longest_side:
        return image
    scale = longest_side / max(height, width)
    target = (max(1, round(height * scale)), max(1, round(width * scale)))
    return F.interpolate(image.movedim(-1, 1), size=target, mode="area").movedim(1, -1)


def fit_reference_pixels(
    image: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """Apply the v1.2 training-matched pixel-space ``fit`` geometry.

    Input and output are BHWC. Near-matched aspect ratios use a minimal center
    crop and fill the target. Genuine mismatches fit inside and snap dimensions
    down to the /16 geometry used while training the v1.2 weights.
    """
    source = image.movedim(-1, 1)
    height, width = source.shape[-2:]
    scale = min(target_height / height, target_width / width)

    crop_tolerance = 0.08
    if height * scale >= target_height * (
        1 - crop_tolerance
    ) and width * scale >= target_width * (1 - crop_tolerance):
        fill_scale = max(target_height / height, target_width / width)
        crop_height = min(height, int(round(target_height / fill_scale)))
        crop_width = min(width, int(round(target_width / fill_scale)))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        source = source[..., top : top + crop_height, left : left + crop_width]
        resized_height, resized_width = target_height, target_width
    else:
        resized_height = min(
            max(16, int(height * scale) // 16 * 16), max(16, target_height // 16 * 16)
        )
        resized_width = min(
            max(16, int(width * scale) // 16 * 16), max(16, target_width // 16 * 16)
        )

    source = F.interpolate(
        source.float(),
        size=(resized_height, resized_width),
        mode="bicubic",
        antialias=True,
    )
    return source.movedim(1, -1).clamp(0, 1)


def fit_reference_mask(
    mask: torch.Tensor,
    reference_image: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Apply the reference pixel-fit transform to a BHW attention mask."""
    source_height, source_width = reference_image.shape[1:3]
    source = mask.unsqueeze(1).float()
    if source.shape[-2:] != (source_height, source_width):
        source = F.interpolate(
            source,
            size=(source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )

    scale = min(target_height / source_height, target_width / source_width)
    crop_tolerance = 0.08
    if source_height * scale >= target_height * (
        1 - crop_tolerance
    ) and source_width * scale >= target_width * (1 - crop_tolerance):
        fill_scale = max(target_height / source_height, target_width / source_width)
        crop_height = min(source_height, int(round(target_height / fill_scale)))
        crop_width = min(source_width, int(round(target_width / fill_scale)))
        top = (source_height - crop_height) // 2
        left = (source_width - crop_width) // 2
        source = source[..., top : top + crop_height, left : left + crop_width]
        resized_height, resized_width = target_height, target_width
    else:
        resized_height = min(
            max(16, int(source_height * scale) // 16 * 16),
            max(16, target_height // 16 * 16),
        )
        resized_width = min(
            max(16, int(source_width * scale) // 16 * 16),
            max(16, target_width // 16 * 16),
        )

    return (
        F.interpolate(
            source,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(1)
        .clamp(0, 1)
    )
