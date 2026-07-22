from __future__ import annotations

import torch
from einops import rearrange

from backend.nn.flux import timestep_embedding
from backend.utils import pad_to_patch_size


def _to_4d(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 5:
        batch, channels, frames, height, width = value.shape
        return value.reshape(batch * frames, channels, height, width)
    return value


def _image_ids(batch: int, frame: int, height: int, width: int, device) -> torch.Tensor:
    ids = torch.zeros(height, width, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, height * width, 3).repeat(batch, 1, 1)


def _offset_image_ids(
    batch: int,
    frame: int,
    ref_height: int,
    ref_width: int,
    target_height: int,
    target_width: int,
    device,
) -> torch.Tensor:
    top = max(0, (target_height - ref_height) // 2)
    left = max(0, (target_width - ref_width) // 2)
    ids = torch.zeros(ref_height, ref_width, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(ref_height, device=device, dtype=torch.float32) + top)[
        :, None
    ]
    ids[..., 2] = (torch.arange(ref_width, device=device, dtype=torch.float32) + left)[
        None, :
    ]
    return ids.reshape(1, ref_height * ref_width, 3).repeat(batch, 1, 1)


def krea2_edit_forward(
    model, x, timesteps, context, source_latent, transformer_options
):
    """Forge port of the v1.2 single-reference Krea2 Edit forward path."""
    patch = model.patch
    temporal = x.ndim == 5
    if temporal:
        batch_5d, _, frames_5d, height_5d, width_5d = x.shape

    x = _to_4d(x)
    batch, _, original_height, original_width = x.shape
    x = pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    height, width = x.shape[-2:]
    target_grid_height, target_grid_width = height // patch, width // patch

    source = _to_4d(source_latent).to(device=x.device, dtype=x.dtype)
    if source.shape[0] != batch:
        source = source[:1].expand(batch, *source.shape[1:])
    source = pad_to_patch_size(source, (patch, patch), padding_mode="replicate")
    source_grid_height, source_grid_width = (
        source.shape[-2] // patch,
        source.shape[-1] // patch,
    )

    # Forge currently carries an extra singleton dimension on Krea conditioning.
    if context.ndim == 4 and context.shape[1] == 1:
        context = context.squeeze(1)
    context = model._unpack_context(context)

    target_tokens = model.first(
        rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    )
    source_tokens = model.first(
        rearrange(source, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    )

    timestep = model.tmlp(
        timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(target_tokens.dtype)
    )
    timestep_vector = model.tproj(timestep)
    context = model.txtfusion(
        context, mask=None, transformer_options=transformer_options
    )
    context = model.txtmlp(context)

    text_length = context.shape[1]
    source_length = source_tokens.shape[1]
    target_length = target_tokens.shape[1]
    combined = torch.cat((context, source_tokens, target_tokens), dim=1)

    device = combined.device
    positions = torch.cat(
        (
            torch.zeros(batch, text_length, 3, device=device, dtype=torch.float32),
            _offset_image_ids(
                batch,
                1,
                source_grid_height,
                source_grid_width,
                target_grid_height,
                target_grid_width,
                device,
            ),
            _image_ids(batch, 0, target_grid_height, target_grid_width, device),
        ),
        dim=1,
    )
    frequencies = model.pe_embedder(positions)

    for block_index, block in enumerate(model.blocks):
        transformer_options["block_index"] = block_index
        combined = block(
            combined,
            timestep_vector,
            frequencies,
            None,
            transformer_options=transformer_options,
        )

    final = model.last(combined, timestep)
    output = final[
        :, text_length + source_length : text_length + source_length + target_length, :
    ]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=target_grid_height,
        w=target_grid_width,
        ph=patch,
        pw=patch,
        c=model.channels,
    )
    output = output[:, :, :original_height, :original_width]
    if temporal:
        output = output.reshape(
            batch_5d, frames_5d, model.channels, height_5d, width_5d
        ).movedim(1, 2)
    return output
