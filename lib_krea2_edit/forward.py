from __future__ import annotations

import math
import logging

import torch
import torch.nn.functional as F
from einops import rearrange

from backend.nn.flux import timestep_embedding
from backend.utils import pad_to_patch_size

logger = logging.getLogger(__name__)


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


def reference_attention_bias(
    boosts,
    boost_mask,
    text_length,
    source_lengths,
    target_length,
    source_grids,
    device,
    dtype,
):
    """Bias target-to-reference attention in ``[text | refs... | target]``.

    The optional mask addresses the last reference (the subject in a two-reference
    job, or the only reference in a single-reference job).
    """
    offsets = [text_length]
    for source_length in source_lengths:
        offsets.append(offsets[-1] + source_length)

    target_start = offsets[-1]
    sequence_length = target_start + target_length
    bias = torch.zeros(
        1, 1, sequence_length, sequence_length, device=device, dtype=dtype
    )
    last_source = len(source_lengths) - 1
    for index, boost in enumerate(boosts):
        if boost == 1.0:
            continue
        offset = offsets[index]
        source_length = source_lengths[index]
        if boost_mask is not None and index == last_source:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = F.interpolate(
                mask.unsqueeze(1).float(), mode="area", size=source_grids[index]
            )[0, 0]
            columns = offset + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[
                0
            ].to(device)
        else:
            columns = torch.arange(offset, offset + source_length, device=device)
        bias[:, :, target_start:, columns] = math.log(max(float(boost), 1e-4))
    return bias


def krea2_edit_forward(
    model,
    x,
    timesteps,
    context,
    source_latent,
    transformer_options,
    ref_boost=1.0,
    ref_boost_a=1.0,
    ref_boost_mask=None,
    attention_bias_cache=None,
):
    """Forge port of the v1.2 one/two-reference Krea2 Edit forward path."""
    patch = model.patch
    temporal = x.ndim == 5
    if temporal:
        batch_5d, _, frames_5d, height_5d, width_5d = x.shape

    x = _to_4d(x)
    batch, _, original_height, original_width = x.shape
    x = pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    height, width = x.shape[-2:]
    target_grid_height, target_grid_width = height // patch, width // patch

    source_latents = (
        list(source_latent)
        if isinstance(source_latent, (list, tuple))
        else [source_latent]
    )
    if not 1 <= len(source_latents) <= 2:
        raise ValueError("Krea2 Edit requires one or two reference latents.")
    sources = []
    for latent in source_latents:
        source = _to_4d(latent).to(device=x.device, dtype=x.dtype)
        if source.shape[0] != batch:
            source = source[:1].expand(batch, *source.shape[1:])
        sources.append(
            pad_to_patch_size(source, (patch, patch), padding_mode="replicate")
        )
    source_grids = [
        (source.shape[-2] // patch, source.shape[-1] // patch) for source in sources
    ]

    # Forge currently carries an extra singleton dimension on Krea conditioning.
    if context.ndim == 4 and context.shape[1] == 1:
        context = context.squeeze(1)
    context = model._unpack_context(context)

    target_tokens = model.first(
        rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    )
    source_tokens = [
        model.first(
            rearrange(
                source,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch,
                pw=patch,
            )
        )
        for source in sources
    ]

    timestep = model.tmlp(
        timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(target_tokens.dtype)
    )
    timestep_vector = model.tproj(timestep)
    context = model.txtfusion(
        context, mask=None, transformer_options=transformer_options
    )
    context = model.txtmlp(context)

    text_length = context.shape[1]
    source_lengths = [tokens.shape[1] for tokens in source_tokens]
    source_length = sum(source_lengths)
    target_length = target_tokens.shape[1]
    combined = torch.cat([context] + source_tokens + [target_tokens], dim=1)

    device = combined.device
    positions = torch.cat(
        (
            torch.zeros(batch, text_length, 3, device=device, dtype=torch.float32),
            *[
                _offset_image_ids(
                    batch,
                    index + 1,
                    source_grid_height,
                    source_grid_width,
                    target_grid_height,
                    target_grid_width,
                    device,
                )
                for index, (source_grid_height, source_grid_width) in enumerate(
                    source_grids
                )
            ],
            _image_ids(batch, 0, target_grid_height, target_grid_width, device),
        ),
        dim=1,
    )
    frequencies = model.pe_embedder(positions)

    attention_bias = None
    if ref_boost != 1.0 or ref_boost_a != 1.0:
        boosts = [ref_boost_a] * (len(source_tokens) - 1) + [ref_boost]
        cache_key = (
            text_length,
            tuple(source_lengths),
            target_length,
            tuple(boosts),
            combined.device,
            combined.dtype,
            None
            if ref_boost_mask is None
            else (ref_boost_mask.data_ptr(), tuple(ref_boost_mask.shape)),
        )
        if attention_bias_cache is not None:
            attention_bias = attention_bias_cache.get(cache_key)
        if attention_bias is None:
            sequence_length = text_length + source_length + target_length
            allocation = sequence_length**2 * combined.element_size()
            if allocation >= 512 * 1024**2:
                logger.warning(
                    "[Krea2 Edit] reference boost attention bias requires %.1f MiB "
                    "at this resolution; reduce output/reference resolution if VRAM is tight.",
                    allocation / 1024**2,
                )
            attention_bias = reference_attention_bias(
                boosts,
                ref_boost_mask,
                text_length,
                source_lengths,
                target_length,
                source_grids,
                combined.device,
                combined.dtype,
            )
            if attention_bias_cache is not None:
                attention_bias_cache[cache_key] = attention_bias

    for block_index, block in enumerate(model.blocks):
        transformer_options["block_index"] = block_index
        combined = block(
            combined,
            timestep_vector,
            frequencies,
            attention_bias,
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
