from __future__ import annotations

import torch

from .forward import _to_4d, krea2_edit_forward
from .image_utils import fit_reference_mask, fit_reference_pixels


def patch_krea2_unet(
    unet,
    vae,
    reference_images,
    ref_boost=1.0,
    ref_boost_a=1.0,
    ref_boost_mask=None,
):
    """Clone and attach the Krea2 Edit model-function wrapper."""
    references = (
        list(reference_images)
        if isinstance(reference_images, (list, tuple))
        else [reference_images]
    )
    if not 1 <= len(references) <= 2:
        raise ValueError("Krea2 Edit requires one or two reference images.")

    patched = unet.clone()
    kmodel = patched.model
    previous_wrapper = patched.model_options.get("model_function_wrapper")
    latent_cache = {}
    attention_bias_cache = {}

    def source_for_target(x):
        target = _to_4d(x)
        height, width = target.shape[-2:]
        key = (height, width)
        if key not in latent_cache:
            latents = []
            for reference in references:
                fitted = fit_reference_pixels(reference, height * 8, width * 8)
                latent = vae.encode(fitted)
                latents.append(vae.first_stage_model.process_in(latent))
            latent_cache[key] = latents
        return latent_cache[key]

    def mask_for_target(x):
        if ref_boost_mask is None:
            return None
        target = _to_4d(x)
        height, width = target.shape[-2:]
        key = ("mask", height, width)
        if key not in latent_cache:
            latent_cache[key] = fit_reference_mask(
                ref_boost_mask,
                references[-1],
                height * 8,
                width * 8,
            )
        return latent_cache[key]

    def edit_apply_model(
        x,
        sigma,
        c_concat=None,
        c_crossattn=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        transformer_options = {} if transformer_options is None else transformer_options
        input_x = kmodel.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            input_x = torch.cat([input_x] + [c_concat], dim=1)

        dtype = kmodel.computation_dtype
        input_x = input_x.to(dtype)
        model_timestep = kmodel.predictor.timestep(sigma).float()
        context = c_crossattn.to(dtype)
        source = source_for_target(input_x)
        model_output = krea2_edit_forward(
            kmodel.diffusion_model,
            input_x,
            model_timestep,
            context,
            source,
            transformer_options,
            ref_boost=ref_boost,
            ref_boost_a=ref_boost_a,
            ref_boost_mask=mask_for_target(input_x),
            attention_bias_cache=attention_bias_cache,
        ).float()
        return kmodel.predictor.calculate_denoised(sigma, model_output, x)

    def wrapper(_model_function, call):
        if previous_wrapper is not None:
            return previous_wrapper(edit_apply_model, call)
        return edit_apply_model(call["input"], call["timestep"], **call["c"])

    patched.set_model_unet_function_wrapper(wrapper)
    return patched
