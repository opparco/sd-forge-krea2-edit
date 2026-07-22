from __future__ import annotations

import torch

from .forward import _to_4d, krea2_edit_forward
from .image_utils import fit_reference_pixels


def patch_krea2_unet(unet, vae, reference_image):
    """Clone and attach the Krea2 Edit model-function wrapper."""
    patched = unet.clone()
    kmodel = patched.model
    previous_wrapper = patched.model_options.get("model_function_wrapper")
    latent_cache = {}

    def source_for_target(x):
        target = _to_4d(x)
        height, width = target.shape[-2:]
        key = (height, width)
        if key not in latent_cache:
            fitted = fit_reference_pixels(reference_image, height * 8, width * 8)
            latent = vae.encode(fitted)
            latent_cache[key] = vae.first_stage_model.process_in(latent)
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
        ).float()
        return kmodel.predictor.calculate_denoised(sigma, model_output, x)

    def wrapper(_model_function, call):
        if previous_wrapper is not None:
            return previous_wrapper(edit_apply_model, call)
        return edit_apply_model(call["input"], call["timestep"], **call["c"])

    patched.set_model_unet_function_wrapper(wrapper)
    return patched
