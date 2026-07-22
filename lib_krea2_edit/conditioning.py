from __future__ import annotations

from contextlib import contextmanager

import torch


def expand_grounded_multipliers(tokens, multipliers, embeds_info):
    """Expand prompt weights when one image placeholder becomes many embeddings."""
    expanded = []
    embed_iter = iter(embeds_info)
    for token, multiplier in zip(tokens, multipliers):
        try:
            int(token)
            expanded.append(multiplier)
        except (TypeError, ValueError):
            info = next(embed_iter, None)
            if info is not None:
                expanded.extend([multiplier] * int(info["size"]))
    return expanded


@contextmanager
def grounded_token_multiplier_compat(engine, tap_layers=None):
    """Make Forge prompt-emphasis weights follow expanded Qwen image tokens."""
    if tap_layers is None:
        from backend.text_processing.qwen3vl_engine import KREA2_TAP_LAYERS

        tap_layers = KREA2_TAP_LAYERS

    original = engine.process_tokens

    def process_tokens(batch_tokens, batch_multipliers):
        embeds, mask, count, info = engine.process_embeds(batch_tokens)
        expanded_multipliers = [
            expand_grounded_multipliers(tokens, multipliers, info)
            for tokens, multipliers in zip(batch_tokens, batch_multipliers)
        ]
        multiplier_tensor = torch.asarray(expanded_multipliers).to(embeds)
        if multiplier_tensor.shape != embeds.shape[:-1]:
            raise RuntimeError(
                "Krea2 Edit grounded token expansion mismatch: "
                f"embeddings={tuple(embeds.shape[:-1])}, "
                f"multipliers={tuple(multiplier_tensor.shape)}"
            )

        engine.emphasis.tokens = batch_tokens
        engine.emphasis.multipliers = multiplier_tensor
        engine.emphasis.z = embeds
        engine.emphasis.after_transformers()
        embeds = engine.emphasis.z

        _, output = engine.text_encoder(
            None,
            embeds=embeds,
            attention_mask=mask,
            num_tokens=count,
            embeds_info=info,
            intermediate_output=tap_layers,
            final_layer_norm_intermediate=False,
        )
        return output

    engine.process_tokens = process_tokens
    try:
        yield
    finally:
        engine.process_tokens = original


@contextmanager
def forge_qwen_vision_attention_compat(qwen_module=None, selected_attention=None):
    """Adapt Forge's selected attention function to Qwen's selector API.

    Forge exposes ``backend.attention.attention_function`` as the already selected
    q/k/v implementation. The imported Comfy Qwen vision code expects a callable
    selector and invokes it first with ``(device, mask=..., small_input=...)``.
    Keep the patch scoped to grounded encoding and restore the module afterward.
    """
    if selected_attention is None:
        from backend import attention as forge_attention

        selected_attention = forge_attention.attention_function
    if qwen_module is None:
        from backend.nn.llm import qwen35 as qwen_module

    original = qwen_module.attention_function

    def select_attention(device, mask=False, small_input=False):
        return selected_attention

    qwen_module.attention_function = select_attention
    try:
        yield
    finally:
        qwen_module.attention_function = original


class GroundedQwenEngine:
    """Job-local proxy that adds one reference image to every prompt encode."""

    def __init__(self, engine, image):
        self._engine = engine
        self._image = image

    def __call__(self, texts, images=None):
        with grounded_token_multiplier_compat(self._engine):
            return self._engine(texts, images=[self._image])

    def __getattr__(self, name):
        return getattr(self._engine, name)


def install_grounded_setup(processing, grounded_image) -> None:
    """Wrap ``setup_conds`` so positive and negative prompts see the reference.

    The Qwen engine swap exists only while Forge builds conditioning. This avoids
    leaving the globally shared diffusion engine monkey-patched between jobs.
    """
    if getattr(processing, "_krea2_edit_grounding_installed", False):
        return

    original_setup_conds = processing.setup_conds

    def setup_conds_with_reference():
        sd_model = processing.sd_model
        original_engine = sd_model.text_processing_engine_qwen
        sd_model.text_processing_engine_qwen = GroundedQwenEngine(
            original_engine, grounded_image
        )
        try:
            with forge_qwen_vision_attention_compat():
                return original_setup_conds()
        finally:
            sd_model.text_processing_engine_qwen = original_engine

    processing.setup_conds = setup_conds_with_reference
    processing._krea2_edit_grounding_installed = True
