from __future__ import annotations


class GroundedQwenEngine:
    """Job-local proxy that adds one reference image to every prompt encode."""

    def __init__(self, engine, image):
        self._engine = engine
        self._image = image

    def __call__(self, texts, images=None):
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
            return original_setup_conds()
        finally:
            sd_model.text_processing_engine_qwen = original_engine

    processing.setup_conds = setup_conds_with_reference
    processing._krea2_edit_grounding_installed = True
