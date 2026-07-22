from __future__ import annotations

import logging

import gradio as gr

from backend.diffusion_engine.krea import Krea2
from modules import scripts
from modules.ui_components import InputAccordion

from lib_krea2_edit.conditioning import install_grounded_setup
from lib_krea2_edit.image_utils import limit_long_side, pil_to_bhwc
from lib_krea2_edit.patch import patch_krea2_unet

logger = logging.getLogger(__name__)


class Krea2IdentityEditForForge(scripts.ScriptBuiltinUI):
    sorting_priority = 18120

    def title(self):
        return "Krea 2 Identity Edit"

    def show(self, is_img2img):
        return None if is_img2img else scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(
            False, label=self.title(), elem_id=self.elem_id("krea2_edit_enable")
        ) as enabled:
            reference = gr.Image(
                label="Reference image",
                type="pil",
                image_mode="RGB",
                source="upload",
                interactive=True,
            )
            grounding_px = gr.Slider(
                minimum=256,
                maximum=1536,
                value=768,
                step=64,
                label="Grounding resolution",
                info="Lower favors edit adherence; higher favors identity detail.",
            )
            gr.Markdown(
                "Use the normal positive prompt as the edit instruction. "
                "Load `krea2_identity_edit_v1_2` with Forge's normal LoRA syntax."
            )

        for component in (enabled, reference, grounding_px):
            component.do_not_save_to_config = True

        self.infotext_fields = [
            (enabled, "Krea2 Edit"),
            (grounding_px, "Krea2 Edit Grounding"),
        ]
        return enabled, reference, grounding_px

    def process(self, p, enabled, reference, grounding_px):
        if not enabled:
            return
        if reference is None:
            raise ValueError("Krea 2 Identity Edit requires a reference image.")
        if not isinstance(p.sd_model, Krea2):
            raise ValueError(
                "Krea 2 Identity Edit requires a Krea 2 Raw or Turbo model."
            )
        if p.batch_size != 1:
            raise ValueError(
                "Krea 2 Identity Edit Phase 1 supports batch size 1 only. Use n_iter for multiple outputs."
            )
        if getattr(p, "enable_hr", False):
            raise ValueError(
                "Krea 2 Identity Edit Phase 1 does not support Hires. fix. Generate at the final resolution and upscale afterward."
            )
        if p.width * p.height > 2_000_000:
            raise ValueError(
                "Krea 2 Identity Edit supports output resolutions up to 2 megapixels."
            )

        reference_tensor = pil_to_bhwc(reference)
        grounded_reference = limit_long_side(reference_tensor, int(grounding_px))
        install_grounded_setup(p, grounded_reference)
        p._krea2_edit_reference = reference_tensor
        p.extra_generation_params["Krea2 Edit"] = "v1.2 fit"
        p.extra_generation_params["Krea2 Edit Grounding"] = int(grounding_px)

    def process_before_every_sampling(
        self, p, enabled, reference, grounding_px, **kwargs
    ):
        if not enabled:
            return
        unet = p.sd_model.forge_objects.unet
        diffusion_model = unet.model.diffusion_model
        if not getattr(diffusion_model, "loras", []):
            logger.warning(
                "[Krea2 Edit] WARNING: no active LoRA was detected. "
                "Load krea2_identity_edit_v1_2 at model strength 1.0 unless it is merged into the model."
            )
        p.sd_model.forge_objects.unet = patch_krea2_unet(
            unet,
            p.sd_model.forge_objects.vae,
            p._krea2_edit_reference,
        )
