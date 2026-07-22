from __future__ import annotations

import logging

import gradio as gr

from backend.diffusion_engine.krea import Krea2
from modules import scripts
from modules.ui_components import InputAccordion

from lib_krea2_edit.conditioning import install_grounded_setup
from lib_krea2_edit.image_utils import (
    limit_long_side,
    order_reference_images,
    pil_mask_to_bhw,
    pil_to_bhwc,
)
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
            subject_reference = gr.Image(
                label="Subject reference (required)",
                type="pil",
                image_mode="RGB",
                sources="upload",
                interactive=True,
            )
            scene_reference = gr.Image(
                label="Scene reference (optional)",
                type="pil",
                image_mode="RGB",
                sources="upload",
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
            ref_boost = gr.Slider(
                minimum=0.0,
                maximum=16.0,
                value=1.0,
                step=0.05,
                label="Subject reference boost",
                info="Applies to the required subject reference.",
            )
            ref_boost_a = gr.Slider(
                minimum=0.0,
                maximum=16.0,
                value=1.0,
                step=0.05,
                label="Scene reference boost",
                info="Applies only when an optional scene reference is present.",
            )
            ref_boost_mask = gr.Image(
                label="Subject boost mask (optional; white = boosted)",
                type="pil",
                image_mode="L",
                sources="upload",
                interactive=True,
            )
            system_prompt = gr.Textbox(
                label="Grounding system prompt override (advanced)",
                lines=2,
                value="",
                info="Empty preserves Forge's default image-grounding template.",
            )
            gr.Markdown(
                "Use the normal positive prompt as the edit instruction. "
                "Load `krea2_identity_edit_v1_2` with Forge's normal LoRA syntax."
            )

        components = (
            enabled,
            subject_reference,
            scene_reference,
            grounding_px,
            ref_boost,
            ref_boost_a,
            ref_boost_mask,
            system_prompt,
        )
        for component in components:
            component.do_not_save_to_config = True

        self.infotext_fields = [
            (enabled, "Krea2 Edit"),
            (grounding_px, "Krea2 Edit Grounding"),
            (ref_boost, "Krea2 Edit Subject Boost"),
            (ref_boost_a, "Krea2 Edit Scene Boost"),
        ]
        return components

    def process(
        self,
        p,
        enabled,
        subject_reference,
        scene_reference,
        grounding_px,
        ref_boost,
        ref_boost_a,
        ref_boost_mask,
        system_prompt,
    ):
        if not enabled:
            return
        if subject_reference is None:
            raise ValueError("Krea 2 Identity Edit requires a subject reference.")
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

        subject_tensor = pil_to_bhwc(subject_reference)
        scene_tensor = (
            pil_to_bhwc(scene_reference) if scene_reference is not None else None
        )
        references = order_reference_images(subject_tensor, scene_tensor)
        grounded_references = [
            limit_long_side(item, int(grounding_px)) for item in references
        ]
        install_grounded_setup(p, grounded_references, system_prompt)
        p._krea2_edit_references = references
        p._krea2_edit_ref_boost = float(ref_boost)
        p._krea2_edit_ref_boost_a = float(ref_boost_a)
        p._krea2_edit_ref_boost_mask = (
            pil_mask_to_bhw(ref_boost_mask) if ref_boost_mask is not None else None
        )
        p.extra_generation_params["Krea2 Edit"] = (
            "v1.2 fit dual" if scene_reference is not None else "v1.2 fit"
        )
        p.extra_generation_params["Krea2 Edit Grounding"] = int(grounding_px)
        if float(ref_boost) != 1.0:
            p.extra_generation_params["Krea2 Edit Subject Boost"] = float(ref_boost)
        if scene_reference is not None and float(ref_boost_a) != 1.0:
            p.extra_generation_params["Krea2 Edit Scene Boost"] = float(ref_boost_a)
        if ref_boost_mask is not None:
            p.extra_generation_params["Krea2 Edit Boost Mask"] = True
        if system_prompt.strip():
            p.extra_generation_params["Krea2 Edit System Prompt"] = "custom"

    def process_before_every_sampling(
        self,
        p,
        enabled,
        subject_reference,
        scene_reference,
        grounding_px,
        ref_boost,
        ref_boost_a,
        ref_boost_mask,
        system_prompt,
        **kwargs,
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
            p._krea2_edit_references,
            ref_boost=p._krea2_edit_ref_boost,
            ref_boost_a=p._krea2_edit_ref_boost_a,
            ref_boost_mask=p._krea2_edit_ref_boost_mask,
        )
