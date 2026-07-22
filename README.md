# Krea 2 Identity Edit for Forge Neo

Phase 1 Forge Neo adapter for
[ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit).

It implements both paths required by the Krea 2 Identity Edit LoRA:

- clean reference latent tokens in `[text | source(frame=1) | target(frame=0)]`
- image-grounded positive and negative encoding through Forge's Qwen3-VL engine

## Requirements

- Forge Neo with native Krea 2 support
- Krea 2 Raw or Turbo, including its Qwen3-VL text encoder and VAE
- [`krea2_identity_edit_v1_2.safetensors`](https://huggingface.co/conradlocke/krea2-identity-edit),
  obtained separately under its model license

The LoRA is not bundled. Put it in Forge's normal LoRA directory and activate it
at model strength `1.0` using the normal prompt syntax.

## Usage

1. Select Krea 2 Raw or Turbo.
2. Open the **Krea 2 Identity Edit** accordion under txt2img.
3. Upload one reference image.
4. Write an edit instruction in the normal positive prompt.
5. Activate `krea2_identity_edit_v1_2` at strength 1.0.
6. Generate at no more than 2 megapixels.

Recommended starting points:

- Turbo: 8 steps, CFG 1
- Raw: about 20 steps, CFG 3 for removals and other strong edits

At CFG greater than 1, the extension also grounds the negative prompt with the
same reference image.

## Phase 1 limits

- txt2img only
- one reference image
- v1.2 `fit` geometry only
- batch size 1; `n_iter` may be used for multiple outputs
- Hires. fix is disabled; generate at the final resolution and upscale afterward
- no regional reference-boost mask or two-reference editing yet

## Attribution

The edit-forward and fit geometry are adapted from
`lbouaraba/comfyui-krea2edit`, licensed under Apache-2.0. Krea 2 and the Identity
Edit weights have their own licenses and are not redistributed here.
