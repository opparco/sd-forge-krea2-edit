# Krea 2 Identity Edit for Forge Neo

Forge Neo adapter for
[ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit).

It implements both paths required by the Krea 2 Identity Edit LoRA, including
multi-reference controls:

- one or two clean reference latent blocks in the training-matched token order
- image-grounded positive and negative encoding through Forge's Qwen3-VL engine
- per-reference attention boost, an optional subject boost mask, and an advanced
  grounding system-prompt override

```text
single: [text | subject(frame=1) | target(frame=0)]
dual:   [text | scene(frame=1) | subject(frame=2) | target(frame=0)]
```

## Requirements

- Forge Neo with native Krea 2 support
- Krea 2 Raw or Turbo, including its Qwen3-VL text encoder and VAE
- [`krea2_identity_edit_v1_2.safetensors`](https://huggingface.co/conradlocke/krea2-identity-edit),
  obtained separately under its model license

The LoRA is not bundled. Put it in Forge's normal LoRA directory and activate it
at model strength `1.0` using the normal prompt syntax.

```text
<lora:krea2_identity_edit_v1_2:1>
```

## Usage

For detailed Japanese instructions and a feature-by-feature comparison with the
ComfyUI node pack, see [`docs/usage.ja.md`](docs/usage.ja.md).

1. Select Krea 2 Raw or Turbo.
2. Open the **Krea 2 Identity Edit** accordion under txt2img.
3. Upload the required subject reference. Optionally upload a scene reference for
   a two-reference edit.
4. Write an edit instruction in the normal positive prompt.
5. Activate `krea2_identity_edit_v1_2` at strength 1.0.
6. Generate at no more than 2 megapixels.

### Reference modes

| UI input | Required | Single-reference role | Two-reference role |
|---|---|---|---|
| Subject reference | Yes | Subject at frame 1 | Subject at frame 2 |
| Scene reference | No | Not used | Scene at frame 1 |

The UI is subject-first for convenience. In two-reference mode, the extension
internally reorders the inputs to `scene -> subject` for both the VAE-token and
Qwen3-VL paths, matching the ComfyUI implementation and training order.

### Controls

| Control | Behavior |
|---|---|
| Grounding resolution | Lower favors edit adherence; higher favors identity detail. Default: 768 |
| Subject reference boost | Applies to the required subject in both modes. `1.0` is off |
| Scene reference boost | Applies only when a scene reference is supplied. `1.0` is off |
| Subject boost mask | White areas select which subject-reference tokens receive the boost |
| Grounding system prompt override | Optional advanced override; empty preserves Forge's default template |

The boost mask addresses regions in the subject reference; it is not an inpaint
or output-region mask.

### Attention backend requirement for boost

Reference boost is implemented as an additive attention bias. The bundled
SageAttention 1.0.6 accepts but ignores this bias, so boost values have no effect
while SageAttention is active. Disable SageAttention and restart Forge when using
either boost control:

```bat
set COMMANDLINE_ARGS=--uv --api --disable-sage
```

Confirm that the startup log reports `Using PyTorch Cross Attention`.

### Additional LoRAs

Additional character/style LoRAs also affect the joint reference/target stream.
If their text-encoder weights distort reference grounding, load them with an
explicit zero TE strength, for example `<lora:name:te=0:unet=0.3>`.

### Starting settings

Recommended starting points:

- Turbo: 8 steps, CFG 1
- Raw: about 20 steps, CFG 3 for removals and other strong edits

At CFG greater than 1, the extension also grounds the negative prompt with the
same ordered reference image set.

The extension includes a scoped compatibility adapter for Forge Neo's Qwen3-VL
vision tower when SageAttention, FlashAttention, or PyTorch attention is selected.

## ComfyUI correspondence

| Forge Neo | ComfyUI-Krea2Edit |
|---|---|
| Subject only | Main `source_image` / `source_latent` |
| Scene + subject | Scene on main inputs; subject on `_b` inputs |
| Automatic positive/negative grounding | Separate grounded-encode nodes, including empty negative at CFG > 1 |
| Subject reference boost | `ref_boost` |
| Scene reference boost | `ref_boost_a` |
| Uploaded grayscale subject mask | `ref_boost_mask` (`MASK`) |
| v1.2 `fit` | `fit_mode: fit` |

See the [Japanese usage guide](docs/usage.ja.md) for detailed workflows,
differences, and troubleshooting.

## Current limits

- txt2img only
- v1.2 `fit` geometry only
- batch size 1; `n_iter` may be used for multiple outputs
- Hires. fix is disabled; generate at the final resolution and upscale afterward
- extension-specific reference inputs are not exposed through the Web API yet
- reference boost uses an additive attention mask and can require substantially
  more VRAM at high resolutions
- reference boost requires an attention backend that honors additive masks;
  bundled SageAttention 1.0.6 does not

## Development checks

From the Forge Neo repository root:

```powershell
$env:PYTHONPATH = @(
  (Resolve-Path extensions/sd-forge-krea2-edit).Path,
  (Resolve-Path .).Path,
  (Resolve-Path modules_forge/packages).Path
) -join ';'

.\venv\Scripts\python.exe -m unittest discover `
  -s extensions/sd-forge-krea2-edit/tests -v
.\venv\Scripts\python.exe -m ruff check extensions/sd-forge-krea2-edit
.\venv\Scripts\python.exe -m ruff format --check extensions/sd-forge-krea2-edit
```

## Attribution

The edit-forward and fit geometry are adapted from
`lbouaraba/comfyui-krea2edit`, licensed under Apache-2.0. Krea 2 and the Identity
Edit weights have their own licenses and are not redistributed here.
