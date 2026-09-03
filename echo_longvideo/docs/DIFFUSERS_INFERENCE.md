# Diffusers inference backend

Echo 1.5 can run on Hugging Face Diffusers without changing the released BF16
checkpoint's generation semantics. The backend uses Diffusers for the LTX-2.3
Transformer, text connectors, Gemma encoder, video/audio VAEs, vocoder,
offloading, and media post-processing. It intentionally keeps Echo's custom
parts:

- the 8-step stochastic DMD schedule (predict x0, then re-noise with fresh
  Gaussian noise at the next sigma);
- zero-noise image and audio memory tokens;
- paired memory slots at virtual `slot_center` timestamps;
- first-frame tokens fixed at timestep zero throughout sampling.

## Install

Use the same Python 3.11 / PyTorch 2.8 / CUDA 12.8 environment as the reference
backend, then install Diffusers support:

```bash
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
uv pip install --upgrade -r requirements-diffusers.txt
```

The current requirements file tracks Diffusers main because the required
LTX-2.3 condition pipeline landed after older tagged releases. Install the two
files in sequence: the Diffusers step intentionally upgrades Transformers and
Hugging Face Hub beyond the reference backend's pins. A dedicated virtual
environment is recommended when both backends must remain available.

## Convert the Echo BF16 checkpoint

The release checkpoint is in the original LTX combined format. Convert its
Transformer, connectors, video VAE, audio VAE, and 48 kHz BWE vocoder once:

```bash
python scripts/convert_echo_to_diffusers.py \
  --checkpoint checkpoints/echo15_full_dmd \
  --output checkpoints/echo15_full_dmd_diffusers
```

The inference config's `base_model` directory supplies only Gemma/tokenizer
assets (and the Diffusers pipeline metadata). Neural image/audio components are
always loaded from the converted Echo directory; using a generic LTX VAE or
vocoder changes ref/memory latents and is not numerically equivalent.

The converter performs strict state-dict loading. A missing, unexpected, or
shape-mismatched tensor aborts conversion instead of producing a partially
initialized model.

This is not the stock LTX text/image-to-video scheduler. `diffusers_echo`
implements Echo's ref/multishot-memory layout and DMD update loop directly,
including the released fixed sigma sequence, first-frame zero timestep,
cross-modal timestep exchange, and re-noising between DMD steps.

## Run

```bash
# Validate paths and request parsing without loading weights.
python inference_diffusers.py \
  --config configs/inference.diffusers.bf16.yaml \
  --dry-run --limit 1

# Generate one request.
python inference_diffusers.py \
  --config configs/inference.diffusers.bf16.yaml \
  --request examples/the_last_visa/requests/009_01_shot_008_nathan_replies_to_elena_r2v.json

# Process the configured request directory.
python inference_diffusers.py --config configs/inference.diffusers.bf16.yaml
```

Results are written to
`inference_result_diffusers/<work-id>/<shot-id>/result.mp4`, with a
`run_metadata.json` record containing the model paths, sigma schedule, latent
shapes, memory inputs, fingerprints, and runtime.

`inference.offload` accepts `none`, `model`, or `sequential`. `model` is the
balanced default; `sequential` minimizes VRAM at the cost of transfer overhead.

The repository case above is 241 frames at 1280x736 with one conditioning
frame and three paired image/audio memory slots. It is the recommended parity
case against the original `inference.py` backend.

## Precision support

This backend currently converts the BF16 `echo15_full_dmd` checkpoint. The
released FP8 and FP4 variants use the reference backend's custom scaled-matmul
and NVIDIA ModelOpt packed formats. Diffusers does not load those files as
ordinary `LTX2VideoTransformer3DModel` weights, so the converter rejects them
explicitly instead of silently dequantizing or changing numerics.
