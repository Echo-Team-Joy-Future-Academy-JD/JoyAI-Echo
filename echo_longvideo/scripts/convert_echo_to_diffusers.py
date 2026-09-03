#!/usr/bin/env python3
"""Convert all Echo 1.5 BF16 neural components to Diffusers format.

The Echo release is a combined checkpoint.  The converter emits its fine-tuned
transformer and text connectors as well as the exact video VAE, audio VAE, and
48 kHz BWE vocoder.  Only Gemma/tokenizer assets are reused from the configured
base directory at inference time.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors, LTX2VocoderWithBWE


TRANSFORMER_RENAMES = {
    "patchify_proj": "proj_in",
    "audio_patchify_proj": "audio_proj_in",
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
    "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
    "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
    "audio_prompt_adaln_single": "audio_prompt_adaln",
    "prompt_adaln_single": "prompt_adaln",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

CONNECTOR_RENAMES = {
    "connectors.": "",
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    "text_embedding_projection.audio_aggregate_embed": "audio_text_proj_in",
    "text_embedding_projection.video_aggregate_embed": "video_text_proj_in",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

VIDEO_VAE_RENAMES = {
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.1",
    "down_blocks.3": "down_blocks.1.downsamplers.0",
    "down_blocks.4": "down_blocks.2",
    "down_blocks.5": "down_blocks.2.downsamplers.0",
    "down_blocks.6": "down_blocks.3",
    "down_blocks.7": "down_blocks.3.downsamplers.0",
    "down_blocks.8": "mid_block",
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
    "last_time_embedder": "time_embedder",
    "last_scale_shift_table": "scale_shift_table",
    "res_blocks": "resnets",
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

AUDIO_VAE_RENAMES = {
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

VOCODER_RENAMES = {
    "resblocks": "resnets",
    "conv_pre": "conv_in",
    "conv_post": "conv_out",
    "act_post": "act_out",
    "downsample.lowpass": "downsample",
}

TRANSFORMER_CONFIG = {
    "in_channels": 128,
    "out_channels": 128,
    "patch_size": 1,
    "patch_size_t": 1,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "cross_attention_dim": 4096,
    "vae_scale_factors": (8, 32, 32),
    "pos_embed_max_pos": 20,
    "base_height": 2048,
    "base_width": 2048,
    "gated_attn": True,
    "cross_attn_mod": True,
    "audio_in_channels": 128,
    "audio_out_channels": 128,
    "audio_patch_size": 1,
    "audio_patch_size_t": 1,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "audio_scale_factor": 4,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16000,
    "audio_hop_length": 160,
    "audio_gated_attn": True,
    "audio_cross_attn_mod": True,
    "num_layers": 48,
    "activation_fn": "gelu-approximate",
    "qk_norm": "rms_norm_across_heads",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "caption_channels": 3840,
    "attention_bias": True,
    "attention_out_bias": True,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_offset": 1,
    "timestep_scale_multiplier": 1000,
    "cross_attn_timestep_scale_multiplier": 1000,
    "rope_type": "split",
    "use_prompt_embeddings": False,
    "perturbed_attn": True,
}

CONNECTOR_CONFIG = {
    "caption_channels": 3840,
    "text_proj_in_factor": 49,
    "video_connector_num_attention_heads": 32,
    "video_connector_attention_head_dim": 128,
    "video_connector_num_layers": 8,
    "video_connector_num_learnable_registers": 128,
    "video_gated_attn": True,
    "audio_connector_num_attention_heads": 32,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_layers": 8,
    "audio_connector_num_learnable_registers": 128,
    "audio_gated_attn": True,
    "connector_rope_base_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_temporal_positioning": False,
    "rope_type": "split",
    "per_modality_projections": True,
    "video_hidden_dim": 4096,
    "audio_hidden_dim": 2048,
    "proj_bias": True,
}

VIDEO_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    "block_out_channels": (256, 512, 1024, 1024),
    "down_block_types": (
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
    ),
    "decoder_block_out_channels": (256, 512, 512, 1024),
    "layers_per_block": (4, 6, 4, 2, 2),
    "decoder_layers_per_block": (4, 6, 4, 2, 2),
    "spatio_temporal_scaling": (True, True, True, True),
    "decoder_spatio_temporal_scaling": (True, True, True, True),
    "decoder_inject_noise": (False, False, False, False, False),
    "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    "upsample_type": ("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    "upsample_residual": (False, False, False, False),
    "upsample_factor": (2, 2, 1, 2),
    "timestep_conditioning": False,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-6,
    "encoder_causal": True,
    "decoder_causal": False,
    "encoder_spatial_padding_mode": "zeros",
    "decoder_spatial_padding_mode": "zeros",
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
}

AUDIO_VAE_CONFIG = {
    "base_channels": 128,
    "output_channels": 2,
    "ch_mult": (1, 2, 4),
    "num_res_blocks": 2,
    "attn_resolutions": None,
    "in_channels": 2,
    "resolution": 256,
    "latent_channels": 8,
    "norm_type": "pixel",
    "causality_axis": "height",
    "dropout": 0.0,
    "mid_block_add_attention": False,
    "sample_rate": 16000,
    "mel_hop_length": 160,
    "is_causal": True,
    "mel_bins": 64,
    "double_z": True,
}

VOCODER_CONFIG = {
    "in_channels": 128,
    "hidden_channels": 1536,
    "out_channels": 2,
    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
    "upsample_factors": [5, 2, 2, 2, 2, 2],
    "resnet_kernel_sizes": [3, 7, 11],
    "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "act_fn": "snakebeta",
    "leaky_relu_negative_slope": 0.1,
    "antialias": True,
    "antialias_ratio": 2,
    "antialias_kernel_size": 12,
    "final_act_fn": None,
    "final_bias": False,
    "bwe_in_channels": 128,
    "bwe_hidden_channels": 512,
    "bwe_out_channels": 2,
    "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
    "bwe_upsample_factors": [6, 5, 2, 2, 2],
    "bwe_resnet_kernel_sizes": [3, 7, 11],
    "bwe_resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "bwe_act_fn": "snakebeta",
    "bwe_leaky_relu_negative_slope": 0.1,
    "bwe_antialias": True,
    "bwe_antialias_ratio": 2,
    "bwe_antialias_kernel_size": 12,
    "bwe_final_act_fn": None,
    "bwe_final_bias": False,
    "filter_length": 512,
    "hop_length": 80,
    "window_length": 512,
    "num_mel_channels": 64,
    "input_sampling_rate": 16000,
    "output_sampling_rate": 48000,
}


def _rename(state: dict[str, Any], replacements: dict[str, str]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in state.items():
        new_key = key
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        if new_key.startswith("adaln_single."):
            new_key = new_key.replace("adaln_single.", "time_embed.", 1)
        elif new_key.startswith("audio_adaln_single."):
            new_key = new_key.replace("audio_adaln_single.", "audio_time_embed.", 1)
        if new_key in converted:
            raise KeyError(f"conversion produced duplicate key: {new_key}")
        converted[new_key] = value
    return converted


def _split_checkpoint(checkpoint: dict[str, torch.Tensor]):
    dit_prefix = "model.diffusion_model."
    original: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if key.startswith(dit_prefix):
            original[key.removeprefix(dit_prefix)] = value
        elif key.startswith("text_embedding_projection."):
            original[key] = value

    connector_markers = (
        "video_embeddings_connector",
        "audio_embeddings_connector",
        "text_embedding_projection",
        "connectors.",
        "video_connector",
        "audio_connector",
    )
    transformer: dict[str, torch.Tensor] = {}
    connectors: dict[str, torch.Tensor] = {}
    for key, value in original.items():
        target = connectors if any(marker in key for marker in connector_markers) else transformer
        target[key] = value
    components = {
        "transformer": transformer,
        "connectors": connectors,
        "vae": {},
        "audio_vae": {},
        "vocoder": {},
    }
    for key, value in checkpoint.items():
        for prefix in ("vae.", "audio_vae.", "vocoder."):
            if key.startswith(prefix):
                components[prefix[:-1]][key.removeprefix(prefix)] = value
                break
    return components


def _convert_video_vae_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = _rename(state, VIDEO_VAE_RENAMES)
    return {
        key: value
        for key, value in converted.items()
        if "per_channel_statistics.channel" not in key
        and "per_channel_statistics.mean-of-stds" not in key
    }


def _convert_vocoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = _rename(state, VOCODER_RENAMES)
    result: dict[str, torch.Tensor] = {}
    for key, value in converted.items():
        if ".ups." in key and (".weight" in key or ".bias" in key):
            key = key.replace(".ups.", ".upsamplers.")
        result[key] = value
    return result


def _strict_load_and_save(
    model_cls,
    config: dict[str, Any],
    state: dict[str, torch.Tensor],
    output_path: Path,
    *,
    max_shard_size: str,
):
    with init_empty_weights():
        model = model_cls.from_config(config)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{output_path.name} mismatch: {incompatible}")
    model.to(torch.bfloat16).save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    return model


def _manifest_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    manifest_path = path / "checkpoint.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    precision = str(manifest.get("precision", "")).lower()
    if precision not in {"bf16", "bfloat16"}:
        raise ValueError(
            f"Diffusers conversion currently supports Echo BF16 only, got {precision!r}. "
            "The released FP8/FP4 files use custom scaled-mm/ModelOpt packing."
        )
    model_name = manifest.get("files", {}).get("model")
    if not model_name:
        raise ValueError(f"manifest has no files.model: {manifest_path}")
    return path / model_name


def convert(checkpoint_path: Path, output_path: Path, *, max_shard_size: str) -> None:
    source = _manifest_checkpoint(checkpoint_path.expanduser().resolve())
    output_path = output_path.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading {source}", flush=True)
    checkpoint = load_file(str(source), device="cpu")
    components = _split_checkpoint(checkpoint)
    del checkpoint
    missing_components = [name for name, state in components.items() if not state]
    if missing_components:
        raise ValueError(f"checkpoint is missing components: {missing_components}")

    transformer_state = _rename(components.pop("transformer"), TRANSFORMER_RENAMES)
    with init_empty_weights():
        transformer = LTX2VideoTransformer3DModel.from_config(TRANSFORMER_CONFIG)
    incompatible = transformer.load_state_dict(transformer_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"transformer mismatch: {incompatible}")
    transformer.to(torch.bfloat16).save_pretrained(
        output_path / "transformer",
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    transformer_tensor_count = len(transformer_state)
    del transformer, transformer_state
    gc.collect()

    connector_state = _rename(components.pop("connectors"), CONNECTOR_RENAMES)
    with init_empty_weights():
        connectors = LTX2TextConnectors.from_config(CONNECTOR_CONFIG)
    incompatible = connectors.load_state_dict(connector_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"connectors mismatch: {incompatible}")
    connectors.to(torch.bfloat16).save_pretrained(
        output_path / "connectors",
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    connector_tensor_count = len(connector_state)
    del connectors, connector_state
    gc.collect()

    vae_state = _convert_video_vae_state(components.pop("vae"))
    vae = _strict_load_and_save(
        AutoencoderKLLTX2Video,
        VIDEO_VAE_CONFIG,
        vae_state,
        output_path / "vae",
        max_shard_size=max_shard_size,
    )
    vae_tensor_count = len(vae_state)
    del vae, vae_state
    gc.collect()

    audio_vae_state = _rename(components.pop("audio_vae"), AUDIO_VAE_RENAMES)
    audio_vae = _strict_load_and_save(
        AutoencoderKLLTX2Audio,
        AUDIO_VAE_CONFIG,
        audio_vae_state,
        output_path / "audio_vae",
        max_shard_size=max_shard_size,
    )
    audio_vae_tensor_count = len(audio_vae_state)
    del audio_vae, audio_vae_state
    gc.collect()

    vocoder_state = _convert_vocoder_state(components.pop("vocoder"))
    vocoder = _strict_load_and_save(
        LTX2VocoderWithBWE,
        VOCODER_CONFIG,
        vocoder_state,
        output_path / "vocoder",
        max_shard_size=max_shard_size,
    )
    vocoder_tensor_count = len(vocoder_state)
    del vocoder, vocoder_state
    gc.collect()

    report = {
        "schema": "echo15.diffusers.conversion.v2",
        "source": str(source),
        "precision": "bfloat16",
        "transformer_tensors": transformer_tensor_count,
        "connector_tensors": connector_tensor_count,
        "video_vae_tensors": vae_tensor_count,
        "audio_vae_tensors": audio_vae_tensor_count,
        "vocoder_tensors": vocoder_tensor_count,
        "vocoder_output_sample_rate": 48000,
        "base_model_usage": "Gemma text encoder and tokenizer only",
    }
    (output_path / "conversion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[convert] complete: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.checkpoint, args.output, max_shard_size=args.max_shard_size)
