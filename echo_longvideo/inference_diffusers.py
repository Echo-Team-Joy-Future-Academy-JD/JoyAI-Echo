"""Batch Echo 1.5 R2V inference using Hugging Face Diffusers components."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
_DISTILLATION_SRC = str(REPO_ROOT / "ltx-distillation" / "src")
if _DISTILLATION_SRC not in sys.path:
    sys.path.insert(0, _DISTILLATION_SRC)

import torch
import yaml

from diffusers.utils import encode_video

from diffusers_echo import EchoDiffusersPipeline
from ltx_distillation.audio_voice_filter import VoiceFilterConfig
from r2v_schema import load_r2v_request


DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference.diffusers.bf16.yaml"


def _resolve(value: str, *, required: bool = True) -> Path | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError("required path is empty")
        return None
    path = Path(text).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


class Config:
    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        paths = raw.get("paths", {})
        video = raw.get("video", {})
        denoising = raw.get("denoising", {})
        memory = raw.get("memory", {})
        runtime = raw.get("inference", {})
        voice = memory.get("voice_filter", {}) or {}

        self.base_model = _resolve(paths.get("base_model"))
        self.echo_model = _resolve(paths.get("echo_model"))
        self.requests_dir = _resolve(paths.get("requests_dir"))
        self.requests_glob = str(paths.get("requests_glob", "*.json"))
        self.output_root = _resolve(paths.get("output_root", "inference_result_diffusers"))
        self.num_frames = int(video.get("num_frames", 241))
        self.height = int(video.get("height", 736))
        self.width = int(video.get("width", 1280))
        self.fps = float(video.get("fps", 25))
        self.seed = int(video.get("seed", 42))
        self.sigmas = [float(value) for value in denoising.get("sigmas", [])]
        self.memory_max_size = int(memory.get("max_size", 7))
        self.enable_audio_memory = bool(memory.get("enable_audio", True))
        self.position_mode = str(memory.get("position_mode", "slot_center"))
        self.position_offset = float(memory.get("position_offset", 500.0))
        self.slot_stride = float(memory.get("position_slot_stride", 50.0))
        self.device = str(runtime.get("device", "cuda"))
        self.offload = str(runtime.get("offload", "model")).lower()
        self.max_sequence_length = int(runtime.get("max_sequence_length", 1024))
        self.vae_tiling = bool(runtime.get("vae_tiling", True))
        self.voice_filter = VoiceFilterConfig(
            enabled=bool(voice.get("enabled", False)),
            backend=str(voice.get("backend", "msst_speech")),
            min_output_rms=float(voice.get("min_output_rms", 0.004)),
            msst_dir=str(_resolve(voice.get("msst_dir", "third_party/MSST-WebUI"))),
            msst_model_path=str(
                _resolve(
                    voice.get(
                        "msst_model_path",
                        "checkpoints/msst/model_bandit_plus_dnr_sdr_11.47.chpt",
                    )
                )
            ),
            msst_config_path=str(
                _resolve(
                    voice.get(
                        "msst_config_path",
                        "third_party/MSST-WebUI/configs_backup/multi_stem_models/"
                        "model_bandit_plus_dnr_sdr_11.47.chpt.yaml",
                    )
                )
            ),
            msst_model_type=str(voice.get("msst_model_type", "bandit")),
            msst_sample_rate=int(voice.get("msst_sample_rate", 44100)),
            msst_device=str(voice.get("msst_device", "auto")),
            msst_local_rank_env=str(voice.get("msst_local_rank_env", "LOCAL_RANK")),
        )
        self.validate()

    def validate(self) -> None:
        if self.position_mode != "slot_center":
            raise ValueError("Diffusers Echo 1.5 currently requires memory.position_mode=slot_center")
        if self.offload not in {"none", "model", "sequential"}:
            raise ValueError("inference.offload must be none, model, or sequential")
        if len(self.sigmas) < 2 or self.sigmas[-1] != 0:
            raise ValueError("denoising.sigmas must contain at least two values and end in 0")
        if self.max_sequence_length <= 0:
            raise ValueError("inference.max_sequence_length must be positive")


def _load_requests(config: Config, request_path: str | None, limit: int | None):
    if request_path:
        files = [Path(request_path).expanduser().resolve()]
    else:
        files = [Path(value) for value in sorted(glob(str(config.requests_dir / config.requests_glob)))]
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no request files found under {config.requests_dir}")
    requests = [
        load_r2v_request(
            file,
            default_num_frames=config.num_frames,
            default_width=config.width,
            default_height=config.height,
            default_seed=config.seed,
        )
        for file in files
    ]
    return files, requests


def _validate_paths(config: Config) -> None:
    required = {
        "base model_index.json": config.base_model / "model_index.json",
        "Echo transformer config": config.echo_model / "transformer" / "config.json",
        "Echo connectors config": config.echo_model / "connectors" / "config.json",
        "Echo video VAE config": config.echo_model / "vae" / "config.json",
        "Echo audio VAE config": config.echo_model / "audio_vae" / "config.json",
        "Echo BWE vocoder config": config.echo_model / "vocoder" / "config.json",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Diffusers model files:\n" + "\n".join(missing))


def run(args: argparse.Namespace) -> None:
    config = Config(Path(args.config).expanduser().resolve())
    files, requests = _load_requests(config, args.request, args.limit)
    _validate_paths(config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "base_model": str(config.base_model),
                    "echo_model": str(config.echo_model),
                    "requests": [str(value) for value in files],
                    "sigmas": config.sigmas,
                    "offload": config.offload,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    pipeline = EchoDiffusersPipeline.from_pretrained(
        config.base_model,
        config.echo_model,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if config.vae_tiling:
        pipeline.enable_vae_tiling()
    if config.offload == "model":
        pipeline.enable_model_cpu_offload(config.device)
    elif config.offload == "sequential":
        pipeline.enable_sequential_cpu_offload(config.device)
    else:
        pipeline.to(config.device)

    for request_file, request in zip(files, requests, strict=True):
        if len(request.memory_slots) > config.memory_max_size:
            raise ValueError(
                f"{request.shot_id} has {len(request.memory_slots)} memory slots; "
                f"maximum is {config.memory_max_size}"
            )
        output_dir = config.output_root / request.work_id / request.shot_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "result.mp4"
        print(f"[diffusers] generating {request.shot_id}", flush=True)
        started = time.perf_counter()
        result = pipeline(
            request,
            sigmas=config.sigmas,
            frame_rate=config.fps,
            position_offset=config.position_offset,
            slot_stride=config.slot_stride,
            enable_audio_memory=(config.enable_audio_memory and not args.no_audio_memory),
            voice_filter_config=config.voice_filter,
            max_sequence_length=config.max_sequence_length,
            output_type="np",
        )
        sample_rate = int(pipeline.pipe.vocoder.config.output_sampling_rate)
        encode_video(
            result.frames[0],
            fps=int(config.fps),
            audio=result.audio[0].float().cpu(),
            audio_sample_rate=sample_rate,
            output_path=str(output_path),
        )
        elapsed = time.perf_counter() - started
        metadata = {
            "schema": "echo15.diffusers.result.v1",
            "request_file": str(request_file),
            "request": request.as_payload(),
            "base_model": str(config.base_model),
            "echo_model": str(config.echo_model),
            "sigmas": config.sigmas,
            "memory_slots": len(request.memory_slots),
            "has_audio_memory": bool(
                config.enable_audio_memory
                and not args.no_audio_memory
                and any(slot.audio_url for slot in request.memory_slots)
            ),
            "input_fingerprints": result.input_fingerprints,
            "video_latent_shape": list(result.video_latents.shape),
            "audio_latent_shape": list(result.audio_latents.shape),
            "elapsed_seconds": round(elapsed, 3),
            "output_path": str(output_path),
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[diffusers] done in {elapsed:.1f}s: {output_path}", flush=True)
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--request", help="run one request JSON instead of requests_dir")
    parser.add_argument("--limit", type=int, help="limit the sorted request list")
    parser.add_argument("--no-audio-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
