"""Diffusers-native Echo 1.5 DMD inference.

This module intentionally keeps Echo's sampler and memory-token semantics while
using Hugging Face Diffusers for the LTX-2.3 transformer, connectors, VAEs,
text encoder, vocoder, offload hooks, and output processing.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import torch
import torchaudio
from PIL import Image

from r2v_schema import R2VRequest


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 100 * 1024 * 1024
# The released Echo wrapper always builds video RoPE coordinates at the
# training-time rate, independently of the output container frame rate.
MODEL_POSITION_FPS = 24.0


@dataclass
class EchoPipelineOutput:
    """Decoded media plus the final normalized target latents."""

    frames: Any
    audio: torch.Tensor
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    input_fingerprints: dict[str, str]


@dataclass
class _PreparedConditions:
    first_frame_tokens: torch.Tensor | None
    memory_video_tokens: torch.Tensor | None
    memory_video_coords: torch.Tensor | None
    memory_audio_tokens: torch.Tensor | None
    memory_audio_coords: torch.Tensor | None
    input_fingerprints: dict[str, str]


class _ResourceStore:
    def __init__(self) -> None:
        self._bytes: dict[str, bytes] = {}
        self.fingerprints: dict[str, str] = {}

    def read(self, source: str, *, kind: str, max_bytes: int) -> bytes:
        if source in self._bytes:
            return self._bytes[source]
        if source.startswith("data:"):
            header, separator, encoded = source.partition(",")
            if not separator or ";base64" not in header.lower():
                raise ValueError(f"{kind} has an invalid data URL")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"{kind} has invalid base64 data") from exc
        elif source.startswith(("http://", "https://")):
            request = Request(source, headers={"User-Agent": "JoyAI-Echo15-Diffusers/1.0"})
            with urlopen(request, timeout=30) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > max_bytes:
                    raise ValueError(f"{kind} exceeds {max_bytes} bytes: {source}")
                data = response.read(max_bytes + 1)
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"{kind} not found: {path}")
            if path.stat().st_size > max_bytes:
                raise ValueError(f"{kind} exceeds {max_bytes} bytes: {path}")
            data = path.read_bytes()
        if not data or len(data) > max_bytes:
            raise ValueError(f"{kind} must contain 1..{max_bytes} bytes: {source}")
        self._bytes[source] = data
        self.fingerprints[source] = hashlib.sha256(data).hexdigest()
        return data

    def image(self, source: str) -> Image.Image:
        raw = self.read(source, kind="R2V image", max_bytes=MAX_IMAGE_BYTES)
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            return image.convert("RGB")

    def audio(self, source: str) -> tuple[torch.Tensor, int]:
        raw = self.read(source, kind="R2V audio", max_bytes=MAX_AUDIO_BYTES)
        waveform, sample_rate = torchaudio.load(io.BytesIO(raw))
        return waveform.detach().cpu().float().contiguous(), int(sample_rate)


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for value in module.parameters():
        return value.dtype
    for value in module.buffers():
        return value.dtype
    return torch.float32


def _normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(waveform).detach().cpu().float()
    while value.ndim > 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim > 2:
        value = value.reshape(value.shape[-2], value.shape[-1])
    if value.ndim != 2 or value.shape[-1] <= 1:
        raise ValueError(f"R2V audio has no usable samples: shape={tuple(value.shape)}")
    if value.shape[0] == 1:
        value = value.repeat(2, 1)
    elif value.shape[0] > 2:
        value = value[:2]
    return value.contiguous()


class EchoDiffusersPipeline:
    """Run Echo's stochastic few-step DMD sampler on Diffusers components."""

    def __init__(self, pipe) -> None:
        self.pipe = pipe

    @classmethod
    def from_pretrained(
        cls,
        base_model: str | Path,
        echo_model: str | Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
    ) -> "EchoDiffusersPipeline":
        from diffusers import (
            AutoencoderKLLTX2Audio,
            AutoencoderKLLTX2Video,
            LTX2ConditionPipeline,
            LTX2VideoTransformer3DModel,
        )
        from diffusers.pipelines.ltx2 import (
            LTX2TextConnectors,
            LTX2VocoderWithBWE,
        )

        echo_model = Path(echo_model)
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            echo_model,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        connectors = LTX2TextConnectors.from_pretrained(
            echo_model,
            subfolder="connectors",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        # Echo ships these components in the combined release checkpoint.  They
        # are part of the numerical contract: in particular, Echo 1.5 uses the
        # LTX-2.3 video latent statistics and the 48 kHz BWE vocoder.  Loading
        # them from an unrelated base pipeline changes conditioning latents and
        # can corrupt the DMD trajectory even when the DiT weights are correct.
        vae = AutoencoderKLLTX2Video.from_pretrained(
            echo_model,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
            echo_model,
            subfolder="audio_vae",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        vocoder = LTX2VocoderWithBWE.from_pretrained(
            echo_model,
            subfolder="vocoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        pipe = LTX2ConditionPipeline.from_pretrained(
            base_model,
            transformer=transformer,
            connectors=connectors,
            vae=vae,
            audio_vae=audio_vae,
            vocoder=vocoder,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        return cls(pipe)

    @property
    def device(self) -> torch.device:
        return self.pipe._execution_device

    def enable_model_cpu_offload(self, device: str | torch.device = "cuda") -> None:
        self.pipe.enable_model_cpu_offload(device=device)

    def enable_sequential_cpu_offload(self, device: str | torch.device = "cuda") -> None:
        self.pipe.enable_sequential_cpu_offload(device=device)

    def to(self, device: str | torch.device) -> "EchoDiffusersPipeline":
        self.pipe.to(device)
        return self

    def enable_vae_tiling(self) -> None:
        self.pipe.vae.enable_tiling()

    def _image_pixels(self, image: Image.Image, *, height: int, width: int) -> torch.Tensor:
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy()
        pixels = torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0
        return pixels.unsqueeze(0).unsqueeze(2)

    @torch.inference_mode()
    def _encode_image(
        self,
        image: Image.Image,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pixels = self._image_pixels(image, height=height, width=width).to(
            device=device, dtype=self.pipe.vae.dtype
        )
        latents = self.pipe.vae.encode(pixels).latent_dist.mode()
        latents = self.pipe._normalize_latents(
            latents,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        ).to(device=device, dtype=dtype)
        return self.pipe._pack_latents(
            latents,
            self.pipe.transformer_spatial_patch_size,
            self.pipe.transformer_temporal_patch_size,
        )

    def _waveform_to_mel(
        self, waveform: torch.Tensor, sample_rate: int, *, device: torch.device
    ) -> torch.Tensor:
        target_rate = int(self.pipe.audio_vae.config.sample_rate)
        waveform = waveform.to(device=device, dtype=torch.float32)
        if sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=int(self.pipe.audio_vae.config.mel_hop_length),
            f_min=0.0,
            f_max=target_rate / 2.0,
            n_mels=int(self.pipe.audio_vae.config.mel_bins),
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        ).to(device)(waveform)
        return torch.log(torch.clamp(mel, min=1e-5)).permute(0, 2, 1).unsqueeze(0)

    @torch.inference_mode()
    def _encode_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        waveform = _normalize_waveform(waveform)
        # Echo's released R2V conditioner explicitly runs the audio encoder in
        # FP32.  Using the pipeline-wide BF16 dtype changes memory audio tokens
        # enough to perturb the few-step DMD trajectory.
        mel = self._waveform_to_mel(waveform, sample_rate, device=device).to(
            dtype=_module_dtype(self.pipe.audio_vae.encoder)
        )
        latents = self.pipe.audio_vae.encode(mel).latent_dist.mode()
        latents = self.pipe._pack_audio_latents(latents)
        return self.pipe._normalize_audio_latents(
            latents, self.pipe.audio_vae.latents_mean, self.pipe.audio_vae.latents_std
        ).to(device=device, dtype=dtype)

    def _slot_center_video_coords(
        self,
        *,
        slots: int,
        latent_height: int,
        latent_width: int,
        frame_rate: float,
        position_offset: float,
        slot_stride: float,
        device: torch.device,
    ) -> torch.Tensor:
        base = self.pipe.transformer.rope.prepare_video_coords(
            1, 1, latent_height, latent_width, device, fps=frame_rate
        )
        pieces = []
        for slot_idx in range(slots):
            piece = base.clone()
            center = float(position_offset) + slot_idx * float(slot_stride)
            midpoint = (piece[:, 0, :1, 0] + piece[:, 0, :1, 1]) * 0.5
            piece[:, 0] += (center - midpoint).view(1, 1, 1)
            pieces.append(piece)
        return torch.cat(pieces, dim=2)

    def _slot_center_audio_coords(
        self,
        lengths: list[int],
        *,
        position_offset: float,
        slot_stride: float,
        device: torch.device,
    ) -> torch.Tensor:
        pieces = []
        for slot_idx, length in enumerate(lengths):
            piece = self.pipe.transformer.audio_rope.prepare_audio_coords(1, length, device)
            center = float(position_offset) + slot_idx * float(slot_stride)
            midpoint = (piece[:, 0, :1, 0] + piece[:, 0, -1:, 1]) * 0.5
            piece[:, 0] += (center - midpoint).view(1, 1, 1)
            pieces.append(piece)
        return torch.cat(pieces, dim=2)

    @torch.inference_mode()
    def prepare_conditions(
        self,
        request: R2VRequest,
        *,
        frame_rate: float,
        position_offset: float,
        slot_stride: float,
        enable_audio_memory: bool,
        voice_filter_config=None,
    ) -> _PreparedConditions:
        if any(slot.shot_id for slot in request.memory_slots):
            raise ValueError(
                "offline diffusers inference cannot resolve shot_id memory; use image_url/audio_url"
            )
        if any(not slot.image_url for slot in request.memory_slots):
            raise ValueError("every offline R2V memory slot requires image_url")

        device = self.device
        dtype = _module_dtype(self.pipe.transformer)
        resources = _ResourceStore()
        first_frame = (
            self._encode_image(
                resources.image(request.condition_img),
                height=request.height,
                width=request.width,
                device=device,
                dtype=dtype,
            )
            if request.condition_img
            else None
        )

        video_slices = [
            self._encode_image(
                resources.image(slot.image_url),
                height=request.height,
                width=request.width,
                device=device,
                dtype=dtype,
            )
            for slot in request.memory_slots
        ]
        memory_video = torch.cat(video_slices, dim=1) if video_slices else None
        memory_video_coords = None
        if memory_video is not None:
            memory_video_coords = self._slot_center_video_coords(
                slots=len(video_slices),
                latent_height=request.height // self.pipe.vae_spatial_compression_ratio,
                latent_width=request.width // self.pipe.vae_spatial_compression_ratio,
                frame_rate=MODEL_POSITION_FPS,
                position_offset=position_offset,
                slot_stride=slot_stride,
                device=device,
            )

        audio_slices: list[torch.Tensor | None] = []
        if enable_audio_memory:
            encoder_dtype = _module_dtype(self.pipe.audio_vae.encoder)
            self.pipe.audio_vae.encoder.to(dtype=torch.float32)
            try:
                for slot in request.memory_slots:
                    if not slot.audio_url:
                        audio_slices.append(None)
                        continue
                    waveform, sample_rate = resources.audio(slot.audio_url)
                    if voice_filter_config is not None and voice_filter_config.enabled:
                        from ltx_distillation.audio_voice_filter import filter_voice_only

                        waveform = filter_voice_only(waveform, sample_rate, voice_filter_config)
                        if waveform is None:
                            raise ValueError(f"voice filter removed all memory audio: {slot.audio_url}")
                    audio_slices.append(
                        self._encode_audio(
                            waveform, sample_rate, device=device, dtype=dtype
                        )
                    )
            finally:
                self.pipe.audio_vae.encoder.to(dtype=encoder_dtype)

        template = next((value for value in audio_slices if value is not None), None)
        memory_audio = None
        memory_audio_coords = None
        if template is not None:
            aligned = [value if value is not None else torch.zeros_like(template) for value in audio_slices]
            lengths = [int(value.shape[1]) for value in aligned]
            memory_audio = torch.cat(aligned, dim=1)
            memory_audio_coords = self._slot_center_audio_coords(
                lengths,
                position_offset=position_offset,
                slot_stride=slot_stride,
                device=device,
            )

        return _PreparedConditions(
            first_frame_tokens=first_frame,
            memory_video_tokens=memory_video,
            memory_video_coords=memory_video_coords,
            memory_audio_tokens=memory_audio,
            memory_audio_coords=memory_audio_coords,
            input_fingerprints=dict(resources.fingerprints),
        )

    @staticmethod
    def _renoise(
        clean: torch.Tensor, sigma: float, generator: torch.Generator
    ) -> torch.Tensor:
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        return clean * (1.0 - float(sigma)) + noise * float(sigma)

    @torch.inference_mode()
    def __call__(
        self,
        request: R2VRequest,
        *,
        sigmas: list[float],
        frame_rate: float = 25.0,
        position_offset: float = 500.0,
        slot_stride: float = 50.0,
        enable_audio_memory: bool = True,
        voice_filter_config=None,
        max_sequence_length: int = 1024,
        output_type: str = "np",
    ) -> EchoPipelineOutput:
        if len(sigmas) < 2 or float(sigmas[-1]) != 0.0:
            raise ValueError("Echo DMD sigmas must contain at least two values and end at 0")
        if (request.num_frames - 1) % self.pipe.vae_temporal_compression_ratio:
            raise ValueError("num_frames must be 1 + 8*k for Echo 1.5")

        device = self.device
        dtype = _module_dtype(self.pipe.transformer)
        conditions = self.prepare_conditions(
            request,
            frame_rate=frame_rate,
            position_offset=position_offset,
            slot_stride=slot_stride,
            enable_audio_memory=enable_audio_memory,
            voice_filter_config=voice_filter_config,
        )

        prompt_embeds, prompt_mask, _, _ = self.pipe.encode_prompt(
            prompt=request.prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        video_context, audio_context, context_mask = self.pipe.connectors(
            prompt_embeds,
            prompt_mask,
            padding_side=self.pipe.tokenizer_padding_side,
        )

        latent_frames = 1 + (request.num_frames - 1) // self.pipe.vae_temporal_compression_ratio
        latent_height = request.height // self.pipe.vae_spatial_compression_ratio
        latent_width = request.width // self.pipe.vae_spatial_compression_ratio
        video_tokens_count = latent_frames * latent_height * latent_width
        duration = float(request.num_frames) / float(frame_rate)
        audio_fps = (
            self.pipe.audio_sampling_rate
            / self.pipe.audio_hop_length
            / self.pipe.audio_vae_temporal_compression_ratio
        )
        audio_frames = round(duration * audio_fps)

        generator = torch.Generator(device=device).manual_seed(int(request.seed))
        video = torch.randn(
            (1, video_tokens_count, self.pipe.transformer.config.in_channels),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        audio = torch.randn(
            (1, audio_frames, self.pipe.transformer.config.audio_in_channels),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        first_count = latent_height * latent_width
        if conditions.first_frame_tokens is not None:
            video[:, :first_count] = conditions.first_frame_tokens

        target_video_coords = self.pipe.transformer.rope.prepare_video_coords(
            1,
            latent_frames,
            latent_height,
            latent_width,
            device,
            fps=MODEL_POSITION_FPS,
        )
        target_audio_coords = self.pipe.transformer.audio_rope.prepare_audio_coords(
            1, audio_frames, device
        )
        memory_video_count = (
            int(conditions.memory_video_tokens.shape[1])
            if conditions.memory_video_tokens is not None
            else 0
        )
        memory_audio_count = (
            int(conditions.memory_audio_tokens.shape[1])
            if conditions.memory_audio_tokens is not None
            else 0
        )

        for index, sigma in enumerate(sigmas[:-1]):
            sigma = float(sigma)
            scaled_sigma = sigma * float(self.pipe.transformer.config.timestep_scale_multiplier)
            video_timestep = torch.full(
                (1, video_tokens_count), scaled_sigma, device=device, dtype=torch.float32
            )
            if conditions.first_frame_tokens is not None:
                video_timestep[:, :first_count] = 0
            audio_timestep = torch.full(
                (1, audio_frames), scaled_sigma, device=device, dtype=torch.float32
            )

            model_video = video
            model_video_timestep = video_timestep
            video_coords = target_video_coords
            if conditions.memory_video_tokens is not None:
                model_video = torch.cat([conditions.memory_video_tokens, video], dim=1)
                model_video_timestep = torch.cat(
                    [
                        torch.zeros(
                            (1, memory_video_count), device=device, dtype=video_timestep.dtype
                        ),
                        video_timestep,
                    ],
                    dim=1,
                )
                video_coords = torch.cat(
                    [conditions.memory_video_coords, target_video_coords], dim=2
                )

            model_audio = audio
            model_audio_timestep = audio_timestep
            audio_coords = target_audio_coords
            if conditions.memory_audio_tokens is not None:
                model_audio = torch.cat([conditions.memory_audio_tokens, audio], dim=1)
                model_audio_timestep = torch.cat(
                    [
                        torch.zeros(
                            (1, memory_audio_count), device=device, dtype=audio_timestep.dtype
                        ),
                        audio_timestep,
                    ],
                    dim=1,
                )
                audio_coords = torch.cat(
                    [conditions.memory_audio_coords, target_audio_coords], dim=2
                )

            velocity_video, velocity_audio = self.pipe.transformer(
                hidden_states=model_video,
                audio_hidden_states=model_audio,
                encoder_hidden_states=video_context,
                audio_encoder_hidden_states=audio_context,
                timestep=model_video_timestep,
                audio_timestep=model_audio_timestep,
                # Match the release wrapper's global-sigma semantics exactly.
                # With first-frame conditioning, video_timestep[:, 0] is zero;
                # that zero drives video prompt modulation while audio retains
                # the current sampling sigma. Cross-modal modulation swaps the
                # two values because use_cross_timestep=True.
                sigma=video_timestep[:, 0],
                audio_sigma=audio_timestep[:, 0],
                encoder_attention_mask=context_mask,
                audio_encoder_attention_mask=context_mask,
                num_frames=latent_frames,
                height=latent_height,
                width=latent_width,
                fps=frame_rate,
                audio_num_frames=audio_frames,
                video_coords=video_coords,
                audio_coords=audio_coords,
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=True,
                return_dict=False,
            )
            velocity_video = velocity_video[:, memory_video_count:].to(dtype=dtype)
            velocity_audio = velocity_audio[:, memory_audio_count:].to(dtype=dtype)
            timestep_scale = float(self.pipe.transformer.config.timestep_scale_multiplier)
            pred_video = (
                video.float()
                - velocity_video.float() * video_timestep.unsqueeze(-1) / timestep_scale
            ).to(dtype)
            pred_audio = (
                audio.float()
                - velocity_audio.float() * audio_timestep.unsqueeze(-1) / timestep_scale
            ).to(dtype)
            if conditions.first_frame_tokens is not None:
                pred_video[:, :first_count] = conditions.first_frame_tokens

            next_sigma = float(sigmas[index + 1])
            if next_sigma > 0:
                video = self._renoise(pred_video, next_sigma, generator)
                audio = self._renoise(pred_audio, next_sigma, generator)
                if conditions.first_frame_tokens is not None:
                    video[:, :first_count] = conditions.first_frame_tokens
            else:
                video, audio = pred_video, pred_audio

        video_5d = self.pipe._unpack_latents(
            video,
            latent_frames,
            latent_height,
            latent_width,
            self.pipe.transformer_spatial_patch_size,
            self.pipe.transformer_temporal_patch_size,
        )
        video_5d = self.pipe._denormalize_latents(
            video_5d,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        )
        latent_mel_bins = self.pipe.audio_mel_bins // self.pipe.audio_vae_mel_compression_ratio
        audio_4d = self.pipe._denormalize_audio_latents(
            audio, self.pipe.audio_vae.latents_mean, self.pipe.audio_vae.latents_std
        )
        audio_4d = self.pipe._unpack_audio_latents(
            audio_4d, audio_frames, num_mel_bins=latent_mel_bins
        )

        if output_type == "latent":
            frames, waveform = video_5d, audio_4d
        else:
            decoded = self.pipe.vae.decode(
                video_5d.to(self.pipe.vae.dtype), None, return_dict=False
            )[0]
            frames = self.pipe.video_processor.postprocess_video(
                decoded, output_type=output_type
            )
            mel = self.pipe.audio_vae.decode(
                audio_4d.to(self.pipe.audio_vae.dtype), return_dict=False
            )[0]
            waveform = self.pipe.vocoder(mel)

        return EchoPipelineOutput(
            frames=frames,
            audio=waveform,
            video_latents=video,
            audio_latents=audio,
            input_fingerprints=conditions.input_fingerprints,
        )
