"""Exact VideoMAE architecture and preprocessing used by abd-videomae.ipynb."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import decord
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEConfig, VideoMAEForVideoClassification


NUM_FRAMES = 16
CROP_SIZE = 224
SHORT_SIDE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _videomae_config() -> VideoMAEConfig:
    """Create the offline equivalent of the notebook's Kinetics config."""
    return VideoMAEConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=16,
        tubelet_size=2,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        use_mean_pooling=True,
        num_labels=400,
    )


class VideoMAEAggression(nn.Module):
    def __init__(self, dropout: float = 0.35) -> None:
        super().__init__()
        self.backbone = VideoMAEForVideoClassification(_videomae_config())
        self.embedding_dim = int(self.backbone.config.hidden_size)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 1),
        )

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(pixel_values=clip).logits
        return self.classifier(embedding).squeeze(-1), embedding


@dataclass(frozen=True)
class ModelRuntime:
    model: VideoMAEAggression
    device: torch.device
    temperature: float

    @classmethod
    def load(cls, checkpoint_path: str, requested_device: str = "auto") -> "ModelRuntime":
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"VideoMAE checkpoint not found: {path}. Expected best_model.pt "
                "exported by abd-videomae.ipynb."
            )

        if requested_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested_device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("DEVICE requests CUDA, but CUDA is unavailable")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise KeyError("best_model.pt must contain a 'model_state' mapping")

        model = VideoMAEAggression()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        temperature = float(checkpoint.get("temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("checkpoint temperature must be a positive number")
        return cls(model=model, device=device, temperature=temperature)

    def predict(self, clips: torch.Tensor) -> list[float]:
        clips = clips.to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            logits, _ = self.model(clips)
        return torch.sigmoid(logits.float() / self.temperature).cpu().tolist()


def decode_frame_context(
    reader: decord.VideoReader,
    center_frame: int,
    fps: float,
    context_seconds: float,
) -> np.ndarray:
    """Sample a moving temporal context centered on one source frame.

    Boundary indices are replicated instead of shrinking the context. This
    keeps the five-second temporal scale used for training while still giving
    frames near the start and end their own predictions.
    """
    frame_count = len(reader)
    context_frames = max(1.0, context_seconds * fps)
    start_frame = float(center_frame) - context_frames / 2
    end_frame = start_frame + context_frames
    edges = np.linspace(start_frame, end_frame, NUM_FRAMES + 1)
    indices = [
        round((lower + upper - 1) / 2)
        for lower, upper in zip(edges[:-1], edges[1:])
    ]
    indices = np.clip(indices, 0, frame_count - 1).astype(np.int64)
    return reader.get_batch(indices).asnumpy()


def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    height, width = clip.shape[-2:]
    scale = SHORT_SIDE / min(height, width)
    output_size = (round(height * scale), round(width * scale))
    clip = F.interpolate(
        clip, size=output_size, mode="bilinear", align_corners=False
    )
    height, width = clip.shape[-2:]
    top = (height - CROP_SIZE) // 2
    left = (width - CROP_SIZE) // 2
    clip = clip[:, :, top : top + CROP_SIZE, left : left + CROP_SIZE]
    clip = clip.contiguous()
    return (clip - IMAGENET_MEAN) / IMAGENET_STD
