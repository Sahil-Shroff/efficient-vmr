from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - handled at runtime
    Image = None  # type: ignore[assignment]

try:
    from transformers import AutoProcessor, CLIPModel
except ImportError:  # pragma: no cover - handled at runtime
    AutoProcessor = None  # type: ignore[assignment]
    CLIPModel = None  # type: ignore[assignment]


def _require_clip_stack() -> None:
    missing: list[str] = []
    if torch is None:
        missing.append("torch")
    if AutoProcessor is None or CLIPModel is None:
        missing.append("transformers")
    if missing:
        raise ImportError(
            "CLIP text/image encoding requires: "
            + ", ".join(missing)
            + ". Install requirements with `pip install -r requirements.txt`."
        )


def _require_image_stack() -> None:
    _require_clip_stack()
    if Image is None:
        raise ImportError(
            "Image encoding requires Pillow. Install requirements with `pip install -r requirements.txt`."
        )


def _require_video_stack() -> None:
    if cv2 is None:
        raise ImportError(
            "Video frame extraction requires opencv-python-headless. "
            "Install requirements with `pip install -r requirements.txt`."
        )


@dataclass
class ClipFeatureEncoder:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
    batch_size: int = 16

    def __post_init__(self) -> None:
        _require_clip_stack()
        resolved_device = self.device
        if not resolved_device:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _normalize_output(self, output: object, *, tensor_attr: str) -> np.ndarray:
        if isinstance(output, torch.Tensor):
            tensor = output
        elif hasattr(output, tensor_attr):
            tensor = getattr(output, tensor_attr)
        else:
            raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=-1)
        return tensor.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(texts), self.batch_size):
                batch = texts[offset : offset + self.batch_size]
                inputs = self.processor(text=batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                text_outputs = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                text_features = self.model.text_projection(text_outputs.pooler_output)
                embeddings.append(self._normalize_output(text_features, tensor_attr="text_embeds"))
        return np.concatenate(embeddings, axis=0)

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)

        _require_image_stack()
        pil_images = [Image.fromarray(image) for image in images]
        embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(pil_images), self.batch_size):
                batch = pil_images[offset : offset + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
                image_features = self.model.visual_projection(vision_outputs.pooler_output)
                embeddings.append(self._normalize_output(image_features, tensor_attr="image_embeds"))
        return np.concatenate(embeddings, axis=0)


def sample_video_frames(
    video_path: str | Path,
    *,
    sample_stride_sec: float,
    max_frames: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    _require_video_stack()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = (frame_count / fps) if fps > 0.0 and frame_count > 0.0 else 0.0
    if duration <= 0.0:
        duration = sample_stride_sec

    timestamps = np.arange(0.0, duration + 1e-6, max(sample_stride_sec, 1e-3), dtype=np.float32)
    if max_frames > 0:
        timestamps = timestamps[:max_frames]
    if timestamps.size == 0:
        timestamps = np.asarray([0.0], dtype=np.float32)

    frames: list[np.ndarray] = []
    actual_timestamps: list[float] = []
    for timestamp in timestamps.tolist():
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        success, frame = capture.read()
        if not success:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        actual_timestamps.append(float(timestamp))

    capture.release()
    return np.asarray(actual_timestamps, dtype=np.float32), frames


def save_visual_feature_file(
    output_path: str | Path,
    *,
    timestamps: np.ndarray,
    embeddings: np.ndarray,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=np.asarray(timestamps, dtype=np.float32),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    return path


def load_visual_feature_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(Path(path))
    return (
        np.asarray(payload["timestamps"], dtype=np.float32),
        np.asarray(payload["embeddings"], dtype=np.float32),
    )


def iter_video_feature_paths(features_dir: str | Path) -> Iterable[Path]:
    return sorted(Path(features_dir).glob("*.npz"))
