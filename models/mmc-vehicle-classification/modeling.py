import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import faiss
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

VEHICLE_MODEL_FILENAME = "vehicle_detector.pt"
COLOR_MODEL_FILENAME = "color_best.pt"
MAKE_MODEL_FILENAME = "make_model_best.pt"
VECTOR_DB_DIRECTORY = "vector_db"

DETECTION_CONFIDENCE = 0.30
DETECTION_IOU = 0.60
DETECTION_IMAGE_SIZE = 1280
EMBEDDING_DIMENSION = 128
DEFAULT_TOP_K = 5
EXPECTED_VEHICLE_CLASSES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
}

COLOR_INDEX_NAME = "color_index"
PREFERRED_MAKE_INDEX_NAME = "make_model_registry"
FALLBACK_MAKE_INDEX_NAME = "make_model_index"


class CarEmbeddingModel(nn.Module):
    def __init__(self, embedding_dimension: int = EMBEDDING_DIMENSION) -> None:
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained=None,
        )
        self.visual = clip_model.visual
        self.head = nn.Linear(self.visual.output_dim, embedding_dimension)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        embeddings = self.head(features)
        return F.normalize(embeddings, dim=-1)


class FaissStore:
    def __init__(self, root: Path) -> None:
        self.faiss_directory = root / "faiss" if (root / "faiss").is_dir() else root
        self.indices: dict[str, faiss.Index] = {}
        self.metadata: dict[str, list[dict]] = {}

    def exists(self, index_name: str) -> bool:
        return (self.faiss_directory / f"{index_name}.faiss").is_file()

    def load(self, index_name: str) -> tuple[faiss.Index, list[dict]]:
        if index_name in self.indices:
            return self.indices[index_name], self.metadata[index_name]

        index_path = self.faiss_directory / f"{index_name}.faiss"
        metadata_path = self.faiss_directory / f"{index_name}_metadata.json"
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing FAISS artifacts for '{index_name}'")

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if index.ntotal != len(metadata):
            raise ValueError(
                f"FAISS/metadata count mismatch for '{index_name}': "
                f"{index.ntotal} vectors and {len(metadata)} rows"
            )

        self.indices[index_name] = index
        self.metadata[index_name] = metadata
        return index, metadata

    def query(
        self,
        index_name: str,
        embedding: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        index, metadata = self.load(index_name)
        query = embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = index.search(query, min(top_k, index.ntotal))

        uses_inner_product = index.metric_type == faiss.METRIC_INNER_PRODUCT
        score_name = "similarity" if uses_inner_product else "distance"
        matches = []
        for rank, raw_index in enumerate(indices[0], start=1):
            index_position = int(raw_index)
            if index_position < 0:
                continue
            item = metadata[index_position]
            match = {
                "rank": rank,
                "label_name": str(item.get("label_name", "unknown")),
                score_name: float(distances[0][rank - 1]),
            }
            for field in ("source", "market", "make_model"):
                if field in item:
                    match[field] = item[field]
            matches.append(match)
        return matches


@dataclass
class MMCRuntime:
    vehicle_detector: YOLO
    color_model: CarEmbeddingModel
    make_model: CarEmbeddingModel
    vector_store: FaissStore
    make_index_name: str
    device: torch.device
    transform: transforms.Compose

    def embed(self, crop_bgr: np.ndarray, task: str) -> np.ndarray:
        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        model = self.color_model if task == "color" else self.make_model
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ),
        ):
            embedding = model(tensor)
        return embedding.float().cpu().numpy().squeeze().astype("float32")


def _load_embedding_model(
    checkpoint_path: Path,
    device: torch.device,
) -> CarEmbeddingModel:
    model = CarEmbeddingModel()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    cleaned_state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned_state, strict=True)
    del checkpoint, state, cleaned_state
    model.to(device)
    model.eval()
    return model


def load_runtime(
    model_directory: str,
    vector_db_directory: str | Path | None = None,
) -> MMCRuntime:
    model_root = Path(model_directory)
    vector_db_root = (
        Path(vector_db_directory)
        if vector_db_directory is not None
        else model_root / VECTOR_DB_DIRECTORY
    )
    required_paths = [
        model_root / VEHICLE_MODEL_FILENAME,
        model_root / COLOR_MODEL_FILENAME,
        model_root / MAKE_MODEL_FILENAME,
        vector_db_root,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing model artifacts: {missing_paths}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vector_store = FaissStore(vector_db_root)
    make_index_name = (
        PREFERRED_MAKE_INDEX_NAME
        if vector_store.exists(PREFERRED_MAKE_INDEX_NAME)
        else FALLBACK_MAKE_INDEX_NAME
    )
    vector_store.load(COLOR_INDEX_NAME)
    vector_store.load(make_index_name)

    transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    vehicle_detector = YOLO(str(model_root / VEHICLE_MODEL_FILENAME))
    if vehicle_detector.names != EXPECTED_VEHICLE_CLASSES:
        raise ValueError(
            "Expected four-class vehicle detector mapping "
            f"{EXPECTED_VEHICLE_CLASSES}, found {vehicle_detector.names}"
        )
    vehicle_detector.to(device)

    return MMCRuntime(
        vehicle_detector=vehicle_detector,
        color_model=_load_embedding_model(
            model_root / COLOR_MODEL_FILENAME,
            device,
        ),
        make_model=_load_embedding_model(
            model_root / MAKE_MODEL_FILENAME,
            device,
        ),
        vector_store=vector_store,
        make_index_name=make_index_name,
        device=device,
        transform=transform,
    )
