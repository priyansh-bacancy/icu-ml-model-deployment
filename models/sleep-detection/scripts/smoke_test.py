"""Load the checkpoint, verify its contract, and run one synthetic inference."""

import os
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from builder import build_detections
from config import EVENT_TYPE
from extractor import extract_result


model_path = os.getenv("MODEL_PATH", "weights/best.pt")
image_size = int(os.getenv("IMAGE_SIZE", "640"))
threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
target_label = os.getenv("TARGET_LABEL", "SLEEPING")
model = YOLO(model_path)
assert model.names == {0: "SLEEPING", 1: "AWAKE"}, model.names
result = model.predict(
    np.zeros((image_size, image_size, 3), dtype=np.uint8),
    conf=threshold,
    imgsz=image_size,
    classes=[0],
    verbose=False,
)[0]
flat = extract_result(
    result,
    model.names,
    0,
    target_label=target_label,
    confidence_threshold=threshold,
)
grouped = build_detections(flat, is_video=False)
assert all(item["label"] == target_label for item in grouped)
assert all(
    instance["confidence"] > threshold
    for item in grouped
    for instance in item["instances"]
)
assert EVENT_TYPE == "sleep_event"
print(
    f"smoke test ok: {model_path}; imgsz={image_size}; "
    f"threshold=>{threshold:.2f}; target={target_label}; classes={model.names}"
)
