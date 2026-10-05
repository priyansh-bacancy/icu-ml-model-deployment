"""Load the baked weights and perform one inference without external files."""

import os

import numpy as np
from ultralytics import YOLO

from builder import build_detections
from config import EVENT_TYPE
from extractor import extract_result


model_path = os.getenv("MODEL_PATH", "weights/best.pt")
image_size = int(os.getenv("IMAGE_SIZE", "1280"))
model = YOLO(model_path)
result = model.predict(
    np.zeros((image_size, image_size, 3), dtype=np.uint8),
    imgsz=image_size,
    verbose=False,
)[0]
flat = extract_result(result, model.names, 0)
build_detections(flat, is_video=False)
assert model.names == {0: "in_hand", 1: "on_ear"}, model.names
assert EVENT_TYPE == "phone_usage_event"
print(f"smoke test ok: {model_path}; imgsz={image_size}; classes={model.names}")
