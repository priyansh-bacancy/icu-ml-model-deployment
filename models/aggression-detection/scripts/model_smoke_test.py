"""Load the baked VideoMAE checkpoint without requiring a running HTTP server."""

from __future__ import annotations

import os

from modeling import ModelRuntime


model_path = os.getenv("MODEL_PATH", "/app/weights/best_model.pt")
runtime = ModelRuntime.load(model_path, "cpu")
assert runtime.device.type == "cpu"
print(
    "model smoke test ok: "
    f"{model_path}; device={runtime.device}; temperature={runtime.temperature:.6f}"
)
