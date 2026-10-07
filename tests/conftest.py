"""Load recovered runtime modules with ML/cloud imports stubbed at boundaries."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("fall-detection", "restricted-zone", "mmc-vehicle-classification")


@pytest.fixture(params=MODELS)
def runtime(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    model = request.param
    directory = ROOT / "models" / model
    monkeypatch.syspath_prepend(str(directory))
    for dependency in ("boto3", "torch", "ultralytics"):
        monkeypatch.setitem(sys.modules, dependency, ModuleType(dependency))
    sys.modules["ultralytics"].YOLO = object
    if model == "mmc-vehicle-classification":
        modeling = ModuleType("modeling")
        constants = {
            "COLOR_INDEX_NAME": "color_index",
            "DEFAULT_TOP_K": 5,
            "DETECTION_CONFIDENCE": 0.30,
            "DETECTION_IMAGE_SIZE": 1280,
            "DETECTION_IOU": 0.60,
            "EXPECTED_VEHICLE_CLASSES": {
                0: "car",
                1: "truck",
                2: "bus",
                3: "motorcycle",
            },
            "MMCRuntime": dict,
            "load_runtime": lambda *args, **kwargs: None,
        }
        for name, value in constants.items():
            setattr(modeling, name, value)
        monkeypatch.setitem(sys.modules, "modeling", modeling)
    spec = importlib.util.spec_from_file_location(
        model.replace("-", "_") + "_inference", directory / "inference.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
