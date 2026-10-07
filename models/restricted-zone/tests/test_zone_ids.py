"""Protect camera rule identity across polygon preparation and response generation."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from zone_pipeline import prepare_zones, process_video

FRAME_SIZE = 64
FPS = 10.0
FRAME_COUNT = 2
ZONE_TYPES = ("restricted", "entry", "exit", "dwell", "capacity")


def zone(zone_id: str, zone_type: str = "restricted") -> dict:
    return {
        "id": zone_id,
        "type": zone_type,
        "applies_to": "both",
        "points": [
            [0, 0],
            [FRAME_SIZE - 1, 0],
            [FRAME_SIZE - 1, FRAME_SIZE - 1],
            [0, FRAME_SIZE - 1],
        ],
    }


def config(zones: list[dict]) -> dict:
    return {"orig_w": FRAME_SIZE, "orig_h": FRAME_SIZE, "zones": zones}


@pytest.mark.parametrize("zone_type", ZONE_TYPES)
def test_preserves_caller_id_for_each_zone_type(zone_type: str) -> None:
    supplied_id = "BK065C9PAG92CE3:6"

    prepared = prepare_zones(
        config([zone(supplied_id, zone_type)]), FRAME_SIZE, FRAME_SIZE
    )

    assert prepared[0]["zone_id"] == supplied_id


def test_ids_survive_reordering_and_removal() -> None:
    first, second = zone("camera:6"), zone("camera:7")

    reordered = prepare_zones(config([second, first]), FRAME_SIZE, FRAME_SIZE)
    remaining = prepare_zones(config([second]), FRAME_SIZE, FRAME_SIZE)

    assert [item["zone_id"] for item in reordered] == ["camera:7", "camera:6"]
    assert remaining[0]["zone_id"] == "camera:7"


def test_legacy_configuration_without_id_keeps_index_fallback() -> None:
    legacy_zone = zone("unused")
    del legacy_zone["id"]

    prepared = prepare_zones(config([legacy_zone]), FRAME_SIZE, FRAME_SIZE)

    assert prepared[0]["zone_id"] == "zone-0001"


@pytest.mark.parametrize("invalid_id", [None, "", "  ", 6, True])
def test_rejects_invalid_supplied_id(invalid_id: object) -> None:
    invalid_zone = {**zone("unused"), "id": invalid_id}

    with pytest.raises(ValueError, match="non-empty string"):
        prepare_zones(config([invalid_zone]), FRAME_SIZE, FRAME_SIZE)


def test_rejects_duplicate_ids() -> None:
    duplicate_config = config([zone("camera:6"), zone("camera:6")])

    with pytest.raises(ValueError, match="Duplicate zone id"):
        prepare_zones(duplicate_config, FRAME_SIZE, FRAME_SIZE)


def test_rejects_supplied_id_collision_with_legacy_fallback() -> None:
    legacy_zone = zone("unused")
    del legacy_zone["id"]

    with pytest.raises(ValueError, match="Duplicate zone id"):
        prepare_zones(config([legacy_zone, zone("zone-0001")]), FRAME_SIZE, FRAME_SIZE)


class Coordinates:
    def cpu(self) -> np.ndarray:
        return np.array([20, 20, 40, 40])


class TrackedBoxes(list):
    def __init__(self) -> None:
        super().__init__(
            [
                SimpleNamespace(
                    cls=np.array(1),
                    id=np.array(1),
                    conf=np.array(0.9),
                    xyxy=[Coordinates()],
                )
            ]
        )
        self.id = np.array([1])


class FixedDetector:
    def track(self, frame: np.ndarray, **options: object) -> list[SimpleNamespace]:
        del frame, options
        return [SimpleNamespace(boxes=TrackedBoxes())]


def test_response_observations_echo_all_five_supplied_ids(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (FRAME_SIZE, FRAME_SIZE)
    )
    assert writer.isOpened()
    for _ in range(FRAME_COUNT):
        writer.write(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))
    writer.release()
    zone_config = config(
        [zone(f"camera:{index}", kind) for index, kind in enumerate(ZONE_TYPES)]
    )

    result = process_video(
        FixedDetector(),
        video,
        tmp_path / "annotated.mp4",
        zone_config,
        "cpu",
        tmp_path / "prelabels",
        "prelabels/test/restricted-zone",
    )

    attributes = result.detections[0]["instances"][0]["attributes"]
    assert [item["zone_id"] for item in attributes["zone_observations"]] == [
        f"camera:{index}" for index in range(len(ZONE_TYPES))
    ]
    assert attributes["has_zone_violation"] is True
    assert result.prelabel_samples[0]["manifest_entry"]["frame_number"] == 0
    assert (tmp_path / "annotated.mp4").stat().st_size > 0
