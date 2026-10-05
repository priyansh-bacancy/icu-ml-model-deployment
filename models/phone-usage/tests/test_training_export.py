import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np

    from media import _normalize_yolo_box, export_training_artifacts
except ImportError:
    cv2 = None
    np = None


def grouped_detection(label, instances):
    return {
        "detection_id": "d-0001",
        "label": label,
        "best_frame_number": instances[0]["attributes"].get("frame_number", 0),
        "first_seen_ms": 0,
        "last_seen_ms": 0,
        "count": len(instances),
        "instances": instances,
    }


@unittest.skipIf(cv2 is None, "OpenCV is verified inside the runtime container")
class TrainingExportTests(unittest.TestCase):
    def test_yolo_bbox_is_clipped_and_normalized(self):
        row = _normalize_yolo_box(
            1,
            {"x": -10, "y": 5, "w": 30, "h": 10},
            100,
            50,
        )
        self.assertEqual(row, "1 0.100000 0.200000 0.200000 0.200000")

    def test_exports_every_fifteenth_frame_with_empty_labels_when_needed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for value in range(31):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()

            detections = [
                grouped_detection(
                    "in_hand",
                    [
                        {
                            "confidence": 0.8,
                            "bbox": {"x": 10, "y": 5, "w": 20, "h": 10},
                            "attributes": {"frame_number": 0, "timestamp_ms": 0},
                        },
                        {
                            "confidence": 0.7,
                            "bbox": {"x": 0, "y": 0, "w": 32, "h": 24},
                            "attributes": {"frame_number": 16, "timestamp_ms": 3200},
                        },
                    ],
                ),
                grouped_detection(
                    "on_ear",
                    [
                        {
                            "confidence": 0.6,
                            "bbox": {"x": 32, "y": 24, "w": 32, "h": 24},
                            "attributes": {"frame_number": 30, "timestamp_ms": 6000},
                        }
                    ],
                ),
            ]
            training_root = root / "training"
            frames = export_training_artifacts(
                source,
                detections,
                is_video=True,
                class_names={0: "in_hand", 1: "on_ear"},
                training_root=training_root,
            )

            self.assertEqual(frames, [0, 15, 30])
            self.assertFalse((training_root / "images").exists())
            self.assertEqual(
                sorted(path.name for path in (training_root / "labels").iterdir()),
                ["frame_00000.txt", "frame_00015.txt", "frame_00030.txt"],
            )
            self.assertEqual(
                (training_root / "labels" / "frame_00000.txt").read_text(),
                "0 0.312500 0.208333 0.312500 0.208333",
            )
            self.assertEqual(
                (training_root / "labels" / "frame_00015.txt").read_text(), ""
            )
            self.assertEqual(
                (training_root / "labels" / "frame_00030.txt")
                .read_text()
                .splitlines(),
                [
                    "1 0.750000 0.750000 0.500000 0.500000",
                ],
            )

    def test_no_detections_create_empty_sampled_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for value in range(16):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()
            frames = export_training_artifacts(
                source,
                [],
                is_video=True,
                class_names={0: "in_hand", 1: "on_ear"},
                training_root=root / "training",
            )
            self.assertEqual(frames, [0, 15])
            for label_path in (root / "training" / "labels").iterdir():
                self.assertEqual(label_path.read_text(), "")


if __name__ == "__main__":
    unittest.main()
