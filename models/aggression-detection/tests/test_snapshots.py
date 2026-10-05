from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from snapshots import _ranked_instances, save_ranked_snapshots


class SnapshotTests(unittest.TestCase):
    def test_ranks_top_five_and_keeps_actual_instance_frames(self):
        detections = [
            {
                "label": "event",
                "best_frame_number": 99,
                "instances": [
                    {
                        "confidence": confidence,
                        "bbox": {"x": index * 5, "y": 5, "w": 4, "h": 4},
                        "attributes": {"frame_number": index},
                    }
                    for index, confidence in enumerate(
                        [0.2, 0.9, 0.5, 0.8, 0.7, 0.6]
                    )
                ],
            }
        ]
        ranked = _ranked_instances(detections, is_video=True)
        self.assertEqual([item["confidence"] for item in ranked], [0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertEqual([item["frame_number"] for item in ranked], [1, 3, 4, 5, 2])

    def test_each_image_snapshot_draws_only_its_own_instance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jpg"
            cv2.imwrite(str(source), np.zeros((100, 100, 3), dtype=np.uint8))
            detections = [
                {
                    "label": "high",
                    "instances": [
                        {
                            "confidence": 0.9,
                            "bbox": {"x": 10, "y": 10, "w": 15, "h": 15},
                            "attributes": {},
                        }
                    ],
                },
                {
                    "label": "low",
                    "instances": [
                        {
                            "confidence": 0.5,
                            "bbox": {"x": 60, "y": 60, "w": 15, "h": 15},
                            "attributes": {},
                        }
                    ],
                },
            ]

            paths = save_ranked_snapshots(
                source, detections, is_video=False, workdir=root
            )

            self.assertEqual([path.name for path in paths], ["frame_1.jpg", "frame_2.jpg"])
            first = cv2.imread(str(paths[0]))
            self.assertGreater(int(first[10, 10, 2]), 150)
            self.assertLess(int(first[60, 60].max()), 50)

    def test_no_detections_produces_no_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(
                save_ranked_snapshots(
                    Path(temp) / "unused.mp4",
                    [],
                    is_video=True,
                    workdir=Path(temp),
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
