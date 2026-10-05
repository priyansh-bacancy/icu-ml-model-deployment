import unittest

from builder import build_detections, select_representative_frame


def detection(frame, confidence, x=10):
    return {
        "frame": frame,
        "class_id": 0,
        "label": "SLEEPING",
        "confidence": confidence,
        "bbox": {"x": x, "y": 20, "w": 30, "h": 40},
        "attributes": {},
    }


class BuilderTests(unittest.TestCase):
    def test_video_keeps_all_instances_and_selects_best_frame(self):
        flat = [
            detection(2, 0.75),
            detection(1, 0.90004),
            detection(2, 0.8, x=50),
        ]
        result = build_detections(flat, is_video=True, fps=10)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["label"], "SLEEPING")
        self.assertEqual(result[0]["best_frame_number"], 2)
        self.assertEqual(result[0]["count"], 3)
        self.assertEqual(result[0]["first_seen_ms"], 100)
        self.assertEqual(result[0]["last_seen_ms"], 200)
        self.assertEqual(
            result[0]["instances"][0]["attributes"],
            {"frame_number": 1, "timestamp_ms": 100},
        )

    def test_image_omits_video_only_fields(self):
        result = build_detections([detection(0, 0.75)], is_video=False)[0]
        self.assertEqual(result["count"], 1)
        self.assertNotIn("best_frame_number", result)
        self.assertEqual(result["instances"][0]["attributes"], {})

    def test_representative_frame_defaults_to_zero(self):
        self.assertEqual(select_representative_frame([]), 0)


if __name__ == "__main__":
    unittest.main()
