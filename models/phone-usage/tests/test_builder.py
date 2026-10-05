import unittest

from builder import build_detections, select_representative_frame


def detection(frame, label, confidence, x=10):
    return {
        "frame": frame,
        "label": label,
        "confidence": confidence,
        "bbox": {"x": x, "y": 20, "w": 30, "h": 40},
        "attributes": {},
    }


class BuilderTests(unittest.TestCase):
    def test_video_keeps_all_frame_instances_and_selects_best_frame(self):
        flat = [
            detection(2, "in_hand", 0.5),
            detection(1, "in_hand", 0.90004),
            detection(2, "in_hand", 0.5, x=50),
            detection(4, "on_ear", 0.8),
        ]

        result = build_detections(flat, is_video=True, fps=10)

        self.assertEqual(result[0]["best_frame_number"], 2)
        self.assertEqual(result[0]["count"], 3)
        self.assertEqual(result[0]["count"], len(result[0]["instances"]))
        self.assertEqual(result[0]["first_seen_ms"], 100)
        self.assertEqual(result[0]["last_seen_ms"], 200)
        self.assertEqual(
            result[0]["instances"][0]["attributes"],
            {"frame_number": 1, "timestamp_ms": 100},
        )
        self.assertEqual(result[0]["instances"][0]["confidence"], 0.9)
        self.assertEqual(
            result[0]["instances"][2]["attributes"],
            {"frame_number": 2, "timestamp_ms": 200},
        )
        self.assertEqual(result[1]["detection_id"], "d-0002")

    def test_tie_breaks_on_instance_count_then_earliest_frame(self):
        flat = [
            detection(3, "in_hand", 0.5),
            detection(3, "in_hand", 0.5),
            detection(7, "in_hand", 1.0),
            detection(9, "on_ear", 0.7),
            detection(10, "on_ear", 0.7),
        ]

        result = build_detections(flat, is_video=True)

        self.assertEqual(result[0]["best_frame_number"], 3)
        self.assertEqual(result[1]["best_frame_number"], 9)

    def test_image_omits_video_only_fields(self):
        result = build_detections(
            [detection(0, "on_ear", 0.75)], is_video=False
        )[0]

        self.assertEqual(result["count"], 1)
        self.assertNotIn("best_frame_number", result)
        self.assertNotIn("first_seen_ms", result)
        self.assertNotIn("last_seen_ms", result)
        self.assertEqual(result["instances"][0]["attributes"], {})

    def test_video_without_fps_keeps_frame_and_null_timestamp(self):
        result = build_detections(
            [detection(12, "in_hand", 0.75)], is_video=True
        )[0]

        self.assertEqual(
            result["instances"][0]["attributes"],
            {"frame_number": 12, "timestamp_ms": None},
        )

    def test_representative_frame_defaults_to_zero(self):
        self.assertEqual(select_representative_frame([]), 0)


if __name__ == "__main__":
    unittest.main()
