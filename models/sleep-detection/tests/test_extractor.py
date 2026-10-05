import unittest

from extractor import class_id_for_label, extract_result


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Box:
    def __init__(self, class_id, confidence, xywh=(100, 80, 40, 20)):
        self.cls = Scalar(class_id)
        self.conf = Scalar(confidence)
        self.xywh = [Vector(xywh)]


class Result:
    def __init__(self, boxes):
        self.boxes = boxes


class ExtractorTests(unittest.TestCase):
    def test_only_sleeping_strictly_above_point_seven_is_returned(self):
        result = Result(
            [
                Box(0, 0.70000),
                Box(0, 0.70004),  # rounds to 0.7000 and is excluded
                Box(0, 0.70006),  # rounds to 0.7001 and is included
                Box(0, 0.92345),
                Box(1, 0.99999),  # AWAKE is excluded regardless of confidence
            ]
        )

        detections = extract_result(
            result,
            {0: "SLEEPING", 1: "AWAKE"},
            12,
            target_label="SLEEPING",
            confidence_threshold=0.70,
        )

        self.assertEqual([item["confidence"] for item in detections], [0.7001, 0.9234])
        self.assertEqual({item["label"] for item in detections}, {"SLEEPING"})
        self.assertEqual(detections[0]["bbox"], {"x": 80, "y": 70, "w": 40, "h": 20})

    def test_resolves_target_class_id_and_rejects_missing_label(self):
        self.assertEqual(class_id_for_label({0: "SLEEPING", 1: "AWAKE"}, "SLEEPING"), 0)
        with self.assertRaises(ValueError):
            class_id_for_label({0: "SLEEPING", 1: "AWAKE"}, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
