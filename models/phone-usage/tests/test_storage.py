import json
import tempfile
import unittest
from pathlib import Path

from storage import LocalStorage, normalize_key


class StorageTests(unittest.TestCase):
    def test_rejects_traversal_and_absolute_keys(self):
        for key in ("../secret", "/absolute", "folder/../../secret", ""):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_key(key)

    def test_local_storage_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            storage = LocalStorage(str(root))
            source = root / "source.txt"
            source.write_text("hello", encoding="utf-8")

            storage.upload_file(source, "jobs/job-1/file.txt")
            destination = root / "downloaded.txt"
            storage.download_file("jobs/job-1/file.txt", destination)
            storage.put_json("jobs/job-1/result.json", {"status": "SUCCESS"})

            self.assertEqual(destination.read_text(encoding="utf-8"), "hello")
            archived = json.loads((root / "jobs/job-1/result.json").read_text())
            self.assertEqual(archived["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
