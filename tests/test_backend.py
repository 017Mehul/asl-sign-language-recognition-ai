"""Unit tests for pure backend behavior; model artifacts are not required."""
import unittest

from backend.app import SentenceStore, decode_base64_image


class SentenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = SentenceStore()

    def test_sentence_actions(self):
        self.store.apply_action("session", "append", "a")
        self.store.apply_action("session", "append", "B")
        self.store.apply_action("session", "space", None)
        self.store.apply_action("session", "append", "c")
        self.assertEqual(self.store.get("session"), "AB C")
        self.store.apply_action("session", "backspace", None)
        self.assertEqual(self.store.get("session"), "AB ")
        self.store.apply_action("session", "clear", None)
        self.assertEqual(self.store.get("session"), "")

    def test_sessions_are_isolated(self):
        self.store.apply_action("one", "append", "A")
        self.assertEqual(self.store.get("two"), "")

    def test_append_requires_one_ascii_letter(self):
        for invalid in (None, "", "AB", "1", "é"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.store.apply_action("s", "append", invalid)

    def test_invalid_base64_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_base64_image("not-base64***")


if __name__ == "__main__":
    unittest.main()
