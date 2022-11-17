import unittest

import torch
from transformers import RealmScorer, RealmTokenizer


class TestDocumentEmbeddings(unittest.TestCase):
    def setUp(self):
        self.tokenizer = RealmTokenizer.from_pretrained(
            "google/realm-cc-news-pretrained-scorer"
        )
        self.input_texts = ["How are you?", "What is the item in the picture?"]

    def test_tokenize(self):
        result = self.tokenizer(
            self.input_texts, return_tensors="pt", padding="longest"
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
