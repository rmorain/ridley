import unittest

import torch
from ridley.document_embeddings import score
from transformers import RealmScorer, RealmTokenizer


class TestDocumentEmbeddings(unittest.TestCase):
    def setUp(self):
        self.model = RealmScorer.from_pretrained(
            "google/realm-cc-news-pretrained-scorer", num_candidates=2
        )
        self.tokenizer = RealmTokenizer.from_pretrained(
            "google/realm-cc-news-pretrained-scorer"
        )
        self.input_texts = ["How are you?", "What is the item in the picture?"]
        self.candidate_texts = [
            ["Hello world!", "Nice to meet you!"],
            ["A cute cat.", "An adorable dog"],
        ]

    def test_tokenize(self):
        result = self.tokenizer(
            self.input_texts, return_tensors="pt", padding="longest"
        )
        self.assertIsNotNone(result)

    def test_score(self):
        score(self.model, self.tokenizer, self.input_texts, self.candidate_texts)


if __name__ == "__main__":
    unittest.main()
