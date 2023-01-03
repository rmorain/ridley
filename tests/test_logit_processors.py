import unittest

import requests
import torch
from ridley.logit_processors import RhymeLogitsProcessor
from ridley.riddle_generation import generate
from transformers import GPT2Tokenizer


class TestRhymeLogitsProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.rhyme_lp = RhymeLogitsProcessor(self.tokenizer)

    def test_generate(self):
        result = generate(logits_processor=[RhymeLogitsProcessor()], do_sample=True)
        self.assertIsNotNone(result)

    def test_request_rhymes(self):
        result = self.rhyme_lp.request_rhymes("hello")
        self.assertIsInstance(result, requests.models.Response)

    def test_rhyming_prior(self):
        input_ids = self.tokenizer("Hello world").input_ids
        scores = torch.rand(self.tokenizer.vocab_size)
        result = self.rhyme_lp.rhyming_prior(input_ids, scores)
        self.assertIsNotNone(result)
        self.assertEqual(self.tokenizer.vocab_size, len(result))
