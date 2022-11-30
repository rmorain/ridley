import unittest

from ridley.logit_processors import RhymeLogitsProcessor
from ridley.riddle_generation import generate


class TestRhymeLogitsProcessor(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        rhyme_lp = RhymeLogitsProcessor()
        self.assertIsInstance(rhyme_lp, RhymeLogitsProcessor)

    def test_generate(self):
        result = generate(logits_processor=[RhymeLogitsProcessor()], do_sample=True)
        self.assertIsNotNone(result)
