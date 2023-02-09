import pickle
import unittest

from GPT2ForwardBackward.padded_encoder import Encoder
from ridley.logit_processors import BackwardsRhymeLogitsProcessor
from ridley.riddle_generation import generate
from transformers import GenerationConfig


class TestBackwardRhymeLogitsProcessor(unittest.TestCase):
    def setUp(self):
        self.prompt = "What am I? An elephant."
        self.seed = 42
        self.generation_config = GenerationConfig(
            max_new_tokens=20,
            eos_token_id=50256,
            bos_token_id=50256,
            do_sample=True,
        )
        self.rhyme_lp = BackwardsRhymeLogitsProcessor(
            Encoder(), self.generation_config.max_new_tokens
        )

    def test_generate_backward_rhyme(self):
        with open("tests/data/generate_backward_rhyme_result.pkl", "rb") as f:
            correct_output = pickle.load(f)
            result = generate(
                self.prompt,
                generation_config=self.generation_config,
                logits_processor=[self.rhyme_lp],
                seed=self.seed,
                backward=True,
            )
            self.assertEqual(result, correct_output)


if __name__ == "__main__":
    unittest.main()
