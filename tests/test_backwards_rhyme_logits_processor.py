import pickle
import unittest

from GPT2ForwardBackward.padded_encoder import Encoder
from ridley.logit_processors import BackwardsRhymeLogitsProcessor
from ridley.riddle_generation import generate_backward


class TestBackwardRhymeLogitsProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Encoder()
        self.prompt = "\n What am I? An elephant."
        self.num_results = 1
        self.seed = 42
        self.max_length = 2 * len(self.prompt.split())
        self.rhyme_lp = BackwardsRhymeLogitsProcessor(self.tokenizer, self.max_length)

    def test_generate_backward_rhyme(self):
        with open("tests/data/generate_backward_rhyme_result.pkl", "rb") as f:
            correct_output = pickle.load(f)
            result = generate_backward(
                input_text=self.prompt,
                num_return_sequences=self.num_results,
                num_beams=1,
                logits_processor=[self.rhyme_lp],
                max_length=self.max_length,
            )
            self.assertEqual(result, correct_output)


if __name__ == "__main__":
    unittest.main()
