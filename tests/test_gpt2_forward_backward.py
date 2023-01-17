import pickle
import unittest

import torch
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder


class TestGPT2ForwardBackward(unittest.TestCase):
    def setUp(self):
        self.path_to_backward = "models/opengpt2_pytorch_backward"
        self.model_backward = OpenGPT2LMHeadModel.from_pretrained(self.path_to_backward)
        self.encoder = Encoder()
        self.device = "cpu"
        # if torch.cuda.is_available():
        # self.device = "cuda:0"
        # else:
        # self.device = "cpu"

    def test_generate_backward_text(self):
        with open("tests/data/backwards_test.pkl", "rb") as f:
            correct_output = pickle.load(f)
            input_text = " And that was the last I heard from her."
            input_tokens = self.encoder.encode(input_text)[::-1]

            output = self.model_backward.generate(
                torch.tensor([input_tokens]).to(self.device)
            )
            output_tokens = output.tolist()[0][::-1]
            result = self.encoder.decode(output_tokens)
            self.assertEqual(result, correct_output)
