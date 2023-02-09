import unittest

from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder


class TestGPT2ForwardBackward(unittest.TestCase):
    def setUp(self):
        self.path_to_backward = "models/opengpt2_pytorch_backward"
        self.model_backward = OpenGPT2LMHeadModel.from_pretrained(self.path_to_backward)
        self.encoder = Encoder()
        self.device = "cpu"

    def test_tokenize_decode(self):
        input_text = "Hello world"
        tokens = self.encoder(input_text, return_tensors="pt")
        decode_text = self.encoder.decode(tokens.input_ids[0])
        self.assertEqual(input_text, decode_text)
