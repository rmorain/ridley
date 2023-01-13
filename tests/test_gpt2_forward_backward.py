import unittest

from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder


class TestGPT2ForwardBackward(unittest.TestCase):
    def setUp(self):
        self.path_to_backward = "models/opengpt2_pytorch_backward.tar.gz"

    def init(self):
        model_backward = OpenGPT2LMHeadModel.from_pretrained(self.path_to_backward)
        self.assertIsNotNone(model_backward)
        encoder = Encoder()
        self.assertIsNotNone(encoder)
