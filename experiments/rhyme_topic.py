from GPT2ForwardBackward.padded_encoder import Encoder
from ridley.logit_processors import (BackwardsRhymeLogitsProcessor,
                                     TopicalPriorLogitsProcessor)
from ridley.riddle_generation import generate_lines
from transformers import GenerationConfig

input_text = "Hello world"
generation_config = GenerationConfig(
    max_new_tokens=10,
    do_sample=True,
    eos_token_id=50256,
    bos_token_id=50256,
)
tokenizer = Encoder()
max_new_tokens = 10
topics = ["Pudding"]
booster = 3
rhyme_lp = BackwardsRhymeLogitsProcessor(tokenizer, max_new_tokens, do_sample=True)
topic_lp = TopicalPriorLogitsProcessor(tokenizer, max_new_tokens, topics, booster)
logits_processor = [rhyme_lp, topic_lp]


result = generate_lines(
    input_text,
    num_lines=3,
    generation_config=generation_config,
    logits_processor=logits_processor,
    backward=True,
)
print(result)
