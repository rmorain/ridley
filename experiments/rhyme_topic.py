from GPT2ForwardBackward.padded_encoder import Encoder
from ridley.ConceptNetAPiAccess import GetAllCommonNeighbors
from ridley.logit_processors import (BackwardsRhymeLogitsProcessor,
                                     TopicalLogitsProcessor)
from ridley.riddle_generation import generate_lines
from transformers import GenerationConfig

input_text = "What am I? The stock market."
tokenizer = Encoder()
bad_words = ["stock", "market"]
force_words = ["pudding"]
bad_words_ids = tokenizer(
    bad_words,
    add_prefix_space=True,
    add_special_tokens=False,
    return_tensors="pt",
).input_ids.tolist()
force_words_ids = [
    tokenizer(
        force_words,
        add_prefix_space=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.tolist()
]

generation_config = GenerationConfig(
    max_new_tokens=30,
    eos_token_id=50256,
    bos_token_id=50256,
    bad_words_ids=bad_words_ids,
    # force_words_ids=force_words_ids,
    # do_sample=True,
    num_beams=10,
    repetition_penalty=1.2,
)
max_new_tokens = 20
topics = ["the stock market"]
booster = 2
rhyme_lp = BackwardsRhymeLogitsProcessor(tokenizer, max_new_tokens, do_sample=True)
topic_lp = TopicalLogitsProcessor(tokenizer, max_new_tokens, topics, booster)
logits_processor = [topic_lp, rhyme_lp]
# logits_processor = [rhyme_lp]
result = generate_lines(
    input_text,
    num_lines=4,
    generation_config=generation_config,
    logits_processor=logits_processor,
    backward=True,
)
print(result)
