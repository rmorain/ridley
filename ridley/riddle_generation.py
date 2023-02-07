from time import time

import numpy as np
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder
from transformers import (GenerationConfig, GPT2Tokenizer, RealmScorer,
                          RealmTokenizer, pipeline, set_seed)

from ridley.document_embeddings import score_riddle
from ridley.logit_processors import (BackwardsRhymeLogitsProcessor,
                                     RhymeLogitsProcessor)
from ridley.pipelines import BackwardsTextGenerationPipeline


def generate(
    inputs,
    model=None,
    tokenizer=None,
    generation_config=GenerationConfig(pad_token_id=50256, eos_token_id=50256),
    logits_processor=[],
    stopping_criteria=[],
    prefix_allowed_tokens_fn=None,
    synced_gpus=False,
    seed=None,
    backward=False,
):
    if not seed:
        seed = np.random.randint(100000)
    if model and tokenizer:
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    elif backward:
        path_to_backward = "models/opengpt2_pytorch_backward"
        model_backward = OpenGPT2LMHeadModel.from_pretrained(path_to_backward)
        tokenizer = Encoder()
        generator = BackwardsTextGenerationPipeline(
            model=model_backward, tokenizer=tokenizer
        )
    else:
        generator = pipeline("text-generation", model="gpt2")
    set_seed(seed)

    result = generator(
        inputs,
        generation_config=generation_config,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        synced_gpus=synced_gpus,
    )
    seqs = [r["generated_text"] for r in result]
    return seqs


def generate_until_done():
    done = False
    start = time()
    bssf = (None, float("-inf"), 0)
    scorer = RealmScorer.from_pretrained(
        "google/realm-cc-news-pretrained-scorer", num_candidates=2
    )
    eval_tokenizer = RealmTokenizer.from_pretrained(
        "google/realm-cc-news-pretrained-embedder"
    )
    candidate_file1 = "data/kaggle_riddles/riddles.csv"
    candidate_file2 = "data/jeopardy.csv"
    num_candidates = 2
    prompt = "Here is a riddle I really enjoy: "
    num_return_sequences = 5
    seed = None
    max_length = 100
    constraints = None
    gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gen_tokenizer.padding_token = gen_tokenizer.eos_token
    do_sample = True
    threshold = 0.5
    lamda = 0.9
    time_limit = 60
    # metric = "cosine similarity"
    metric = "euclidean distance"
    while not done or (start - time() > time_limit):
        riddles = generate(
            prompt,
            num_return_sequences,
            seed,
            max_length,
            constraints,
            gen_tokenizer,
            do_sample,
        )
        for r in riddles:
            result = score_riddle(
                scorer,
                eval_tokenizer,
                r,
                candidate_file1,
                candidate_file2,
                num_candidates,
                lamda,
                metric,
            )
            if result > bssf[1]:
                bssf = (r, result, bssf[2] + 1)
        if bssf[1] <= threshold:
            return bssf

    return bssf


def generate_rhyming_lines(prompt, num_lines=5, max_length=5, do_sample=False):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    rhyme_lp = RhymeLogitsProcessor(tokenizer, max_length, do_sample=do_sample)
    lines = prompt + "\n "
    prev_len = len(prompt.split())
    for i in range(num_lines):
        result = generate(
            lines,
            num_return_sequences=1,
            max_length=max_length,
            do_sample=do_sample,
            logits_processor=[rhyme_lp],
            num_beams=1,
        )[0]
        line = " ".join(result.split()[prev_len:])
        prev_len = len(result.split())
        lines += line + "\n "

    return lines


def generate_rhyming_lines_backward(
    prompt, num_lines=5, max_new_tokens=5, do_sample=False
):
    tokenizer = Encoder()
    rhyme_lp = BackwardsRhymeLogitsProcessor(
        tokenizer, max_new_tokens, do_sample=do_sample
    )
    lines = "\n" + prompt
    prev_len = len(prompt.split())
    for i in range(num_lines):
        result = generate_backward(
            lines,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            logits_processor=[rhyme_lp],
            num_beams=1,
        )
        line = " ".join(result.split()[: len(result.split()) - prev_len])
        prev_len = len(result.split())
        lines = "\n" + line + lines

    return lines


if __name__ == "__main__":
    result = generate_until_done()
    print(result)
