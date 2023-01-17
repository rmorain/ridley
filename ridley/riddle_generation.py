from time import time

import numpy as np
import torch
from GPT2ForwardBackward.modeling_opengpt2 import OpenGPT2LMHeadModel
from GPT2ForwardBackward.padded_encoder import Encoder
from transformers import (GPT2Tokenizer, PhrasalConstraint, RealmScorer,
                          RealmTokenizer, pipeline, set_seed)

from ridley.document_embeddings import score_riddle
from ridley.logit_processors import RhymeLogitsProcessor


def generate(
    prompt="Hello world!",
    num_return_sequences=5,
    seed=None,
    max_length=100,
    constraints=None,
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
    do_sample=False,
    logits_processor=[],
    num_beams=5,
):
    if not seed:
        seed = np.random.randint(100000)
    if constraints:
        constraint_tokens = tokenizer(constraints).input_ids
        constraints = [PhrasalConstraint(token_ids=t) for t in constraint_tokens]

    generator = pipeline("text-generation", model="gpt2")
    set_seed(seed)

    result = generator(
        prompt,
        max_new_tokens=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        temperature=0.9,
        do_sample=do_sample,
        constraints=constraints,
        repetition_penalty=10.1,
        logits_processor=logits_processor,
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
    rhyme_lp = RhymeLogitsProcessor(tokenizer, max_length)
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


def generate_rhyming_lines_backward(prompt, num_lines=5, max_length=5, do_sample=False):
    pass


def generate_backward(
    input_text="Hello World!",
    num_return_sequences=1,
    max_length=20,
    num_beams=1,
    logits_processor=[],
):
    path_to_backward = "models/opengpt2_pytorch_backward"
    model_backward = OpenGPT2LMHeadModel.from_pretrained(path_to_backward)
    tokenizer = Encoder()
    input_tokens = tokenizer.encode(input_text)[::-1]  # Reverse

    output = model_backward.generate(
        torch.tensor([input_tokens]),
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        logits_processor=logits_processor,
    )
    output_tokens = output.tolist()[0][::-1]
    output_text = tokenizer.decode(output_tokens)
    return output_text


if __name__ == "__main__":
    result = generate_until_done()
    print(result)
