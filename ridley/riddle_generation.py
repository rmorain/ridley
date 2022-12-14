from pprint import pprint
from time import time

import numpy as np
import pudb
from transformers import (GPT2Tokenizer, PhrasalConstraint, RealmScorer,
                          RealmTokenizer, pipeline, set_seed)

from ridley.document_embeddings import score_riddle


def generate(
    prompt="Hello world!",
    num_return_sequences=5,
    seed=None,
    max_length=100,
    constraints=None,
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
    do_sample=False,
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
        num_beams=5,
        temperature=0.9,
        do_sample=do_sample,
        constraints=constraints,
        repetition_penalty=10.1,
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


if __name__ == "__main__":
    result = generate_until_done()
    print(result)
