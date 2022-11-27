from pprint import pprint

import numpy as np
import pudb
from transformers import GPT2Tokenizer, PhrasalConstraint, pipeline, set_seed


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


if __name__ == "__main__":
    prompt = (
        "Riddle 1\n\n"
        "Thirty white horses on a red hill, first they champ, then they "
        "stamp, then they stand still"
        "\n\nRiddle 2:"
    )
    # prompt = "Riddle me this! Riddle me that!"
    # constraints = ["What am I?", " dog "]
    constraints = None
    result = generate(prompt=prompt, constraints=constraints, do_sample=True)
    pprint(result)
