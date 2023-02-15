import time

from ridley.riddle_generation import generate_rhyming_lines_backward

max_length = [5, 10, 12, 15]
# prompt = "There's just one thing I don't understand"
prompt = "I know you may think this silly"
with open(
    f"experiments/results/rhyming_lines_backward{int(time.time())}.txt", "w"
) as f:
    for i in range(len(max_length)):
        lines = generate_rhyming_lines_backward(
            prompt, max_new_tokens=max_length[i], do_sample=True
        )
        f.writelines(f"Line length: {max_length[i]}\n\n")
        f.writelines(lines)
