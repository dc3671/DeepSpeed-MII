import argparse
import mii
import time
import torch
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/datadisk/share/llama2-7b", help="model name or path.")
parser.add_argument("--tp", type=int, default=2, help="Tensor-Parallel Size.")
parser.add_argument(
    "--prompts", type=str, nargs="+", default=[
        "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
        "DeepSpeed is",
        "Seattle is",
        '<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-bystep and justify your answer.\n<</SYS>>\n\nGiven the sentence "A woman with a fairy tattoo on her back is carrying a purse with a red floral print." can we conclude that "The woman\'s purse has red flowers on it."?\nOptions:\n- yes\n- it is not possible to tell\n- no Now, let\'s be accurate as possible. Some thinking first: [/INST]',
    ]
)
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--skip_decode", action="store_true", help="response tokens w/o tokenzier.decode")
args = parser.parse_args()

inputs = args.prompts
inputs = pickle.load(open("./bad_prompts.pvc_vs_cuda.pkl", "rb"))
# inputs = [pipe.tokenizer.encode(input) for input in inputs]
# print(f"inputs::{inputs}", flush=True)

pipe = mii.pipeline(
    args.model,
    tensor_parallel=args.tp,
    skip_decode=args.skip_decode,
    profile_model_time=True,
)

# output_file = open("blk_attn_output.log", "w")
# for i, inp in enumerate(inputs):
start_time = time.time()
responses = pipe(
    inputs[:7],
    max_new_tokens=args.max_new_tokens,
    do_sample=False,  # Greedy
    # return_full_text=True,
)
end_time = time.time()

if pipe.is_rank_0:
    # for _, r in enumerate(responses):
    for i, r in enumerate(responses):
        print(f"response {i}\nspend {end_time - start_time}\ngenerated_text:{r.generated_text}\ngenerated_tokens:{r.generated_tokens}\n", "-" * 80, "\n")
        # output_file.write(f"*** In: {inputs[i]}\n*** Out: {r.generated_text}\n\n")
