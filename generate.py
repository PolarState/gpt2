import argparse
import os
import re

from cfg import cfg_defines, cfg_generator

import transformers
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel


parser = argparse.ArgumentParser(
    description="Options for main",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-m",
    "--model",
    default="GPT2",
    help="name of the model to use. 'GPT2' and 'GPTNeoX' are the only supported options currently.",
)

args = parser.parse_args()
config = vars(args)
model_name = config["model"]


checkpoint_regex = r"(?<=checkpoint\-)*.?"


cfg = cfg_defines.cfg3b
cfg_start_symbol = "22"


def validate_samples(tokenizer, model, n_samples, cfg, cfg_start_symbol):
    valid_count = 0
    for _ in range(n_samples):
        inputs = tokenizer(
            tokenizer.bos_token, return_tensors="pt", padding=True
        ).input_ids
        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            do_sample=True,
            top_k=10,
            top_p=0.95,
        )
        tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if cfg_generator.validate_string(tokenized_output[0], cfg_start_symbol, cfg):
            valid_count += 1
    print(f"{valid_count}/{n_samples} = {valid_count/n_samples}")


if model_name == "GPT2":
    output_dir = ("gpt2-cfg3b/polm-0/",)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

elif model_name == "GPTNeoX":
    output_dir = "gptneox-cfg3b/polm-0/"
    tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
        "openai-community/gpt2"
    )
    tokenizer.pad_token = tokenizer.eos_token

else:
    raise NotImplementedError(f"no implementation for {model_name}")

checkpoint_names = os.listdir(f"{output_dir}")
for checkpoint_name in checkpoint_names:
    print(checkpoint_name)
    if re.findall(checkpoint_regex, checkpoint_name):
        model = GPT2LMHeadModel.from_pretrained(f"{output_dir}/{checkpoint_name}/")
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        validate_samples(tokenizer, model, 10, cfg, cfg_start_symbol)
