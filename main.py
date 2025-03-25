import argparse
from datetime import datetime
import random
import time
import torch
import transformers
import os

# from hf
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

# EPOCHS = 1

parser = argparse.ArgumentParser(
    description="Options for main",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# parser.add_argument("-o", "--output", required=True, help="Path to output folder.")
parser.add_argument(
    "-r",
    "--resume",
    default=False,
    help="Resume training.",
    action=argparse.BooleanOptionalAction,
)

args = parser.parse_args()
config = vars(args)
resume = config["resume"]
# output_path = config["output"]


cfg3b = {
    "22": [["21", "20"], ["20", "19"]],
    "21": [["18", "16"], ["16", "18", "17"]],
    "20": [["17", "16", "18"], ["16", "17"]],
    "19": [["16", "17", "18"], ["17", "18", "16"]],
    "18": [["15", "14", "13"], ["14", "13"]],
    "17": [["14", "13", "15"], ["15", "13", "14"]],
    "16": [["15", "13"], ["13", "15", "14"]],
    "15": [["12", "11", "10"], ["11", "12", "10"]],
    "14": [["11", "10", "12"], ["10", "11", "12"]],
    "13": [["11", "12"], ["12", "11"]],
    "12": [["8", "9", "7"], ["9", "7", "8"]],
    "11": [["8", "7", "9"], ["7", "8", "9"]],
    "10": [["7", "9", "8"], ["9", "8", "7"]],
    "9": [["3", "2", "1"], ["2", "1"]],
    "8": [["3", "2"], ["3", "1", "2"]],
    "7": [["3", "1"], ["1", "2", "3"]],
}

cfg = cfg3b
cfg_start_symbol = "22"


def _flatten_symbols(symbols):
    flat_symbols = []
    for symbol in symbols:
        if isinstance(symbol, list):
            flat_symbols.extend(_flatten_symbols(symbol))
        else:
            flat_symbols.append(symbol)
    return flat_symbols


def get_longest_sequence(start_symbol, cfg_rules):
    terminal_symbols = set(get_terminal_symbols(cfg_rules))
    cfg_lengths = {ts: 1 for ts in terminal_symbols}
    rules = list(cfg_rules.keys())

    while rules:
        next_rule = rules.pop()
        nts_set = set(_flatten_symbols(cfg_rules[next_rule]))
        calculate_length = True
        for nts in nts_set:
            # if we don't find all the symbols we need in our lengths, skip for now.
            if nts not in cfg_lengths:
                calculate_length = False
                break

        if calculate_length:
            generation_lengths = []
            for generation_list in cfg_rules[next_rule]:
                list_length = 0
                for generation_rule in generation_list:
                    list_length += cfg_lengths[generation_rule]
                generation_lengths.append(list_length)
            cfg_lengths[next_rule] = max(generation_lengths)
        else:
            rules.append(next_rule)

    return cfg_lengths[start_symbol]


def generate_from_cfg(
    symbol: str,
    cfg_rules: dict[str, str],
) -> str:
    """Generate a string from the CFG using recursive expansion.

    Args:
        symbol: next symbol to select from.
        cfg_rules: dictionary of tokens to recursively sample from.

    Returns:
        cfg string of length max_depth or exclusively terminal symbols.
    """
    if symbol not in cfg_rules:  # Terminal symbol reached.
        return symbol
    production = random.choice(cfg_rules[symbol])  # Randomly pick a production rule
    return "".join(generate_from_cfg(sym, cfg_rules) for sym in production)


def get_terminal_symbols(cfg_rules):
    # Find all terminal symbols in cfg.
    terminal_symbols = []
    for values in cfg_rules.values():
        for value in values:
            for v in value:
                if v not in cfg_rules:
                    terminal_symbols.append(v)

    return terminal_symbols


def validate_string(input: str, start_symbol: str, cfg_rules: dict[str, str]):
    # TODO: derive the start symbol(s)

    # Reverse dictionary.
    reverse_cfg_rules = {}
    for k, values in cfg_rules.items():
        for v in values:
            reverse_cfg_rules[tuple(v)] = k

    # Find all terminal symbols in cfg.
    terminal_symbols = get_terminal_symbols(cfg_rules)

    # Split input string into array of terminal symbols.
    # WARNING: this won't work if terminal symbols overlap ('1' and '11').
    #   We can adopt a DP approach like parse_cfg if this is needed.
    max_t = max(len(t) for t in terminal_symbols)
    start = 0
    stop = 1
    tape = []
    while stop <= len(input):
        found_t = False
        while stop - start <= max_t and not found_t:
            if input[start:stop] in terminal_symbols:
                tape.append(input[start:stop])
                start = stop
                found_t = True
            stop += 1
        # If there is a non-terminal symbol then it's not a valid cfg.
        if not found_t:
            return False

    max_symbol_len = max([len(key) for key in reverse_cfg_rules.keys()])

    def parse_cfg(input_tape, output_tape):
        # If our input tape is empty and we still have an output tape:
        if not input_tape and output_tape:
            # Check if the output tape is the start symbol.
            if output_tape == [start_symbol]:
                return True
            # Otherwise parse the output tokens.
            else:
                return parse_cfg(output_tape, [])
        # If there is no valid input or output tape exit.
        elif not input_tape and not output_tape:
            return False

        # Try matching every window length from the input to the rules.
        for window in range(1, max_symbol_len + 1):
            next_tuple = tuple(input_tape[:window])
            # If we get a window match, continue parsing.
            if next_tuple in reverse_cfg_rules and parse_cfg(
                input_tape[window:], output_tape + [reverse_cfg_rules[next_tuple]]
            ):
                return True

        # If there are no true matches, return false.
        return False

    # return parse_cfg(tape, [start_symbol], reverse_cfg_rules)
    return parse_cfg(tape, [])


# use cpu or gpu based on your system
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


# load the gpt-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")


class CFGDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        cfg_rules: dict[str, str],
        start_symbol: str,
        num_generations: int,
        window_length: int = 512,
    ):
        """Each CFG could be drawn from infinite times. To satisfy PyTorch Dataset, we ask for the length."""
        super().__init__()
        self.cfg_rules = cfg_rules
        self.start_symbol = start_symbol
        self.num_generations = num_generations
        self.idx = 0
        self.generation_buffer = []
        self.window_length = window_length

    def __len__(self):
        return self.num_generations

    def __iter__(self):
        # Reset our internal count when we're asked to iterate again.
        self.idx = 0
        return self

    def __next__(self):

        # Exit if we've completed all iterations.
        if self.idx >= len(self):
            raise StopIteration

        # Fill our generation buffer up to our widow length.
        while len(self.generation_buffer) < self.window_length:
            self.generation_buffer.extend(
                tokenizer.encode(
                    tokenizer.bos_token
                    + generate_from_cfg(self.start_symbol, self.cfg_rules)
                    + tokenizer.eos_token
                )
            )

        # Update our fake iterator length.
        self.idx += 1

        # Generate tensors from our window.
        next_item = torch.tensor(
            self.generation_buffer[: self.window_length],
            device=device,
        )

        # Trim outgoing tokens from our window.
        self.generation_buffer = self.generation_buffer[self.window_length :]

        return next_item


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

gpt_config = transformers.GPT2Config()

model = transformers.GPT2LMHeadModel(gpt_config)

batch_size = 22
training_args = TrainingArguments(
    output_dir="gpt-2-cfg3b/polm-0/",
    evaluation_strategy="steps",
    eval_steps=500,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=0.003,
    lr_scheduler_type="linear",
    adam_beta1=0.9,
    adam_beta2=0.98,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=500,
    save_steps=5000,
    save_total_limit=10,
    report_to="wandb",
    dataloader_pin_memory=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=CFGDataset(cfg, cfg_start_symbol, 100000 * 96),
    eval_dataset=CFGDataset(cfg, cfg_start_symbol, 10000),
    data_collator=data_collator,
)

checkpoint_names = os.listdir("./gpt-2-cfg3b/polm-0")
if resume:
    last_checkpoint = list(reversed(sorted(checkpoint_names)))[0]
    trainer.train(f"gpt-2-cfg3b/polm-0/{last_checkpoint}/")
else:
    trainer.train()
