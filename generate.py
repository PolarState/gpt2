import os
import re

# from hf
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel


checkpoint_regex = r"(?<=checkpoint\-)*.?"

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


# load the gpt-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token


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
        if validate_string(tokenized_output[0], cfg_start_symbol, cfg):
            valid_count += 1
    print(f"{valid_count}/{n_samples} = {valid_count/n_samples}")


checkpoint_names = os.listdir("./gpt-2-cfg3b/polm-0")
for checkpoint_name in checkpoint_names:
    print(checkpoint_name)
    if re.findall(checkpoint_regex, checkpoint_name):
        model = GPT2LMHeadModel.from_pretrained(
            f"gpt-2-cfg3b/polm-0/{checkpoint_name}/"
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        validate_samples(tokenizer, model, 10, cfg, cfg_start_symbol)

