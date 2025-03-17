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

import random
import transformers


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
    cfg_lengths = {ts:1 for ts in terminal_symbols}
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
                    
    return (terminal_symbols)


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

stop_symbol = '0'
# Get all unique characters in the text as vocabulary
chars = list(set(get_terminal_symbols(cfg)))
chars.append(stop_symbol)
vocab_size = len(chars)

# build the character level tokenizer
chr_to_idx = {c:i for i, c in enumerate(chars)}
idx_to_chr = {i:c for i, c in enumerate(chars)}

def encode(input_text: str) -> list[int]:
    return [chr_to_idx[t] for t in input_text]

def decode(input_tokens: list[int]) -> str:
    return "".join([idx_to_chr[i] for i in input_tokens])

print(chars)
print(vocab_size)

import torch

# use cpu or gpu based on your system
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

train_batch_size = 16  # training batch size
eval_batch_size = 8  # evaluation batch size
train_split = 0.8  # percentage of data to use from total data for training

class DataLoader:
    def __init__(self, batch_size, cfg, start_symbol) -> None:
        self.batch_size = batch_size
        self.cfg = cfg
        self.start_symbol = start_symbol
        self.context_length = get_longest_sequence(start_symbol, cfg) + 1 # stop symbol

        self.current_position = 0

    def get_batch(self) -> torch.tensor:
        b, c = self.batch_size, self.context_length

        # Genereate cfg strings with terminating tokens until we're full.
        sequences = ""
        while len(sequences) <= b * c:
            sequences += generate_from_cfg(self.start_symbol, self.cfg) + stop_symbol
        
        data = torch.tensor(encode(sequences[:b*c+1]), device=device)

        x = data[:-1].view(b, c)
        y = data[1:].view(b, c)

        self.current_position += b
        return x, y

train_loader = DataLoader(train_batch_size, cfg, "22")
eval_loader = DataLoader(eval_batch_size, cfg, "22")

xb, yb = train_loader.get_batch()
print(xb.size(), yb.size())

# used to define size of embeddings
d_model = vocab_size 


import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings

    def forward(self, inputs, targets = None):
        logits = self.wte(inputs) # dim -> batch_size, sequence_length, d_model
        loss = None
        if targets != None:
            batch_size, sequence_length, d_model = logits.shape
            # to calculate loss for all token embeddings in a batch
            # kind of a requirement for cross_entropy
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # this will store the model outputs along with the initial input sequence
        # make a copy so that it doesn't interfare with model 
        for _ in range(max_new_tokens):
            # we only pass targets on training to calculate loss
            logits, _ = self(inputs)  
            # for all the batches, get the embeds for last predicted sequence
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=1)            
            # get the probable token based on the input probs
            idx_next = torch.multinomial(probs, num_samples=1) 

            inputs = torch.cat([inputs, idx_next], dim=1)
        # as the inputs has all model outputs + initial inputs, we can use it as final output
        return inputs

m = GPT(vocab_size=vocab_size, d_model=d_model).to(device)

lr = 1e-3
optim = torch.optim.AdamW(m.parameters(), lr=lr)

epochs = 5000
eval_steps = 1000 # perform evaluation in every n steps
for ep in range(epochs):
    xb, yb = train_loader.get_batch()

    logits, loss = m(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if ep % eval_steps == 0 or ep == epochs-1:
        m.eval()
        with torch.no_grad():
            xvb, yvb = eval_loader.get_batch()
            _, e_loss = m(xvb, yvb)

            print(f"Epoch: {ep}\tlr: {lr}\ttrain_loss: {loss}\teval_loss: {e_loss}")
        m.train() # back to training mode
