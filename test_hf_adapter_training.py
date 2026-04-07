"""Smoke test: train a small GPTNeoX model using the HFTokenizerAdapter.

Generates a small dataset from cfg3b on the fly, tokenizes it with the
CFGCharacterTokenizer wrapped in HFTokenizerAdapter, and runs a handful
of training steps to verify everything plugs together.
"""

import torch
import transformers
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from cfg.cfg_grammar import CFGrammar
from cfg.cfg_tokenizers import CFGCharacterTokenizer
from cfg.hf_adapter import HFTokenizerAdapter


def build_tokenizer(grammar: CFGrammar) -> HFTokenizerAdapter:
    """Build an HF-compatible tokenizer from a grammar's terminal symbols."""
    char_tok = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    return HFTokenizerAdapter(char_tok)


def generate_token_sequences(grammar, tokenizer, n_sequences, max_length):
    """Generate a list of padded/truncated token-id tensors from the grammar."""
    sequences = []
    for _ in range(n_sequences):
        # Generate a valid CFG string and wrap with bos/eos.
        cfg_string = grammar.generate()
        text = tokenizer.bos_token + cfg_string + tokenizer.eos_token

        # Encode through the HF interface (no extra special tokens — we
        # added bos/eos manually to mirror the existing training pipeline).
        ids = tokenizer.encode(text, add_special_tokens=False)

        # Pad or truncate to max_length.
        if len(ids) < max_length:
            ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))
        else:
            ids = ids[:max_length]

        sequences.append(ids)
    return sequences


class SimpleListDataset(torch.utils.data.Dataset):
    """Minimal dataset wrapping a list of token-id lists."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx])


def main():
    grammar = CFGrammar.from_name("cfg3b")
    tokenizer = build_tokenizer(grammar)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"bos_token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
    print(f"eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # Verify encode/decode roundtrip on a generated string.
    sample = grammar.generate()
    ids = tokenizer.encode(sample, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"\nSample:  {sample!r}")
    print(f"Ids:     {ids}")
    print(f"Decoded: {decoded!r}")
    assert decoded == sample, f"roundtrip failed: {decoded!r} != {sample!r}"

    # Generate small train and eval datasets.
    max_length = 64
    train_seqs = generate_token_sequences(grammar, tokenizer, n_sequences=200, max_length=max_length)
    eval_seqs = generate_token_sequences(grammar, tokenizer, n_sequences=20, max_length=max_length)

    train_dataset = SimpleListDataset(train_seqs)
    eval_dataset = SimpleListDataset(eval_seqs)

    # Configure a tiny GPTNeoX so it trains fast.
    gpt_config = transformers.GPTNeoXConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=max_length,
    )
    model = transformers.GPTNeoXForCausalLM(gpt_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="/tmp/hf_adapter_test",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Quick sanity check: generate from the trained model.
    print("\nGenerating from trained model...")
    input_ids = tokenizer(tokenizer.bos_token, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=True,
            top_k=tokenizer.vocab_size,
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: {generated!r}")

    is_valid = grammar.validate(generated)
    print(f"Valid CFG: {is_valid}")
    print("\nTraining smoke test PASSED.")


if __name__ == "__main__":
    main()
