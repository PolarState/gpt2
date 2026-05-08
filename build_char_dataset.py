"""Build a character-tokenized CFG dataset.

Two modes:

  --num-examples N   Generate N examples, hold them in memory, write the
                     packed file, and round-trip every example through
                     CFGFileDataset to verify the on-disk format. Suitable
                     for small N (verification).

  --num-windows N    Stream-generate examples until N complete windows
                     have been written to disk. The id stream is built
                     directly from CFGCharacterTokenizer's encode_vocab
                     (skipping HF per-call overhead) and flushed in
                     batches as numpy big-endian int32. After writing,
                     a small spot check decodes the head of the file
                     and validates a few examples. Suitable for the
                     train/val sets at training scale.

Both modes write the same on-disk layout that ``CFGFileDataset`` reads:
fixed-size windows of big-endian signed int32 ids, packed back-to-back
with no per-window header. Each example is laid out as
``bos_id + char_ids + eos_id`` in a continuous stream that is then
sliced into windows; example boundaries do not align to window
boundaries.
"""

import argparse
import os
import random
import struct
import sys
import time

import numpy as np

from cfg.cfg_datasets import CFGFileDataset
from cfg.cfg_grammar import CFGrammar
from cfg.cfg_tokenizers import CFGCharacterTokenizer
from cfg.hf_adapter import HFTokenizerAdapter


# ── Small mode (with full verification) ────────────────────────────────


def build_tokenizer(grammar: CFGrammar) -> HFTokenizerAdapter:
    char_tok = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    return HFTokenizerAdapter(char_tok)


def parse_mask_rule(spec: str, rules: dict) -> tuple[str, int]:
    """Parse '<NT>:<INDEX>' (e.g. '7:0') and validate against the grammar."""
    if ":" not in spec:
        raise ValueError(f"--mask-rule must be 'NT:INDEX', got {spec!r}")
    nt, idx_str = spec.split(":", 1)
    if nt not in rules:
        raise ValueError(f"NT {nt!r} not in grammar; valid: {sorted(rules.keys())}")
    idx = int(idx_str)
    if not (0 <= idx < len(rules[nt])):
        raise ValueError(
            f"NT {nt!r} has {len(rules[nt])} rules, index {idx} out of range"
        )
    return nt, idx


def build_mask_weights(rules: dict, mask_nt: str, mask_idx: int) -> dict:
    """Weights dict: 1.0 for every rule except the masked one (0.0).

    Passed to CFGrammar.generate(weights=...). random.choices treats a
    weight of 0 as never-selected, so the masked production is fully
    suppressed without modifying the grammar object.
    """
    return {
        nt: [
            0.0 if (nt == mask_nt and i == mask_idx) else 1.0
            for i in range(len(prods))
        ]
        for nt, prods in rules.items()
    }


def tokenize_examples(grammar, tokenizer, num_examples, weights=None):
    examples = []
    for _ in range(num_examples):
        text = grammar.generate(weights=weights)
        ids = (
            [tokenizer.bos_token_id]
            + tokenizer.encode(text, add_special_tokens=False)
            + [tokenizer.eos_token_id]
        )
        examples.append((text, ids))
    return examples


def pack_windows(examples, pad_token_id, window_length):
    flat = []
    for _text, ids in examples:
        flat.extend(ids)

    remainder = len(flat) % window_length
    if remainder != 0:
        flat.extend([pad_token_id] * (window_length - remainder))

    return [flat[i : i + window_length] for i in range(0, len(flat), window_length)]


def write_dataset(windows, output_path, window_length):
    fmt = "!" + "i" * window_length
    with open(output_path, "wb") as f:
        for window in windows:
            assert len(window) == window_length, len(window)
            f.write(struct.pack(fmt, *window))


def read_flat_stream(output_path, window_length):
    """Read the file back through CFGFileDataset and concatenate windows."""
    dataset = CFGFileDataset(output_path, device="cpu", window_length=window_length)
    file_size = dataset._mmap.size()
    num_windows = file_size // (window_length * 4)
    flat = []
    for i in range(num_windows):
        offset = i * window_length * 4
        buf = dataset._mmap[offset : offset + window_length * 4]
        flat.extend(np.frombuffer(buf, dtype=">i4").astype(np.int64).tolist())
    return flat


def verify_roundtrip(flat, examples, tokenizer, grammar):
    """Walk the flat id stream and assert each example is recovered intact."""
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    cursor = 0
    for idx, (orig_text, orig_ids) in enumerate(examples):
        if cursor >= len(flat) or flat[cursor] != bos_id:
            raise AssertionError(
                f"example {idx}: expected bos at position {cursor}, got "
                f"{flat[cursor] if cursor < len(flat) else 'EOF'}"
            )
        end = cursor + 1
        while end < len(flat) and flat[end] != eos_id:
            end += 1
        if end >= len(flat):
            raise AssertionError(f"example {idx}: ran off end before eos")

        recovered_ids = flat[cursor : end + 1]
        if recovered_ids != orig_ids:
            raise AssertionError(
                f"example {idx}: id mismatch\n"
                f"  expected: {orig_ids}\n"
                f"  got:      {recovered_ids}"
            )

        body_ids = recovered_ids[1:-1]
        decoded = tokenizer.decode(body_ids, skip_special_tokens=False)
        if decoded != orig_text:
            raise AssertionError(
                f"example {idx}: text mismatch\n"
                f"  expected: {orig_text!r}\n"
                f"  got:      {decoded!r}"
            )
        if not grammar.validate(decoded):
            raise AssertionError(f"example {idx}: decoded text not in grammar: {decoded!r}")

        cursor = end + 1

    pad_id = tokenizer.pad_token_id
    tail = flat[cursor:]
    if any(t != pad_id for t in tail):
        non_pad = [t for t in tail if t != pad_id]
        raise AssertionError(
            f"unexpected non-pad tokens after last example: {non_pad[:10]}..."
        )

    return len(tail)


def run_small_mode(args):
    grammar = CFGrammar.from_name(args.cfg)
    tokenizer = build_tokenizer(grammar)
    print(
        f"grammar={args.cfg}  vocab_size={tokenizer.vocab_size}  "
        f"bos={tokenizer.bos_token!r}({tokenizer.bos_token_id})  "
        f"eos={tokenizer.eos_token!r}({tokenizer.eos_token_id})"
    )

    weights = None
    if args.mask_rule is not None:
        nt, idx = parse_mask_rule(args.mask_rule, grammar.rules)
        print(f"masking NT {nt!r} rule index {idx}: {grammar.rules[nt][idx]}")
        weights = build_mask_weights(grammar.rules, nt, idx)

    examples = tokenize_examples(grammar, tokenizer, args.num_examples, weights=weights)
    total_ids = sum(len(ids) for _, ids in examples)
    print(
        f"generated {len(examples)} examples, "
        f"{total_ids} total ids "
        f"(avg {total_ids / len(examples):.1f}/example)"
    )

    windows = pack_windows(examples, tokenizer.pad_token_id, args.window_length)
    write_dataset(windows, args.output, args.window_length)
    print(
        f"wrote {len(windows)} windows of {args.window_length} tokens "
        f"to {args.output} ({os.path.getsize(args.output)} bytes)"
    )

    flat = read_flat_stream(args.output, args.window_length)
    pad_count = verify_roundtrip(flat, examples, tokenizer, grammar)
    print(
        f"roundtrip OK: all {len(examples)} examples recovered, "
        f"{pad_count} trailing pad tokens"
    )


# ── Large mode (streaming, with spot check) ───────────────────────────


def run_streaming_mode(args):
    grammar = CFGrammar.from_name(args.cfg)
    char_tok = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    encode_vocab = char_tok.encode_vocab
    bos_id = encode_vocab[char_tok.bos_string]
    eos_id = encode_vocab[char_tok.eos_string]

    print(
        f"grammar={args.cfg}  vocab_size={len(char_tok)}  "
        f"bos={char_tok.bos_string!r}({bos_id})  "
        f"eos={char_tok.eos_string!r}({eos_id})"
    )

    weights = None
    if args.mask_rule is not None:
        nt, idx = parse_mask_rule(args.mask_rule, grammar.rules)
        print(f"masking NT {nt!r} rule index {idx}: {grammar.rules[nt][idx]}")
        weights = build_mask_weights(grammar.rules, nt, idx)

    window_length = args.window_length
    num_windows = args.num_windows
    flush_windows = max(1, args.flush_windows)
    flush_size = flush_windows * window_length  # ids per flush

    print(
        f"target: {num_windows} windows × {window_length} tokens "
        f"= {num_windows * window_length:,} tokens "
        f"({num_windows * window_length * 4 / 1e9:.2f} GB)"
    )
    print(f"flush every {flush_windows} windows ({flush_size:,} ids per flush)")

    written = 0
    buf = []  # int ids waiting to be flushed
    examples_generated = 0
    t0 = time.time()
    last_progress_t = t0

    with open(args.output, "wb") as f:
        while written < num_windows:
            windows_remaining = num_windows - written
            target_chunk_ids = min(flush_size, windows_remaining * window_length)

            # Generate examples until we have enough ids to flush.
            while len(buf) < target_chunk_ids:
                text = grammar.generate(weights=weights)
                buf.append(bos_id)
                # Inline the encode loop: dict lookup per char, no HF overhead.
                for c in text:
                    buf.append(encode_vocab[c])
                buf.append(eos_id)
                examples_generated += 1

            # Slice off exactly target_chunk_ids ids and write as
            # big-endian int32. Anything beyond is leftover for next chunk.
            chunk = buf[:target_chunk_ids]
            del buf[:target_chunk_ids]

            arr = np.asarray(chunk, dtype=np.int32).astype(">i4", copy=False)
            arr.tofile(f)

            written += target_chunk_ids // window_length

            now = time.time()
            if now - last_progress_t >= args.progress_seconds or written == num_windows:
                elapsed = now - t0
                rate = written / elapsed if elapsed > 0 else 0
                eta = (num_windows - written) / rate if rate > 0 else 0
                print(
                    f"  {written:>10,}/{num_windows:,} windows "
                    f"({100 * written / num_windows:5.1f}%)  "
                    f"{rate:.0f} win/s  "
                    f"ex={examples_generated:,}  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                    flush=True,
                )
                last_progress_t = now

    elapsed = time.time() - t0
    file_size = os.path.getsize(args.output)
    print(
        f"done: {written:,} windows, {examples_generated:,} examples, "
        f"{file_size:,} bytes ({file_size / 1e9:.2f} GB), "
        f"{elapsed:.1f}s wall ({written / elapsed:.0f} win/s)"
    )

    # Spot check: decode the first window and validate any complete
    # examples it contains against the grammar.
    spot_check_streaming(args.output, window_length, grammar, char_tok, bos_id, eos_id)


def spot_check_streaming(output_path, window_length, grammar, char_tok, bos_id, eos_id):
    print("spot check (first window):")
    dataset = CFGFileDataset(output_path, device="cpu", window_length=window_length)
    arr = dataset[0].cpu().numpy().tolist()

    # Find every complete bos..eos span in this window.
    spans = []
    i = 0
    while i < len(arr):
        if arr[i] == bos_id:
            j = i + 1
            while j < len(arr) and arr[j] != eos_id:
                j += 1
            if j < len(arr):
                spans.append((i, j))
                i = j + 1
            else:
                break
        else:
            i += 1

    if not spans:
        print("  no complete bos..eos spans in first window (window may split an example)")
        return

    valid = 0
    for start, end in spans:
        body_ids = arr[start + 1 : end]
        decoded = "".join(char_tok.decode_vocab[i] for i in body_ids)
        if grammar.validate(decoded):
            valid += 1
        else:
            print(f"  INVALID: {decoded!r}")

    print(f"  {valid}/{len(spans)} complete examples in first window pass grammar.validate()")
    print(f"  first example: {''.join(char_tok.decode_vocab[i] for i in arr[spans[0][0] + 1 : spans[0][1]])!r}")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", default="cfg3b", help="grammar name from cfg_defines")
    p.add_argument("--output", required=True, help="output .bin path")
    p.add_argument("--window-length", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--num-examples", type=int, help="small mode: N examples + full verification")
    g.add_argument("--num-windows", type=int, help="streaming mode: target N windows on disk")

    p.add_argument(
        "--flush-windows",
        type=int,
        default=2000,
        help="streaming mode: flush every K windows (≈4 MB at K=2000)",
    )
    p.add_argument(
        "--progress-seconds",
        type=float,
        default=10.0,
        help="streaming mode: print progress every N seconds",
    )
    p.add_argument(
        "--mask-rule",
        default=None,
        help="Suppress one production rule by setting its sampling weight to 0. "
             "Format: 'NT:INDEX' (e.g. '7:0'). Other rules keep weight 1.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        sys.exit(f"refusing to overwrite {args.output} (pass --overwrite)")

    random.seed(args.seed)

    if args.num_examples is not None:
        run_small_mode(args)
    else:
        run_streaming_mode(args)


if __name__ == "__main__":
    main()
