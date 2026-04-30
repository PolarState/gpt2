"""Build a suffix-array infinigram from a CFGFileDataset .bin file.

Mirrors analysis/build_dawg.py in structure but uses pydivsufsort to
construct an in-memory suffix array instead of a CDAWG. The SA supports
the same arbitrary-length count queries as the CDAWG (which is the
"infinigram" use case from Liu et al. 2024) but builds much faster on
small-alphabet data: ~5s for the 49M-token val set vs ~240s for the
CDAWG.

The on-disk layout under <output_dir>/ is:
  tokens.bin        — raw little-endian uint8 of the token stream
  suffix_array.bin  — raw little-endian int32 (or int64) suffix array
  meta.txt          — small text file recording dtype + lengths

Usage:
    # Build from validation set:
    python analysis/build_infinigram.py --dataset ../CFG/datasets/cfg3b_val_dataset.bin

    # Build from training set:
    python analysis/build_infinigram.py --dataset ../CFG/datasets/cfg3b_train_dataset.bin
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pydivsufsort


def convert_dataset(dataset_path, window_length=512, max_windows=None):
    """Read big-endian int32 .bin file into a uint8 numpy array.

    The CFG .bin files store token IDs as big-endian int32 in 512-token
    windows (see cfg.cfg_datasets.CFGFileDataset). For the cfg3b
    character vocabulary the only token IDs are {0, 1, 2, 3, 4}, so the
    int32 → uint8 downcast is lossless and lets pydivsufsort use its
    fast bytes path. We assert that downcast holds before writing.
    """
    file_size = os.path.getsize(dataset_path)
    total_ints = file_size // 4
    total_windows = total_ints // window_length

    if max_windows is not None:
        n_windows = min(max_windows, total_windows)
    else:
        n_windows = total_windows

    n_tokens = n_windows * window_length
    print(f"  File has {total_windows:,} windows; reading {n_windows:,} ({n_tokens:,} tokens)")

    chunk_size = 50_000_000  # 50M tokens per chunk to keep peak memory bounded.
    if n_tokens <= chunk_size:
        raw = np.fromfile(dataset_path, dtype=">i4", count=n_tokens)
        max_id = int(raw.max())
        if max_id > 255:
            raise ValueError(f"token id {max_id} exceeds uint8 range — use a wider dtype")
        tokens = raw.astype(np.uint8)
        del raw
    else:
        tokens = np.empty(n_tokens, dtype=np.uint8)
        n_chunks = (n_tokens + chunk_size - 1) // chunk_size
        bar_width = 40
        with open(dataset_path, "rb") as f:
            max_seen = 0
            for i in range(n_chunks):
                offset = i * chunk_size
                count = min(chunk_size, n_tokens - offset)
                chunk = np.frombuffer(f.read(count * 4), dtype=">i4")
                cmax = int(chunk.max())
                if cmax > max_seen:
                    max_seen = cmax
                if cmax > 255:
                    raise ValueError(f"token id {cmax} exceeds uint8 range")
                tokens[offset:offset + count] = chunk.astype(np.uint8)

                frac = (offset + count) / n_tokens
                filled = int(bar_width * frac)
                bar = "█" * filled + "░" * (bar_width - filled)
                sys.stderr.write(
                    f"\r  Reading: [{bar}] {100 * frac:5.1f}%  "
                    f"({offset + count:,}/{n_tokens:,} tokens)"
                )
                sys.stderr.flush()
        sys.stderr.write("\n")

    return tokens


def build_sa(tokens):
    """Build a suffix array over `tokens` using libdivsufsort (via pydivsufsort).

    Uses int32 SA for inputs ≤ 2^31 tokens, and int64 (`force64=True`)
    for larger inputs. The SA has the same length as `tokens` (no
    sentinel — libdivsufsort handles the implicit end-of-text).
    """
    force64 = len(tokens) >= 2**31
    print(f"  Building SA over {len(tokens):,} tokens "
          f"({'int64' if force64 else 'int32'} SA)...")
    sa = pydivsufsort.divsufsort(tokens, force64=force64)
    return sa


def save_index(output_dir, tokens, sa):
    """Write tokens + SA as raw little-endian binary blobs."""
    tokens_path = os.path.join(output_dir, "tokens.bin")
    sa_path = os.path.join(output_dir, "suffix_array.bin")
    meta_path = os.path.join(output_dir, "meta.txt")

    tokens.tofile(tokens_path)
    sa.tofile(sa_path)

    with open(meta_path, "w") as f:
        f.write(f"n_tokens {len(tokens)}\n")
        f.write(f"tokens_dtype {tokens.dtype.str}\n")
        f.write(f"sa_dtype {sa.dtype.str}\n")

    return tokens_path, sa_path, meta_path


def count(tokens, sa, pattern):
    """Count occurrences of `pattern` (numpy array, same dtype as tokens)
    in `tokens` via SA binary search. Wraps pydivsufsort.sa_search,
    which returns (count, first_position)."""
    n, _ = pydivsufsort.sa_search(tokens, sa, pattern)
    return int(n)


def next_token_distribution(tokens, sa, context, vocab):
    """Return (ctx_count, [(token, ext_count, prob)]) for each candidate
    next token. Probabilities are conditioned on the context appearing
    in the corpus; if it doesn't (ctx_count == 0), returns 0.0 probs."""
    ctx = np.asarray(context, dtype=tokens.dtype)
    ctx_count = count(tokens, sa, ctx) if len(ctx) > 0 else len(tokens)

    extended = np.empty(len(ctx) + 1, dtype=tokens.dtype)
    extended[:len(ctx)] = ctx

    dist = []
    for tok in vocab:
        extended[-1] = tok
        ext_count = count(tokens, sa, extended)
        prob = ext_count / ctx_count if ctx_count > 0 else 0.0
        dist.append((int(tok), ext_count, prob))
    return ctx_count, dist


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to .bin dataset")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: analysis/sa_<dataset_stem>/)")
    parser.add_argument("--window-length", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Optionally cap number of 512-token windows read")
    args = parser.parse_args()

    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.dataset))[0]
        args.output_dir = os.path.join("analysis", f"sa_{stem}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1: Read dataset into uint8 numpy array ──
    print(f"Reading dataset: {args.dataset}")
    t0 = time.time()
    tokens = convert_dataset(args.dataset, args.window_length, args.max_windows)
    t_read = time.time() - t0
    print(f"  Read {len(tokens):,} tokens in {t_read:.1f}s "
          f"(unique IDs: {sorted(set(tokens.tolist())) if len(tokens) < 1_000_000 else sorted(np.unique(tokens).tolist())})\n")

    # ── Phase 2: Build SA ──
    t0 = time.time()
    sa = build_sa(tokens)
    t_build = time.time() - t0
    print(f"  SA built in {t_build:.1f}s (dtype={sa.dtype}, len={len(sa):,})\n")

    # ── Phase 3: Save to disk ──
    print("Saving index to disk...")
    t0 = time.time()
    tokens_path, sa_path, meta_path = save_index(args.output_dir, tokens, sa)
    t_save = time.time() - t0
    tok_mb = os.path.getsize(tokens_path) / 1e6
    sa_mb = os.path.getsize(sa_path) / 1e6
    print(f"  Tokens: {tok_mb:.1f} MB  SA: {sa_mb:.1f} MB  Total: {tok_mb + sa_mb:.1f} MB")
    print(f"  Save: {t_save:.1f}s\n")

    # ── Phase 4: Sanity check — same patterns as build_dawg.py ──
    # Token IDs for cfg3b (CFGCharacterTokenizer over terminal symbols
    # ['1','2','3'] sorted, plus eos='E'(3) and bos='B'(4)):
    #   0 = '1', 1 = '2', 2 = '3'
    print("Sanity check — count queries:")
    test_seqs = [
        ([0],         "'1'"),
        ([1],         "'2'"),
        ([2],         "'3'"),
        ([0, 1, 2],   "'123'"),
        ([2, 0, 1],   "'312'"),
    ]
    for seq, label in test_seqs:
        pat = np.asarray(seq, dtype=tokens.dtype)
        c = count(tokens, sa, pat)
        print(f"  {label:>8s} {seq}: count = {c:,}")

    # ── Phase 5: Next-token distribution at a few context lengths ──
    print("\nNext-token distributions (P(next | context) over terminals 0,1,2):")
    vocab = [0, 1, 2]
    test_contexts = [
        ([], "empty"),
        ([0], "'1'"),
        ([0, 1], "'12'"),
        ([0, 1, 2], "'123'"),
        ([0, 1, 2, 0, 1], "'12312'"),
        ([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], "'1231231231'"),
    ]
    for ctx, label in test_contexts:
        ctx_count, dist = next_token_distribution(tokens, sa, ctx, vocab)
        parts = " ".join(f"{tok}={prob:.4f}({cnt:,})" for tok, cnt, prob in dist)
        print(f"  ctx={label:<14s} count={ctx_count:>12,}  {parts}")

    total = t_read + t_build + t_save
    print(f"\nDone. Total: {total:.1f}s. Index at: {args.output_dir}/")


if __name__ == "__main__":
    main()
