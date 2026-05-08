"""Dump CFG derivation traces (parse trees) as JSONL.

Mirrors build_char_dataset.py but emits the *production-rule choices* made
during generation instead of the resulting terminal stream. Each line of the
output file is one Schema B parse tree:

  Internal node:  {"nt": "<NT>", "rule": <int>, "children": [...]}
  Terminal leaf:  {"t": "<terminal>"}

The trace is produced by ``CFGrammar.generate_traced``, which consumes RNG
identically to ``CFGrammar.generate`` — so passing the same ``--seed`` here
that you'd use for ``build_char_dataset.py`` produces the *same* derivation,
just with the tree exposed instead of flattened to a terminal string.

Usage:

    # 100 trees from cfg3b, seed 0, no masking:
    python analysis/dump_traces.py --num-samples 100 --output /tmp/cfg3b_seed0.jsonl

    # Same but with one rule masked (matches build_char_dataset.py --mask-rule):
    python analysis/dump_traces.py --num-samples 100 --seed 0 --mask-rule 7:0 \\
        --output /tmp/cfg3b_seed0_mask7a.jsonl

The output streams one tree per line, so any line-oriented tool (jq, pandas
read_json with lines=True, or `for line in open(...)` in Python) works on
files of arbitrary size.
"""

import argparse
import json
import os
import random
import sys
import time

from cfg.cfg_grammar import CFGrammar


def parse_mask_rule(spec: str, rules: dict) -> tuple[str, int]:
    """Parse '<NT>:<INDEX>' (e.g. '7:0') and validate against the grammar.

    Same semantics as build_char_dataset.parse_mask_rule, kept duplicated
    here so this script has no dependency on that module.
    """
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
    """Weights dict suppressing one rule (weight 0 for masked, 1 for kept)."""
    return {
        nt: [
            0.0 if (nt == mask_nt and i == mask_idx) else 1.0
            for i in range(len(prods))
        ]
        for nt, prods in rules.items()
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cfg", default="cfg3b", help="grammar name from cfg_defines")
    p.add_argument("--output", required=True, help="output .jsonl path")
    p.add_argument("--num-samples", type=int, required=True,
                   help="number of derivation trees to write")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed; matches build_char_dataset.py")
    p.add_argument(
        "--mask-rule",
        default=None,
        help="Suppress one production rule (sets weight to 0). Format 'NT:INDEX' (e.g. '7:0').",
    )
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite --output if it exists")
    p.add_argument("--progress-seconds", type=float, default=10.0,
                   help="print progress every N seconds")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        sys.exit(f"refusing to overwrite {args.output} (pass --overwrite)")

    random.seed(args.seed)

    grammar = CFGrammar.from_name(args.cfg)
    print(
        f"grammar={args.cfg}  start_symbols={grammar.start_symbols}  "
        f"NTs={len(grammar.nonterminal_symbols)}  terminals={len(grammar.terminal_symbols)}"
    )

    weights = None
    if args.mask_rule is not None:
        nt, idx = parse_mask_rule(args.mask_rule, grammar.rules)
        print(f"masking NT {nt!r} rule index {idx}: {grammar.rules[nt][idx]}")
        weights = build_mask_weights(grammar.rules, nt, idx)

    t0 = time.time()
    last_progress_t = t0
    written = 0

    # json.dumps with no whitespace keeps each JSONL line tight.
    dumps = lambda obj: json.dumps(obj, separators=(",", ":"))

    with open(args.output, "w") as f:
        for _ in range(args.num_samples):
            tree = grammar.generate_traced(weights=weights)
            f.write(dumps(tree))
            f.write("\n")
            written += 1

            now = time.time()
            if (now - last_progress_t >= args.progress_seconds
                    or written == args.num_samples):
                elapsed = now - t0
                rate = written / elapsed if elapsed > 0 else 0
                eta = (args.num_samples - written) / rate if rate > 0 else 0
                print(
                    f"  {written:>10,}/{args.num_samples:,} samples  "
                    f"{rate:.0f} sample/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                    flush=True,
                )
                last_progress_t = now

    elapsed = time.time() - t0
    file_size = os.path.getsize(args.output)
    print(
        f"done: {written:,} samples, "
        f"{file_size:,} bytes ({file_size / 1e6:.1f} MB), "
        f"{elapsed:.1f}s ({written / elapsed:.0f} sample/s)"
    )


if __name__ == "__main__":
    main()
