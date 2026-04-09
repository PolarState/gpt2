"""Build a CDAWG from a pre-tokenized CFG binary dataset.

Reads the same .bin files used by main.py for training (big-endian int32
windows of GPT2 token IDs) and constructs a disk-backed CDAWG via the
rusty-dawg library.  The DiskCdawg variant memory-maps the graph to disk,
so RAM usage stays bounded regardless of corpus size.

The built CDAWG is persisted to an output directory and can be reloaded
later with DiskCdawg.load() for querying.

Usage:
    # Build from validation set (fast, ~5M tokens):
    python analysis/build_dawg.py --dataset ../CFG/datasets/cfg3b_val_dataset.bin

    # Build from full training set (~4.9B tokens):
    python analysis/build_dawg.py --dataset ../CFG/datasets/cfg3b_train_dataset.bin

    # Build from a subset:
    python analysis/build_dawg.py --dataset ../CFG/datasets/cfg3b_train_dataset.bin --max-windows 1000000
"""

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np

from rusty_dawg import Cdawg, DiskCdawg


def _monitor_sparse_progress(graph_dir, stop_event, label="Building"):
    """Monitor sparse file block growth in a child process to show build progress.

    The DiskCdawg memory-maps pre-allocated files (nodes.vec, edges.vec).
    As data is written, the kernel allocates actual disk blocks for dirty pages.
    We track allocated blocks vs. apparent size as a progress estimate.
    """
    nodes_path = os.path.join(graph_dir, "nodes.vec")
    edges_path = os.path.join(graph_dir, "edges.vec")

    # Wait for files to appear.
    while not (os.path.exists(nodes_path) and os.path.exists(edges_path)):
        if stop_event.is_set():
            return
        time.sleep(0.2)

    # Apparent size (pre-allocated capacity) in 512-byte blocks.
    nodes_cap = os.stat(nodes_path).st_size // 512
    edges_cap = os.stat(edges_path).st_size // 512
    total_cap = nodes_cap + edges_cap
    if total_cap == 0:
        return

    t0 = time.time()
    prev_blocks = 0
    bar_width = 40

    while not stop_event.is_set():
        try:
            nodes_blocks = os.stat(nodes_path).st_blocks
            edges_blocks = os.stat(edges_path).st_blocks
        except OSError:
            break
        cur_blocks = nodes_blocks + edges_blocks
        frac = cur_blocks / total_cap
        pct = 100.0 * frac

        elapsed = time.time() - t0
        filled = int(bar_width * frac)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        # Estimate remaining time from average rate.
        if cur_blocks > 0 and elapsed > 0:
            rate = cur_blocks / elapsed  # blocks/sec
            remaining_blocks = total_cap - cur_blocks
            eta = remaining_blocks / rate if rate > 0 else 0
            eta_str = _fmt_time(eta)
        else:
            eta_str = "??:??"

        elapsed_str = _fmt_time(elapsed)
        sys.stderr.write(
            f"\r  {label}: [{bar}] {pct:5.1f}%  "
            f"elapsed {elapsed_str}  eta {eta_str}   "
        )
        sys.stderr.flush()

        prev_blocks = cur_blocks
        stop_event.wait(5)

    # Final line.
    sys.stderr.write("\n")
    sys.stderr.flush()


def _fmt_time(seconds):
    """Format seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def convert_dataset_to_diskvec(dataset_path, tokens_output_path, window_length=512, max_windows=None):
    """Convert a .bin dataset to a DiskVec<u16> file for rusty-dawg.

    Reads big-endian int32 token IDs and writes little-endian uint16
    values (bincode fixint format).  Appends a CDAWG EOS sentinel
    (65535) at the end of the stream.

    Returns the total number of tokens written (including the EOS).
    """
    # Calculate how many tokens to read.
    file_size = os.path.getsize(dataset_path)
    total_ints = file_size // 4
    total_windows = total_ints // window_length

    if max_windows is not None:
        n_windows = min(max_windows, total_windows)
    else:
        n_windows = total_windows

    n_tokens = n_windows * window_length
    print(f"  File has {total_windows:,} windows; reading {n_windows:,} ({n_tokens:,} tokens)")

    # Read big-endian int32 in chunks and convert to native-endian uint16,
    # showing progress for large datasets.
    chunk_size = 50_000_000  # 50M tokens per chunk
    if n_tokens <= chunk_size:
        raw = np.fromfile(dataset_path, dtype=">i4", count=n_tokens)
        tokens = raw.astype(np.dtype("<u2"))
        del raw
    else:
        tokens = np.empty(n_tokens, dtype="<u2")
        n_chunks = (n_tokens + chunk_size - 1) // chunk_size
        bar_width = 40
        with open(dataset_path, "rb") as f:
            for i in range(n_chunks):
                offset = i * chunk_size
                count = min(chunk_size, n_tokens - offset)
                chunk = np.frombuffer(f.read(count * 4), dtype=">i4")
                tokens[offset:offset + count] = chunk.astype(np.dtype("<u2"))

                frac = (offset + count) / n_tokens
                filled = int(bar_width * frac)
                bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
                sys.stderr.write(
                    f"\r  Reading: [{bar}] {100*frac:5.1f}%  "
                    f"{offset + count:,}/{n_tokens:,} tokens   "
                )
                sys.stderr.flush()
        sys.stderr.write("\n")

    # Append the CDAWG end-of-stream sentinel.
    tokens = np.append(tokens, np.array([DiskCdawg.EOS], dtype="<u2"))

    # Write as flat little-endian uint16 — this is the DiskVec<u16> format.
    tokens.tofile(tokens_output_path)

    n_total = len(tokens)
    unique = sorted(set(tokens.tolist()))
    del tokens

    print(f"  Wrote {n_total:,} tokens to {tokens_output_path}")
    print(f"  Unique token IDs ({len(unique)}): {unique}")
    return n_total


def main():
    parser = argparse.ArgumentParser(
        description="Build a disk-backed CDAWG from a pre-tokenized CFG dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the .bin dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CDAWG output files.  Defaults to analysis/cdawg_<dataset_stem>/.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit the number of 512-token windows to read (default: all).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=512,
        help="Tokens per window in the binary file.",
    )
    args = parser.parse_args()

    # Derive output directory from dataset name if not specified.
    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.dataset))[0]
        args.output_dir = os.path.join("analysis", f"cdawg_{stem}")

    # Create output directory structure.
    os.makedirs(args.output_dir, exist_ok=True)
    tokens_path = os.path.join(args.output_dir, "tokens.diskvec")
    graph_dir = os.path.join(args.output_dir, "graph")

    # Step 1: Convert the binary dataset to a DiskVec<u16> file.
    print(f"Reading dataset: {args.dataset}")
    t0 = time.time()

    # Clean up stale files from a previous run so DiskVec::new doesn't fail.
    if os.path.exists(tokens_path):
        os.remove(tokens_path)
    if os.path.exists(graph_dir):
        import shutil
        shutil.rmtree(graph_dir)

    n_tokens = convert_dataset_to_diskvec(
        args.dataset,
        tokens_path,
        window_length=args.window_length,
        max_windows=args.max_windows,
    )
    t_read = time.time() - t0
    print(f"Conversion complete in {t_read:.1f}s\n")

    # Estimate node/edge capacity from our scaling measurements:
    # ~0.38 nodes per token, ~0.82 edges per token (from 512M-token run).
    est_nodes = int(n_tokens * 0.40)
    est_edges = int(n_tokens * 0.85)
    print(f"Estimated capacity: {est_nodes:,} nodes, {est_edges:,} edges")

    # Step 2: Build the disk-backed CDAWG.
    print("Building DiskCdawg...")
    t0 = time.time()
    cdawg = DiskCdawg(tokens_path, graph_dir, est_nodes, est_edges)

    # build() holds the GIL, so we use a child process to monitor file growth.
    stop_evt = multiprocessing.Event()
    monitor = multiprocessing.Process(
        target=_monitor_sparse_progress,
        args=(graph_dir, stop_evt, "Build"),
        daemon=True,
    )
    monitor.start()
    try:
        cdawg.build()
    finally:
        stop_evt.set()
        monitor.join(timeout=5)

    t_build = time.time() - t0
    print(f"Build complete in {t_build:.1f}s")
    print(f"  Nodes: {cdawg.node_count():,}")
    print(f"  Edges: {cdawg.edge_count():,}")

    # Step 3: Fill n-gram counts via topological traversal.
    print("\nFilling counts...")
    t0 = time.time()
    # Use RAM-based counter — the counter itself is small relative to the graph.
    # fill_counts_ram() also holds the GIL; reuse the file monitor for its phase.
    stop_evt2 = multiprocessing.Event()
    monitor2 = multiprocessing.Process(
        target=_monitor_sparse_progress,
        args=(graph_dir, stop_evt2, "Counts"),
        daemon=True,
    )
    monitor2.start()
    try:
        cdawg.fill_counts_ram()
    finally:
        stop_evt2.set()
        monitor2.join(timeout=5)

    t_counts = time.time() - t0
    print(f"Counts filled in {t_counts:.1f}s")

    # Step 4: Quick sanity check — query a few short token sequences.
    print("\nSanity check — querying short sequences:")
    test_sequences = [
        ([16],           "'1'"),
        ([17],           "'2'"),
        ([18],           "'3'"),
        ([16, 17, 18],   "'123'"),
        ([18, 16, 17],   "'312'"),
    ]
    for seq, label in test_sequences:
        # Walk the CDAWG to compute suffix context and count.
        state = cdawg.get_initial()
        for token in seq:
            state = cdawg.transition_and_count(state, token)
        count = cdawg.get_suffix_count(state)
        print(f"  {label:>8s} {seq}: count = {count:,}")

    total_time = t_read + t_build + t_counts
    print(f"\nDone. Total time: {total_time:.1f}s")
    print(f"CDAWG saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
