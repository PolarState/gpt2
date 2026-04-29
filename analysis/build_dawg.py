"""Build a CDAWG from a pre-tokenized CFG binary dataset.

Reads the same .bin files used by main.py for training (big-endian int32
windows of GPT2 token IDs) and constructs a disk-backed CDAWG via the
rusty-dawg library.  The DiskCdawg variant memory-maps the graph to disk,
so RAM usage stays bounded regardless of corpus size.

The built CDAWG is persisted to an output directory and can be reloaded
later with DiskCdawg.load() for querying.

Background
----------
A Compacted Directed Acyclic Word Graph (CDAWG) is a space-efficient
index over every substring of a corpus.  Given any query string, the
CDAWG can report in O(|query|) time:
  - whether the query appears in the corpus,
  - how many times it appears (after fill_counts), and
  - the longest suffix of any prefix that matches the corpus
    (the "non-novel suffix length" from the Rusty-DAWG paper).

We use the *disk-backed* variant (DiskCdawg) because the in-memory
Cdawg requires storing the full graph in RAM.  For our 4.9 billion-
token training set the graph alone would need ~140+ GB, exceeding
available memory.  DiskCdawg memory-maps the node/edge arrays to
files on disk; the OS page cache decides what stays resident.

Data flow
---------
1. Read the .bin file (big-endian int32) → convert to little-endian
   uint16 → write as a flat "DiskVec<u16>" file (rusty-dawg's on-disk
   vector format, which is just raw bincode-serialized u16 values).
2. Pass the tokens file path + a graph output directory to DiskCdawg,
   which memory-maps pre-allocated node/edge files and builds the
   CDAWG in Rust.
3. Run fill_counts_ram() to propagate substring frequencies through
   the graph via a topological traversal.
4. The resulting directory can be reloaded with DiskCdawg.load().

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

    DiskCdawg pre-allocates large files (nodes.vec, edges.vec) via
    mmap.  These start as *sparse* files — the apparent size reflects
    the full capacity, but actual disk blocks are only allocated as
    the Rust build loop writes data.  By comparing st_blocks (real
    disk blocks) to st_size (apparent size), we get a rough progress
    fraction.

    This runs in a separate process because cdawg.build() holds the
    GIL for the entire duration — no Python code can execute in the
    main process until it returns.
    """
    nodes_path = os.path.join(graph_dir, "nodes.vec")
    edges_path = os.path.join(graph_dir, "edges.vec")

    # Wait for the Rust side to create the memory-mapped files.
    while not (os.path.exists(nodes_path) and os.path.exists(edges_path)):
        if stop_event.is_set():
            return
        time.sleep(0.2)

    # Get apparent file sizes in 512-byte blocks.  These represent the
    # full pre-allocated capacity (est_nodes * node_size + est_edges *
    # edge_size).  Actual block count starts near zero and grows as the
    # build writes into the mmap region.
    nodes_cap = os.stat(nodes_path).st_size // 512
    edges_cap = os.stat(edges_path).st_size // 512
    total_cap = nodes_cap + edges_cap
    if total_cap == 0:
        return

    t0 = time.time()
    prev_blocks = 0
    bar_width = 40

    while not stop_event.is_set():
        # Read the number of actually-allocated 512-byte blocks for
        # each file.  This is a Linux-specific stat field (st_blocks).
        try:
            nodes_blocks = os.stat(nodes_path).st_blocks
            edges_blocks = os.stat(edges_path).st_blocks
        except OSError:
            break
        cur_blocks = nodes_blocks + edges_blocks

        # Compute progress as fraction of capacity filled.
        frac = cur_blocks / total_cap
        pct = 100.0 * frac

        elapsed = time.time() - t0
        filled = int(bar_width * frac)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        # Estimate remaining time from average write rate.
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
        # Poll every 5 seconds — frequent enough to be useful,
        # infrequent enough to avoid stat() overhead.
        stop_event.wait(5)

    # Clear the progress line.
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

    The CFG .bin files store token IDs as big-endian int32 values in
    contiguous 512-token windows (see cfg.cfg_datasets.CFGFileDataset).
    Rusty-dawg's DiskVec<u16> expects flat little-endian uint16 values
    (bincode with fixint encoding).  All token IDs in our data are
    <= 91, so the int32 → uint16 downcast is lossless.

    A CDAWG EOS sentinel (u16::MAX = 65535) is appended at the very
    end.  This marks the corpus boundary so the CDAWG construction
    algorithm knows where the input ends.  It is distinct from any
    real token in the vocabulary.

    Returns the total number of tokens written (including the EOS).
    """
    # Determine how many tokens to read from the file.
    file_size = os.path.getsize(dataset_path)
    total_ints = file_size // 4              # Each token is 4 bytes (int32).
    total_windows = total_ints // window_length  # Each window is 512 tokens.

    # Optionally cap the number of windows for faster iteration or
    # memory-constrained environments.
    if max_windows is not None:
        n_windows = min(max_windows, total_windows)
    else:
        n_windows = total_windows

    n_tokens = n_windows * window_length
    print(f"  File has {total_windows:,} windows; reading {n_windows:,} ({n_tokens:,} tokens)")

    # Read big-endian int32 values and convert to little-endian uint16.
    # For small datasets, do it in one shot.  For large datasets, read
    # in 50M-token chunks to keep peak memory bounded — only one chunk
    # plus the output array need to be in memory at once.
    chunk_size = 50_000_000  # 50M tokens (~200MB as int32) per chunk.
    if n_tokens <= chunk_size:
        # Small dataset: single bulk read + type conversion.
        raw = np.fromfile(dataset_path, dtype=">i4", count=n_tokens)
        tokens = raw.astype(np.dtype("<u2"))
        del raw
    else:
        # Large dataset: pre-allocate the output array, then fill it
        # chunk-by-chunk to avoid doubling memory with a temp copy.
        tokens = np.empty(n_tokens, dtype="<u2")
        n_chunks = (n_tokens + chunk_size - 1) // chunk_size
        bar_width = 40
        with open(dataset_path, "rb") as f:
            for i in range(n_chunks):
                offset = i * chunk_size
                count = min(chunk_size, n_tokens - offset)

                # Read raw bytes and interpret as big-endian int32.
                chunk = np.frombuffer(f.read(count * 4), dtype=">i4")
                # Downcast to little-endian uint16 directly into the
                # output array — no intermediate full-size copy.
                tokens[offset:offset + count] = chunk.astype(np.dtype("<u2"))

                # Show a progress bar on stderr so it doesn't mix with
                # the structured stdout output.
                frac = (offset + count) / n_tokens
                filled = int(bar_width * frac)
                bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
                sys.stderr.write(
                    f"\r  Reading: [{bar}] {100*frac:5.1f}%  "
                    f"{offset + count:,}/{n_tokens:,} tokens   "
                )
                sys.stderr.flush()
        sys.stderr.write("\n")

    # Append the CDAWG end-of-stream sentinel (u16::MAX = 65535).
    # This tells the CDAWG builder where the token stream ends.
    tokens = np.append(tokens, np.array([DiskCdawg.EOS], dtype="<u2"))

    # Write the numpy array directly to disk as raw bytes.  Because
    # the dtype is "<u2" (little-endian uint16), the file is byte-for-
    # byte what DiskVec<u16>::load() expects: len = file_size / 2.
    tokens.tofile(tokens_output_path)

    n_total = len(tokens)
    unique = sorted(set(tokens.tolist()))
    del tokens  # Free the numpy buffer.

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
    # E.g. cfg3b_train_dataset.bin → analysis/cdawg_cfg3b_train_dataset/
    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.dataset))[0]
        args.output_dir = os.path.join("analysis", f"cdawg_{stem}")

    # The output directory will contain:
    #   tokens.diskvec  — flat u16 file (DiskVec format) of the token stream
    #   graph/          — directory with nodes.vec and edges.vec (memory-mapped)
    os.makedirs(args.output_dir, exist_ok=True)
    tokens_path = os.path.join(args.output_dir, "tokens.diskvec")
    graph_dir = os.path.join(args.output_dir, "graph")

    # ── Step 1: Convert the binary dataset to a DiskVec<u16> file ──
    # This is the only step that touches the original .bin file.  After
    # this, everything works off the tokens.diskvec and graph/ directory.
    print(f"Reading dataset: {args.dataset}")
    t0 = time.time()

    # DiskVec::new() in Rust will fail if the target file already exists,
    # so remove stale artifacts from any previous run first.
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

    # ── Step 2: Build the disk-backed CDAWG ──
    # DiskCdawg needs up-front capacity hints for nodes and edges so it
    # can pre-allocate the memory-mapped files.  We measured the scaling
    # ratio empirically on subsets of this dataset:
    #   4.9M tokens  → 1.92M nodes (0.39/tok), 4.14M edges (0.84/tok)
    #   51M tokens   → 19.7M nodes (0.38/tok), 42.2M edges (0.82/tok)
    #   512M tokens  → 195M nodes  (0.38/tok), 418M edges  (0.82/tok)
    # We round up slightly (0.40, 0.85) for safety margin.
    est_nodes = int(n_tokens * 0.40)
    est_edges = int(n_tokens * 0.85)
    print(f"Estimated capacity: {est_nodes:,} nodes, {est_edges:,} edges")

    print("Building DiskCdawg...")
    t0 = time.time()

    # Construct the DiskCdawg object.  This opens the tokens file as a
    # read-only DiskVec<u16>, creates the graph/ directory, and mmap's
    # two sparse files (nodes.vec, edges.vec) pre-sized to the capacity.
    cdawg = DiskCdawg(tokens_path, graph_dir, est_nodes, est_edges)

    # cdawg.build() runs the online CDAWG construction algorithm entirely
    # in Rust.  It holds the Python GIL for the whole duration, so no
    # Python threads can run.  To show progress we spawn a *child process*
    # that monitors disk block allocation on the sparse graph files.
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
        # Signal the monitor to stop and wait for it to exit cleanly.
        stop_evt.set()
        monitor.join(timeout=5)

    t_build = time.time() - t0
    print(f"Build complete in {t_build:.1f}s")
    print(f"  Nodes: {cdawg.node_count():,}")
    print(f"  Edges: {cdawg.edge_count():,}")

    # ── Step 3: Fill n-gram counts ──
    # After build(), each CDAWG node represents a set of equivalent
    # substrings (an "equivalence class") but doesn't yet know how many
    # times those substrings appear in the corpus.  fill_counts_ram()
    # does a topological traversal from leaves to root, propagating
    # occurrence counts upward.  We use the RAM-based counter variant
    # because the counter's own working memory is small (proportional
    # to node count, not corpus size) even though the graph is on disk.
    print("\nFilling counts...")
    t0 = time.time()

    # Same GIL-holding situation, so we use a child-process monitor.
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

    # ── Step 4: Sanity check ──
    # Query a few short token sequences to verify the CDAWG was built
    # correctly.  We walk the CDAWG from the initial state, feeding one
    # token at a time via transition_and_count(), then read off the
    # suffix count at the final state.
    #
    # Token IDs for cfg3b (GPT2 encoding of single characters):
    #   16 = '1', 17 = '2', 18 = '3'
    print("\nSanity check — querying short sequences:")
    test_sequences = [
        ([16],           "'1'"),
        ([17],           "'2'"),
        ([18],           "'3'"),
        ([16, 17, 18],   "'123'"),
        ([18, 16, 17],   "'312'"),
    ]
    for seq, label in test_sequences:
        # Start at the CDAWG initial state (root / empty string).
        state = cdawg.get_initial()
        # Feed each token — the CDAWG follows edges and tracks the
        # longest matching suffix context internally.
        for token in seq:
            state = cdawg.transition_and_count(state, token)
        # Read the count: how many times this exact sequence appears
        # in the training corpus.
        count = cdawg.get_suffix_count(state)
        print(f"  {label:>8s} {seq}: count = {count:,}")

    total_time = t_read + t_build + t_counts
    print(f"\nDone. Total time: {total_time:.1f}s")
    print(f"CDAWG saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
