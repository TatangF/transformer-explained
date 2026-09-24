"""Example 3: why a longer context is expensive (the O(N^2) bottleneck).

For each sequence length N we measure:
  * the size of the N x N score matrix (exact: N * N * 4 bytes in float32),
  * the wall-clock time of one attention call on your machine.

The memory column is exact. The time column is a real measurement: on a CPU
with small N it can be noisy or dominated by overhead, and it will approach a
4x jump per doubling only once the computation is the bottleneck.

Note: optimized kernels (e.g. FlashAttention) avoid materializing the full
N x N matrix in memory, but the amount of computation still grows as N^2.
This script uses the plain implementation on purpose.

Run:  python examples/03_complexity_benchmark.py [--plot]
"""

import argparse
import os
import time

import torch

from tinytransformer import causal_mask, scaled_dot_product_attention

parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=64, help="head dimension d_k")
parser.add_argument("--max-n", type=int, default=4096)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--plot", action="store_true", help="save a log-log plot to outputs/complexity.png")
args = parser.parse_args()

torch.manual_seed(0)
lengths = []
n = 128
while n <= args.max_n:
    lengths.append(n)
    n *= 2

rows = []
print(f"{'N':>6} | {'score matrix':>13} | {'x vs prev':>9} | {'time (ms)':>10} | {'x vs prev':>9}")
print("-" * 60)
prev_mem = prev_t = None
for n in lengths:
    q = k = v = torch.randn(1, n, args.d)
    mask = causal_mask(n)
    scaled_dot_product_attention(q, k, v, mask)  # warm-up
    best = float("inf")
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        scaled_dot_product_attention(q, k, v, mask)
        best = min(best, time.perf_counter() - t0)
    mem = n * n * 4  # bytes of the float32 score matrix
    mem_ratio = f"{mem / prev_mem:.1f}x" if prev_mem else "-"
    t_ratio = f"{best / prev_t:.1f}x" if prev_t else "-"
    print(f"{n:>6} | {mem / 2**20:>10.1f} MB | {mem_ratio:>9} | {best * 1000:>10.2f} | {t_ratio:>9}")
    rows.append((n, mem, best))
    prev_mem, prev_t = mem, best

print("\nDoubling N multiplies the score-matrix memory by exactly 4 (N^2).")

if args.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("outputs", exist_ok=True)
    ns = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(ns, [r[2] * 1000 for r in rows], "o-", label="measured time (ms)")
    ref = rows[0][2] * 1000
    ax.loglog(ns, [ref * (m / ns[0]) ** 2 for m in ns], "--", color="gray", label="N^2 reference")
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("time (ms)")
    ax.legend()
    ax.set_title("Attention cost vs sequence length")
    fig.tight_layout()
    fig.savefig("outputs/complexity.png", dpi=150)
    print("Saved outputs/complexity.png")
