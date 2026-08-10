"""Measure what output-tensor allocation actually costs on the decode path.

Answers whether an out-variant op with caller-preallocated buffers is worth the
aliasing hazard, before adding that complexity.
"""

import time

import torch

# (name, N, calls per decode token) for Bonsai-8B, M=1
SHAPES = [
    ("qkv", 6144, 36),
    ("o", 4096, 36),
    ("gate_up", 24576, 36),
    ("down", 4096, 36),
    ("lm_head", 151680, 1),
]

STEP_US = 30420.0  # measured decode step


def main() -> None:
    torch.set_num_threads(24)
    print(f"{'name':10s} {'N':>8s} {'KiB':>8s} {'us/alloc':>10s} {'n/tok':>7s} {'us/tok':>8s}")
    total = 0.0
    for name, n, count in SHAPES:
        iters = 20000
        # touch the first element, as the real consumer would
        t0 = time.perf_counter()
        for _ in range(iters):
            t = torch.empty((1, n), dtype=torch.bfloat16)
            t[0, 0] = 0
        dt = (time.perf_counter() - t0) / iters * 1e6
        per_tok = dt * count
        total += per_tok
        print(f"{name:10s} {n:8d} {n * 2 / 1024:8.1f} {dt:10.3f} {count:7d} {per_tok:8.1f}")

    print(f"\ntotal allocation cost per token : {total:8.1f} us")
    print(f"decode step                     : {STEP_US:8.1f} us")
    print(f"=> allocation is {100 * total / STEP_US:.2f}% of the decode step")

    # how much of that is the >128 KiB mmap threshold?
    print("\n--- effect of glibc mmap threshold (lm_head output is 296 KiB) ---")
    for n in (4096, 65536, 151680):
        iters = 20000
        t0 = time.perf_counter()
        for _ in range(iters):
            t = torch.empty((1, n), dtype=torch.bfloat16)
            t[0, 0] = 0
        dt = (time.perf_counter() - t0) / iters * 1e6
        print(f"  N={n:7d} ({n * 2 / 1024:7.1f} KiB): {dt:8.3f} us")


if __name__ == "__main__":
    main()
