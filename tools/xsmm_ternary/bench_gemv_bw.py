"""DRAM-streaming benchmark for the ternary GEMV/GEMM kernel.

A single weight matrix (4 MB packed at 4096x4096) fits in this CPU's 36 MB L3, so
re-timing one matrix measures cache bandwidth, not memory bandwidth. This script
allocates a working set far larger than L3 and cycles through it, so every GEMV
reads weights cold from DRAM -- the regime that actually matters during decode.
"""

import argparse
import time

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, default=1)
    p.add_argument("-n", type=int, default=4096)
    p.add_argument("-k", type=int, default=4096)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--working-set-mb", type=int, default=768,
                   help="total packed-weight bytes to cycle through (>> L3)")
    p.add_argument("--iters", type=int, default=0, help="0 = one full sweep")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--fp32-scales", action="store_true",
                   help="fp32 weight scales instead of the default fp16")
    p.add_argument("--sweeps", type=int, default=3)
    args = p.parse_args()

    import vllm._custom_ops  # noqa: F401

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    m, n, k, g = args.m, args.n, args.k, args.group_size

    packed_bytes = n * k // 4
    scale_elem = 4 if args.fp32_scales else 2
    scale_bytes = n * (k // g) * scale_elem
    per_mat = packed_bytes + scale_bytes
    n_mat = max(1, (args.working_set_mb * 2**20) // per_mat)

    print(f"matrix: n={n} k={k}  packed={packed_bytes / 2**20:.2f} MiB "
          f"+ scales({'fp32' if args.fp32_scales else 'fp16'})="
          f"{scale_bytes / 2**20:.2f} MiB = {per_mat / 2**20:.2f} MiB")
    print(f"allocating {n_mat} matrices = {n_mat * per_mat / 2**30:.2f} GiB "
          f"(L3 is 36 MiB, so this streams from DRAM)")

    mats = []
    for _ in range(n_mat):
        w = torch.randint(-1, 2, (n, k), dtype=torch.float32).to(dtype)
        packed, scales = torch.ops._C.xsmm_ternary_pack_weight(
            w, g, not args.fp32_scales
        )
        mats.append((packed, scales))
        del w

    x = torch.randn(m, k, dtype=dtype)

    # warmup: primes the JIT cache, never timed
    torch.ops._C.xsmm_ternary_gemm(x, mats[0][0], mats[0][1], None, g)

    iters = args.iters or n_mat
    print(f"\nthreads={torch.get_num_threads()}  m={m}  iters/sweep={iters}")
    for s in range(args.sweeps):
        t0 = time.perf_counter()
        for i in range(iters):
            packed, scales = mats[i % n_mat]
            torch.ops._C.xsmm_ternary_gemm(x, packed, scales, None, g)
        dt = time.perf_counter() - t0

        bytes_read = iters * per_mat
        gbs = bytes_read / dt / 1e9
        gflops = 2.0 * m * n * k * iters / dt / 1e9
        per_call_us = dt / iters * 1e6
        print(f"  sweep {s}: {per_call_us:8.1f} us/call   "
              f"{gbs:6.1f} GB/s   {gflops:7.1f} GFLOP/s")


if __name__ == "__main__":
    main()
