"""Achieved bandwidth of the fused ternary MoE op.

Decode is weight-streaming bound, so the figure that matters is how fast the op
pulls the selected experts' packed weights out of DRAM. Compare against the
plain GEMV roofline from bench_gemv_bw.py (~97 GB/s on this box).

Experts are chosen at random per iteration so the working set is all E experts,
not just the ones that happen to stay in L3.
"""

import argparse
import time

import torch

import vllm._custom_ops  # noqa: F401

BLOCK_N = 32


def make_expert_weights(num_experts, n, k, g, half_scales):
    packed, scales = [], []
    for _ in range(num_experts):
        codes = torch.randint(-1, 2, (n, k), dtype=torch.float32)
        scale = torch.rand(n, k // g, dtype=torch.float32) * 0.05 + 0.01
        w = (codes.view(n, k // g, g) * scale.unsqueeze(-1)).view(n, k)
        p, s = torch.ops._C.xsmm_ternary_pack_weight(w, g, half_scales)
        packed.append(p)
        scales.append(s)
    return torch.stack(packed), torch.stack(scales)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=1536)
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--fp32-scales", action="store_true")
    args = p.parse_args()

    g = args.group_size
    dt = getattr(torch, args.dtype)
    half_scales = not args.fp32_scales
    E, K, H, I, T = args.experts, args.top_k, args.hidden, args.inter, args.tokens

    w13, w13s = make_expert_weights(E, 2 * I, H, g, half_scales)
    w2, w2s = make_expert_weights(E, H, I, g, half_scales)

    ws_bytes = 2 if half_scales else 4
    per_expert = (
        (2 * I * H) // 4 + (2 * I * H // g) * ws_bytes  # w13 packed + scales
        + (H * I) // 4 + (H * I // g) * ws_bytes  # w2
    )
    # each of the T*K (token, expert) pairs touches one expert's weights, but a
    # repeated expert within a step is only streamed once
    x = torch.randn(T, H, dtype=torch.float32).to(dt)

    ids_list, wts_list = [], []
    for _ in range(args.iters):
        ids = torch.stack(
            [torch.randperm(E)[:K] for _ in range(T)]
        ).to(torch.int32)
        ids_list.append(ids)
        wts_list.append(torch.softmax(torch.randn(T, K), dim=-1))

    args_w = (w13, w13s, w2, w2s, g)
    for i in range(3):
        torch.ops._C.xsmm_ternary_moe(x, ids_list[0], wts_list[0], *args_w)

    t0 = time.perf_counter()
    for i in range(args.iters):
        torch.ops._C.xsmm_ternary_moe(x, ids_list[i], wts_list[i], *args_w)
    dt_s = (time.perf_counter() - t0) / args.iters

    uniq = sum(len(set(ids_list[i].flatten().tolist())) for i in range(args.iters))
    uniq /= args.iters
    bytes_moved = uniq * per_expert
    flops = 2.0 * T * K * (2 * I * H + H * I)

    print(
        f"E={E} topk={K} H={H} I={I} T={T} {args.dtype}: "
        f"{dt_s * 1e6:8.1f} us/call  "
        f"{bytes_moved / dt_s / 1e9:6.1f} GB/s  "
        f"{flops / dt_s / 1e9:7.1f} GFLOP/s  "
        f"(working set {E * per_expert / 1024**2:.0f} MiB, "
        f"{uniq:.1f} experts/call)"
    )


if __name__ == "__main__":
    main()
