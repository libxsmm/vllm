"""Correctness + throughput check for the LIBXSMM ternary GEMM op.

Compares `xsmm_ternary_gemm` against a reference computed from the *same*
ternary quantization, so any mismatch is a kernel/layout bug rather than
quantization error.
"""

import argparse
import time

import torch

BLOCK_N = 32
VNNI = 4
LANES = 8
PACK = 4


def reference_quantize(weight: torch.Tensor, group_size: int, half_scales: bool):
    """Ternary-quantize [N, C] weights the same way the C++ packer does."""
    n, c = weight.shape
    w = weight.float().view(n, c // group_size, group_size)
    scale = w.abs().amax(dim=-1).clamp_min(1e-8)
    codes = torch.round(w / scale.unsqueeze(-1)).clamp(-1, 1)
    if half_scales:
        # match the kernel's stored precision, so the test measures the kernel
        # rather than fp16 scale rounding
        scale = scale.half().float()
    return codes.view(n, c), scale  # scale: [N, C/G]


def reference_gemm(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, group_size: int):
    """Emulate int8 activation quantization + ternary GEMM in fp32."""
    m, c = x.shape
    n = codes.shape[0]
    xg = x.float().view(m, c // group_size, group_size)
    xs = xg.abs().amax(dim=-1) / 127.0
    xq = torch.round(xg / xs.clamp_min(1e-30).unsqueeze(-1)).clamp(-127, 127)
    xq = torch.where(xs.unsqueeze(-1) == 0, torch.zeros_like(xq), xq)

    cg = codes.view(n, c // group_size, group_size)
    out = torch.zeros(m, n, dtype=torch.float32)
    for g in range(c // group_size):
        acc = xq[:, g, :] @ cg[:, g, :].t()  # [M, N]
        out += acc * xs[:, g].unsqueeze(1) * scale[:, g].unsqueeze(0)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, default=17)
    p.add_argument("-n", type=int, default=256)
    p.add_argument("-k", type=int, default=512)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--ternary-weights", action="store_true",
                   help="build exactly-ternary weights (as a real ternary checkpoint has)")
    p.add_argument("--fp32-scales", action="store_true",
                   help="use fp32 weight scales instead of the default fp16")
    args = p.parse_args()

    import vllm._custom_ops  # noqa: F401  (loads the right _C variant)

    print("ternary target isa:", torch.ops._C.xsmm_target_arch())

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    m, n, k, g = args.m, args.n, args.k, args.group_size

    if args.ternary_weights:
        codes = torch.randint(-1, 2, (n, k), dtype=torch.float32)
        wscale = torch.rand(n, k // g, dtype=torch.float32) * 0.05 + 0.01
        weight = (codes.view(n, k // g, g) * wscale.unsqueeze(-1)).view(n, k).to(dtype)
    else:
        weight = (torch.randn(n, k) * 0.02).to(dtype)

    x = torch.randn(m, k, dtype=dtype)

    packed, scales = torch.ops._C.xsmm_ternary_pack_weight(
        weight, g, not args.fp32_scales
    )
    print("packed:", tuple(packed.shape), packed.dtype,
          "scales:", tuple(scales.shape), scales.dtype)
    expected_bytes = n * k // 4
    assert packed.numel() == expected_bytes, (packed.numel(), expected_bytes)

    out = torch.ops._C.xsmm_ternary_gemm(x, packed, scales, None, g)

    ref_codes, ref_scale = reference_quantize(weight, g, not args.fp32_scales)
    ref = reference_gemm(x, ref_codes, ref_scale, g)

    got = out.float()
    denom = ref.abs().max().clamp_min(1e-6)
    rel = (got - ref).abs().max() / denom
    print(f"max |ref| = {ref.abs().max():.4f}   max rel err = {rel:.3e}")

    tol = 3e-2 if dtype == torch.bfloat16 else 3e-3
    if rel > tol:
        bad = (got - ref).abs().argmax().item()
        print(f"MISMATCH at flat index {bad}: got={got.view(-1)[bad]:.5f} "
              f"ref={ref.view(-1)[bad]:.5f}")
        raise SystemExit(1)
    print("PASS")

    if args.bench:
        for _ in range(5):
            torch.ops._C.xsmm_ternary_gemm(x, packed, scales, None, g)
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.ops._C.xsmm_ternary_gemm(x, packed, scales, None, g)
        dt = (time.perf_counter() - t0) / iters
        gflops = 2.0 * m * n * k / dt / 1e9
        print(f"[bench] m={m} n={n} k={k}: {dt * 1e6:.1f} us  {gflops:.1f} GFLOP/s")

        wref = weight.clone()
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.nn.functional.linear(x, wref)
        dt_ref = (time.perf_counter() - t0) / iters
        print(f"[bench] torch {args.dtype} linear: {dt_ref * 1e6:.1f} us "
              f"({2.0 * m * n * k / dt_ref / 1e9:.1f} GFLOP/s)  "
              f"speedup x{dt_ref / dt:.2f}")


if __name__ == "__main__":
    main()
