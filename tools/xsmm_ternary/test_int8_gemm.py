"""Correctness check for the int8 x int8 -> int32 GEMM path.

Compares torch.ops._C.xsmm_int8_gemm against an emulation of the same
pipeline (per-(row, group) int8 weights, per-(row, group) int8 activations,
int32 accumulation, dequantized with act_scale * weight_scale).
"""

import argparse

import torch

import vllm._custom_ops  # noqa: F401  (registers the ops / loads the right _C)

CASES = [
    # M,   N,     K,     G,   dtype
    (1, 256, 512, 128, torch.float32),
    (1, 256, 512, 128, torch.float16),
    (2, 256, 512, 128, torch.bfloat16),
    (17, 256, 512, 128, torch.bfloat16),
    (32, 256, 128, 128, torch.float32),
    (33, 4096, 4096, 128, torch.bfloat16),
    (1, 4096, 4096, 64, torch.float32),
    (100, 256, 512, 64, torch.float16),
    (8, 1024, 4096, 256, torch.float32),
    (1, 151936, 4096, 128, torch.float16),  # lm_head shape
]


def reference(x: torch.Tensor, w: torch.Tensor, g: int, half_scales: bool):
    """Mirror the kernel exactly.

    Two details matter or the comparison is misleading:
      * the kernel accumulates in int32 and scales once per group, so
        dequantizing both operands and doing a float matmul is *less* accurate
        and disagrees at ~2e-3;
      * the kernel multiplies by a precomputed reciprocal (w * (1/s)), which
        rounds differently from w / s on exact ties -- about 1 weight in 2e6.
    """
    n, k = w.shape
    m = x.shape[0]

    wq = w.float().view(n, k // g, g)
    ws = wq.abs().amax(-1).clamp_min(1e-12) / 127.0
    wcodes = torch.round(wq * (1.0 / ws).unsqueeze(-1)).clamp(-127, 127)
    if half_scales:
        ws = ws.half().float()

    xg = x.float().view(m, k // g, g)
    xs = xg.abs().amax(-1) / 127.0
    xinv = torch.where(xs > 0, 1.0 / xs, torch.zeros_like(xs))
    xcodes = torch.round(xg * xinv.unsqueeze(-1)).clamp(-127, 127)

    out = torch.zeros(m, n, dtype=torch.float32)
    for gi in range(k // g):
        acc = xcodes[:, gi, :] @ wcodes[:, gi, :].T
        out += acc * xs[:, gi].unsqueeze(1) * ws[:, gi].unsqueeze(0)
    return out


# output dtype dominates the residual error
TOL = {
    torch.float32: 1e-6,
    torch.float16: 1.5e-3,
    torch.bfloat16: 8e-3,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fp32-scales", action="store_true")
    args = p.parse_args()
    half_scales = not args.fp32_scales

    torch.manual_seed(0)
    print("target isa:", torch.ops._C.xsmm_target_arch())
    fails = 0

    for m, n, k, g, dt in CASES:
        w = torch.randn(n, k, dtype=torch.float32)
        x = torch.randn(m, k, dtype=torch.float32).to(dt)

        packed, scales = torch.ops._C.xsmm_int8_pack_weight(w, g, half_scales)
        got = torch.ops._C.xsmm_int8_gemm(x, packed, scales, None, g)
        ref = reference(x, w, g, half_scales)

        denom = max(ref.abs().max().item(), 1e-9)
        err = (got.float() - ref).abs().max().item() / denom
        tol = TOL[dt]
        ok = err < tol
        fails += not ok
        name = str(dt).split(".")[-1]
        status = "PASS" if ok else "FAIL"
        print(
            f"  M={m:<5} N={n:<7} K={k:<5} G={g:<4} {name:<9} "
            f"rel_err={err:.3e}  {status}"
        )

    # bias
    w = torch.randn(256, 512, dtype=torch.float32)
    x = torch.randn(4, 512, dtype=torch.float32)
    b = torch.randn(256, dtype=torch.float32)
    packed, scales = torch.ops._C.xsmm_int8_pack_weight(w, 128, half_scales)
    got = torch.ops._C.xsmm_int8_gemm(x, packed, scales, b, 128)
    ref = reference(x, w, 128, half_scales) + b
    err = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    ok = err < 1e-5
    fails += not ok
    print(f"  bias                                        rel_err={err:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")

    print("ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
