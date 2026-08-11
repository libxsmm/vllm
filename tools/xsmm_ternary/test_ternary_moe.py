"""Correctness check for the ternary MoE expert loop.

The int2 GEMM itself is covered by test_ternary_gemm.py; what this exercises is
the MoE plumbing that sits on top of it -- expert gather/scatter, the gate/up
split inside w13, the SiLU-gated activation, and how router weights are applied.

The reference recomputes the same thing one token at a time using the same GEMM
op, so any disagreement is a plumbing bug (wrong expert index, wrong w13 half,
misapplied routing weight) rather than kernel numerics.
"""

import argparse
import types

import torch

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.xsmm_ternary import (
    XSMM_TERNARY_BLOCK_N,
    XsmmTernaryConfig,
    XsmmTernaryMoEMethod,
)


def make_ternary(n: int, k: int, g: int) -> torch.Tensor:
    codes = torch.randint(-1, 2, (n, k), dtype=torch.float32)
    scale = torch.rand(n, k // g, dtype=torch.float32) * 0.05 + 0.01
    return (codes.view(n, k // g, g) * scale.unsqueeze(-1)).view(n, k)


def build_layer(num_experts, hidden, inter, g, half_scales, dtype):
    """A stand-in for RoutedExperts holding packed weights."""
    w13_dense = [make_ternary(2 * inter, hidden, g) for _ in range(num_experts)]
    w2_dense = [make_ternary(hidden, inter, g) for _ in range(num_experts)]

    w13_packed, w13_scale, w2_packed, w2_scale = [], [], [], []
    for e in range(num_experts):
        p, s = torch.ops._C.xsmm_ternary_pack_weight(w13_dense[e], g, half_scales)
        w13_packed.append(p)
        w13_scale.append(s)
        p, s = torch.ops._C.xsmm_ternary_pack_weight(w2_dense[e], g, half_scales)
        w2_packed.append(p)
        w2_scale.append(s)

    layer = types.SimpleNamespace(
        w13_weight=torch.stack(w13_packed),
        w2_weight=torch.stack(w2_packed),
        xsmm_w13_scale=torch.stack(w13_scale),
        xsmm_w2_scale=torch.stack(w2_scale),
        activation=MoEActivation.SILU,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    return layer


def reference(layer, x, topk_weights, topk_ids, g):
    """One token at a time, using the same GEMM op."""
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            xe = x[t : t + 1]
            h = torch.ops._C.xsmm_ternary_gemm(
                xe, layer.w13_weight[e], layer.xsmm_w13_scale[e], None, g
            )
            half = h.shape[-1] // 2
            act = torch.nn.functional.silu(h[:, :half]) * h[:, half:]
            y = torch.ops._C.xsmm_ternary_gemm(
                act, layer.w2_weight[e], layer.xsmm_w2_scale[e], None, g
            )
            out[t] += (y * topk_weights[t, j].to(y.dtype))[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fp32-scales", action="store_true")
    args = p.parse_args()
    half_scales = not args.fp32_scales

    torch.manual_seed(0)
    cfg = XsmmTernaryConfig(group_size=128, half_scales=half_scales)
    fails = 0

    # (tokens, experts, top_k, hidden, inter, dtype)
    cases = [
        (1, 8, 2, 256, 128, torch.float32),
        (1, 128, 8, 2048, 768, torch.float32),  # cat-q MoE decode shape
        (7, 16, 4, 256, 128, torch.float32),
        (32, 16, 4, 512, 256, torch.float32),
        (5, 8, 8, 256, 128, torch.float16),  # every expert hit
        (13, 8, 1, 256, 128, torch.bfloat16),
    ]

    for tokens, num_experts, top_k, hidden, inter, dtype in cases:
        g = cfg.group_size
        layer = build_layer(num_experts, hidden, inter, g, half_scales, dtype)
        method = XsmmTernaryMoEMethod.__new__(XsmmTernaryMoEMethod)
        method.quant_config = cfg
        method.group_size = g

        x = (torch.randn(tokens, hidden, dtype=torch.float32) * 0.5).to(dtype)
        logits = torch.randn(tokens, num_experts, dtype=torch.float32)
        vals, ids = torch.topk(logits, k=top_k, dim=-1)
        weights = torch.softmax(vals, dim=-1).to(torch.float32)
        ids = ids.to(torch.int32)

        got = method._run_experts(layer, x, weights, ids)
        ref = reference(layer, x, weights, ids, g)

        denom = max(ref.abs().max().item(), 1e-9)
        err = (got.float() - ref.float()).abs().max().item() / denom
        tol = 1e-6 if dtype == torch.float32 else 8e-3
        ok = err < tol
        fails += not ok
        name = str(dtype).split(".")[-1]
        print(
            f"  T={tokens:<4} E={num_experts:<4} topk={top_k:<2} H={hidden:<5} "
            f"I={inter:<4} {name:<9} rel_err={err:.3e}  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # router weights applied to the input instead of the output
    g = cfg.group_size
    layer = build_layer(8, 256, 128, g, half_scales, torch.float32)
    layer.apply_router_weight_on_input = True
    method = XsmmTernaryMoEMethod.__new__(XsmmTernaryMoEMethod)
    method.quant_config, method.group_size = cfg, g
    x = torch.randn(4, 256, dtype=torch.float32) * 0.5
    ids = torch.randint(0, 8, (4, 1), dtype=torch.int32)
    weights = torch.rand(4, 1, dtype=torch.float32)
    got = method._run_experts(layer, x, weights, ids)
    ref = torch.zeros_like(x)
    for t in range(4):
        e = int(ids[t, 0])
        xe = x[t : t + 1] * weights[t, 0]
        h = torch.ops._C.xsmm_ternary_gemm(
            xe, layer.w13_weight[e], layer.xsmm_w13_scale[e], None, g
        )
        half = h.shape[-1] // 2
        act = torch.nn.functional.silu(h[:, :half]) * h[:, half:]
        ref[t] = torch.ops._C.xsmm_ternary_gemm(
            act, layer.w2_weight[e], layer.xsmm_w2_scale[e], None, g
        )[0]
    err = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    ok = err < 1e-6
    fails += not ok
    print(f"  apply_router_weight_on_input                  rel_err={err:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")

    print("ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
