"""Honest decode-throughput measurement for the ternary path.

Reports PER-SEQUENCE decode tok/s at a given batch size, separating prefill from
decode, with prefix caching disabled so nothing is reused between runs.
Also prints the achieved DRAM bandwidth implied by the weight footprint.
"""

import argparse
import time

import torch

from vllm import LLM, SamplingParams

PROMPT = (
    "Explain in detail why matrix multiplication dominates the cost of large "
    "language model inference on modern CPUs, and what can be done about it."
)


def weight_bytes_per_token(hidden, inter, layers, kv_heads, head_dim, bits, group):
    q = hidden * hidden
    kv = 2 * (kv_heads * head_dim) * hidden
    o = hidden * hidden
    mlp = 3 * inter * hidden
    per_layer = q + kv + o + mlp
    params = per_layer * layers
    return params * bits / 8 + (params / group) * 4, params


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--bits", type=float, default=2.0)
    p.add_argument("--group", type=int, default=128)
    args = p.parse_args()

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=512 * 1024 * 1024,
    )

    cfg = llm.llm_engine.vllm_config.model_config.hf_config
    wb, params = weight_bytes_per_token(
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        args.bits,
        args.group,
    )
    print(f"\n[model] linear params = {params / 1e9:.3f} G, "
          f"weight bytes streamed per decode step = {wb / 2**30:.3f} GiB "
          f"(at {args.bits} bits + fp32 scales / {args.group})")

    prompts = [f"{PROMPT} (variant {i})" for i in range(args.batch)]

    # warmup: primes the LIBXSMM JIT cache so it is never timed
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4))

    # Two-point slope: time(N) = fixed + N * step. Timing a separate max_tokens=1
    # run as "prefill" absorbs one-time warmup and over-subtracts, which
    # overstates tok/s; differencing two decode lengths cancels all fixed cost.
    def timed_run(max_tokens):
        torch.ops._C.xsmm_ternary_reset_stats()
        t0 = time.perf_counter()
        outs = llm.generate(
            prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens)
        )
        dt = time.perf_counter() - t0
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        return dt, n, torch.ops._C.xsmm_ternary_stats()

    n_short = max(4, args.max_tokens // 4)
    t_short, tok_short, st_short = timed_run(n_short)
    t_long, tok_long, st_long = timed_run(args.max_tokens)

    steps = (tok_long - tok_short) / args.batch
    t_step = (t_long - t_short) / steps
    per_seq = 1.0 / t_step
    agg = per_seq * args.batch
    bw = per_seq * wb / 1e9

    # ternary GEMM cost attributable to those extra steps
    gemm_sec = st_long[3] - st_short[3]
    gemm_bytes = st_long[2] - st_short[2]

    print(f"\n[decode] batch={args.batch}  slope over {n_short}->{args.max_tokens} tokens "
          f"({t_short:.2f}s -> {t_long:.2f}s)")
    print(f"[decode] step time      : {t_step * 1e3:.2f} ms")
    print(f"[decode] per sequence   : {per_seq:.2f} tok/s   <-- single-stream speed")
    print(f"[decode] aggregate      : {agg:.2f} tok/s")
    print(f"[decode] implied DRAM BW: {bw:.1f} GB/s (ternary weights only)")

    if gemm_sec > 0:
        print(f"\n[gemm-only] {gemm_bytes / 2**30:.2f} GiB in {gemm_sec:.2f} s")
        print(f"[gemm-only] bandwidth   : {gemm_bytes / gemm_sec / 1e9:.1f} GB/s")
        print(f"[gemm-only] per step    : {gemm_sec / steps * 1e3:.2f} ms "
              f"({100 * gemm_sec / (t_long - t_short):.1f}% of step)")
        print(f"[other    ] per step    : {(t_step - gemm_sec / steps) * 1e3:.2f} ms "
              "(lm_head + attention + norms + dispatch)")


if __name__ == "__main__":
    main()
