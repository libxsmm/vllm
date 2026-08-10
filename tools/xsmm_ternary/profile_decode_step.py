"""Profile where a decode step actually goes: ternary GEMMs vs lm_head vs the rest.

Resolves the apparent contradiction that the measured ternary-GEMM time plus the
isolated lm_head time exceeds the measured decode step time.
"""

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from vllm import LLM, SamplingParams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--top", type=int, default=18)
    args = p.parse_args()

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=512 * 1024 * 1024,
    )

    prompt = "Explain why matrix multiplication dominates LLM inference cost."
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4))

    torch.ops._C.xsmm_ternary_reset_stats()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        outs = llm.generate(
            [prompt], SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        )
    wall = time.perf_counter() - t0
    calls, rows, wbytes, wsec = torch.ops._C.xsmm_ternary_stats()
    n_tok = len(outs[0].outputs[0].token_ids)

    print(f"\nwall={wall:.3f}s for {n_tok} tokens ({wall / max(n_tok, 1) * 1e3:.1f} ms/tok)")
    print(f"ternary op: {calls} calls, {wbytes / 2**30:.2f} GiB, {wsec:.3f}s "
          f"({wbytes / wsec / 1e9:.1f} GB/s)")

    print(f"\n=== top {args.top} ops by self CPU time ===")
    ka = prof.key_averages()
    rows_ = sorted(ka, key=lambda e: -e.self_cpu_time_total)[: args.top]
    print(f"{'op':45s} {'self ms':>10s} {'total ms':>10s} {'calls':>7s}")
    for e in rows_:
        print(f"{e.key[:45]:45s} {e.self_cpu_time_total / 1e3:10.1f} "
              f"{e.cpu_time_total / 1e3:10.1f} {e.count:7d}")


if __name__ == "__main__":
    main()
