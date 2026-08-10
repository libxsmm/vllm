"""Clean single-stream decode measurement.

Measures steady-state decode by differencing two generation lengths, so prefill,
warmup and any other fixed cost cancels exactly:

    time(N) = fixed + N * step   =>   step = (time(N2) - time(N1)) / (N2 - N1)

Reports the ternary-GEMM share (from in-op counters) and the residual, so the
lm_head / attention / norm cost is explicit rather than inferred.
"""

import argparse
import statistics
import time

import torch

from vllm import LLM, SamplingParams

PROMPT = (
    "Explain in detail why matrix multiplication dominates the cost of large "
    "language model inference on modern CPUs, and what can be done about it."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--n1", type=int, default=16)
    p.add_argument("--n2", type=int, default=80)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=1024)
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

    def run(n):
        torch.ops._C.xsmm_ternary_reset_stats()
        t0 = time.perf_counter()
        outs = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=n))
        dt = time.perf_counter() - t0
        ntok = len(outs[0].outputs[0].token_ids)
        _, _, wbytes, wsec = torch.ops._C.xsmm_ternary_stats()
        return dt, ntok, wbytes, wsec

    run(8)  # warmup, primes the LIBXSMM JIT cache

    steps, gemm_ms, other_ms = [], [], []
    for _ in range(args.reps):
        t1, k1, b1, s1 = run(args.n1)
        t2, k2, b2, s2 = run(args.n2)
        d = k2 - k1
        if d <= 0:
            continue
        step = (t2 - t1) / d
        g = (s2 - s1) / d
        steps.append(step)
        gemm_ms.append(g)
        other_ms.append(step - g)

    step = statistics.median(steps)
    g = statistics.median(gemm_ms)
    o = statistics.median(other_ms)

    print(f"\n=== steady-state decode, batch=1, {args.reps} reps ===")
    print(f"  step time        : {step * 1e3:7.2f} ms   ({1 / step:.2f} tok/s)")
    print(f"  ternary GEMMs    : {g * 1e3:7.2f} ms   ({100 * g / step:.1f}%)")
    print(f"  everything else  : {o * 1e3:7.2f} ms   ({100 * o / step:.1f}%)")
    print(f"  spread (min-max) : {min(steps) * 1e3:.2f} - {max(steps) * 1e3:.2f} ms")
    print(f"\n  SINGLE-STREAM DECODE: {1 / step:.2f} tok/s")


if __name__ == "__main__":
    main()
