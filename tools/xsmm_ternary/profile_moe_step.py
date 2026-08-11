"""Break down a decode step of the pre-packed MoE model by torch op.

The ternary GEMMs are ~76% of the step (their own counters say so); this
attributes the rest, which is the only part left to optimize.
"""

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from vllm import LLM, SamplingParams


def patch(cfg):
    qc = dict(getattr(cfg, "quantization_config", None) or {})
    qc.setdefault("quant_method", "xsmm_ternary")
    qc["lm_head_mode"] = "int8"
    return cfg if setattr(cfg, "quantization_config", qc) else cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--warmup-tokens", type=int, default=128)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    p.add_argument("--kv-cache-memory-bytes", type=int, default=4 * 1024**3)
    p.add_argument("--rows", type=int, default=28)
    args = p.parse_args()

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        hf_overrides=patch,
    )
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain what a CPU cache is."}],
        tokenize=False,
        add_generation_prompt=True,
    )

    llm.generate([prompt], SamplingParams(temperature=0, max_tokens=args.warmup_tokens))

    torch.ops._C.xsmm_ternary_reset_stats()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = llm.generate(
            [prompt], SamplingParams(temperature=0, max_tokens=args.tokens)
        )
    wall = time.perf_counter() - t0
    n = len(out[0].outputs[0].token_ids)
    _, _, _, secs = torch.ops._C.xsmm_ternary_stats()

    print(f"\ntokens={n}  wall={wall / n * 1e3:.1f} ms/tok  "
          f"ternary_ops={secs / n * 1e3:.1f} ms/tok "
          f"({100 * secs / wall:.0f}%)  other={(wall - secs) / n * 1e3:.1f} ms/tok\n")

    ev = prof.key_averages()
    total_self = sum(e.self_cpu_time_total for e in ev)
    print(f"{'op':<44}{'self ms/tok':>12}{'% step':>9}{'calls/tok':>11}")
    print("-" * 76)
    for e in sorted(ev, key=lambda x: -x.self_cpu_time_total)[: args.rows]:
        ms = e.self_cpu_time_total / 1e3 / n
        print(f"{e.key[:43]:<44}{ms:>12.2f}{100 * ms / (wall / n * 1e3):>8.1f}%"
              f"{e.count / n:>11.1f}")
    print("-" * 76)
    print(f"{'sum of self time':<44}{total_self / 1e3 / n:>12.2f}")


if __name__ == "__main__":
    main()
