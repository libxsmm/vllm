"""Small offline generation + timing driver used for baselines and ternary comparisons."""

import argparse
import time

from vllm import LLM, SamplingParams

PROMPTS = [
    "Explain in three sentences why matrix multiplication dominates LLM inference cost.",
    "Write a Python function that computes the n-th Fibonacci number iteratively.",
    "What is the capital of France, and what river runs through it?",
    "Summarize the idea of ternary (1.58-bit) neural network weights.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantization", default=None)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    p.add_argument("--prompt", default=None, help="use a single explicit prompt")
    p.add_argument("--chat", action="store_true",
                   help="wrap prompts in the model's chat template (instruct models)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--lm-head-mode", choices=("ternary", "skip", "int8"),
                   default=None,
                   help="override how lm_head is quantized: keep it ternary, "
                        "leave it dense, or quantize it to int8 at load time")
    p.add_argument("--lm-head-group-size", type=int, default=None,
                   help="group size for --lm-head-mode int8 (default: the "
                        "checkpoint's group_size)")
    # on the CPU backend this is the fraction of host RAM vLLM reserves
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    args = p.parse_args()

    llm_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.kv_cache_memory_bytes:
        llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    lm_head_overrides = {}
    if args.lm_head_mode is not None:
        lm_head_overrides["lm_head_mode"] = args.lm_head_mode
    if args.lm_head_group_size is not None:
        lm_head_overrides["lm_head_group_size"] = args.lm_head_group_size
    if lm_head_overrides:
        def _patch(cfg, extra=lm_head_overrides):
            qc = dict(getattr(cfg, "quantization_config", None) or {})
            qc.update(extra)
            cfg.quantization_config = qc
            return cfg

        llm_kwargs["hf_overrides"] = _patch

    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    t_load = time.perf_counter() - t0
    print(f"[timing] model load + warmup: {t_load:.1f} s")

    prompts = [args.prompt] if args.prompt else PROMPTS[: args.num_prompts]
    if args.chat:
        tok = llm.get_tokenizer()
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # warmup so the LIBXSMM JIT cache is primed before we time anything
    llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=8))

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    dt = time.perf_counter() - t0

    n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n[timing] generated {n_out} tokens in {dt:.2f} s -> {n_out / dt:.2f} tok/s")
    for o in outputs:
        print("-" * 70)
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text.strip())


if __name__ == "__main__":
    main()
