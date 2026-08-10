"""Inspect how each linear-ish layer of a loaded model is executed, and time the
lm_head GEMV specifically.

lm_head is [vocab, hidden] and is NOT a LinearBase in vLLM (it is a
ParallelLMHead / VocabParallelEmbedding), so the ternary quant method never sees
it. At vocab=151669, hidden=4096 that is 1.24 GB of bf16 read on every decode
step -- potentially comparable to the whole ternary body of the model.
"""

import argparse
import time

import torch

from vllm import LLM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
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

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    print("\n=== modules carrying weights ===")
    total_dense = 0
    total_packed = 0
    for name, mod in model.named_modules():
        packed = getattr(mod, "xsmm_packed_weight", None)
        w = getattr(mod, "weight", None)
        if packed is not None:
            sc = mod.xsmm_weight_scale
            b = packed.numel() + sc.numel() * 4
            total_packed += b
            if "layers.0." in name or "layers" not in name:
                print(f"  [TERNARY] {name:55s} {tuple(packed.shape)} "
                      f"{b / 2**20:8.2f} MiB")
        elif isinstance(w, torch.Tensor) and w.dim() == 2:
            b = w.numel() * w.element_size()
            total_dense += b
            if "layers.0." in name or "layers" not in name:
                print(f"  [DENSE  ] {name:55s} {tuple(w.shape)} "
                      f"{w.dtype} {b / 2**20:8.2f} MiB")

    print(f"\ntotal ternary weights: {total_packed / 2**30:.3f} GiB")
    print(f"total dense weights  : {total_dense / 2**30:.3f} GiB")

    # time the lm_head GEMV in isolation
    lm_head = None
    for name, mod in model.named_modules():
        if name.endswith("lm_head"):
            lm_head = mod
            break
    if lm_head is None or not hasattr(lm_head, "weight"):
        print("\nno lm_head weight found")
        return

    W = lm_head.weight.data
    print(f"\n=== lm_head GEMV: {tuple(W.shape)} {W.dtype} "
          f"({W.numel() * W.element_size() / 2**30:.3f} GiB) ===")
    x = torch.randn(1, W.shape[1], dtype=W.dtype)
    for _ in range(3):
        torch.nn.functional.linear(x, W)
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.linear(x, W)
    dt = (time.perf_counter() - t0) / iters
    nbytes = W.numel() * W.element_size()
    print(f"  {dt * 1e3:.2f} ms/call   {nbytes / dt / 1e9:.1f} GB/s   "
          f"{2 * W.numel() / dt / 1e9:.1f} GFLOP/s")
    print(f"  => {dt * 1e3:.2f} ms of every decode step is lm_head alone")


if __name__ == "__main__":
    main()
