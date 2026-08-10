"""Offline ternary packer for HF checkpoints whose weights are already ternary
but stored unpacked as 16-bit floats (e.g. prism-ml/Ternary-Bonsai-8B-unpacked).

Rewrites each linear weight [N, C] as
    <name>.weight_packed : uint8   [N/32, C/G, G/4, 8, 4]   (2 bits per weight)
    <name>.weight_scale  : float32 [N, C/G]
and tags config.json with a `xsmm_ternary` quantization_config, so vLLM loads the
packed representation directly instead of materializing the 16-bit weights.

Runs one shard at a time, so peak memory stays at roughly one shard.
"""

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK_N = 32

# only the transformer's projection matrices; embeddings / lm_head stay dense
TARGET_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)

NON_WEIGHT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "LICENSE",
    "NOTICE.txt",
    "README.md",
)


def is_target(name: str, tensor_shape) -> bool:
    return name.endswith(TARGET_SUFFIXES) and len(tensor_shape) == 2


def ternary_stats(w: torch.Tensor, group_size: int) -> tuple[float, float]:
    """Fraction of weights that are exactly ternary, and the max relative
    reconstruction error. Tells us whether the checkpoint really is ternary."""
    n, c = w.shape
    wf = w.float().view(n, c // group_size, group_size)
    scale = wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    codes = torch.round(wf / scale)
    exact = (codes.abs() <= 1).float().mean().item()
    err = (wf - codes.clamp(-1, 1) * scale).abs().max().item()
    denom = max(wf.abs().max().item(), 1e-8)
    return exact, err / denom


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="source HF checkpoint directory")
    p.add_argument("--dst", required=True, help="destination directory")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--fp32-scales", action="store_true",
                   help="store weight scales as fp32 (default is fp16, which "
                        "halves the scale bandwidth during dequantization)")
    p.add_argument("--check", action="store_true",
                   help="report how ternary the source weights actually are")
    args = p.parse_args()

    import vllm._custom_ops  # noqa: F401  (registers the op / loads the right _C)

    g = args.group_size
    os.makedirs(args.dst, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    if not shards:
        raise SystemExit(f"no .safetensors found in {args.src}")

    weight_map: dict[str, str] = {}
    total_size = 0
    n_packed = 0
    n_skipped: list[str] = []

    for shard in shards:
        out_name = os.path.basename(shard)
        out_tensors: dict[str, torch.Tensor] = {}

        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)

                if not is_target(name, t.shape):
                    out_tensors[name] = t
                    continue

                n, c = t.shape
                if n % BLOCK_N != 0 or c % g != 0:
                    n_skipped.append(f"{name} {tuple(t.shape)}")
                    out_tensors[name] = t
                    continue

                if args.check:
                    exact, rel = ternary_stats(t, g)
                    print(f"  {name} {tuple(t.shape)}: ternary={exact * 100:.2f}% "
                          f"max_rel_recon_err={rel:.3e}")

                packed, scales = torch.ops._C.xsmm_ternary_pack_weight(
                    t, g, not args.fp32_scales
                )
                base = name[: -len(".weight")]
                out_tensors[f"{base}.weight_packed"] = packed
                # [N/32, C/G, 32] -> [N, C/G] so vLLM's row/column sharding works
                out_tensors[f"{base}.weight_scale"] = (
                    scales.permute(0, 2, 1).reshape(n, c // g).contiguous()
                )
                n_packed += 1
                del t, packed, scales

        for k, v in out_tensors.items():
            weight_map[k] = out_name
            total_size += v.numel() * v.element_size()

        save_file(out_tensors, os.path.join(args.dst, out_name),
                  metadata={"format": "pt"})
        print(f"wrote {out_name} ({len(out_tensors)} tensors)")
        del out_tensors

    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": weight_map}, f, indent=2)

    for fname in NON_WEIGHT_FILES:
        src = os.path.join(args.src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.dst, fname))

    cfg_path = os.path.join(args.dst, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "quant_method": "xsmm_ternary",
        "group_size": g,
        "pre_packed": True,
        "half_scales": not args.fp32_scales,
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\npacked {n_packed} linear weights (group_size={g}) -> {args.dst}")
    print(f"total size: {total_size / 2**30:.2f} GiB")
    if n_skipped:
        print(f"left dense (shape not supported): {n_skipped}")


if __name__ == "__main__":
    main()
