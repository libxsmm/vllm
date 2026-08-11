# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-the-fly ternary (1.58-bit) weight quantization backed by LIBXSMM int2 kernels.

Checkpoints such as `prism-ml/Ternary-Bonsai-8B-unpacked` store ternary weights
*unpacked* as 16-bit floats (each group of `group_size` input channels is
`code * scale` with `code` in {-1, 0, +1}). This method re-packs those weights
into the 2-bit VNNI4-interleaved layout expected by the LIBXSMM int2 x int8
microkernel at load time, so inference runs on the packed representation.

Applying it to a non-ternary checkpoint is lossy, but exercises the same kernel
path and is useful for performance bring-up.
"""

from typing import Any

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

# fixed by the LIBXSMM int2 microkernel: one 32-byte ymm of packed weights
XSMM_TERNARY_BLOCK_N = 32

# Opt-in: reuse a per-layer output buffer instead of allocating per call.
# Measured worth ~0.3% of a decode step, and unsafe under graph capture.
_REUSE_OUTPUT_BUFFER = os.environ.get("VLLM_XSMM_TERNARY_REUSE_OUT", "0") == "1"


class XsmmTernaryConfig(QuantizationConfig):
    """Config for LIBXSMM ternary weight / int8 activation linear layers."""

    def __init__(
        self,
        group_size: int = 128,
        pre_packed: bool = False,
        ignored_layers: list[str] | None = None,
        quantize_lm_head: bool = True,
        half_scales: bool = True,
        lm_head_mode: str | None = None,
        lm_head_group_size: int | None = None,
    ) -> None:
        super().__init__()
        if group_size % 4 != 0:
            raise ValueError(f"group_size must be a multiple of 4, got {group_size}")
        self.group_size = group_size
        # True when the checkpoint already stores weight_packed/weight_scale
        self.pre_packed = pre_packed
        self.ignored_layers = ignored_layers or []
        self.quantize_lm_head = quantize_lm_head
        # fp16 weight scales: 1 per 128 values, so this is 11% of weight traffic
        self.half_scales = half_scales
        self.scale_dtype = torch.float16 if half_scales else torch.float32

        # How to treat lm_head. Checkpoints such as cat-q keep lm_head and the
        # embeddings in 16-bit even though every projection is ternary, so
        # quantizing it to ternary would corrupt the logits.
        #   "ternary" -- pack to 2-bit (only valid if lm_head really is ternary)
        #   "skip"    -- leave dense, unquantized
        #   "int8"    -- quantize to int8 at load time, halving its bandwidth
        if lm_head_mode is None:
            lm_head_mode = "ternary" if quantize_lm_head else "skip"
        if lm_head_mode not in ("ternary", "skip", "int8"):
            raise ValueError(
                "lm_head_mode must be one of 'ternary', 'skip', 'int8'; "
                f"got {lm_head_mode!r}"
            )
        self.lm_head_mode = lm_head_mode
        self.lm_head_group_size = lm_head_group_size or group_size
        if self.lm_head_group_size % 4 != 0:
            raise ValueError(
                "lm_head_group_size must be a multiple of 4, got "
                f"{self.lm_head_group_size}"
            )

    def __repr__(self) -> str:
        return (
            f"XsmmTernaryConfig(group_size={self.group_size}, "
            f"pre_packed={self.pre_packed}, "
            f"lm_head_mode={self.lm_head_mode}, "
            f"lm_head_group_size={self.lm_head_group_size}, "
            f"half_scales={self.half_scales})"
        )

    def get_name(self) -> str:
        return "xsmm_ternary"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "XsmmTernaryConfig":
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 128),
            pre_packed=cls.get_from_keys_or(config, ["pre_packed"], False),
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None),
            quantize_lm_head=cls.get_from_keys_or(config, ["quantize_lm_head"], True),
            half_scales=cls.get_from_keys_or(config, ["half_scales"], True),
            lm_head_mode=cls.get_from_keys_or(config, ["lm_head_mode"], None),
            lm_head_group_size=cls.get_from_keys_or(
                config, ["lm_head_group_size"], None
            ),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if any(prefix.endswith(name) for name in self.ignored_layers):
            return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None

        # lm_head is the single largest GEMV in the model. It is a
        # ParallelLMHead (not a LinearBase) and vLLM pads the vocab, so it is
        # always packed at load time rather than pre-packed.
        if isinstance(layer, ParallelLMHead):
            if self.lm_head_mode == "skip":
                return None
            if self.lm_head_mode == "int8":
                return XsmmInt8LinearMethod(self)
            return XsmmTernaryLinearMethod(self, runtime_pack=True)

        if not isinstance(layer, LinearBase):
            return None
        return XsmmTernaryLinearMethod(self)


class XsmmTernaryLinearMethod(LinearMethodBase):
    def __init__(
        self, quant_config: XsmmTernaryConfig, runtime_pack: bool = False
    ) -> None:
        self.quant_config = quant_config
        # load dense and pack after loading, ignoring `pre_packed`
        self.runtime_pack = runtime_pack

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_loader = extra_weight_attrs.pop("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.quant_config.group_size

        layer.xsmm_ternary = _can_pack(
            output_size_per_partition, input_size_per_partition, group_size
        )
        if not layer.xsmm_ternary:
            logger.warning_once(
                "Shape [%d, %d] is not compatible with the LIBXSMM ternary "
                "kernel (needs N %% %d == 0 and K %% %d == 0); falling back to "
                "an unquantized GEMM for this layer.",
                output_size_per_partition,
                input_size_per_partition,
                XSMM_TERNARY_BLOCK_N,
                group_size,
            )

        if not (self.quant_config.pre_packed and layer.xsmm_ternary) or self.runtime_pack:
            # dense checkpoint; packing happens after loading
            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, extra_weight_attrs)
            return

        n_blocks = output_size_per_partition // XSMM_TERNARY_BLOCK_N
        n_groups = input_size_per_partition // group_size

        # dim 0 is blocked by 32 output rows, so vLLM's column/QKV sharding must
        # divide shard offsets by 32; dim 1 is in whole groups, which row
        # sharding handles as-is.
        weight_packed = PackedvLLMParameter(
            data=torch.empty(
                n_blocks,
                n_groups,
                group_size // 4,
                8,
                4,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=XSMM_TERNARY_BLOCK_N,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                n_groups,
                dtype=self.quant_config.scale_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight_packed)
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_packed, extra_weight_attrs)
        set_weight_attrs(weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "xsmm_ternary", False):
            UnquantizedLinearMethod().process_weights_after_loading(layer)
            return

        if self.quant_config.pre_packed and not self.runtime_pack:
            packed = layer.weight_packed.data
            n_blocks, n_groups = packed.shape[0], packed.shape[1]
            # [N, C/G] -> [N/32, C/G, 32], the layout the kernel indexes
            scales = (
                layer.weight_scale.data.view(n_blocks, XSMM_TERNARY_BLOCK_N, n_groups)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer.register_parameter("weight_packed", None)
            layer.register_parameter("weight_scale", None)
        else:
            packed, scales = torch.ops._C.xsmm_ternary_pack_weight(
                layer.weight.data,
                self.quant_config.group_size,
                self.quant_config.half_scales,
            )
            # the dense copy is no longer needed
            layer.register_parameter("weight", None)

        layer.register_buffer("xsmm_packed_weight", packed.contiguous())
        layer.register_buffer("xsmm_weight_scale", scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(layer, "xsmm_ternary", False):
            return UnquantizedLinearMethod().apply(layer, x, bias)

        if _REUSE_OUTPUT_BUFFER:
            return self._apply_reuse_out(layer, x, bias)

        return torch.ops._C.xsmm_ternary_gemm(
            x,
            layer.xsmm_packed_weight,
            layer.xsmm_weight_scale,
            bias,
            self.quant_config.group_size,
        )

    def _apply_reuse_out(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Write into a per-layer buffer instead of allocating an output.

        Only safe while nothing holds this layer's previous output past the next
        call to the same layer -- true for eager execution, NOT for graph
        capture. Hence opt-in via VLLM_XSMM_TERNARY_REUSE_OUT.
        """
        n = layer.xsmm_packed_weight.size(0) * XSMM_TERNARY_BLOCK_N
        m = x.numel() // x.size(-1)

        buf = getattr(layer, "_xsmm_out_buf", None)
        if buf is None or buf.numel() < m * n or buf.dtype != x.dtype:
            buf = torch.empty(m * n, dtype=x.dtype)
            layer._xsmm_out_buf = buf

        out = buf.narrow(0, 0, m * n).view(m, n)
        torch.ops._C.xsmm_ternary_gemm_out(
            x,
            layer.xsmm_packed_weight,
            layer.xsmm_weight_scale,
            bias,
            self.quant_config.group_size,
            out,
        )
        return out.view(*x.shape[:-1], n)


def _can_pack(n: int, k: int, group_size: int) -> bool:
    return (
        current_platform.is_cpu()
        and n % XSMM_TERNARY_BLOCK_N == 0
        and k % group_size == 0
    )


class XsmmInt8LinearMethod(LinearMethodBase):
    """Dynamic int8 weight quantization, sharing the int8-activation pipeline.

    Used for lm_head in checkpoints whose projections are ternary but whose
    output head is not. The weights are loaded dense and quantized once, so the
    head costs 1 byte per weight instead of 2.
    """

    def __init__(self, quant_config: XsmmTernaryConfig) -> None:
        self.quant_config = quant_config
        self.group_size = quant_config.lm_head_group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_loader = extra_weight_attrs.pop("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        layer.xsmm_int8 = _can_pack(
            output_size_per_partition, input_size_per_partition, self.group_size
        )
        if not layer.xsmm_int8:
            logger.warning_once(
                "Shape [%d, %d] is not compatible with the int8 kernel "
                "(needs N %% %d == 0 and K %% %d == 0); falling back to an "
                "unquantized GEMM for this layer.",
                output_size_per_partition,
                input_size_per_partition,
                XSMM_TERNARY_BLOCK_N,
                self.group_size,
            )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "xsmm_int8", False):
            UnquantizedLinearMethod().process_weights_after_loading(layer)
            return

        packed, scales = torch.ops._C.xsmm_int8_pack_weight(
            layer.weight.data,
            self.group_size,
            self.quant_config.half_scales,
        )
        layer.register_parameter("weight", None)
        layer.register_buffer("xsmm_packed_weight", packed.contiguous())
        layer.register_buffer("xsmm_weight_scale", scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(layer, "xsmm_int8", False):
            return UnquantizedLinearMethod().apply(layer, x, bias)

        return torch.ops._C.xsmm_int8_gemm(
            x,
            layer.xsmm_packed_weight,
            layer.xsmm_weight_scale,
            bias,
            self.group_size,
        )
