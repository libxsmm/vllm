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
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
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
        quantize_moe: bool = True,
    ) -> None:
        super().__init__()
        if group_size % 4 != 0:
            raise ValueError(f"group_size must be a multiple of 4, got {group_size}")
        self.group_size = group_size
        # True when the checkpoint already stores weight_packed/weight_scale
        self.pre_packed = pre_packed
        # The MoE router is a small dense Linear whose shape happens to be
        # packable, but it is never ternary; quantizing it corrupts routing.
        self.ignored_layers = list(ignored_layers) if ignored_layers else ["mlp.gate"]
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
        # MoE expert weights are packed while they load, one expert at a time,
        # so the dense 16-bit copies are never all resident.
        self.quantize_moe = quantize_moe

    def __repr__(self) -> str:
        return (
            f"XsmmTernaryConfig(group_size={self.group_size}, "
            f"pre_packed={self.pre_packed}, "
            f"lm_head_mode={self.lm_head_mode}, "
            f"lm_head_group_size={self.lm_head_group_size}, "
            f"quantize_moe={self.quantize_moe}, "
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
            quantize_moe=cls.get_from_keys_or(config, ["quantize_moe"], True),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if any(prefix.endswith(name) for name in self.ignored_layers):
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None

        if isinstance(layer, RoutedExperts):
            if not self.quantize_moe:
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return XsmmTernaryMoEMethod(self, layer.moe_config)

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


class XsmmTernaryMoEMethod(FusedMoEMethodBase):
    """Ternary expert weights for MoE, sharing the int2 x int8 GEMM.

    The stacked expert parameters hold the packed 2-bit layout directly and are
    filled by a custom weight loader, so the 16-bit expert weights are packed
    one expert at a time as they stream in and never all exist at once (they
    would be ~58 GiB for Qwen3-MoE-30B-A3B).
    """

    def __init__(self, quant_config: XsmmTernaryConfig, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.group_size = quant_config.group_size

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        g = self.group_size
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition

        ok = _can_pack(w13_up_dim, hidden_size, g) and _can_pack(
            hidden_size, intermediate_size_per_partition, g
        )
        if not ok:
            raise ValueError(
                f"MoE shapes (w13 [{w13_up_dim}, {hidden_size}], "
                f"w2 [{hidden_size}, {intermediate_size_per_partition}]) are not "
                f"compatible with the ternary kernel (need N % "
                f"{XSMM_TERNARY_BLOCK_N} == 0 and K % {g} == 0)"
            )

        w13 = _empty_packed(num_experts, w13_up_dim, hidden_size, g)
        w13_scale = _empty_scales(
            num_experts, w13_up_dim, hidden_size, g, self.quant_config.scale_dtype
        )
        w2 = _empty_packed(
            num_experts, hidden_size, intermediate_size_per_partition, g
        )
        w2_scale = _empty_scales(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            g,
            self.quant_config.scale_dtype,
        )

        # The expert params mapping targets the names "w13_weight"/"w2_weight",
        # so the packed tensors have to use them; the scales ride along on the
        # parameter because nothing in the checkpoint maps to them.
        w13_param = torch.nn.Parameter(w13, requires_grad=False)
        w2_param = torch.nn.Parameter(w2, requires_grad=False)
        w13_param.xsmm_scale = w13_scale
        w2_param.xsmm_scale = w2_scale

        layer.register_parameter("w13_weight", w13_param)
        layer.register_parameter("w2_weight", w2_param)

        extra_weight_attrs = dict(extra_weight_attrs)
        extra_weight_attrs["weight_loader"] = self._pack_and_load
        set_weight_attrs(w13_param, extra_weight_attrs)
        set_weight_attrs(w2_param, extra_weight_attrs)

        layer.xsmm_w13_scale = w13_scale
        layer.xsmm_w2_scale = w2_scale

    def _pack_and_load(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ):
        packed, scales = torch.ops._C.xsmm_ternary_pack_weight(
            loaded_weight, self.group_size, self.quant_config.half_scales
        )
        if shard_id == "w2":
            param.data[expert_id].copy_(packed)
            param.xsmm_scale[expert_id].copy_(scales)
        else:
            # w13 stacks gate then up along dim 0; the packed layout blocks that
            # dim by 32, and the split is block aligned, so each half can be
            # packed on its own.
            blocks = loaded_weight.shape[0] // XSMM_TERNARY_BLOCK_N
            off = 0 if shard_id == "w1" else blocks
            param.data[expert_id, off : off + blocks].copy_(packed)
            param.xsmm_scale[expert_id, off : off + blocks].copy_(scales)
        if return_success:
            return True
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        return

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    @property
    def is_monolithic(self) -> bool:
        # Route internally: the modular path calls torch.ops._moe_C.topk_softmax,
        # which is not built for CPU.
        return True

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            select_experts,
        )

        num_expert_group = layer.num_expert_group
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=layer.top_k,
            renormalize=self.moe.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=layer.topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                layer.routed_scaling_factor
                if layer.routed_scaling_factor is not None
                else 1.0
            ),
            e_score_correction_bias=layer.e_score_correction_bias,
        )
        return self._run_experts(layer, x, topk_weights, topk_ids)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if shared_experts is not None:
            raise NotImplementedError(
                "xsmm_ternary MoE does not handle shared experts yet"
            )
        return self._run_experts(layer, x, topk_weights, topk_ids)

    def _run_experts(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if layer.activation != MoEActivation.SILU:
            raise NotImplementedError(
                f"xsmm_ternary MoE only implements SiLU, got {layer.activation}"
            )

        g = self.group_size
        w13, w2 = layer.w13_weight, layer.w2_weight
        w13_scale, w2_scale = layer.xsmm_w13_scale, layer.xsmm_w2_scale

        x2 = x.reshape(-1, x.shape[-1])
        out = torch.zeros_like(x2)
        expert_map = layer.expert_map
        route_on_input = layer.apply_router_weight_on_input

        for e in torch.unique(topk_ids).tolist():
            local_e = e if expert_map is None else int(expert_map[e])
            if local_e < 0:
                continue
            tok, slot = (topk_ids == e).nonzero(as_tuple=True)
            xe = x2.index_select(0, tok)
            w = topk_weights[tok, slot].unsqueeze(-1).to(xe.dtype)
            if route_on_input:
                xe = xe * w

            h = torch.ops._C.xsmm_ternary_gemm(
                xe, w13[local_e], w13_scale[local_e], None, g
            )
            half = h.shape[-1] // 2
            act = torch.nn.functional.silu(h[:, :half]) * h[:, half:]
            y = torch.ops._C.xsmm_ternary_gemm(
                act, w2[local_e], w2_scale[local_e], None, g
            )
            if not route_on_input:
                y = y * w
            out.index_add_(0, tok, y.to(out.dtype))

        return out.view_as(x)


def _packed_shape(n: int, k: int, g: int) -> tuple[int, ...]:
    return (n // XSMM_TERNARY_BLOCK_N, k // g, g // 4, 8, 4)


def _empty_packed(num_experts: int, n: int, k: int, g: int) -> torch.Tensor:
    return torch.empty((num_experts, *_packed_shape(n, k, g)), dtype=torch.uint8)


def _empty_scales(
    num_experts: int, n: int, k: int, g: int, dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(
        num_experts,
        n // XSMM_TERNARY_BLOCK_N,
        k // g,
        XSMM_TERNARY_BLOCK_N,
        dtype=dtype,
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
