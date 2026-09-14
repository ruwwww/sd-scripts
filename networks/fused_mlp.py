"""Explicit-VJP GELU MLP for frozen base weights plus LoRA branches."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from .fp8_kernels import (
    TRITON_AVAILABLE,
    dequantize_gelu_backward_fp8_triton,
    pack_rowwise_fp8_triton,
    unpack_rowwise_fp8_triton,
)

FP8_ACTIVATION_DTYPE = torch.float8_e4m3fn
_LOW_RANK_BASIS_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}
_FP8_BACKEND_LOGGED: set[tuple[str, str, bool]] = set()
logger = logging.getLogger(__name__)


def pack_rowwise_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the last feature dimension with one FP16 scale per row."""

    fp8_max = torch.finfo(FP8_ACTIVATION_DTYPE).max
    scale = x.detach().float().abs().amax(dim=-1, keepdim=True).div(fp8_max)
    scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
    quantized = x.detach().float().div(scale).clamp(-fp8_max, fp8_max).to(FP8_ACTIVATION_DTYPE)
    return quantized, scale.to(torch.float16)


def unpack_rowwise_fp8(quantized: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Restore a rowwise-FP8 activation to the original compute dtype."""

    return (quantized.float() * scale.float()).to(dtype=dtype)


def _pack_fp8_with_backend(x: torch.Tensor, backend: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    if backend not in ("auto", "eager", "triton"):
        raise ValueError(f"unsupported FP8 kernel backend: {backend}")
    if backend in ("auto", "triton") and TRITON_AVAILABLE and x.is_cuda:
        try:
            quantized, scale = pack_rowwise_fp8_triton(x)
            log_key = (backend, "triton", bool(x.is_cuda))
            if log_key not in _FP8_BACKEND_LOGGED:
                logger.info("fused MLP FP8 activation backend selected: triton")
                _FP8_BACKEND_LOGGED.add(log_key)
            return quantized, scale, "triton"
        except torch.cuda.OutOfMemoryError:
            # An OOM is a real capacity failure, not a backend capability
            # probe. Do not hide it by retrying the larger eager path.
            raise
        except Exception as error:
            if backend == "triton":
                raise
            log_key = (backend, "eager", bool(x.is_cuda))
            if log_key not in _FP8_BACKEND_LOGGED:
                logger.warning("fused MLP FP8 Triton pack failed; falling back to eager: %s", error)
                _FP8_BACKEND_LOGGED.add(log_key)
    quantized, scale = pack_rowwise_fp8(x)
    log_key = (backend, "eager", bool(x.is_cuda))
    if log_key not in _FP8_BACKEND_LOGGED:
        logger.info("fused MLP FP8 activation backend selected: eager")
        _FP8_BACKEND_LOGGED.add(log_key)
    return quantized, scale, "eager"


def _randomized_low_rank_storage(pre_gelu: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Store a deterministic randomized range approximation of ``pre_gelu``.

    CUDA QR does not support BF16/FP16 on this runtime, so the range finder is
    performed in FP32 and the two factors are stored back in the compute dtype.
    This is intentionally a research path: it trades backward fidelity and
    extra projection/QR work for a much smaller saved activation.
    """

    matrix = pre_gelu.reshape(-1, pre_gelu.shape[-1])
    rows, columns = matrix.shape
    if rank <= 0 or rank > min(rows, columns):
        raise ValueError(f"low-rank activation rank must be in [1, {min(rows, columns)}], got {rank}")
    cache_key = (f"{matrix.device.type}:{matrix.device.index}", int(columns), int(rank))
    omega = _LOW_RANK_BASIS_CACHE.get(cache_key)
    if omega is None or omega.device != matrix.device:
        generator = torch.Generator(device=matrix.device)
        generator.manual_seed((0x4D4C5000 + columns * 131 + rank) & 0x7FFFFFFF)
        omega = torch.randn((columns, rank), device=matrix.device, dtype=torch.float32, generator=generator)
        omega = F.normalize(omega, dim=0)
        _LOW_RANK_BASIS_CACHE[cache_key] = omega

    matrix_fp32 = matrix.float()
    range_matrix = torch.mm(matrix_fp32, omega)
    q, _ = torch.linalg.qr(range_matrix, mode="reduced")
    coefficient = torch.mm(q.transpose(0, 1), matrix_fp32)
    return q.to(dtype=pre_gelu.dtype), coefficient.to(dtype=pre_gelu.dtype)


def _unpack_low_rank_storage(q: torch.Tensor, coefficient: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.mm(q.float(), coefficient.float()).to(dtype=dtype)


def _validate_linear_shapes(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
) -> None:
    if base_weight.requires_grad:
        raise ValueError("FusedGELUMLP requires frozen base weights")
    if x.shape[-1] != base_weight.shape[-1] or lora_down.shape[1] != x.shape[-1]:
        raise ValueError("input and first MLP projection dimensions do not match")
    if lora_up.shape[1] != lora_down.shape[0] or lora_up.shape[0] != base_weight.shape[0]:
        raise ValueError("first MLP LoRA dimensions do not match")


def _linear_lora_forward(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    low_rank = F.linear(x, lora_down)
    output = F.linear(x, base_weight)
    output.add_(F.linear(low_rank, lora_up).mul_(scale))
    return output, low_rank


class _FusedGELUMLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        base1: torch.Tensor,
        down1: torch.Tensor,
        up1: torch.Tensor,
        scale1: float,
        base2: torch.Tensor,
        down2: torch.Tensor,
        up2: torch.Tensor,
        scale2: float,
        activation_storage: str,
        low_rank_rank: int,
        fp8_backend: str,
        store_input_fp8: bool,
    ) -> torch.Tensor:
        _validate_linear_shapes(x, base1, down1, up1)
        if base2.requires_grad:
            raise ValueError("FusedGELUMLP requires frozen second base weight")
        if base2.shape[1] != base1.shape[0] or down2.shape[1] != base1.shape[0]:
            raise ValueError("second MLP projection input dimensions do not match")
        if up2.shape[1] != down2.shape[0] or up2.shape[0] != base2.shape[0]:
            raise ValueError("second MLP LoRA dimensions do not match")

        first, low_rank1 = _linear_lora_forward(x, base1, down1, up1, scale1)
        pre_gelu = first
        activated = F.gelu(pre_gelu, approximate="none")
        output, low_rank2 = _linear_lora_forward(activated, base2, down2, up2, scale2)

        if activation_storage == "bf16":
            stored_pre, pre_scale = pre_gelu, pre_gelu.new_empty((0,))
            selected_fp8_backend = "eager"
        elif activation_storage == "fp8":
            stored_pre, pre_scale, selected_fp8_backend = _pack_fp8_with_backend(pre_gelu, fp8_backend)
        elif activation_storage == "lowrank":
            stored_pre, pre_scale = _randomized_low_rank_storage(pre_gelu, int(low_rank_rank))
            selected_fp8_backend = "eager"
        else:
            raise ValueError(f"unsupported activation storage mode: {activation_storage}")

        if store_input_fp8:
            if activation_storage != "fp8":
                raise ValueError("FP8 input storage is only supported with FP8 MLP activation storage")
            stored_x, x_scale, input_fp8_backend = _pack_fp8_with_backend(x, fp8_backend)
        else:
            stored_x, x_scale, input_fp8_backend = x, x.new_empty((0,)), "eager"

        ctx.save_for_backward(
            stored_x,
            x_scale,
            low_rank1,
            stored_pre,
            pre_scale,
            low_rank2,
            base1,
            down1,
            up1,
            base2,
            down2,
            up2,
        )
        ctx.scale1 = float(scale1)
        ctx.scale2 = float(scale2)
        ctx.activation_storage = activation_storage
        ctx.low_rank_rank = int(low_rank_rank)
        ctx.fp8_backend = selected_fp8_backend
        ctx.input_storage = "fp8" if store_input_fp8 else "original"
        ctx.input_fp8_backend = input_fp8_backend
        ctx.input_dtype = x.dtype
        ctx.pre_dtype = pre_gelu.dtype
        ctx.autocast_enabled = torch.is_autocast_enabled(x.device.type)
        ctx.compute_dtype = output.dtype
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        stored_x, x_scale, low_rank1, stored_pre, pre_scale, low_rank2, base1, down1, up1, base2, down2, up2 = (
            ctx.saved_tensors
        )
        if ctx.input_storage == "fp8" and ctx.input_fp8_backend == "triton":
            x = unpack_rowwise_fp8_triton(stored_x, x_scale, ctx.input_dtype)
        elif ctx.input_storage == "fp8":
            x = unpack_rowwise_fp8(stored_x, x_scale, ctx.input_dtype)
        else:
            x = stored_x
        if ctx.activation_storage == "fp8" and ctx.fp8_backend == "triton":
            pre_gelu = None
        elif ctx.activation_storage == "fp8":
            pre_gelu = unpack_rowwise_fp8(stored_pre, pre_scale, ctx.pre_dtype)
        elif ctx.activation_storage == "lowrank":
            pre_gelu = _unpack_low_rank_storage(stored_pre, pre_scale, ctx.pre_dtype)
        else:
            pre_gelu = stored_pre
        scale1 = ctx.scale1
        scale2 = ctx.scale2

        x_2d = x.reshape(-1, x.shape[-1])
        low_rank1_2d = low_rank1.reshape(-1, low_rank1.shape[-1])
        pre_gelu_2d = None if pre_gelu is None else pre_gelu.reshape(-1, pre_gelu.shape[-1])
        low_rank2_2d = low_rank2.reshape(-1, low_rank2.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        def vjp():
            grad_up2 = torch.mm(grad_output_2d.transpose(0, 1), low_rank2_2d).mul_(scale2)
            grad_low_rank2 = torch.mm(grad_output_2d, up2).mul_(scale2)
            grad_activated = torch.mm(grad_output_2d, base2)
            grad_activated.addmm_(grad_low_rank2, down2)
            if ctx.activation_storage == "fp8" and ctx.fp8_backend == "triton":
                activated_2d, grad_pre_gelu = dequantize_gelu_backward_fp8_triton(
                    stored_pre,
                    pre_scale,
                    grad_activated,
                    ctx.pre_dtype,
                )
            else:
                activated_2d = F.gelu(pre_gelu_2d, approximate="none")
                grad_pre_gelu = torch.ops.aten.gelu_backward(grad_activated, pre_gelu_2d, approximate="none")
            grad_down2 = torch.mm(grad_low_rank2.transpose(0, 1), activated_2d)

            grad_up1 = torch.mm(grad_pre_gelu.transpose(0, 1), low_rank1_2d).mul_(scale1)
            grad_low_rank1 = torch.mm(grad_pre_gelu, up1).mul_(scale1)
            grad_down1 = torch.mm(grad_low_rank1.transpose(0, 1), x_2d)
            grad_x = torch.mm(grad_pre_gelu, base1)
            grad_x.addmm_(grad_low_rank1, down1)
            return grad_x, grad_down1, grad_up1, grad_down2, grad_up2

        if ctx.autocast_enabled:
            with torch.autocast(device_type=x.device.type, dtype=ctx.compute_dtype):
                grad_x, grad_down1, grad_up1, grad_down2, grad_up2 = vjp()
        else:
            grad_x, grad_down1, grad_up1, grad_down2, grad_up2 = vjp()

        if grad_x.dtype != x.dtype:
            grad_x = grad_x.to(dtype=x.dtype)
        if grad_down1.dtype != down1.dtype:
            grad_down1 = grad_down1.to(dtype=down1.dtype)
        if grad_up1.dtype != up1.dtype:
            grad_up1 = grad_up1.to(dtype=up1.dtype)
        if grad_down2.dtype != down2.dtype:
            grad_down2 = grad_down2.to(dtype=down2.dtype)
        if grad_up2.dtype != up2.dtype:
            grad_up2 = grad_up2.to(dtype=up2.dtype)

        return (
            grad_x.reshape_as(x),
            None,
            grad_down1.reshape_as(down1),
            grad_up1.reshape_as(up1),
            None,
            None,
            grad_down2.reshape_as(down2),
            grad_up2.reshape_as(up2),
            None,
            None,
            None,
            None,
            None,
        )


def fused_gelu_mlp(
    x: torch.Tensor,
    base1: torch.Tensor,
    lora_down1: torch.Tensor,
    lora_up1: torch.Tensor,
    scale1: float,
    base2: torch.Tensor,
    lora_down2: torch.Tensor,
    lora_up2: torch.Tensor,
    scale2: float,
) -> torch.Tensor:
    """Frozen-base `Linear -> GELU -> Linear` with explicit LoRA VJP."""

    return _FusedGELUMLP.apply(
        x,
        base1,
        lora_down1,
        lora_up1,
        scale1,
        base2,
        lora_down2,
        lora_up2,
        scale2,
        "bf16",
        0,
        "eager",
        False,
    )


def fused_gelu_mlp_fp8(
    x: torch.Tensor,
    base1: torch.Tensor,
    lora_down1: torch.Tensor,
    lora_up1: torch.Tensor,
    scale1: float,
    base2: torch.Tensor,
    lora_down2: torch.Tensor,
    lora_up2: torch.Tensor,
    scale2: float,
    backend: str = "auto",
    store_input_fp8: bool = False,
) -> torch.Tensor:
    """Same MLP with optional row-wise FP8 storage for pre-GELU and input."""

    return _FusedGELUMLP.apply(
        x,
        base1,
        lora_down1,
        lora_up1,
        scale1,
        base2,
        lora_down2,
        lora_up2,
        scale2,
        "fp8",
        0,
        backend,
        bool(store_input_fp8),
    )


def fused_gelu_mlp_lowrank(
    x: torch.Tensor,
    base1: torch.Tensor,
    lora_down1: torch.Tensor,
    lora_up1: torch.Tensor,
    scale1: float,
    base2: torch.Tensor,
    lora_down2: torch.Tensor,
    lora_up2: torch.Tensor,
    scale2: float,
    rank: int,
) -> torch.Tensor:
    """Research path storing pre-GELU as a low-rank ``Q @ B`` approximation."""

    return _FusedGELUMLP.apply(
        x,
        base1,
        lora_down1,
        lora_up1,
        scale1,
        base2,
        lora_down2,
        lora_up2,
        scale2,
        "lowrank",
        int(rank),
        "eager",
        False,
    )
