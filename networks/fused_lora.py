"""Explicit-VJP LoRA linear primitive used by the opt-in Anima fast path."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class _FusedLoRALinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if base_weight.requires_grad:
            raise ValueError("FusedLoRALinear requires a frozen base weight")
        if x.shape[-1] != base_weight.shape[-1]:
            raise ValueError("input and base weight feature dimensions do not match")
        if lora_down.shape[1] != x.shape[-1]:
            raise ValueError("LoRA down weight and input feature dimensions do not match")
        if lora_up.shape[1] != lora_down.shape[0]:
            raise ValueError("LoRA up/down ranks do not match")
        if lora_up.shape[0] != base_weight.shape[0]:
            raise ValueError("LoRA up and base output dimensions do not match")

        low_rank = F.linear(x, lora_down)
        output = F.linear(x, base_weight)
        output.add_(F.linear(low_rank, lora_up).mul_(scale))
        ctx.save_for_backward(x, low_rank, base_weight, lora_down, lora_up)
        ctx.scale = float(scale)
        ctx.autocast_enabled = torch.is_autocast_enabled(x.device.type)
        ctx.compute_dtype = output.dtype
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        x, low_rank, base_weight, lora_down, lora_up = ctx.saved_tensors
        scale = ctx.scale

        x_2d = x.reshape(-1, x.shape[-1])
        low_rank_2d = low_rank.reshape(-1, low_rank.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.autocast_enabled:
            with torch.autocast(device_type=x.device.type, dtype=ctx.compute_dtype):
                grad_up = torch.mm(grad_output_2d.transpose(0, 1), low_rank_2d)
                grad_low_rank = torch.mm(grad_output_2d, lora_up).mul(scale)
                grad_down = torch.mm(grad_low_rank.transpose(0, 1), x_2d)
                grad_x = torch.mm(grad_output_2d, base_weight)
            lora_down_compute = lora_down.to(grad_low_rank.dtype)
            grad_x.addmm_(grad_low_rank, lora_down_compute)
        else:
            grad_up = torch.mm(grad_output_2d.transpose(0, 1), low_rank_2d)
            grad_low_rank = torch.mm(grad_output_2d, lora_up).mul(scale)
            grad_down = torch.mm(grad_low_rank.transpose(0, 1), x_2d)
            grad_x = torch.mm(grad_output_2d, base_weight)
            grad_x.addmm_(grad_low_rank, lora_down)

        # Autograd usually casts custom-function returns to the input dtype, but
        # make that contract explicit for mixed BF16-autocast/FP32-parameter use.
        if grad_x.dtype != x.dtype:
            grad_x = grad_x.to(dtype=x.dtype)
        if grad_down.dtype != lora_down.dtype:
            grad_down = grad_down.to(dtype=lora_down.dtype)
        grad_up = grad_up.mul(scale)
        if grad_up.dtype != lora_up.dtype:
            grad_up = grad_up.to(dtype=lora_up.dtype)

        return (
            grad_x.reshape_as(x),
            None,
            grad_down.reshape_as(lora_down),
            grad_up.reshape_as(lora_up),
            None,
        )


def fused_lora_linear(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Frozen-base Linear plus LoRA with an explicit backward VJP."""

    return _FusedLoRALinear.apply(x, base_weight, lora_down, lora_up, scale)
