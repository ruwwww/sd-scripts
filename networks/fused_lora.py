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
        out_2d = output.reshape(-1, output.shape[-1])
        lr_2d = low_rank.reshape(-1, low_rank.shape[-1])
        up_compute = lora_up.to(dtype=output.dtype) if lora_up.dtype != output.dtype else lora_up
        out_2d.addmm_(lr_2d, up_compute.transpose(0, 1), beta=1.0, alpha=float(scale))
        ctx.save_for_backward(x, low_rank, base_weight, lora_down, lora_up)
        ctx.scale = float(scale)
        ctx.autocast_enabled = torch.is_autocast_enabled(x.device.type)
        ctx.compute_dtype = output.dtype
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        x, low_rank, base_weight, lora_down, lora_up = ctx.saved_tensors
        scale = ctx.scale
        need_grad_x = ctx.needs_input_grad[0]

        x_2d = x.reshape(-1, x.shape[-1])
        low_rank_2d = low_rank.reshape(-1, low_rank.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.autocast_enabled:
            with torch.autocast(device_type=x.device.type, dtype=ctx.compute_dtype):
                grad_up = torch.mm(grad_output_2d.transpose(0, 1), low_rank_2d)
                grad_low_rank = torch.mm(grad_output_2d, lora_up)
                if scale != 1.0:
                    grad_low_rank.mul_(scale)
                grad_down = torch.mm(grad_low_rank.transpose(0, 1), x_2d)
                grad_x = torch.mm(grad_output_2d, base_weight) if need_grad_x else None
            if need_grad_x:
                lora_down_compute = lora_down.to(grad_low_rank.dtype)
                grad_x.addmm_(grad_low_rank, lora_down_compute)
        else:
            grad_up = torch.mm(grad_output_2d.transpose(0, 1), low_rank_2d)
            grad_low_rank = torch.mm(grad_output_2d, lora_up)
            if scale != 1.0:
                grad_low_rank.mul_(scale)
            grad_down = torch.mm(grad_low_rank.transpose(0, 1), x_2d)
            if need_grad_x:
                grad_x = torch.mm(grad_output_2d, base_weight)
                grad_x.addmm_(grad_low_rank, lora_down.to(grad_low_rank.dtype))
            else:
                grad_x = None

        # Autograd usually casts custom-function returns to the input dtype, but
        # make that contract explicit for mixed BF16-autocast/FP32-parameter use.
        if grad_x is not None and grad_x.dtype != x.dtype:
            grad_x = grad_x.to(dtype=x.dtype)
        if grad_down.dtype != lora_down.dtype:
            grad_down = grad_down.to(dtype=lora_down.dtype)
        if scale != 1.0:
            grad_up = grad_up.mul(scale)
        if grad_up.dtype != lora_up.dtype:
            grad_up = grad_up.to(dtype=lora_up.dtype)

        return (
            grad_x.reshape_as(x) if grad_x is not None else None,
            None,
            grad_down.reshape_as(lora_down),
            grad_up.reshape_as(lora_up),
            None,
        )


class _FusedSelfAttnQKVLoRA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        wq: torch.Tensor,
        dq: torch.Tensor,
        uq: torch.Tensor,
        sq: float,
        wk: torch.Tensor,
        dk: torch.Tensor,
        uk: torch.Tensor,
        sk: float,
        wv: torch.Tensor,
        dv: torch.Tensor,
        uv: torch.Tensor,
        sv: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r_q, r_k, r_v = dq.shape[0], dk.shape[0], dv.shape[0]
        d_qkv = torch.cat((dq, dk, dv), dim=0)
        low_qkv = F.linear(x, d_qkv)
        low_q, low_k, low_v = low_qkv.split((r_q, r_k, r_v), dim=-1)

        q = F.linear(x, wq)
        k = F.linear(x, wk)
        v = F.linear(x, wv)

        dtype = q.dtype
        uq_c = uq.to(dtype) if uq.dtype != dtype else uq
        uk_c = uk.to(dtype) if uk.dtype != dtype else uk
        uv_c = uv.to(dtype) if uv.dtype != dtype else uv

        q.reshape(-1, q.shape[-1]).addmm_(low_q.reshape(-1, r_q), uq_c.transpose(0, 1), beta=1.0, alpha=float(sq))
        k.reshape(-1, k.shape[-1]).addmm_(low_k.reshape(-1, r_k), uk_c.transpose(0, 1), beta=1.0, alpha=float(sk))
        v.reshape(-1, v.shape[-1]).addmm_(low_v.reshape(-1, r_v), uv_c.transpose(0, 1), beta=1.0, alpha=float(sv))

        ctx.save_for_backward(x, low_q, low_k, low_v, wq, dq, uq, wk, dk, uk, wv, dv, uv)
        ctx.scales = (float(sq), float(sk), float(sv))
        ctx.compute_dtype = dtype
        return q, k, v

    @staticmethod
    def backward(ctx: Any, gq: torch.Tensor, gk: torch.Tensor, gv: torch.Tensor):
        x, low_q, low_k, low_v, wq, dq, uq, wk, dk, uk, wv, dv, uv = ctx.saved_tensors
        sq, sk, sv = ctx.scales
        dtype = ctx.compute_dtype
        need_grad_x = ctx.needs_input_grad[0]

        x_2d = x.reshape(-1, x.shape[-1]).to(dtype)
        gq_2d = gq.reshape(-1, gq.shape[-1]).to(dtype)
        gk_2d = gk.reshape(-1, gk.shape[-1]).to(dtype)
        gv_2d = gv.reshape(-1, gv.shape[-1]).to(dtype)

        low_q_2d = low_q.reshape(-1, low_q.shape[-1]).to(dtype)
        low_k_2d = low_k.reshape(-1, low_k.shape[-1]).to(dtype)
        low_v_2d = low_v.reshape(-1, low_v.shape[-1]).to(dtype)

        uq_c = uq.to(dtype) if uq.dtype != dtype else uq
        uk_c = uk.to(dtype) if uk.dtype != dtype else uk
        uv_c = uv.to(dtype) if uv.dtype != dtype else uv

        grad_uq = torch.mm(gq_2d.transpose(0, 1), low_q_2d)
        grad_uk = torch.mm(gk_2d.transpose(0, 1), low_k_2d)
        grad_uv = torch.mm(gv_2d.transpose(0, 1), low_v_2d)
        if sq != 1.0:
            grad_uq.mul_(sq)
        if sk != 1.0:
            grad_uk.mul_(sk)
        if sv != 1.0:
            grad_uv.mul_(sv)

        glq = torch.mm(gq_2d, uq_c)
        glk = torch.mm(gk_2d, uk_c)
        glv = torch.mm(gv_2d, uv_c)
        if sq != 1.0:
            glq.mul_(sq)
        if sk != 1.0:
            glk.mul_(sk)
        if sv != 1.0:
            glv.mul_(sv)

        gl_qkv = torch.cat((glq, glk, glv), dim=-1)
        grad_d_qkv = torch.mm(gl_qkv.transpose(0, 1), x_2d)
        r_q, r_k, r_v = dq.shape[0], dk.shape[0], dv.shape[0]
        grad_dq, grad_dk, grad_dv = grad_d_qkv.split((r_q, r_k, r_v), dim=0)

        if need_grad_x:
            d_qkv_c = torch.cat((dq, dk, dv), dim=0).to(dtype)
            grad_x = torch.mm(gq_2d, wq)
            grad_x.addmm_(gk_2d, wk)
            grad_x.addmm_(gv_2d, wv)
            grad_x.addmm_(gl_qkv, d_qkv_c)
            grad_x = grad_x.to(x.dtype).reshape_as(x)
        else:
            grad_x = None

        return (
            grad_x,
            None,
            grad_dq.to(dq.dtype),
            grad_uq.to(uq.dtype),
            None,
            None,
            grad_dk.to(dk.dtype),
            grad_uk.to(uk.dtype),
            None,
            None,
            grad_dv.to(dv.dtype),
            grad_uv.to(uv.dtype),
            None,
        )


class _FusedCrossAttnKVLoRA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        context: torch.Tensor,
        wk: torch.Tensor,
        dk: torch.Tensor,
        uk: torch.Tensor,
        sk: float,
        wv: torch.Tensor,
        dv: torch.Tensor,
        uv: torch.Tensor,
        sv: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r_k, r_v = dk.shape[0], dv.shape[0]
        d_kv = torch.cat((dk, dv), dim=0)
        low_kv = F.linear(context, d_kv)
        low_k, low_v = low_kv.split((r_k, r_v), dim=-1)

        k = F.linear(context, wk)
        v = F.linear(context, wv)

        dtype = k.dtype
        uk_c = uk.to(dtype) if uk.dtype != dtype else uk
        uv_c = uv.to(dtype) if uv.dtype != dtype else uv

        k.reshape(-1, k.shape[-1]).addmm_(low_k.reshape(-1, r_k), uk_c.transpose(0, 1), beta=1.0, alpha=float(sk))
        v.reshape(-1, v.shape[-1]).addmm_(low_v.reshape(-1, r_v), uv_c.transpose(0, 1), beta=1.0, alpha=float(sv))

        ctx.save_for_backward(context, low_k, low_v, wk, dk, uk, wv, dv, uv)
        ctx.scales = (float(sk), float(sv))
        ctx.compute_dtype = dtype
        return k, v

    @staticmethod
    def backward(ctx: Any, gk: torch.Tensor, gv: torch.Tensor):
        context, low_k, low_v, wk, dk, uk, wv, dv, uv = ctx.saved_tensors
        sk, sv = ctx.scales
        dtype = ctx.compute_dtype
        need_grad_c = ctx.needs_input_grad[0]

        c_2d = context.reshape(-1, context.shape[-1]).to(dtype)
        gk_2d = gk.reshape(-1, gk.shape[-1]).to(dtype)
        gv_2d = gv.reshape(-1, gv.shape[-1]).to(dtype)

        low_k_2d = low_k.reshape(-1, low_k.shape[-1]).to(dtype)
        low_v_2d = low_v.reshape(-1, low_v.shape[-1]).to(dtype)

        uk_c = uk.to(dtype) if uk.dtype != dtype else uk
        uv_c = uv.to(dtype) if uv.dtype != dtype else uv

        grad_uk = torch.mm(gk_2d.transpose(0, 1), low_k_2d)
        grad_uv = torch.mm(gv_2d.transpose(0, 1), low_v_2d)
        if sk != 1.0:
            grad_uk.mul_(sk)
        if sv != 1.0:
            grad_uv.mul_(sv)

        glk = torch.mm(gk_2d, uk_c)
        glv = torch.mm(gv_2d, uv_c)
        if sk != 1.0:
            glk.mul_(sk)
        if sv != 1.0:
            glv.mul_(sv)

        gl_kv = torch.cat((glk, glv), dim=-1)
        grad_d_kv = torch.mm(gl_kv.transpose(0, 1), c_2d)
        r_k, r_v = dk.shape[0], dv.shape[0]
        grad_dk, grad_dv = grad_d_kv.split((r_k, r_v), dim=0)

        if need_grad_c:
            d_kv_c = torch.cat((dk, dv), dim=0).to(dtype)
            grad_c = torch.mm(gk_2d, wk)
            grad_c.addmm_(gv_2d, wv)
            grad_c.addmm_(gl_kv, d_kv_c)
            grad_c = grad_c.to(context.dtype).reshape_as(context)
        else:
            grad_c = None

        return (
            grad_c,
            None,
            grad_dk.to(dk.dtype),
            grad_uk.to(uk.dtype),
            None,
            None,
            grad_dv.to(dv.dtype),
            grad_uv.to(uv.dtype),
            None,
        )


def fused_self_attn_qkv_lora(
    x: torch.Tensor,
    wq: torch.Tensor,
    dq: torch.Tensor,
    uq: torch.Tensor,
    sq: float,
    wk: torch.Tensor,
    dk: torch.Tensor,
    uk: torch.Tensor,
    sk: float,
    wv: torch.Tensor,
    dv: torch.Tensor,
    uv: torch.Tensor,
    sv: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _FusedSelfAttnQKVLoRA.apply(x, wq, dq, uq, sq, wk, dk, uk, sk, wv, dv, uv, sv)


def fused_cross_attn_kv_lora(
    context: torch.Tensor,
    wk: torch.Tensor,
    dk: torch.Tensor,
    uk: torch.Tensor,
    sk: float,
    wv: torch.Tensor,
    dv: torch.Tensor,
    uv: torch.Tensor,
    sv: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _FusedCrossAttnKVLoRA.apply(context, wk, dk, uk, sk, wv, dv, uv, sv)


def fused_lora_linear(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Frozen-base Linear plus LoRA with an explicit backward VJP."""

    return _FusedLoRALinear.apply(x, base_weight, lora_down, lora_up, scale)
