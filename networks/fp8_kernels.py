"""Small fused kernels for FP8 activation caching.

These kernels accelerate the *storage* path only. They do not claim to turn a
BF16 GEMM into an FP8 GEMM; the caller still performs its linear algebra in the
compute dtype.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on CPU-only installations
    triton = None
    tl = None
    TRITON_AVAILABLE = False


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
FP16_TINY = float(torch.finfo(torch.float16).tiny)


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


if TRITON_AVAILABLE:

    @triton.jit
    def _pack_rowwise_fp8_kernel(
        x_ptr,
        q_ptr,
        scale_ptr,
        n_columns,
        x_stride,
        q_stride,
        FP8_MAX_VALUE: tl.constexpr,
        FP16_TINY_VALUE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_columns
        x = tl.load(x_ptr + row * x_stride + columns, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.where(mask, tl.abs(x), 0.0), axis=0)
        scale = tl.maximum(amax / FP8_MAX_VALUE, FP16_TINY_VALUE)
        quantized = tl.minimum(tl.maximum(x / scale, -FP8_MAX_VALUE), FP8_MAX_VALUE)
        tl.store(q_ptr + row * q_stride + columns, quantized, mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _dequantize_fp8_kernel(
        q_ptr,
        scale_ptr,
        output_ptr,
        n_columns,
        q_stride,
        output_stride,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_columns
        quantized = tl.load(q_ptr + row * q_stride + columns, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)
        tl.store(output_ptr + row * output_stride + columns, quantized * scale, mask=mask)

    @triton.jit
    def _dequantize_gelu_backward_fp8_kernel(
        q_ptr,
        scale_ptr,
        grad_activated_ptr,
        activated_ptr,
        grad_pre_ptr,
        n_columns,
        q_stride,
        grad_stride,
        activated_stride,
        grad_pre_stride,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_columns
        quantized = tl.load(q_ptr + row * q_stride + columns, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)
        pre = quantized * scale
        grad_activated = tl.load(
            grad_activated_ptr + row * grad_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # Exact-GELU derivative (approximate="none") evaluated in FP32.
        inv_sqrt_two = 0.7071067811865476
        inv_sqrt_two_pi = 0.3989422804014327
        cdf = 0.5 * (1.0 + tl.erf(pre * inv_sqrt_two))
        activated = pre * cdf
        derivative = cdf + pre * tl.exp(-0.5 * pre * pre) * inv_sqrt_two_pi
        grad_pre = grad_activated * derivative

        tl.store(activated_ptr + row * activated_stride + columns, activated, mask=mask)
        tl.store(grad_pre_ptr + row * grad_pre_stride + columns, grad_pre, mask=mask)

    @triton.jit
    def _dequantize_gelu_grad_pre_fp8_kernel(
        q_ptr,
        scale_ptr,
        grad_activated_ptr,
        grad_pre_ptr,
        n_columns,
        q_stride,
        grad_stride,
        grad_pre_stride,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_columns
        quantized = tl.load(q_ptr + row * q_stride + columns, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)
        pre = quantized * scale
        grad_activated = tl.load(
            grad_activated_ptr + row * grad_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        inv_sqrt_two = 0.7071067811865476
        inv_sqrt_two_pi = 0.3989422804014327
        cdf = 0.5 * (1.0 + tl.erf(pre * inv_sqrt_two))
        derivative = cdf + pre * tl.exp(-0.5 * pre * pre) * inv_sqrt_two_pi
        tl.store(grad_pre_ptr + row * grad_pre_stride + columns, grad_activated * derivative, mask=mask)

    @triton.jit
    def _grad_down2_from_fp8_kernel(
        q_ptr,
        scale_ptr,
        grad_low_rank_ptr,
        output_ptr,
        n_rows,
        n_columns,
        rank,
        q_stride,
        grad_low_stride_rows,
        grad_low_stride_rank,
        output_stride_rank,
        output_stride_columns,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for row_start in range(0, n_rows, BLOCK_K):
            row_offsets = row_start + tl.arange(0, BLOCK_K)
            row_mask = row_offsets < n_rows
            q_mask = row_mask[:, None] & (columns[None, :] < n_columns)
            quantized = tl.load(
                q_ptr + row_offsets[:, None] * q_stride + columns[None, :],
                mask=q_mask,
                other=0.0,
            ).to(tl.float32)
            scales = tl.load(scale_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
            pre = quantized * scales[:, None]
            activated = pre * (0.5 * (1.0 + tl.erf(pre * 0.7071067811865476)))
            grad_low_rank = tl.load(
                grad_low_rank_ptr
                + row_offsets[None, :] * grad_low_stride_rows
                + rows[:, None] * grad_low_stride_rank,
                mask=(rows[:, None] < rank) & row_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(grad_low_rank, activated)

        tl.store(
            output_ptr + rows[:, None] * output_stride_rank + columns[None, :] * output_stride_columns,
            acc,
            mask=(rows[:, None] < rank) & (columns[None, :] < n_columns),
        )


def _can_use_triton(matrix: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> bool:
    return bool(
        TRITON_AVAILABLE
        and matrix.is_cuda
        and matrix.dim() == 2
        and matrix.is_contiguous()
        and matrix.dtype in (torch.bfloat16, torch.float16, torch.float32, FP8_DTYPE)
        and (output_dtype is None or output_dtype in (torch.bfloat16, torch.float16, torch.float32))
    )


def pack_rowwise_fp8_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse rowwise amax, scaling, clamp, and FP8 conversion into one kernel."""

    matrix = x.reshape(-1, x.shape[-1])
    if not _can_use_triton(matrix):
        raise RuntimeError("Triton FP8 pack requires a contiguous CUDA matrix")
    rows, columns = matrix.shape
    quantized = torch.empty((rows, columns), device=x.device, dtype=FP8_DTYPE)
    scale = torch.empty((rows,), device=x.device, dtype=torch.float16)
    block = _next_power_of_two(columns)
    _pack_rowwise_fp8_kernel[(rows,)](
        matrix,
        quantized,
        scale,
        columns,
        matrix.stride(0),
        quantized.stride(0),
        FP8_MAX_VALUE=FP8_MAX,
        FP16_TINY_VALUE=FP16_TINY,
        BLOCK=block,
        num_warps=8 if block >= 1024 else 4,
    )
    return quantized.reshape(x.shape), scale.reshape(*x.shape[:-1], 1)


def unpack_rowwise_fp8_triton(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fuse FP8 load, scale load, dequantization, and output conversion."""

    matrix = quantized.reshape(-1, quantized.shape[-1])
    if not _can_use_triton(matrix, output_dtype=dtype):
        raise RuntimeError("Triton FP8 unpack requires a contiguous CUDA matrix")
    scale_matrix = scale.reshape(-1).contiguous()
    output = torch.empty(matrix.shape, device=matrix.device, dtype=dtype)
    block = _next_power_of_two(matrix.shape[-1])
    _dequantize_fp8_kernel[(matrix.shape[0],)](
        matrix,
        scale_matrix,
        output,
        matrix.shape[-1],
        matrix.stride(0),
        output.stride(0),
        BLOCK=block,
        num_warps=8 if block >= 1024 else 4,
    )
    return output.reshape_as(quantized).to(dtype=dtype)


def dequantize_gelu_backward_fp8_triton(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    grad_activated: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse FP8 dequantization with exact-GELU forward and derivative."""

    q_matrix = quantized.reshape(-1, quantized.shape[-1])
    grad_matrix = grad_activated.reshape(-1, grad_activated.shape[-1])
    if not _can_use_triton(q_matrix, output_dtype=dtype) or not grad_matrix.is_contiguous():
        raise RuntimeError("Triton FP8 GELU backward requires contiguous CUDA matrices")
    scale_matrix = scale.reshape(-1).contiguous()
    activated = torch.empty_like(grad_matrix, dtype=dtype)
    grad_pre = torch.empty_like(grad_matrix, dtype=grad_activated.dtype)
    block = _next_power_of_two(q_matrix.shape[-1])
    _dequantize_gelu_backward_fp8_kernel[(q_matrix.shape[0],)](
        q_matrix,
        scale_matrix,
        grad_matrix,
        activated,
        grad_pre,
        q_matrix.shape[-1],
        q_matrix.stride(0),
        grad_matrix.stride(0),
        activated.stride(0),
        grad_pre.stride(0),
        BLOCK=block,
        num_warps=8 if block >= 1024 else 4,
    )
    return activated.reshape_as(grad_activated), grad_pre.reshape_as(grad_activated)


def dequantize_gelu_grad_pre_fp8_triton(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    grad_activated: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize FP8 pre-GELU and write only the GELU input gradient."""

    q_matrix = quantized.reshape(-1, quantized.shape[-1])
    grad_matrix = grad_activated.reshape(-1, grad_activated.shape[-1])
    if not _can_use_triton(q_matrix, output_dtype=dtype) or not grad_matrix.is_contiguous():
        raise RuntimeError("Triton FP8 GELU gradient requires contiguous CUDA matrices")
    scale_matrix = scale.reshape(-1).contiguous()
    grad_pre = torch.empty_like(grad_matrix, dtype=grad_activated.dtype)
    block = _next_power_of_two(q_matrix.shape[-1])
    _dequantize_gelu_grad_pre_fp8_kernel[(q_matrix.shape[0],)](
        q_matrix,
        scale_matrix,
        grad_matrix,
        grad_pre,
        q_matrix.shape[-1],
        q_matrix.stride(0),
        grad_matrix.stride(0),
        grad_pre.stride(0),
        BLOCK=block,
        num_warps=8 if block >= 1024 else 4,
    )
    return grad_pre.reshape_as(grad_activated)


def grad_down2_from_fp8_triton(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    grad_low_rank: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute ``grad_low_rank.T @ GELU(dequantized_pre)`` without materializing GELU."""

    q_matrix = quantized.reshape(-1, quantized.shape[-1])
    grad_matrix = grad_low_rank.reshape(-1, grad_low_rank.shape[-1])
    if not _can_use_triton(q_matrix, output_dtype=dtype) or not grad_matrix.is_contiguous():
        raise RuntimeError("Triton FP8 direct dDown2 requires contiguous CUDA matrices")
    rows, columns = q_matrix.shape
    rank = grad_matrix.shape[-1]
    output = torch.empty((rank, columns), device=q_matrix.device, dtype=dtype)
    block_m = _next_power_of_two(rank)
    block_n = 128
    block_k = 128
    _grad_down2_from_fp8_kernel[(triton.cdiv(rank, block_m), triton.cdiv(columns, block_n))](
        q_matrix,
        scale.reshape(-1).contiguous(),
        grad_matrix,
        output,
        rows,
        columns,
        rank,
        q_matrix.stride(0),
        grad_matrix.stride(0),
        grad_matrix.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


def kernel_metadata() -> dict[str, Any]:
    return {
        "triton_available": bool(TRITON_AVAILABLE),
        "fp8_dtype": str(FP8_DTYPE),
        "fp8_max": FP8_MAX,
        "fp16_tiny": FP16_TINY,
    }
