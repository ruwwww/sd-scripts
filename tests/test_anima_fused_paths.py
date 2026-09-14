from __future__ import annotations

import torch
import torch.nn.functional as F

from networks.fused_lora import fused_lora_linear
from networks.fused_mlp import fused_gelu_mlp, fused_gelu_mlp_fp8, fused_gelu_mlp_lowrank
from networks.lora_anima import LoRAModule


def _linear_reference(x, base, down, up, scale):
    return F.linear(x, base) + F.linear(F.linear(x, down), up) * scale


def _mlp_reference(x, base1, down1, up1, scale1, base2, down2, up2, scale2):
    hidden = _linear_reference(x, base1, down1, up1, scale1)
    hidden = F.gelu(hidden, approximate="none")
    return _linear_reference(hidden, base2, down2, up2, scale2)


def _mlp_inputs(dtype=torch.float64):
    torch.manual_seed(123)
    x = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    base1 = torch.randn(11, 7, dtype=dtype)
    down1 = torch.randn(3, 7, dtype=dtype, requires_grad=True)
    up1 = torch.randn(11, 3, dtype=dtype, requires_grad=True)
    base2 = torch.randn(7, 11, dtype=dtype)
    down2 = torch.randn(2, 11, dtype=dtype, requires_grad=True)
    up2 = torch.randn(7, 2, dtype=dtype, requires_grad=True)
    return x, base1, down1, up1, base2, down2, up2


def test_fused_lora_linear_matches_reference_vjp():
    torch.manual_seed(123)
    x_ref = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    x_test = x_ref.detach().clone().requires_grad_(True)
    base = torch.randn(11, 7, dtype=torch.float64)
    down_ref = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    up_ref = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    down_test = down_ref.detach().clone().requires_grad_(True)
    up_test = up_ref.detach().clone().requires_grad_(True)

    y_ref = _linear_reference(x_ref, base, down_ref, up_ref, 0.625)
    y_test = fused_lora_linear(x_test, base, down_test, up_test, 0.625)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_test.backward(grad)

    assert torch.equal(y_test, y_ref)
    assert torch.allclose(x_test.grad, x_ref.grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(down_test.grad, down_ref.grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(up_test.grad, up_ref.grad, atol=1e-12, rtol=1e-12)


def test_fused_gelu_mlp_matches_reference_vjp():
    inputs_ref = _mlp_inputs()
    inputs_test = [value.detach().clone().requires_grad_(value.requires_grad) for value in inputs_ref]
    x_ref, base1, down1, up1, base2, down2, up2 = inputs_ref
    x_test, _, down1_test, up1_test, _, down2_test, up2_test = inputs_test

    args_ref = (x_ref, base1, down1, up1, 0.5, base2, down2, up2, 0.75)
    args_test = (x_test, base1, down1_test, up1_test, 0.5, base2, down2_test, up2_test, 0.75)
    y_ref = _mlp_reference(*args_ref)
    y_test = fused_gelu_mlp(*args_test)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_test.backward(grad)

    assert torch.equal(y_test, y_ref)
    for candidate, reference in (
        (x_test.grad, x_ref.grad),
        (down1_test.grad, down1.grad),
        (up1_test.grad, up1.grad),
        (down2_test.grad, down2.grad),
        (up2_test.grad, up2.grad),
    ):
        assert torch.allclose(candidate, reference, atol=1e-12, rtol=1e-12)


def test_fp8_and_lowrank_storage_keep_forward_and_gradient_direction():
    inputs_ref = _mlp_inputs(dtype=torch.float32)
    x_ref, base1, down1, up1, base2, down2, up2 = inputs_ref
    grad = torch.randn(2, 3, 7, dtype=torch.float32)
    reference = _mlp_reference(x_ref, base1, down1, up1, 0.5, base2, down2, up2, 0.75)
    reference.backward(grad)
    reference_grad = torch.cat(
        [x_ref.grad.flatten(), down1.grad.flatten(), up1.grad.flatten(), down2.grad.flatten(), up2.grad.flatten()]
    )

    for operation in (
        lambda values: fused_gelu_mlp_fp8(*values, backend="eager"),
        lambda values: fused_gelu_mlp_fp8(*values, backend="eager", store_input_fp8=True),
        lambda values: fused_gelu_mlp_lowrank(*values, rank=4),
    ):
        inputs = [value.detach().clone().requires_grad_(value.requires_grad) for value in inputs_ref]
        x, base1, down1, up1, base2, down2, up2 = inputs
        output = operation((x, base1, down1, up1, 0.5, base2, down2, up2, 0.75))
        output.backward(grad)
        candidate_grad = torch.cat(
            [x.grad.flatten(), down1.grad.flatten(), up1.grad.flatten(), down2.grad.flatten(), up2.grad.flatten()]
        )
        cosine = F.cosine_similarity(reference_grad, candidate_grad, dim=0)

        assert torch.equal(output, _mlp_reference(x, base1, down1, up1, 0.5, base2, down2, up2, 0.75))
        assert torch.isfinite(output).all()
        assert float(cosine) > 0.5


def test_lora_module_fused_path_is_opt_in_and_trainable():
    base = torch.nn.Linear(7, 11, bias=False, dtype=torch.float64)
    base.requires_grad_(False)
    lora = LoRAModule("test", base, lora_dim=3, alpha=2.0)
    lora.apply_to()
    lora.to(dtype=torch.float64)

    assert lora._use_fused_lora is False
    lora.enable_fused_lora()
    x = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    output = lora(x)
    expected = _linear_reference(x, lora.org_forward.__self__.weight, lora.lora_down.weight, lora.lora_up.weight, lora.scale)
    assert torch.equal(output, expected)
    output.square().mean().backward()
    assert lora.lora_down.weight.grad is not None
    assert lora.lora_up.weight.grad is not None
    assert x.grad is not None
