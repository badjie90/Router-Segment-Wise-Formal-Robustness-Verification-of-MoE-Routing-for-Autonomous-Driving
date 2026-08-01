from __future__ import annotations

import time
import copy
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class VerificationLayerNorm(nn.Module):
    """LayerNorm expressed with LiRPA-supported primitive operations."""

    def __init__(self, layer: nn.LayerNorm, channels_first_2d: bool = False):
        super().__init__()
        self.normalized_shape = tuple(layer.normalized_shape)
        self.eps = float(layer.eps)
        self.channels_first_2d = channels_first_2d
        if layer.weight is None:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(layer.weight.detach().clone())
            self.bias = nn.Parameter(layer.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_first_2d:
            x = x.permute(0, 2, 3, 1)
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        centered = x - mean
        variance = (centered * centered).mean(dim=dims, keepdim=True)
        inv_std = torch.reciprocal(torch.sqrt(variance + self.eps))
        output = centered * inv_std
        if self.weight is not None:
            output = output * self.weight + self.bias
        if self.channels_first_2d:
            output = output.permute(0, 3, 1, 2)
        return output


class VerificationGlobalResponseNorm(nn.Module):
    """ConvNeXtV2 GRN without the unsupported ONNX ReduceL2 operator."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.eps = float(layer.eps)
        self.spatial_dim = tuple(layer.spatial_dim)
        self.channel_dim = int(layer.channel_dim)
        self.wb_shape = tuple(layer.wb_shape)
        self.weight = nn.Parameter(layer.weight.detach().clone())
        self.bias = nn.Parameter(layer.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sqrt(sum(x^2)) is mathematically identical to norm(p=2), but exports
        # as supported Mul/ReduceSum/Sqrt primitives rather than ReduceL2.
        x_g = torch.sqrt((x * x).sum(dim=self.spatial_dim, keepdim=True))
        denominator = x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps
        x_n = x_g * torch.reciprocal(denominator)
        weight = self.weight.view(self.wb_shape)
        bias = self.bias.view(self.wb_shape)
        return x + bias + weight * (x * x_n)


def make_verification_compatible(model: nn.Module) -> nn.Module:
    """Create an evaluation-equivalent graph using supported bound operators."""
    from auto_LiRPA.operators.gelu import GELU

    converted = copy.deepcopy(model).eval()

    def convert(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            class_name = child.__class__.__name__
            module_name = child.__class__.__module__
            replacement = None
            if class_name == "DropPath" and module_name.startswith("timm"):
                replacement = nn.Identity()  # exact because the graph is in eval mode
            elif isinstance(child, nn.Dropout):
                replacement = nn.Identity()  # exact because the graph is in eval mode
            elif (isinstance(child, nn.GELU)
                  or (class_name == "GELU" and module_name.startswith("timm"))):
                replacement = GELU()  # exact GELU with auto_LiRPA's custom ONNX op
            elif class_name == "GlobalResponseNorm" and module_name.startswith("timm"):
                replacement = VerificationGlobalResponseNorm(child)
            elif isinstance(child, nn.LayerNorm):
                is_2d = class_name.startswith("LayerNorm2d")
                replacement = VerificationLayerNorm(child, channels_first_2d=is_2d)
            if replacement is not None:
                setattr(parent, name, replacement)
            else:
                convert(child)

    convert(converted)
    return converted


def assert_equivalent(reference: nn.Module, converted: nn.Module, x: torch.Tensor,
                      atol: float = 2e-5, rtol: float = 2e-5) -> None:
    with torch.no_grad():
        expected, actual = reference(x), converted(x)
    if not torch.allclose(expected, actual, atol=atol, rtol=rtol):
        error = float((expected - actual).abs().max())
        raise RuntimeError(
            f"Verifier graph is not numerically equivalent to the trained gate "
            f"(maximum absolute error {error:.3e})"
        )


def require_auto_lirpa():
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except Exception as exc:
        raise RuntimeError(
            "auto_LiRPA is required for sound certificates. Install it from the "
            "official Alpha-Beta-CROWN checkout; no approximate fallback is used."
        ) from exc
    return BoundedModule, BoundedTensor, PerturbationLpNorm


def bound_output(model: nn.Module, x: torch.Tensor, eps: float, method: str,
                 bound_opts: Dict[str, Any] | None = None) -> Tuple[np.ndarray, np.ndarray, float]:
    BoundedModule, BoundedTensor, PerturbationLpNorm = require_auto_lirpa()
    # The formal verifier operates on signed backbone feature vectors, not raw
    # pixels. Clipping this box to [0, 1] would exclude the clean feature when
    # a coordinate is negative or greater than one and would invalidate the
    # claimed feature-space certificate.
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=x - eps, x_U=x + eps)
    bounded_x = BoundedTensor(x, ptb)
    bounded_model = BoundedModule(model, (x,), bound_opts=bound_opts or {}, device=x.device)
    start = time.perf_counter()
    lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method=method)
    elapsed = time.perf_counter() - start
    return lb.detach().cpu().numpy(), ub.detach().cpu().numpy(), elapsed


def margin_spec(label: int, classes: int, device: torch.device) -> torch.Tensor:
    rows = []
    for other in range(classes):
        if other == label:
            continue
        row = torch.zeros(classes, device=device)
        row[label], row[other] = 1.0, -1.0
        rows.append(row)
    return torch.stack(rows).unsqueeze(0)


def bound_route_margins(model: nn.Module, x: torch.Tensor, label: int, eps: float,
                        method: str, bound_opts: Dict[str, Any] | None = None):
    BoundedModule, BoundedTensor, PerturbationLpNorm = require_auto_lirpa()
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=x - eps, x_U=x + eps)
    bx = BoundedTensor(x, ptb)
    bounded_model = BoundedModule(model, (x,), bound_opts=bound_opts or {}, device=x.device)
    C = margin_spec(label, 3, x.device)
    start = time.perf_counter()
    lb, ub = bounded_model.compute_bounds(x=(bx,), C=C, method=method)
    return lb.detach().cpu().numpy(), ub.detach().cpu().numpy(), time.perf_counter() - start
