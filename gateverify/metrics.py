from __future__ import annotations

import numpy as np
import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a, b = a.flatten(1), b.flatten(1)
    return torch.nn.functional.cosine_similarity(a, b, dim=1).detach().cpu().numpy()


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a, b = a.flatten(1), b.flatten(1)
    return ((a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)).detach().cpu().numpy()


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(1).double(), b.flatten(1).double()
    a, b = a - a.mean(0), b - b.mean(0)
    hsic = (a.T @ b).pow(2).sum()
    denom = (a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt()
    return float((hsic / denom.clamp_min(1e-18)).cpu())


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> np.ndarray:
    m = 0.5 * (p + q)
    kl1 = (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(1)
    kl2 = (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(1)
    return (0.5 * (kl1 + kl2)).detach().cpu().numpy()

