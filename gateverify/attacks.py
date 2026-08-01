from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttackResult:
    adversarial: torch.Tensor
    success: torch.Tensor
    queries: torch.Tensor
    status: str = "ok"


def hopskipjump(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
                 steps: int, query_budget: int) -> AttackResult:
    """Foolbox HopSkipJump with the clean route as untargeted reference."""
    # Foolbox imports optional Brendel-Bethge/numba code even when only HSJ is used.
    bb = types.ModuleType("foolbox.attacks.brendel_bethge")
    bb.__spec__ = importlib.machinery.ModuleSpec("foolbox.attacks.brendel_bethge", loader=None)
    class Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Brendel-Bethge attacks are not enabled")
    for name in ("BrendelBethgeAttack", "L0BrendelBethgeAttack", "L1BrendelBethgeAttack",
                 "L2BrendelBethgeAttack", "LinfBrendelBethgeAttack"):
        setattr(bb, name, Unavailable)
    bb.LinfinityBrendelBethgeAttack = Unavailable
    sys.modules.setdefault("foolbox.attacks.brendel_bethge", bb)
    import foolbox as fb
    fmodel = fb.PyTorchModel(model, bounds=(0.0, 1.0), device=x.device)
    attack = fb.attacks.HopSkipJumpAttack(steps=steps)
    try:
        _, clipped, success = attack(
            fmodel, x, fb.criteria.Misclassification(y), epsilons=eps
        )
    except ValueError as exc:
        # Foolbox raises when its random initialization cannot find a point on
        # the other side of the decision boundary. This is an inconclusive
        # attack-initialization failure, not evidence of robustness.
        if "init_attack failed" not in str(exc):
            raise
        return AttackResult(
            x.detach().clone(),
            torch.zeros(len(x), dtype=torch.bool, device=x.device),
            torch.full((len(x),), float(query_budget), device=x.device),
            status="initialization_failed",
        )
    if clipped.ndim == x.ndim + 1:
        clipped, success = clipped[0], success[0]
    # Foolbox does not expose exact per-sample query counts for HSJ.
    queries = torch.full((len(x),), float(query_budget), device=x.device)
    return AttackResult(clipped.detach(), success.detach().bool(), queries)


def _project(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, x0 + eps), x0 - eps).clamp(0, 1)


def square_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
                  queries: int, p_init: float = 0.2, restarts: int = 1) -> AttackResult:
    """Published Square Attack through torchattacks, on raw [0,1] pixels."""
    try:
        from torchattacks.attacks.square import Square
    except ImportError as exc:
        raise RuntimeError("Install torchattacks to run the published Square Attack") from exc
    attack = Square(model, norm="Linf", eps=eps, n_queries=queries, n_restarts=restarts,
                    p_init=p_init, seed=0, verbose=False, loss="margin", resc_schedule=True)
    adv = attack(x, y)
    with torch.no_grad():
        success = model(adv).argmax(1).ne(y)
    # torchattacks does not return per-sample early-stop counts; report budget.
    used = torch.full((len(x),), float(queries * restarts), device=x.device)
    return AttackResult(adv.detach(), success.detach(), used)


class SurrogateRouter(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(192, 128), nn.GELU(), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def pgd_transfer(surrogate: nn.Module, target: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 eps: float, steps: int, step_size: float, restarts: int) -> AttackResult:
    best = x.clone()
    success = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    if step_size <= 0:
        step_size = 2.5 * eps / max(steps, 1)
    for _ in range(restarts):
        adv = _project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(surrogate(adv), y)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = _project(adv.detach() + step_size * grad.sign(), x, eps).detach()
        with torch.no_grad():
            current = target(adv).argmax(1).ne(y)
        choose = current & ~success
        best[choose] = adv[choose]
        success |= current
    return AttackResult(best.detach(), success.detach(), torch.full((len(x),), steps * restarts, device=x.device))
