import torch

from gateverify.metrics import cosine_similarity, js_divergence, relative_l2


def test_identity_metrics():
    x = torch.randn(4, 8)
    assert (abs(cosine_similarity(x, x) - 1) < 1e-6).all()
    assert (relative_l2(x, x) == 0).all()
    p = torch.softmax(torch.randn(4, 3), dim=1)
    assert (abs(js_divergence(p, p)) < 1e-7).all()

