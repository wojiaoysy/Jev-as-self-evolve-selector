import copy

import torch
from torch import nn
from torch.nn import functional as F


def delta_to_target(delta, floor=1e-5):
    positive = torch.where(delta > floor, delta, torch.zeros_like(delta))
    maximum = positive.amax(dim=-1, keepdim=True)
    return positive / maximum.clamp_min(1e-12)


def pairwise_ranking_loss(logits, delta, floor=1e-5):
    indices = torch.triu_indices(logits.shape[-1], logits.shape[-1], offset=1, device=logits.device)
    difference = delta[:, indices[0]] - delta[:, indices[1]]
    mask = difference.abs() > floor
    if not mask.any():
        return logits.sum() * 0  # remains differentiable for all-tied batches
    predicted = logits[:, indices[0]] - logits[:, indices[1]]
    return F.binary_cross_entropy_with_logits(predicted[mask], (difference[mask] > 0).float())


class Router(nn.Module):
    def __init__(self, input_dim, K, hidden=128):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, K))

    def logits(self, features):
        return self.net(((features - self.mean) / self.scale).clamp(-10, 10))

    def forward(self, features):
        return self.logits(features).sigmoid()


def select_topk(alpha, top_k=2, threshold=0.5):
    alpha = alpha.detach()
    if alpha.ndim != 1 or not torch.isfinite(alpha).all() or not 0 <= threshold <= 1:
        raise ValueError("Invalid route")
    if not 1 <= top_k <= alpha.numel():
        raise ValueError("Invalid top_k")
    return [int(i) for i in torch.argsort(alpha, descending=True, stable=True)[:top_k]
            if float(alpha[i]) > threshold]


def select_forced_top2(alpha):
    if alpha.ndim != 1 or alpha.numel() < 2 or not torch.isfinite(alpha).all():
        raise ValueError("Forced Top-2 needs at least two finite scores")
    return torch.argsort(alpha, descending=True, stable=True)[:2].tolist()


def objective(router, x, delta, settings):
    logits = router.logits(x)
    alpha = logits.sigmoid()
    regression = F.smooth_l1_loss(alpha, delta_to_target(delta, settings["gain_floor"]))
    ranking = pairwise_ranking_loss(logits, delta, settings["gain_floor"])
    return regression + settings["rank_weight"] * ranking + settings["sparsity_weight"] * alpha.mean()


@torch.no_grad()
def metrics(router, x, delta, settings):
    alpha = router(x)
    selected, regret, positives = [], [], []
    for a, d in zip(alpha, delta):
        ids = select_topk(a, settings["top_k"], settings["threshold"])
        selected.append(len(ids))
        best = max(0.0, float(d.max()))
        chosen = max(0.0, float(d[ids].max())) if ids else 0.0
        regret.append(best - chosen)
        positives.extend([float(d[i] > settings["gain_floor"]) for i in ids])
    return {"loss": float(objective(router, x, delta, settings)),
            "mean_best_single_regret": sum(regret) / len(regret),
            "mean_selected": sum(selected) / len(selected),
            "selected_positive_fraction": sum(positives) / len(positives) if positives else None,
            "positive_episode_fraction": float((delta.max(-1).values > settings["gain_floor"]).float().mean())}


def train_router(train_records, val_records, settings, seed=42):
    if not train_records or not val_records:
        raise ValueError("Need disjoint nonempty router train/validation episode sets")
    torch.manual_seed(seed)
    x = torch.tensor([r["jev_features"] for r in train_records], dtype=torch.float32)
    d = torch.tensor([r["delta_reward"] for r in train_records], dtype=torch.float32)
    vx = torch.tensor([r["jev_features"] for r in val_records], dtype=torch.float32)
    vd = torch.tensor([r["delta_reward"] for r in val_records], dtype=torch.float32)
    if not all(torch.isfinite(t).all() for t in (x, d, vx, vd)):
        raise ValueError("Non-finite router data")
    router = Router(x.shape[1], d.shape[1], settings["hidden"])
    router.mean.copy_(x.mean(0))
    router.scale.copy_(x.std(0, unbiased=False).clamp_min(0.01))
    optimizer = torch.optim.AdamW(router.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(settings["epochs"]):
        router.train()
        permutation = torch.randperm(len(x))
        for start in range(0, len(x), settings["batch_size"]):
            index = permutation[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(router, x[index], d[index], settings)
            loss.backward()
            optimizer.step()
        router.eval()
        report = metrics(router, vx, vd, settings)
        history.append({"epoch": epoch + 1, **report})
        if report["loss"] < best_loss:
            best_loss, best_state = report["loss"], copy.deepcopy(router.state_dict())
    if best_state is None:
        raise ValueError("No router epochs ran")
    router.load_state_dict(best_state)
    return router, history
