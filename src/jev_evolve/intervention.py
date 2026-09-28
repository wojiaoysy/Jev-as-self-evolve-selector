import math
import random

import numpy as np
import torch


class Transaction:
    """Restore adapter, buffers, modes, grad flags, and RNG; never clone the base weights."""

    def __init__(self, model, adapter):
        self.model, self.adapter = model, adapter
        allowed = {id(p) for p in adapter.R}
        if any(p.requires_grad and id(p) not in allowed for p in model.parameters()):
            raise ValueError("Every non-adapter parameter must be frozen")
        self.values = [p.detach().clone() for p in adapter.R]
        self.flags = [(p, p.requires_grad) for p in model.parameters()]
        self.modes = [(m, m.training) for m in model.modules()]
        self.buffers = [(m, {k: None if v is None else v.detach().clone()
                             for k, v in m._buffers.items()}) for m in model.modules()]
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        self.committed = False

    def __enter__(self):
        return self

    def restore(self, parameters=True):
        if parameters:
            with torch.no_grad():
                for p, old in zip(self.adapter.R, self.values):
                    p.copy_(old)
        for p, flag in self.flags:
            p.requires_grad_(flag)
            p.grad = None
        for module, values in self.buffers:
            module._buffers.clear()
            module._buffers.update({k: None if v is None else v.clone() for k, v in values.items()})
        for module, mode in self.modes:
            module.training = mode
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])
        if self.rng[3] is not None:
            torch.cuda.set_rng_state_all(self.rng[3])

    def __exit__(self, exc_type, exc, tb):
        self.restore(parameters=exc_type is not None or not self.committed)


def checked_score(eval_fn):
    with torch.no_grad():
        value = float(eval_fn())
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite validation score")
    return value


def train_selected(model, adapter, indices, loss_fn, settings):
    if not indices:
        return []
    params = adapter.select(indices)
    if settings["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, lr=settings["lr"], momentum=0)
    elif settings["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=0)
    else:
        raise ValueError("optimizer must be sgd or adamw")
    if settings["lr"] <= 0 or settings["steps"] < 1 or settings["max_update_norm"] <= 0:
        raise ValueError("Invalid update settings")
    old = [p.detach().clone() for p in params]
    losses = []
    # Training mode enables HF gradient checkpointing. Disable module dropout; the
    # supported Qwen config also has attention_dropout=0 (checked at load time).
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    for _ in range(settings["steps"]):
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = loss_fn()
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise FloatingPointError("Invalid training loss")
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, settings["grad_clip"], error_if_nonfinite=True)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        # Orthogonal bases make this the exact Frobenius norm of the weight update.
        with torch.no_grad():
            norm = torch.stack([(p - v).square().sum() for p, v in zip(params, old)]).sum().sqrt()
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite adapter update")
            factor = min(1.0, settings["max_update_norm"] / max(norm.item(), 1e-12))
            for p, v in zip(params, old):
                p.copy_(v + (p - v) * factor)
        losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    return losses


def measure_interventions(model, adapter, loss_fn, eval_fn, settings, order=None):
    order = list(range(adapter.K)) if order is None else list(order)
    if sorted(order) != list(range(adapter.K)):
        raise ValueError("Intervention order must be a permutation of all K indices")
    scores = [0.0] * adapter.K
    losses = [None] * adapter.K
    with Transaction(model, adapter) as tx:
        model.eval()
        baseline = checked_score(eval_fn)
        for i in order:
            tx.restore()
            losses[i] = train_selected(model, adapter, [i], loss_fn, settings)
            model.eval()
            scores[i] = checked_score(eval_fn)
    return {"baseline": baseline, "scores": scores,
            "delta_reward": [s - baseline for s in scores], "train_losses": losses}


def adapt_with_guard(model, adapter, indices, loss_fn, eval_fn, guard_fn, settings,
                     min_gain=1e-5, guard_tolerance=0.0):
    if not indices:
        return {"accepted": False, "reason": "no_positive_route", "selected": []}
    with Transaction(model, adapter) as tx:
        model.eval()
        before = checked_score(eval_fn)
        guard_before = checked_score(guard_fn)
        tx.restore()
        losses = train_selected(model, adapter, indices, loss_fn, settings)
        model.eval()
        after = checked_score(eval_fn)
        guard_after = checked_score(guard_fn)
        accepted = after - before > min_gain and guard_after >= guard_before - guard_tolerance
        tx.committed = accepted
        return {"selected": indices, "accepted": accepted,
                "reason": "accepted" if accepted else "validation_or_guard_rejected",
                "before": before, "after": after, "guard_before": guard_before,
                "guard_after": guard_after, "train_losses": losses}


def adapt_direct(model, adapter, indices, loss_fn, settings):
    """Commit every finite selected update; no probe/guard scores are consulted.

    Transaction cleanup restores modes and RNG. Only an execution error rolls
    parameters back; a numerically valid update is never rejected by its gain.
    """
    if not indices:
        return {"accepted": False, "reason": "no_route", "selected": []}
    with Transaction(model, adapter) as tx:
        losses = train_selected(model, adapter, indices, loss_fn, settings)
        tx.committed = True
        return {"accepted": True, "reason": "direct_update", "selected": indices,
                "train_losses": losses}
