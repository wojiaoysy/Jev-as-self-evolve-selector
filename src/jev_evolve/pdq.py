"""Jev-conditioned weight residual: W(e) = W0 + P diag(e) Q."""
import math
import copy
import torch
from torch import nn


class PDQLinear(nn.Module):
    """Keep the common starting adapter frozen; train P and Q in float32.

    D is represented as a vector. Feature order and dimension must match the
    saved Jev schema. Multiplication below is exactly P @ diag(e) @ Q, without
    materializing either a diagonal matrix or a dense weight update.
    """
    def __init__(self, base, feature_dim):
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.base = base
        base.requires_grad_(False)
        out_dim, in_dim = base.weight.shape
        self.feature_dim, self.K = feature_dim, 1
        self.R = nn.ParameterList([
            nn.Parameter(torch.zeros(out_dim, feature_dim, device=base.weight.device)),
            nn.Parameter(torch.empty(feature_dim, in_dim, device=base.weight.device)),
        ])
        nn.init.uniform_(self.Q, -1 / math.sqrt(in_dim), 1 / math.sqrt(in_dim))
        self.register_buffer("D", torch.zeros(feature_dim, device=base.weight.device))
        self.register_buffer("accumulated_D", torch.zeros(feature_dim, device=base.weight.device))
        self.pdq_mode = "dynamic"

    @property
    def P(self):
        return self.R[0]

    @property
    def Q(self):
        return self.R[1]

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def set_features(self, features):
        value = torch.as_tensor(features, device=self.D.device, dtype=torch.float32)
        if value.shape != self.D.shape or not torch.isfinite(value).all():
            raise ValueError("Invalid Jev feature vector")
        self.D.copy_(value.detach())

    def select(self, indices):
        if list(indices) not in ([], [0]):
            raise ValueError("PDQ selects P and Q jointly using index 0")
        for p in self.R:
            p.requires_grad_(bool(indices))
            p.grad = None
        return list(self.R) if indices else []

    def forward(self, x):
        y = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = ((x.float() @ self.Q.T) * (self.accumulated_D + self.D)) @ self.P.T
        return (y.float() + delta).to(y.dtype)

    def export(self):
        return {"kind": "pdq", "feature_dim": self.feature_dim,
                "base_adapter": self.base.export(),
                "P": self.P.detach().cpu().clone(), "Q": self.Q.detach().cpu().clone(),
                "D": self.D.detach().cpu().clone(),
                "accumulated_D": self.accumulated_D.detach().cpu().clone(),
                "pdq_mode": self.pdq_mode,
                "R": torch.cat([p.detach().cpu().flatten() for p in self.R])}

    def restore_payload(self, payload):
        for p, key in ((self.P, "P"), (self.Q, "Q")):
            value = payload[key]
            if value.shape != p.shape or not torch.isfinite(value).all():
                raise ValueError("Invalid PDQ checkpoint")
            with torch.no_grad():
                p.copy_(value)
        self.set_features(payload["D"])
        accumulated = payload.get("accumulated_D", torch.zeros_like(self.accumulated_D))
        if accumulated.shape != self.accumulated_D.shape or not torch.isfinite(accumulated).all():
            raise ValueError("Invalid accumulated PDQ state")
        self.accumulated_D.copy_(accumulated.to(self.accumulated_D))
        self.pdq_mode = payload.get("pdq_mode", "dynamic")
        if self.pdq_mode == "accumulate":
            self.select([])

    def basis_id(self):
        return self.base.basis_id()


def configure_agent(agent, adapter, judge_settings, resolved_judge, expected_schema_id=None):
    from .judge import JevClient
    agent.pdq_adapter = adapter
    agent.pdq_judge = JevClient(judge_settings)
    if expected_schema_id is not None and agent.pdq_judge.schema.id != expected_schema_id:
        raise ValueError("PDQ checkpoint feature schema differs from the current judge implementation")
    agent.pdq_resolved_judge = resolved_judge
    adapter.pdq_judge_settings = copy.deepcopy(judge_settings)
    adapter.pdq_resolved_judge = resolved_judge
    adapter.pdq_schema_id = agent.pdq_judge.schema.id
    if len(agent.pdq_judge.schema.names) != adapter.feature_dim:
        raise ValueError("PDQ feature dimension differs from saved judge schema")


def get_features(agent, row):
    """Generate a draft with committed state, without gold answers."""
    adapter = agent.pdq_adapter
    adapter.set_features(torch.zeros_like(adapter.D))
    state = agent.raw_run(row, agent.pdq_judge.settings["token_slots"])
    features, meta = agent.pdq_judge.evaluate(state)
    if meta["model"] != agent.pdq_resolved_judge:
        raise ValueError("PDQ resolved judge changed")
    features = transform_features(features, getattr(adapter, "pdq_variant", "real"),
                                  getattr(adapter, "pdq_shuffle_seed", 42), row)
    return features, meta


def transform_features(features, variant, seed, row):
    """A within-example permutation fixed across checkpoints; no global RNG."""
    import random
    from .io import digest
    values = list(features)
    if variant == "identity":
        return [1.0] * len(values)
    if variant == "shuffle":
        key = digest(["pdq-shuffle-v1", seed, row.get("id"), row["question"]])
        order = list(range(len(values)))
        random.Random(int(key[:16], 16)).shuffle(order)
        return [values[i] for i in order]
    if variant != "real":
        raise ValueError("Unknown PDQ D variant")
    return values


def conditioned_run(agent, row, token_slots):
    adapter = agent.pdq_adapter
    if adapter.pdq_mode == "accumulate":
        # Evaluation reads committed weights; it never learns from dev examples.
        return agent.raw_run(row, token_slots)
    old = adapter.D.detach().clone()
    try:
        features, meta = get_features(agent, row)
        adapter.set_features(features)
        result = agent.raw_run(row, token_slots)
        result["pdq_judge_meta"] = meta
        return result
    finally:
        adapter.set_features(old)


@torch.no_grad()
def accumulate_episode(agent, row, scale=1.0):
    """Commit W_t = W_(t-1) + scale * P D_t Q with fixed P/Q.

    Linearity permits storing sum(scale * D_t) instead of a dense weight matrix.
    The draft/Jev sees the state committed by all preceding episodes.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("PDQ accumulation scale must be finite and positive")
    adapter = agent.pdq_adapter
    adapter.select([])
    old = adapter.D.detach().clone()
    try:
        features, meta = get_features(agent, row)
        value = torch.as_tensor(features, device=adapter.D.device, dtype=adapter.D.dtype)
        if value.shape != adapter.D.shape or not torch.isfinite(value).all():
            raise ValueError("Invalid Jev feature vector")
        candidate = adapter.accumulated_D + scale * value
        if not torch.isfinite(candidate).all():
            raise ValueError("Nonfinite accumulated PDQ state")
        adapter.accumulated_D.copy_(candidate)
        return {"accepted": True, "reason": "fixed_PQ_accumulated_D", "selected": [],
                "accumulation_scale": scale, "accumulated_D_norm": candidate.norm().item()}, meta
    finally:
        adapter.set_features(old)


def training_settings(cfg):
    settings = {**cfg["intervention"], "lr": 1e-4, "steps": 1}
    settings.update(cfg.get("pdq", {}).get("optimizer", {}))
    return settings


def fit(args):
    """Train on exactly the Router training IDs; select epoch on Router val IDs."""
    import random
    from pathlib import Path
    from .cli import settings, experiment_contract
    from .data import load_split
    from .intervention import adapt_direct, Transaction
    from .io import read_json, digest, require_empty, write_json
    from .judge import JevClient
    from .model import load_agent, save_adapter
    cfg = settings(args.config)
    require_empty(args.output)
    epoch_dir = Path(str(args.output) + ".epochs") if getattr(args, "save_epochs", False) else None
    if epoch_dir is not None and epoch_dir.exists() and any(epoch_dir.iterdir()):
        raise ValueError("Epoch snapshot directory is not empty; use a new output path")
    records_dir = Path(args.records)
    rows = {r["id"]: r for r in load_split(args.data, "offline")}
    records = [read_json(p) for p in sorted((records_dir / "episodes").glob("*.json"))]
    judge = JevClient(cfg["judge"])
    agent, origin, identity = load_agent(cfg["model"], cfg["adapter"], args.adapter)
    contract = experiment_contract(cfg, identity, origin.export()["R"], read_json(Path(args.data) / "manifest.json"), judge)
    if read_json(records_dir / "contract.json") != contract:
        raise ValueError("PDQ training records differ from this experiment")
    if not records or any(r["contract_id"] != digest(contract) for r in records):
        raise ValueError("Missing or incompatible PDQ feature records")
    resolved = {r["judge_meta"]["model"] for r in records}
    if len(resolved) != 1:
        raise ValueError("Mixed PDQ judge models")
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    if not train or not val or {r["episode_id"] for r in records} != set(rows):
        raise ValueError("PDQ requires complete offline records with train and validation IDs")
    adapter = PDQLinear(origin, len(judge.schema.names))
    parent, child = cfg["model"]["target_module"].rsplit(".", 1)
    setattr(agent.model.get_submodule(parent), child, adapter)
    configure_agent(agent, adapter, cfg["judge"], resolved.pop())
    epochs = cfg.get("pdq", {}).get("epochs", 3)
    if epochs < 1:
        raise ValueError("PDQ epochs must be positive")
    common_meta = {"contract": contract, "feature_schema_id": judge.schema.id,
        "pdq_settings": cfg.get("pdq", {}), "train_ids": [r["episode_id"] for r in train],
        "val_ids": [r["episode_id"] for r in val]}
    if epoch_dir is not None:
        save_adapter(epoch_dir / "epoch_000.pt", adapter, identity,
                     {**common_meta, "phase": "pdq_training_epoch", "epoch": 0, "history": []})
    best, best_state, history = float("inf"), None, []
    best_epoch = None
    for epoch in range(epochs):
        order = list(train)
        random.Random(cfg["seed"] + epoch).shuffle(order)
        losses = []
        for record in order:
            adapter.set_features(record["jev_features"])
            row = rows[record["episode_id"]]
            result = adapt_direct(agent.model, adapter, [0], lambda: agent.loss(row), training_settings(cfg))
            losses.extend(result["train_losses"])
        with Transaction(agent.model, adapter), torch.no_grad():
            agent.model.eval()
            scores = []
            for record in val:
                adapter.set_features(record["jev_features"])
                scores.append(float(agent.loss(rows[record["episode_id"]])))
        adapter.set_features(torch.zeros_like(adapter.D))
        score = sum(scores) / len(scores)
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite PDQ validation loss")
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), "val_loss": score})
        if score < best:
            best, best_state = score, adapter.export()
            best_epoch = epoch + 1
        if epoch_dir is not None:
            save_adapter(epoch_dir / f"epoch_{epoch+1:03d}.pt", adapter, identity,
                         {**common_meta, "phase": "pdq_training_epoch", "epoch": epoch+1, "history": history})
        print(f"PDQ epoch {epoch+1}/{epochs}: val loss={score:.6f}", flush=True)
    adapter.restore_payload(best_state)
    save_adapter(args.output, adapter, identity, {**common_meta, "phase": "pdq_offline",
                                               "epoch": best_epoch, "history": history})
    write_json(str(args.output) + ".metrics.json", {"history": history,
        "trainable_parameters": sum(p.numel() for p in adapter.R), "feature_dim": adapter.feature_dim})
