import torch
from torch import nn

from .io import tensor_digest


class OrthogonalSubspaceLinear(nn.Module):
    """A fixed SVD basis; the routing decision changes training, never memory."""

    def __init__(self, base: nn.Linear, K=8, rank=8, basis=None):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Target must be an unquantized torch.nn.Linear")
        if K < 1 or rank < 1 or K * rank > min(base.weight.shape):
            raise ValueError("Require 1 <= K * rank <= min(d_in, d_out)")
        self.base, self.K, self.rank = base, K, rank
        self.base.requires_grad_(False)
        if basis is None:
            # One selected projection only. CPU fp32 SVD avoids a large GPU workspace.
            U, _, Vh = torch.linalg.svd(base.weight.detach().float().cpu(), full_matrices=False)
            U = U[:, :K * rank].reshape(base.out_features, K, rank).permute(1, 0, 2)
            V = Vh[:K * rank].T.reshape(base.in_features, K, rank).permute(1, 0, 2)
        else:
            U, V = basis
        if tuple(U.shape) != (K, base.out_features, rank) or tuple(V.shape) != (K, base.in_features, rank):
            raise ValueError("Incompatible basis dimensions")
        for matrix in (U, V):
            flat = matrix.permute(1, 0, 2).reshape(matrix.shape[1], K * rank).float()
            if not torch.allclose(flat.T @ flat, torch.eye(K * rank, device=flat.device), atol=2e-4, rtol=2e-4):
                raise ValueError("Basis is not orthonormal")
        device = base.weight.device
        self.register_buffer("U", U.to(device=device, dtype=torch.float32).contiguous())
        self.register_buffer("V", V.to(device=device, dtype=torch.float32).contiguous())
        self.R = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, rank, device=device, dtype=torch.float32), requires_grad=False)
            for _ in range(K)
        ])

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def select(self, indices):
        indices = list(indices)
        if len(set(indices)) != len(indices) or any(i < 0 or i >= self.K for i in indices):
            raise ValueError("Invalid or repeated subspace index")
        for i, parameter in enumerate(self.R):
            parameter.grad = None
            parameter.requires_grad_(i in indices)
        return [self.R[i] for i in indices]

    def forward(self, x):
        y = self.base(x)
        # fp32 master R and fp32 adapter arithmetic survive small updates with a bf16 base.
        # All K contributions stay present even when only a subset can receive gradients.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            delta = torch.zeros_like(y, dtype=torch.float32)
            for i in range(self.K):
                delta = delta + (xf @ self.V[i] @ self.R[i].T) @ self.U[i].T
            return (y.float() + delta).to(y.dtype)

    def export(self):
        return {"U": self.U.detach().cpu(), "V": self.V.detach().cpu(),
                "R": torch.stack([p.detach().cpu() for p in self.R]),
                "K": self.K, "rank": self.rank}

    def import_R(self, values):
        if tuple(values.shape) != (self.K, self.rank, self.rank) or not torch.isfinite(values).all():
            raise ValueError("Invalid R checkpoint")
        with torch.no_grad():
            for p, v in zip(self.R, values):
                p.copy_(v)

    def basis_id(self):
        return tensor_digest([("U", self.U), ("V", self.V)])


def attach_adapter(model, path, K, rank, payload=None):
    model.requires_grad_(False)
    parent_path, child = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    base = getattr(parent, child)
    adapter = OrthogonalSubspaceLinear(base, K, rank,
                                       None if payload is None else (payload["U"], payload["V"]))
    if payload is not None:
        adapter.import_R(payload["R"])
    setattr(parent, child, adapter)
    return adapter
