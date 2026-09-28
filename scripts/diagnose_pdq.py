"""Read-only CPU diagnostics of stored training D and checkpoint P/Q.

No model loading, generation, Jev requests, or parameter updates.
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch
from jev_evolve.io import digest, load_torch, read_json, tensor_digest, write_json


EPS = 1e-12


def stats(values):
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    x = x[torch.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "std": None, "min": None, "median": None, "p90": None, "max": None}
    return {"n": len(x), "mean": x.mean().item(), "std": x.std(unbiased=False).item(),
            "min": x.min().item(), "median": torch.quantile(x, .5).item(),
            "p90": torch.quantile(x, .9).item(), "max": x.max().item()}


def spectrum(eigenvalues):
    values = eigenvalues.clamp_min(0).sort(descending=True).values
    total = values.sum().item()
    if total <= EPS ** 2:
        return {"variance_energy": total, "effective_rank": None, "participation_rank": None,
                "components_90pct": 0, "top1_energy_fraction": None, "energy_fractions": []}
    p = values / total
    nonzero = p[p > 1e-15]
    return {"variance_energy": total, "effective_rank": math.exp(float(-(nonzero * nonzero.log()).sum())),
            "participation_rank": float(1 / p.square().sum()),
            "components_90pct": int(torch.searchsorted(p.cumsum(0), torch.tensor(.9, dtype=p.dtype))) + 1,
            "top1_energy_fraction": p[0].item(), "energy_fractions": p.tolist()}


def cosine(inner, norm_a, norm_b):
    denominator = norm_a * norm_b
    return torch.where((norm_a > EPS) & (norm_b > EPS),
                       (inner / denominator.clamp_min(EPS ** 2)).clamp(-1, 1),
                       torch.full_like(inner, float("nan")))


def geometry(E, P, Q, global_c):
    """Exact Frobenius geometry without allocating n dense delta-W matrices."""
    E, P, Q = E.double(), P.double(), Q.double()
    n, m = E.shape
    G = (P.T @ P) * (Q @ Q.T)
    G = (G + G.T) / 2
    Kd, Kw = E @ E.T, E @ G @ E.T
    Kw = (Kw + Kw.T) / 2
    nd = Kd.diag().clamp_min(0).sqrt()
    nw = Kw.diag().clamp_min(0).sqrt()
    ones = torch.ones(m, dtype=E.dtype)
    pq_norm2 = (ones @ G @ ones).clamp_min(0)
    cD = E.mean(1)
    d_cos_I = cosine(E.sum(1), nd, torch.full_like(nd, math.sqrt(m)))
    d_res = torch.where(nd > EPS, (E - cD[:, None]).norm(dim=1) / nd.clamp_min(EPS), torch.nan)
    weight_dot_pq = E @ G @ ones
    w_cos_pq = cosine(weight_dot_pq, nw, pq_norm2.sqrt().expand_as(nw))
    cW = weight_dot_pq / pq_norm2 if pq_norm2 > EPS ** 2 else torch.full_like(nw, torch.nan)
    def relative_weight_residual(c):
        squared = torch.einsum("ni,ij,nj->n", E - c[:, None], G, E - c[:, None]).clamp_min(0)
        return torch.where(nw > EPS, squared.sqrt() / nw.clamp_min(EPS), torch.nan)
    w_res_cD = relative_weight_residual(cD)
    w_res_best = relative_weight_residual(cW)
    w_res_global = relative_weight_residual(torch.full_like(cD, global_c))
    pairs = torch.triu_indices(n, n, offset=1)
    i, j = pairs
    d_cos = cosine(Kd[i, j], nd[i], nd[j])
    w_cos = cosine(Kw[i, j], nw[i], nw[j])
    d_angle = torch.rad2deg(torch.acos(d_cos))
    w_angle = torch.rad2deg(torch.acos(w_cos))
    Ec = E - E.mean(0, keepdim=True)
    centered_w = Ec @ G @ Ec.T
    centered_w = (centered_w + centered_w.T) / 2
    centered_norms = centered_w.diag().clamp_min(0).sqrt()
    centered_cos = cosine(centered_w[i, j], centered_norms[i], centered_norms[j])
    raw_energy = nw.square().mean()
    centered_energy = centered_norms.square().mean()
    summary = {
        "n": n, "feature_dim": m, "global_c_from_train": global_c,
        "D_fro_norm": stats(nd), "D_spectral_norm": stats(E.abs().amax(1)),
        "D_within_sample_variance": stats(E.var(1, unbiased=False)),
        "D_across_sample_variance": stats(E.var(0, unbiased=False)),
        "D_near_constant_columns": int((E.std(0, unbiased=False) <= 1e-6).sum()),
        "D_cosine_I": stats(d_cos_I), "D_relative_residual_best_cI": stats(d_res),
        "D_best_c": stats(cD), "D_pairwise_cosine": stats(d_cos), "D_pairwise_angle_deg": stats(d_angle),
        "D_centered_spectrum": spectrum(torch.linalg.svdvals(Ec).square() / n),
        "deltaW_fro_norm": stats(nw), "PQ_fro_norm": float(pq_norm2.sqrt()),
        "deltaW_cosine_PQ": stats(w_cos_pq), "deltaW_best_cW": stats(cW),
        "deltaW_relative_residual_using_cD": stats(w_res_cD),
        "deltaW_relative_residual_best_cW": stats(w_res_best),
        "deltaW_relative_residual_global_c": stats(w_res_global),
        "deltaW_pairwise_cosine": stats(w_cos), "deltaW_pairwise_angle_deg": stats(w_angle),
        "deltaW_centered_pairwise_angle_deg": stats(torch.rad2deg(torch.acos(centered_cos))),
        "deltaW_centered_spectrum": spectrum(torch.linalg.eigvalsh(centered_w) / n),
        "deltaW_centered_energy_fraction": float(centered_energy / raw_energy) if raw_energy > EPS ** 2 else None,
        "zero_deltaW_count": int((nw <= EPS).sum()),
        "norm_note": "absolute Frobenius norms; base W0 is not loaded, so no deltaW/W0 norm ratio",
    }
    per_sample = [{"D_fro_norm": float(nd[k]), "D_spectral_norm": float(E[k].abs().max()),
        "D_within_variance": float(E[k].var(unbiased=False)), "cD": float(cD[k]),
        "D_cosine_I": float(d_cos_I[k]), "D_relative_residual_best_cI": float(d_res[k]),
        "deltaW_fro_norm": float(nw[k]), "deltaW_cosine_PQ": float(w_cos_pq[k]),
        "cW": float(cW[k]), "deltaW_relative_residual_using_cD": float(w_res_cD[k]),
        "deltaW_relative_residual_best_cW": float(w_res_best[k])} for k in range(n)]
    pair_rows = [{"i": int(i[k]), "j": int(j[k]), "D_cosine": float(d_cos[k]),
                  "D_angle_deg": float(d_angle[k]), "deltaW_cosine": float(w_cos[k]),
                  "deltaW_angle_deg": float(w_angle[k])} for k in range(len(i))]
    return summary, per_sample, pair_rows


def dump_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()} for row in rows)


def feature_rows(E, names):
    lookup = {name: i for i, name in enumerate(names)}
    result = []
    for j, name in enumerate(names):
        presence = lookup.get(name.rsplit(".", 1)[0] + ".present")
        mask = E[:, presence] > 0 if presence is not None else torch.ones(len(E), dtype=torch.bool)
        active = E[mask, j]
        result.append({"feature": name, "is_presence": name.endswith(".present"),
            "mean": float(E[:, j].mean()), "variance": float(E[:, j].var(unbiased=False)),
            "std": float(E[:, j].std(unbiased=False)), "min": float(E[:, j].min()), "max": float(E[:, j].max()),
            "zero_fraction": float((E[:, j] == 0).double().mean()), "available_count": int(mask.sum()),
            "available_mean": float(active.mean()) if len(active) else None,
            "available_variance": float(active.var(unbiased=False)) if len(active) else None})
    return result


def plot(output, summaries, samples, pair_rows):
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    def values(rows, key):
        return [r[key] for r in rows if math.isfinite(r[key])]
    for split, color in (("train", "#0072B2"), ("val", "#D55E00")):
        for ax, key, title, data in (
            (axes[0, 0], "D_fro_norm", "D Frobenius norm", samples),
            (axes[0, 1], "D_cosine_I", "D cosine with I", samples),
            (axes[0, 2], "deltaW_fro_norm", "Weight residual Frobenius norm", samples),
            (axes[1, 0], "deltaW_angle_deg", "Pairwise weight residual angles (degrees)", pair_rows),
            (axes[1, 1], "deltaW_relative_residual_best_cW", "Relative residual after best scalar times PQ", samples)):
            v = values(data[split], key)
            if v:
                ax.hist(v, bins=25, alpha=.5, color=color, label=split, density=True)
            ax.set_title(title)
        p = summaries[split]["deltaW_centered_spectrum"]["energy_fractions"]
        if p:
            axes[1, 2].plot(range(1, len(p) + 1), torch.tensor(p).cumsum(0).tolist(), label=split, color=color)
    axes[1, 2].set_title("Centered weight variation: cumulative energy")
    axes[1, 2].set_xlabel("Number of components")
    for ax in axes.flat:
        ax.grid(alpha=.15)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.suptitle("Stored offline D, mapped through this P/Q checkpoint; pairwise values are not independent samples")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"diagnostics.{ext}", dpi=160)
    plt.close(fig)


def run(records_dir, checkpoint_path, output, make_plot=True):
    records_dir, checkpoint_path, output = map(Path, (records_dir, checkpoint_path, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty diagnostics output directory")
    schema = read_json(records_dir / "feature_schema.json")
    contract = read_json(records_dir / "contract.json")
    records = [read_json(p) for p in sorted((records_dir / "episodes").glob("*.json"))]
    payload = load_torch(checkpoint_path)
    adapter, meta = payload["adapter"], payload["metadata"]
    if adapter.get("kind") != "pdq":
        raise ValueError("Expected a PDQ checkpoint, not router.pt or an R-subspace checkpoint")
    if meta.get("contract") != contract or meta.get("feature_schema_id") != schema["id"]:
        raise ValueError("PDQ checkpoint and offline records have different contracts/schemas; use pdq_offline.pt or its saved epochs")
    if not records or any(r["contract_id"] != digest(contract) for r in records):
        raise ValueError("Missing or mixed offline feature records")
    ids = [r["episode_id"] for r in records]
    if len(set(ids)) != len(ids) or any(r["split"] not in ("train", "val") for r in records):
        raise ValueError("Duplicate IDs or invalid split")
    for split in ("train", "val"):
        if set(meta[split + "_ids"]) != {r["episode_id"] for r in records if r["split"] == split}:
            raise ValueError("Incomplete/mismatched training or validation record IDs")
    if any(r["judge_meta"]["model"] != meta["pdq_resolved_judge"] for r in records):
        raise ValueError("Resolved Jev model mismatch")
    E = torch.tensor([r["jev_features"] for r in records], dtype=torch.float64)
    P, Q = adapter["P"].double(), adapter["Q"].double()
    names = schema["names"]
    if (E.ndim != 2 or E.shape[1] != len(names) or P.ndim != 2 or Q.ndim != 2 or
            P.shape[1] != E.shape[1] or Q.shape[0] != E.shape[1] or
            not all(torch.isfinite(x).all() for x in (E, P, Q))):
        raise ValueError("Invalid feature/P/Q dimensions or nonfinite values")
    train_index = [i for i, r in enumerate(records) if r["split"] == "train"]
    c0 = float(E[train_index].mean())
    summaries, samples, pairs = {}, {}, {}
    for split in ("train", "val"):
        index = [i for i, r in enumerate(records) if r["split"] == split]
        if len(index) < 2:
            raise ValueError("At least two rows per split required for angle diagnostics")
        matrix = E[index]
        summary, per_sample, pair_rows = geometry(matrix, P, Q, c0)
        features = feature_rows(matrix, names)
        nonmask = [i for i, name in enumerate(names) if not name.endswith(".present")]
        centered = matrix[:, nonmask] - matrix[:, nonmask].mean(0) if nonmask else None
        summary["D_nonpresence_centered_spectrum"] = spectrum(torch.linalg.svdvals(centered).square() / len(index)) if nonmask else None
        summary["presence_feature_count"] = len(names) - len(nonmask)
        summary["available_variance_note"] = "feature_stats.csv excludes padded rows when a corresponding .present feature exists"
        summaries[split] = summary
        for k, item in enumerate(per_sample):
            item["episode_id"] = records[index[k]]["episode_id"]
        for item in pair_rows:
            item["episode_id_i"] = records[index[item.pop("i")]]["episode_id"]
            item["episode_id_j"] = records[index[item.pop("j")]]["episode_id"]
        samples[split], pairs[split] = per_sample, pair_rows
        dump_csv(output / split / "sample_metrics.csv", per_sample)
        dump_csv(output / split / "pairwise_angles.csv", pair_rows)
        dump_csv(output / split / "feature_stats.csv", features)
    summary = {"format": 1, "records": str(records_dir.resolve()), "checkpoint": str(checkpoint_path.resolve()),
        "feature_schema_id": schema["id"], "contract_id": digest(contract), "feature_records_hash": digest(records),
        "PQ_hash": tensor_digest([("P", P), ("Q", Q)]), "phase": meta.get("phase"), "epoch": meta.get("epoch"),
        "history": meta.get("history"), "zero_tolerance": EPS,
        "interpretation": "Geometric diversity, not mutual information or proof of predictive usefulness. Offline stored D is fixed during P/Q training. Weight geometry uses this checkpoint only.",
        "formula": "G=(P^T P) elementwise-multiplied by (Q Q^T); <deltaW_i,deltaW_j>=e_i^T G e_j",
        "splits": summaries}
    write_json(output / "summary.json", summary)
    if make_plot:
        plot(output, summaries, samples, pairs)
    print(f"Saved diagnostics to {output}", flush=True)
    for split, s in summaries.items():
        print(split, "D-cI residual:", s["D_relative_residual_best_cI"]["median"],
              "deltaW angle:", s["deltaW_pairwise_angle_deg"]["median"],
              "weight centered energy:", s["deltaW_centered_energy_fraction"], flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True, help="Original seed_N/phase1 directory")
    p.add_argument("--checkpoint", required=True, help="Corresponding pdq_offline.pt or saved training epoch")
    p.add_argument("--output", required=True)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    if a.threads < 1:
        p.error("threads must be positive")
    torch.set_num_threads(a.threads)
    run(a.records, a.checkpoint, a.output, not a.no_plot)


if __name__ == "__main__":
    main()
