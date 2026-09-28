"""Paired retention metrics and target/retention/forgetting learning curves."""
import csv
import statistics
from pathlib import Path
from .io import read_json, write_json


def plot_transfer(root):
    import matplotlib.pyplot as plt
    root = Path(root)
    suite = read_json(root / "learning_curves.json")
    plan = suite["plan"]
    points, reference = [], None
    for point in suite["points"]:
        folder = root / f"seed_{point['seed']}" / point["policy"] / "dev"
        report = read_json(folder / f"episode_{point['episode']:06d}.json")
        retention = report["retention"]
        baseline = read_json(folder / "episode_000000.json")["retention"]
        signature = {k: retention[k] for k in ("dev_id", "protocol", "n")}
        signature["ids"] = [p["id"] for p in retention["predictions"]]
        if reference is None:
            reference = signature
        if signature != reference or retention["dev_id"] != plan["retention"]["id"] or retention["protocol"] != plan["retention"]["protocol"]:
            raise ValueError("Mixed retention sets or protocols")
        predictions, initial = retention["predictions"], baseline["predictions"]
        if [p["id"] for p in initial] != signature["ids"]:
            raise ValueError("Mismatched paired baseline")
        acc = sum(p["correct"] for p in predictions) / len(predictions)
        before = sum(p["correct"] for p in initial) / len(initial)
        frozen = read_json(root / f"seed_{point['seed']}" / "base/dev/episode_000000.json")
        frozen_accuracy = sum(p["correct"] for p in frozen["retention"]["predictions"]) / len(initial)
        points.append({**point, "gsm8k_accuracy": acc, "forgetting_pp": 100 * (before - acc),
                       "gsm8k_drop_vs_base_pp": 100 * (frozen_accuracy - acc),
                       "target_delta_vs_base_pp": 100 * (point["accuracy"] - frozen["accuracy"]),
                       "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in zip(initial, predictions)),
                       "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in zip(initial, predictions))})
    aggregate = []
    for r in suite["aggregate"]:
        row = dict(r)
        selected = [p for p in points if (p["policy"], p["episode"]) == (r["policy"], r["episode"])]
        for metric in ("gsm8k_accuracy", "forgetting_pp", "gsm8k_drop_vs_base_pp", "target_delta_vs_base_pp"):
            values = [p[metric] for p in selected]
            row["mean_" + metric] = statistics.mean(values)
            row["std_" + metric] = statistics.stdev(values) if len(values) > 1 else 0
        aggregate.append(row)
    for name, rows in (("transfer_points.csv", points), ("transfer_mean_std.csv", aggregate)):
        with (root / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(root / "transfer_curves.json", {"plan": plan, "points": points, "aggregate": aggregate})
    write_json(root / "final_dev_summary.json", [r for r in aggregate if r["episode"] == plan["episodes"]])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    colors = dict(base="#555555", router="#0072B2", random="#D55E00", all="#009E73", pdq="#AA4499", router_top2="#56B4E9", pdq_shuffle="#CC6677", pdq_identity="#AA8800")
    for ax, metric, scale, title in zip(axes, ("accuracy", "gsm8k_accuracy", "forgetting_pp", "gsm8k_drop_vs_base_pp"), (100, 100, 1, 1),
            ("Target fixed-dev accuracy (%)", "GSM8K retention accuracy (%)", "GSM8K drop since own episode 0 (pp)", "GSM8K drop vs common frozen base (pp)")):
        for policy, color in colors.items():
            rows = [r for r in aggregate if r["policy"] == policy]
            if not rows:
                continue
            x = [r["episode"] for r in rows]
            means = [scale * r["mean_" + metric] for r in rows]
            std = [scale * r["std_" + metric] for r in rows]
            ax.plot(x, means, label=policy, color=color, marker=".")
            low, high = [m-s for m,s in zip(means,std)], [m+s for m,s in zip(means,std)]
            if metric in ("accuracy", "gsm8k_accuracy"):
                low, high = [max(0,v) for v in low], [min(100,v) for v in high]
            ax.fill_between(x, low, high, color=color, alpha=.15)
        ax.set_title(title)
        ax.set_xlabel("Online episodes processed")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2 if len(plan.get("policies", [])) > 5 else 1)
    axes[2].axhline(0, color="black", linewidth=.6)
    axes[3].axhline(0, color="black", linewidth=.6)
    fig.suptitle("Mean +/- sample SD across seeds; forgetting = episode 0 accuracy - current accuracy", fontsize=10)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(root / f"transfer_curves.{extension}", dpi=180)
    plt.close(fig)
