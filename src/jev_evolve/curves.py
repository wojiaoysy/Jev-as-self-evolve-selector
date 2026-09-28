"""Fixed held-out dev data, observational checkpoints, and seed aggregation."""
import csv
import hashlib
import inspect
import os
import random
import statistics
import time
from pathlib import Path

import requests

from .data import clean_solution, correct, extract_answer, load_split
from .intervention import Transaction
from .io import digest, load_torch, read_json, tensor_digest, write_json
from .model import SYSTEM, save_adapter
from .tasks import SYSTEMS, parse


def parser_id():
    from . import data
    return digest([data.NUMBER, inspect.getsource(extract_answer), inspect.getsource(correct)])


def eval_protocol(cfg, task=None):
    task = task or cfg["model"].get("task", "gsm8k")
    return {"system": SYSTEMS[task], "max_length": cfg["model"]["max_length"],
            "max_new_tokens": cfg["model"]["max_new_tokens"], "do_sample": False,
            "dtype": cfg["model"]["dtype"], "attention": cfg["model"]["attention"],
            "parser_id": parser_id() if task == "gsm8k" else digest(inspect.getsource(parse))}


def prepare_dev(data_dir, output, size=64, seed=2026, source_file=None):
    """Keep the existing data manifest untouched so existing routers remain usable."""
    import json
    manifest = read_json(Path(data_dir) / "manifest.json")
    if size < 1:
        raise ValueError("dev size must be positive")
    if Path(output).exists():
        saved = read_json(output)
        load_dev(output, data_dir)
        if saved["seed"] != seed or len(saved["rows"]) != size:
            raise ValueError("Existing dev file has a different size/seed; choose another path")
        return saved
    if source_file:
        payload = Path(source_file).read_bytes()
    else:
        revision = manifest["source_revision"]
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/{revision}/grade_school_math/data/train.jsonl"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        payload = response.content
    if hashlib.sha256(payload).hexdigest() != manifest["source_sha256"]["train"]:
        raise ValueError("Raw train source differs from the existing dataset; provide the original --source-file")
    used_ids, used_questions = set(), set()
    for split in manifest["split_hashes"]:
        for row in load_split(data_dir, split):
            used_ids.add(row["id"])
            used_questions.add(row["question"].strip())
    candidates = []
    for i, line in enumerate(payload.decode("utf-8").splitlines()):
        row = json.loads(line)
        row_id = f"gsm8k-train-{i}"
        question = row["question"].strip()
        if row_id in used_ids or question in used_questions:
            continue
        used_questions.add(question)
        if extract_answer(row["answer"]) is None:
            raise ValueError("Invalid reference answer in dev source")
        candidates.append({"id": row_id, "question": row["question"], "answer": clean_solution(row["answer"])})
    if len(candidates) < size:
        raise ValueError("Not enough unused training examples for the dev split")
    random.Random(seed).shuffle(candidates)
    result = {"format": 1, "parent_data_id": manifest["id"], "seed": seed,
              "source_sha256": manifest["source_sha256"]["train"], "rows": candidates[:size]}
    result["id"] = digest(result)
    write_json(output, result)
    return result


def load_dev(path, data_dir):
    saved = read_json(path)
    body = {k: v for k, v in saved.items() if k != "id"}
    if saved.get("id") != digest(body) or not saved.get("rows"):
        raise ValueError("Invalid/tampered dev file")
    manifest = read_json(Path(data_dir) / "manifest.json")
    if saved["parent_data_id"] != manifest["id"]:
        raise ValueError("Dev belongs to a different dataset")
    ids, questions = set(), set()
    for split in manifest["split_hashes"]:
        for row in load_split(data_dir, split):
            ids.add(row["id"])
            questions.add(row["question"].strip())
    dev_ids, dev_questions = set(), set()
    for row in saved["rows"]:
        if row.get("task", "gsm8k") != manifest.get("task", "gsm8k"):
            raise ValueError("Dev task differs from its parent dataset")
        q = row["question"].strip()
        if row["id"] in ids or q in questions or row["id"] in dev_ids or q in dev_questions:
            raise ValueError("Dev overlaps another split or contains duplicate examples")
        if parse(row["answer"], row.get("task", "gsm8k")) is None:
            raise ValueError("Dev contains an invalid reference answer")
        dev_ids.add(row["id"])
        dev_questions.add(q)
    return saved


class DevMonitor:
    def __init__(self, out, agent, adapter, identity, run_contract, dev, interval, cfg, seed, retention=None):
        self.out, self.agent, self.adapter = Path(out), agent, adapter
        self.identity, self.run_contract = identity, run_contract
        self.dev, self.interval, self.cfg, self.seed = dev, interval, cfg, seed
        self.retention = retention

    def record(self, events, final=False):
        step = len(events)
        if step % self.interval and not final:
            return
        checkpoint_path = self.out / "checkpoints" / f"episode_{step:06d}.pt"
        report_path = self.out / "dev" / f"episode_{step:06d}.json"
        def state_hash(payload):
            values = [("R", payload["R"])]
            if payload.get("pdq_mode") == "accumulate":
                values += [("accumulated_D", payload["accumulated_D"]), ("D", payload["D"])]
            return tensor_digest(values)
        r_hash = state_hash(self.adapter.export())
        if checkpoint_path.exists():
            saved = load_torch(checkpoint_path)
            if (saved["metadata"]["run_contract"] != self.run_contract or
                    state_hash(saved["adapter"]) != r_hash):
                raise ValueError("Historical checkpoint does not match current progress")
        else:
            save_adapter(checkpoint_path, self.adapter, self.identity,
                         {"run_contract": self.run_contract, "events": events})
        if report_path.exists():
            report = read_json(report_path)
            if report["R_hash"] != r_hash or report["run_id"] != digest(self.run_contract):
                raise ValueError("Existing dev report does not match current progress")
        else:
            start = time.monotonic()
            reused = False
            baseline_path = self.out / "dev" / "episode_000000.json"
            if self.run_contract["policy"] in ("base", "none") and step and baseline_path.exists():
                baseline = read_json(baseline_path)
                if baseline["R_hash"] != r_hash:
                    raise ValueError("Frozen base unexpectedly changed")
                predictions = baseline["predictions"]
                reused = True
            else:
                # Observational only: dev evaluation cannot alter model/RNG state.
                with Transaction(self.agent.model, self.adapter):
                    predictions = self.agent.evaluate(self.dev["rows"])
            report = {"episode": step, "seed": self.seed,
                      "policy": "base" if self.run_contract["policy"] == "none" else self.run_contract["policy"],
                      "run_id": digest(self.run_contract), "dev_id": self.dev["id"],
                      "identity": self.identity, "protocol": eval_protocol(self.cfg),
                      "n": len(predictions), "accuracy": sum(p["correct"] for p in predictions) / len(predictions),
                      "parse_failures": sum(not p["parsed"] for p in predictions),
                      "accepted": sum(e["accepted"] for e in events),
                      "selected_total": sum(len(e["selected"]) for e in events),
                      "R_hash": r_hash, "checkpoint": str(checkpoint_path),
                      "elapsed_seconds": time.monotonic() - start, "reused_frozen_base": reused,
                      "predictions": predictions}
            report["inference"] = "draft_then_jev_then_pdq" if hasattr(self.agent, "pdq_judge") else "single_pass"
            if getattr(self.adapter, "pdq_mode", None) == "accumulate":
                report["inference"] = "single_pass_accumulated_pdq"
            report["update_mode"] = self.run_contract.get("update_mode", "guarded")
            if hasattr(self.agent, "pdq_judge"):
                report["pdq_variant"] = getattr(self.adapter, "pdq_variant", "real")
                report["pdq_shuffle_seed"] = getattr(self.adapter, "pdq_shuffle_seed", 42)
            if self.retention is not None:
                if reused:
                    retained = baseline["retention"]["predictions"]
                else:
                    with Transaction(self.agent.model, self.adapter):
                        retained = self.agent.evaluate(self.retention["rows"])
                initial = retained if step == 0 else read_json(baseline_path)["retention"]["predictions"]
                if [p["id"] for p in initial] != [p["id"] for p in retained]:
                    raise ValueError("Retention example order changed")
                accuracy = sum(p["correct"] for p in retained) / len(retained)
                initial_accuracy = sum(p["correct"] for p in initial) / len(initial)
                report["retention"] = {"dev_id": self.retention["id"], "protocol": eval_protocol(self.cfg, "gsm8k"),
                    "n": len(retained), "accuracy": accuracy, "initial_accuracy": initial_accuracy,
                    "forgetting_pp": 100 * (initial_accuracy - accuracy),
                    "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in zip(initial, retained)),
                    "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in zip(initial, retained)),
                    "parse_failures": sum(not p["parsed"] for p in retained), "predictions": retained}
            write_json(report_path, report)
        # Reports are authoritative; rebuilding the index repairs a crash after report save.
        reports = [read_json(p) for p in sorted((self.out / "dev").glob("episode_*.json"))]
        write_json(self.out / "curve.json", [{k: v for k, v in r.items() if k != "predictions"} for r in reports])
        print(f"Dev episode {step}: accuracy={report['accuracy']:.4f}; checkpoint={checkpoint_path}", flush=True)


def summarize_suite(root):
    """Require the complete predeclared seed x policy x checkpoint grid."""
    root = Path(root)
    plan = read_json(root / "suite.json")
    steps = sorted(set([0, plan["episodes"], *range(plan["interval"], plan["episodes"] + 1, plan["interval"])]))
    all_points, aggregate, reference = [], [], None
    for policy in plan.get("policies", ["base", "router", "random", "all"]):
        for step in steps:
            values = []
            for seed in plan["seeds"]:
                file = root / f"seed_{seed}" / policy / "dev" / f"episode_{step:06d}.json"
                if not file.exists():
                    raise ValueError(f"Missing planned evaluation: {file}; resume suite before plotting")
                report = read_json(file)
                signature = {k: report[k] for k in ("dev_id", "identity", "protocol", "n")}
                signature["ids"] = [p["id"] for p in report["predictions"]]
                if reference is None:
                    reference = signature
                if signature != reference or (report["episode"], report["seed"], report["policy"]) != (step, seed, policy):
                    raise ValueError("Incompatible evaluation protocols, examples, seeds or policies")
                all_points.append({k: report[k] for k in ("seed", "policy", "episode", "accuracy", "parse_failures", "accepted", "selected_total")})
                values.append(report["accuracy"])
            aggregate.append({"policy": policy, "episode": step, "n_seeds": len(values),
                              "mean_accuracy": statistics.mean(values),
                              "std_accuracy": statistics.stdev(values) if len(values) > 1 else 0.0})
    for name, rows in (("curve_points.csv", all_points), ("curve_mean_std.csv", aggregate)):
        with (root / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(root / "learning_curves.json", {"plan": plan, "protocol": reference,
                                                "points": all_points, "aggregate": aggregate})
    return all_points, aggregate


def plot_suite(root):
    points, aggregate = summarize_suite(root)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(root).resolve() / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"base": "#555555", "router": "#0072B2", "random": "#D55E00", "all": "#009E73", "pdq": "#AA4499", "router_top2": "#56B4E9", "pdq_shuffle": "#CC6677", "pdq_identity": "#AA8800"}
    expanded = len({r["policy"] for r in points}) > 5
    fig, axes = plt.subplots(1, 2, figsize=(16, 6) if expanded else (12, 4.6), sharey=True)
    plan = read_json(Path(root) / "suite.json")
    fig.suptitle(f"Mode: {plan.get('mode', 'unspecified')} | seeds: {', '.join(map(str, plan['seeds']))}", fontsize=10)
    for policy, color in colors.items():
        rows = [r for r in aggregate if r["policy"] == policy]
        if not rows:
            continue
        x = [r["episode"] for r in rows]
        mean = [100 * r["mean_accuracy"] for r in rows]
        std = [100 * r["std_accuracy"] for r in rows]
        axes[0].plot(x, mean, label=policy, color=color, marker="o", markersize=3)
        axes[0].fill_between(x, [max(0, m - s) for m, s in zip(mean, std)],
                              [min(100, m + s) for m, s in zip(mean, std)], color=color, alpha=0.15)
        for seed in sorted({r["seed"] for r in points}):
            individual = [r for r in points if r["policy"] == policy and r["seed"] == seed]
            axes[1].plot([r["episode"] for r in individual], [100 * r["accuracy"] for r in individual],
                         color=color, alpha=0.65, label=f"{policy} / {seed}", marker=".")
    axes[0].set_title("Fixed dev: mean +/- sample SD across seeds")
    axes[1].set_title("Individual seed trajectories")
    for ax in axes:
        ax.set_xlabel("Online episodes processed")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2 if expanded else 1)
    axes[0].set_ylabel("Dev accuracy (%)")
    lower, upper = axes[0].get_ylim()
    axes[0].set_ylim(max(0, lower), min(100, upper))
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(Path(root) / f"learning_curves.{extension}", dpi=180)
    plt.close(fig)
    if "retention" in plan:
        from .transfer import plot_transfer
        plot_transfer(root)
