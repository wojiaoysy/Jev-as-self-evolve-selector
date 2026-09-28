import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch

from .data import load_split, prepare_data
from .intervention import adapt_direct, adapt_with_guard, measure_interventions
from .io import digest, load_torch, read_json, require_empty, save_torch, seed_all, tensor_digest, write_json
from .judge import JevClient
from .model import SYSTEM, load_agent, save_adapter
from .tasks import SYSTEMS
from .router import Router, metrics, select_topk, select_forced_top2, train_router
from .curves import DevMonitor, eval_protocol, load_dev, prepare_dev


def settings(path):
    cfg = read_json(path)
    seed_all(cfg["seed"])
    return cfg


def experiment_contract(cfg, identity, initial, data_manifest, judge):
    task = data_manifest.get("task", "gsm8k")
    if cfg["model"].get("task", "gsm8k") != task or cfg["judge"].get("task", "gsm8k") != task:
        raise ValueError("Dataset, model prompt and judge task must match")
    return {"format": 1, "identity": identity, "origin_R": tensor_digest([("R", initial)]),
            "data_id": data_manifest["id"], "judge": judge.identity,
            "intervention": cfg["intervention"], "seed": cfg["seed"],
            "prompt": SYSTEMS[task], "model_behavior": {k: cfg["model"][k] for k in
                ("dtype", "attention", "gradient_checkpointing", "max_length", "max_new_tokens")}}


def probe_rows(pool, row_id, cfg):
    size = cfg["intervention"]["probe_size"]
    if not 1 <= size <= len(pool):
        raise ValueError("probe_size must be in [1, size of probe split]")
    rng = random.Random(int(digest([cfg["seed"], row_id])[:16], 16))
    return rng.sample(pool, size)


def preflight(args):
    import transformers
    import numpy
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__, "numpy": numpy.__version__,
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0),
                    vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2),
                    bf16=torch.cuda.is_bf16_supported())
    print(json.dumps(info, indent=2))
    errors = []
    if sys.version_info[:2] != (3, 10):
        errors.append("Expected Python 3.10")
    if torch.__version__.split("+")[0] != "2.1.2":
        errors.append("Expected torch 2.1.2")
    if transformers.__version__ != "4.44.2":
        errors.append("Expected transformers 4.44.2")
    if int(numpy.__version__.split(".")[0]) >= 2:
        errors.append("torch 2.1.2 requires the pinned NumPy 1.x environment here")
    if not args.cpu:
        if not torch.cuda.is_available() or torch.version.cuda != "11.8":
            errors.append("Expected working CUDA 11.8 PyTorch build")
        elif not torch.cuda.is_bf16_supported():
            errors.append("GPU does not support bf16")
    if errors:
        raise RuntimeError("; ".join(errors))


def download_model(args):
    from huggingface_hub import HfApi, snapshot_download
    # Resolve moving refs first, then download exactly that commit.
    info = HfApi().model_info(args.model, revision=args.revision)
    path = snapshot_download(args.model, revision=info.sha, local_dir=args.output,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE*", "README.md"])
    write_json(Path(path) / "download_manifest.json", {"model": args.model, "revision": info.sha})
    print(f"Downloaded {args.model}@{info.sha} to {path}")


def initialize(args):
    cfg = settings(args.config)
    require_empty(args.output)
    agent, adapter, identity = load_agent(cfg["model"], cfg["adapter"])
    save_adapter(args.output, adapter, identity, {"config": cfg, "phase": "initial"})
    print(json.dumps({"checkpoint": args.output, "identity": identity,
                      "R_parameters": sum(p.numel() for p in adapter.R),
                      "base_parameters": sum(p.numel() for p in agent.model.parameters())}, indent=2))


def check_data(args):
    from transformers import AutoTokenizer
    cfg = settings(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"],
                                              trust_remote_code=False)
    report, failures = {}, []
    splits = {name: load_split(args.data, name) for name in ("offline", "probe", "guard", "online", "test")}
    if args.dev:
        splits["fixed_dev"] = load_dev(args.dev, args.data)["rows"]
    for split, rows in splits.items():
        lengths = []
        for row in rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEMS[row.get("task", "gsm8k")]}, {"role": "user", "content": row["question"]}],
                tokenize=True, add_generation_prompt=True, return_dict=False)
            length = len(prompt) + len(tokenizer.encode(row["answer"], add_special_tokens=False)) + 1
            lengths.append(length)
            if length > cfg["model"]["max_length"]:
                failures.append({"id": row["id"], "tokens": length})
        report[split] = {"count": len(lengths), "max_tokens": max(lengths)}
    print(json.dumps({"splits": report, "max_length": cfg["model"]["max_length"], "over_limit": failures}, indent=2))
    if failures:
        raise ValueError("Supervised samples exceed max_length; adjust before init/collection")


def judge_smoke(args):
    cfg = settings(args.config)
    judge = JevClient(cfg["judge"])
    state = {"question": "What is 2 + 3?", "response": "2 + 3 = 5.\n#### 5",
             "steps": ["2 + 3 = 5.", "#### 5"], "token_samples": [], "tools": [], "memory": []}
    vector, meta = judge.evaluate(state)
    print(json.dumps({"feature_dimension": len(vector), "schema_id": judge.schema.id, **meta}, indent=2))


def collect(args):
    cfg = settings(args.config)
    judge = JevClient(cfg["judge"])
    if cfg["judge"]["provider"] == "synthetic":
        print("WARNING: SYNTHETIC judge: this run cannot support claims about Jev.", flush=True)
    rows = load_split(args.data, "offline")
    pool = load_split(args.data, "probe")
    manifest = read_json(Path(args.data) / "manifest.json")
    agent, adapter, identity = load_agent(cfg["model"], cfg["adapter"], args.adapter)
    contract = experiment_contract(cfg, identity, adapter.export()["R"], manifest, judge)
    out = Path(args.output)
    contract_file = out / "contract.json"
    if contract_file.exists() and read_json(contract_file) != contract:
        raise ValueError("Collection directory belongs to a different experiment")
    write_json(contract_file, contract)
    write_json(out / "feature_schema.json", {"id": judge.schema.id, "names": judge.schema.names,
                                            "questions": judge.schema.questions})
    existing = [read_json(p) for p in (out / "episodes").glob("*.json")]
    resolved = {r["judge_meta"]["model"] for r in existing}
    for index, row in enumerate(rows[:args.max_episodes] if args.max_episodes else rows):
        file = out / "episodes" / (row["id"] + ".json")
        if file.exists():
            cached = read_json(file)
            if cached["contract_id"] != digest(contract):
                raise ValueError("Incompatible existing episode")
            continue
        start = time.monotonic()
        trajectory = agent.run(row, cfg["judge"]["token_slots"])
        features, judge_meta = judge.evaluate(trajectory)
        resolved.add(judge_meta["model"])
        if len(resolved) != 1 or None in resolved:
            raise ValueError("Resolved Jev model changed or is absent; use a pinned model and a new collection")
        probes = probe_rows(pool, row["id"], cfg)
        measured = measure_interventions(agent.model, adapter, lambda: agent.loss(row),
                    lambda: agent.reward(probes, cfg["intervention"]["reward"]), cfg["intervention"])
        record = {"episode_id": row["id"], "split": row["router_split"], "contract_id": digest(contract),
                  "jev_features": features, "judge_meta": judge_meta, "trajectory": trajectory,
                  "probe_ids": [p["id"] for p in probes], **measured,
                  "elapsed_seconds": time.monotonic() - start}
        write_json(file, record)
        print(f"Episode {index + 1}/{len(rows)} saved; max delta={max(measured['delta_reward']):.6g}; "
              f"{record['elapsed_seconds']:.1f}s", flush=True)
    print(f"Collection saved to {out}; rerunning the identical command resumes missing episodes.")


def fit_router(args):
    cfg = settings(args.config)
    require_empty(args.output)
    folder = Path(args.records)
    contract = read_json(folder / "contract.json")
    schema = read_json(folder / "feature_schema.json")
    records = [read_json(p) for p in sorted((folder / "episodes").glob("*.json"))]
    if not records or any(r["contract_id"] != digest(contract) for r in records):
        raise ValueError("Missing or mixed intervention records")
    if len({r["episode_id"] for r in records}) != len(records):
        raise ValueError("Duplicate episodes")
    if len({r["judge_meta"]["model"] for r in records}) != 1:
        raise ValueError("Mixed resolved judge models")
    K = contract["identity"]["K"]
    if any(len(r["jev_features"]) != len(schema["names"]) or len(r["delta_reward"]) != K for r in records):
        raise ValueError("Invalid record dimensions")
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    router, history = train_router(train, val, cfg["router"], cfg["seed"])
    save_torch(args.output, {"format": 1, "state_dict": router.state_dict(), "contract": contract,
                            "feature_schema": schema, "settings": cfg["router"],
                            "resolved_judge": records[0]["judge_meta"]["model"],
                            "record_ids": [r["episode_id"] for r in records],
                            "input_dim": len(schema["names"]), "K": K})
    delta = torch.tensor([r["delta_reward"] for r in records])
    diagnostic = {"train_episodes": len(train), "val_episodes": len(val),
                  "positive_episode_fraction": float((delta.max(-1).values > cfg["router"]["gain_floor"]).float().mean()),
                  "zero_or_tied_episode_fraction": float((delta.max(-1).values - delta.min(-1).values <= cfg["router"]["gain_floor"]).float().mean()),
                  "top1_counts": torch.bincount(delta.argmax(-1), minlength=K).tolist(),
                  "history": history}
    write_json(str(args.output) + ".metrics.json", diagnostic)
    print(json.dumps({k: v for k, v in diagnostic.items() if k != "history"}, indent=2))
    if diagnostic["positive_episode_fraction"] == 0:
        print("WARNING: No positive intervention labels. Inspect reward resolution, LR, precision and target layer.")


def online(args):
    cfg = settings(args.config)
    judge = JevClient(cfg["judge"])
    initial = load_torch(args.adapter)
    manifest = read_json(Path(args.data) / "manifest.json")
    contract = experiment_contract(cfg, initial["identity"], initial["adapter"]["R"], manifest, judge)
    router = None
    router_payload = None
    if args.policy in ("router", "router_top2"):
        if not args.router:
            raise ValueError("--router checkpoint is required for router policy")
        router_payload = load_torch(args.router)
        if router_payload["contract"] != contract:
            raise ValueError("Router training contract differs from this model, data, judge or intervention policy")
        rs = router_payload["settings"]
        if rs != cfg["router"]:
            raise ValueError("Router settings changed; use its saved settings (including threshold and top_k)")
        router = Router(router_payload["input_dim"], router_payload["K"], rs["hidden"])
        router.load_state_dict(router_payload["state_dict"])
        router.eval()
    out = Path(args.output)
    final = out / "adapter.pt"
    run_contract = {"experiment": contract, "policy": args.policy, "router_settings": cfg["router"],
                    "update_mode": args.update_mode,
                    "online_settings": cfg["online"], "router_hash": None if router is None else
                    tensor_digest(router.state_dict().items())}
    if args.policy.startswith("pdq"):
        if not args.pdq or args.update_mode != "direct":
            raise ValueError("PDQ requires --pdq and direct update mode")
        pdq_payload = load_torch(args.pdq)
        if (pdq_payload["metadata"]["contract"] != contract or
                pdq_payload["metadata"]["feature_schema_id"] != judge.schema.id):
            raise ValueError("PDQ checkpoint has a different experiment/schema")
        run_contract["pdq"] = {"initial_hash": tensor_digest([("R", pdq_payload["adapter"]["R"])]),
                               "mode": args.pdq_mode, "settings": cfg.get("pdq", {}),
                               "D_variant": {"pdq": "real", "pdq_shuffle": "shuffle", "pdq_identity": "identity"}[args.policy],
                               "shuffle_seed": cfg["seed"] if args.online_seed is None else args.online_seed}
        if args.pdq_mode == "accumulate":
            import math
            if not math.isfinite(args.pdq_accumulation_scale) or args.pdq_accumulation_scale <= 0:
                raise ValueError("PDQ accumulation scale must be finite and positive")
            run_contract["pdq"]["accumulation_scale"] = args.pdq_accumulation_scale
    online_seed = cfg["seed"] if args.online_seed is None else args.online_seed
    if args.online_seed is not None:
        run_contract["online_seed"] = online_seed
    dev = load_dev(args.dev, args.data) if args.dev else None
    if dev is not None:
        if args.eval_every < 1:
            raise ValueError("eval-every must be positive")
        run_contract["monitor"] = {"dev_id": dev["id"], "interval": args.eval_every,
                                   "protocol": eval_protocol(cfg)}
    retention = None
    if bool(args.retention_dev) != bool(args.retention_data):
        raise ValueError("Supply both retention-dev and retention-data")
    if args.retention_dev:
        if dev is None:
            raise ValueError("Retention monitoring requires --dev")
        retention = load_dev(args.retention_dev, args.retention_data)
        if any(r.get("task", "gsm8k") != "gsm8k" for r in retention["rows"]):
            raise ValueError("Retention set must be GSM8K")
        run_contract["monitor"]["retention"] = {"id": retention["id"], "protocol": eval_protocol(cfg, "gsm8k")}
    events = []
    if final.exists():
        saved = load_torch(final)
        if saved["metadata"].get("run_contract") != run_contract:
            raise ValueError("Online output belongs to a different experiment; choose a new directory")
        events = saved["metadata"]["events"]
    start_adapter = args.pdq if args.policy.startswith("pdq") else args.adapter
    agent, adapter, identity = load_agent(cfg["model"], cfg["adapter"], final if final.exists() else start_adapter)
    if args.policy.startswith("pdq"):
        adapter.pdq_variant = run_contract["pdq"]["D_variant"]
        adapter.pdq_shuffle_seed = run_contract["pdq"]["shuffle_seed"]
        adapter.pdq_mode = args.pdq_mode
        if args.pdq_mode == "accumulate":
            adapter.select([])
    if args.policy == "router_top2" and (args.update_mode != "direct" or adapter.K < 2):
        raise ValueError("Forced Top-2 requires direct mode and at least two subspaces")
    rows = load_split(args.data, "online")
    if args.online_seed is not None:
        random.Random(online_seed).shuffle(rows)
    pool = load_split(args.data, "probe") if args.update_mode == "guarded" else None
    guards = load_split(args.data, "guard") if args.update_mode == "guarded" else None
    chosen_rows = rows[:args.max_episodes] if args.max_episodes else rows
    if len(events) > len(chosen_rows):
        raise ValueError("max-episodes is smaller than already-completed online progress")
    if [e["episode_id"] for e in events] != [r["id"] for r in chosen_rows[:len(events)]]:
        raise ValueError("Saved online order differs from current run")
    monitor = (DevMonitor(out, agent, adapter, identity, run_contract, dev, args.eval_every, cfg, online_seed, retention=retention)
               if dev is not None else None)
    if monitor:
        # Recover a crash between the latest model save and its scheduled dev evaluation.
        monitor.record(events, final=len(events) == len(chosen_rows))
    for index in range(len(events), len(chosen_rows)):
        row = chosen_rows[index]
        start = time.monotonic()
        judge_meta, alpha = None, None
        if args.policy in ("router", "router_top2"):
            trajectory = agent.run(row, cfg["judge"]["token_slots"])
            features, judge_meta = judge.evaluate(trajectory)
            if judge_meta["model"] != router_payload["resolved_judge"]:
                raise ValueError("Resolved Jev model differs from router training")
            with torch.no_grad():
                alpha = router(torch.tensor(features).unsqueeze(0))[0]
            selected = (select_forced_top2(alpha) if args.policy == "router_top2" else
                        select_topk(alpha, cfg["router"]["top_k"], cfg["router"]["threshold"]))
        elif args.policy.startswith("pdq"):
            selected = [0] if args.pdq_mode == "continual" else []
            if selected:
                from .pdq import get_features
                features, judge_meta = get_features(agent, row)
                adapter.set_features(features)
        elif args.policy == "random":
            selected = random.Random(online_seed + index).sample(range(adapter.K), cfg["router"]["top_k"])
        elif args.policy == "all":
            selected = list(range(adapter.K))
        else:
            selected = []
        if args.policy.startswith("pdq") and args.pdq_mode == "accumulate":
            from .pdq import accumulate_episode
            result, judge_meta = accumulate_episode(agent, row, args.pdq_accumulation_scale)
        elif args.update_mode == "direct":
            from .pdq import training_settings
            update_settings = training_settings(cfg) if args.policy.startswith("pdq") else cfg["intervention"]
            result = adapt_direct(agent.model, adapter, selected, lambda: agent.loss(row), update_settings)
            if args.policy.startswith("pdq"):
                adapter.set_features(torch.zeros_like(adapter.D))
                if args.pdq_mode == "dynamic":
                    result["reason"] = "fixed_PQ_dynamic_D"
        else:
            probes = probe_rows(pool, row["id"], cfg)
            result = adapt_with_guard(agent.model, adapter, selected, lambda: agent.loss(row),
                        lambda: agent.reward(probes, cfg["intervention"]["reward"]),
                        lambda: agent.reward(guards, cfg["intervention"]["reward"]), cfg["intervention"],
                        **cfg["online"])
        event = {"episode_id": row["id"], "index": index, **result,
                 "alpha": None if alpha is None else alpha.tolist(), "judge_meta": judge_meta,
                 "elapsed_seconds": time.monotonic() - start}
        events.append(event)
        # This single atomic checkpoint is the source of truth for both state and progress.
        save_adapter(final, adapter, identity, {"run_contract": run_contract, "events": events})
        write_json(out / "events.json", events)
        if monitor:
            monitor.record(events, final=len(events) == len(chosen_rows))
        print(f"Online {index + 1}/{len(chosen_rows)}: {result['reason']}; selected={selected}", flush=True)
    if not events:
        save_adapter(final, adapter, identity, {"run_contract": run_contract, "events": events})
    report = {"policy": args.policy, "processed": len(events),
              "accepted": sum(e["accepted"] for e in events), "checkpoint": str(final)}
    write_json(out / "summary.json", report)
    print(json.dumps(report, indent=2))


def evaluate(args):
    cfg = settings(args.config)
    require_empty(args.output)
    rows = load_split(args.data, "test")
    total = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    agent, adapter, identity = load_agent(cfg["model"], cfg["adapter"], args.adapter)
    start = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    predictions = agent.evaluate(rows)
    n = len(predictions)
    accuracy = sum(r["correct"] for r in predictions) / n
    # Wilson interval on each model's accuracy; compare command uses paired bootstrap.
    z, denominator = 1.96, 1 + 1.96 ** 2 / n
    center = (accuracy + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / denominator
    report = {"label": args.label, "benchmark": read_json(Path(args.data) / "manifest.json")["benchmark"], "test_source": read_json(Path(args.data) / "manifest.json").get("test_source", "official_test"), "full_test": n == total,
              "n": n, "accuracy": accuracy, "accuracy_ci95_wilson": [center - radius, center + radius],
              "parse_failures": sum(not p["parsed"] for p in predictions),
              "elapsed_seconds": time.monotonic() - start,
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "identity": identity, "R_hash": tensor_digest([("R", adapter.export()["R"])]),
              "data_id": read_json(Path(args.data) / "manifest.json")["id"],
              "protocol": eval_protocol(cfg, rows[0].get("task", "gsm8k")),
              "predictions": predictions}
    report["inference"] = "draft_then_jev_then_pdq" if hasattr(agent, "pdq_judge") else "single_pass"
    if getattr(adapter, "pdq_mode", None) == "accumulate":
        report["inference"] = "single_pass_accumulated_pdq"
        report["R_hash"] = tensor_digest([
            ("R", adapter.export()["R"]), ("accumulated_D", adapter.accumulated_D), ("D", adapter.D)])
    if hasattr(agent, "pdq_judge"):
        report["pdq_variant"] = getattr(adapter, "pdq_variant", "real")
        report["pdq_shuffle_seed"] = getattr(adapter, "pdq_shuffle_seed", 42)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("predictions", "identity", "protocol")}, indent=2))


def compare(args):
    reports = [read_json(p) for p in args.reports]
    baseline = reports[0]
    ids = [p["id"] for p in baseline["predictions"]]
    generator = torch.Generator().manual_seed(42)
    rows = []
    for report in reports:
        if any(report[k] != baseline[k] for k in ("data_id", "protocol", "identity")) or [p["id"] for p in report["predictions"]] != ids:
            raise ValueError("Reports must use identical examples, base/basis, and generation protocol")
        differences = torch.tensor([int(a["correct"]) - int(b["correct"])
                                   for a, b in zip(report["predictions"], baseline["predictions"])], dtype=torch.float32)
        samples = torch.randint(len(ids), (2000, len(ids)), generator=generator)
        interval = differences[samples].mean(-1).quantile(torch.tensor([0.025, 0.975])).tolist()
        rows.append({"label": report["label"], "n": report["n"], "full_test": report["full_test"],
                     "accuracy": report["accuracy"], "delta_vs_first": float(differences.mean()),
                     "paired_bootstrap_ci95": interval})
    write_json(args.output, {"baseline": baseline["label"], "results": rows})
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Jev -> router -> single-layer subspace adaptation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train-pdq")
    for name in ("config", "data", "adapter", "records", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--save-epochs", action="store_true", help="Save P/Q checkpoints at epoch 0 and each completed epoch for diagnostics")
    from .pdq import fit
    p.set_defaults(fn=fit)
    p = sub.add_parser("preflight")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(fn=preflight)
    p = sub.add_parser("download-model")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", default="models/qwen2.5-1.5b-instruct")
    p.set_defaults(fn=download_model)
    p = sub.add_parser("prepare-data")
    p.add_argument("--output", default="data/gsm8k")
    p.add_argument("--seed", type=int, default=42)
    for name, default in (("offline", 256), ("probe", 64), ("guard", 64), ("online", 128)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--source-revision", default="master")
    p.add_argument("--source-dir")
    p.set_defaults(fn=lambda a: print(json.dumps(prepare_data(a.output, a.seed, a.offline, a.probe, a.guard,
                                        a.online, a.source_revision, a.source_dir), indent=2)))
    p = sub.add_parser("prepare-dev")
    p.add_argument("--data", default="data/gsm8k")
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--source-file")
    p.set_defaults(fn=lambda a: print(prepare_dev(a.data, a.output, a.size, a.seed, a.source_file)["id"]))
    for name, function in (("init", initialize), ("judge-smoke", judge_smoke), ("collect", collect),
                           ("check-data", check_data), ("train-router", fit_router), ("adapt", online), ("evaluate", evaluate)):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/autodl_4090.json")
        p.set_defaults(fn=function)
        if name not in ("judge-smoke", "check-data"):
            p.add_argument("--output", required=True)
        if name in ("collect", "adapt", "evaluate"):
            p.add_argument("--adapter", required=True)
            p.add_argument("--data", default="data/gsm8k")
        if name == "check-data":
            p.add_argument("--data", default="data/gsm8k")
            p.add_argument("--dev")
        if name in ("collect", "adapt"):
            p.add_argument("--max-episodes", type=int)
        if name == "train-router":
            p.add_argument("--records", required=True)
        if name == "adapt":
            p.add_argument("--update-mode", choices=["direct", "guarded"], default="direct")
            p.add_argument("--router")
            p.add_argument("--policy", choices=["router", "random", "all", "none", "base", "pdq", "router_top2", "pdq_shuffle", "pdq_identity"], default="router")
            p.add_argument("--pdq")
            p.add_argument("--pdq-mode", choices=["dynamic", "continual", "accumulate"], default="accumulate")
            p.add_argument("--pdq-accumulation-scale", type=float, default=1.0)
            p.add_argument("--retention-dev")
            p.add_argument("--retention-data")
            p.add_argument("--dev", help="Fixed held-out dev JSON from prepare-dev")
            p.add_argument("--eval-every", type=int, default=20)
            p.add_argument("--online-seed", type=int, help="Shuffle online order and seed random routing; leaves router contract intact")
        if name == "evaluate":
            p.add_argument("--label", required=True)
            p.add_argument("--limit", type=int)
    p = sub.add_parser("compare")
    p.add_argument("--reports", nargs="+", required=True, help="Put frozen-base report first")
    p.add_argument("--output", required=True)
    p.set_defaults(fn=compare)
    args = parser.parse_args()
    for option in ("max_episodes", "limit"):
        value = getattr(args, option, None)
        if value is not None and value < 1:
            parser.error(f"{option} must be positive")
    args.fn(args)


if __name__ == "__main__":
    main()
