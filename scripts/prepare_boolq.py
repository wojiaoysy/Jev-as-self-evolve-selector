"""Prepare disjoint BoolQ adaptation splits and a fixed train-derived dev set."""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path
from jev_evolve.io import append_jsonl, digest, read_json, write_json


def prepare(output, source_dir=None, seed=42, offline=256, probe=64, guard=64, online=200, dev=128):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new, empty BoolQ data directory")
    if min(offline, probe, guard, online, dev) < 1 or offline < 4:
        raise ValueError("Positive split sizes and offline >= 4 required")
    raw, hashes = {}, {}
    revision = None
    if not source_dir:
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
        revision = HfApi().dataset_info("google/boolq").sha
    for split in ("train", "dev"):
        if source_dir:
            payload = (Path(source_dir) / f"{split}.jsonl").read_bytes()
        else:
            name = "train" if split == "train" else "validation"
            path = hf_hub_download("google/boolq", f"data/{name}-00000-of-00001.parquet",
                                   repo_type="dataset", revision=revision)
            payload = Path(path).read_bytes()
        hashes[split] = hashlib.sha256(payload).hexdigest()
        raw[split] = []
        source_rows = ([json.loads(line) for line in payload.decode("utf-8").splitlines()] if source_dir
                       else pq.read_table(io.BytesIO(payload)).to_pylist())
        for i, r in enumerate(source_rows):
            if type(r.get("answer")) is not bool or not r.get("passage") or not r.get("question"):
                raise ValueError(f"Malformed BoolQ {split} row {i}")
            raw[split].append({"id": f"boolq-{split}-{i}", "task": "boolq",
                "question": f"Passage: {r['passage'].strip()}\nQuestion: {r['question'].strip()}",
                "answer": "#### yes" if r["answer"] else "#### no"})
    seen = {r["question"] for r in raw["dev"]}
    rows = []
    for r in raw["train"]:
        if r["question"] not in seen:
            rows.append(r)
            seen.add(r["question"])
    random.Random(seed).shuffle(rows)
    counts = dict(offline=offline, probe=probe, guard=guard, online=online)
    if sum(counts.values()) + dev > len(rows):
        raise ValueError("Insufficient deduplicated training examples")
    splits, offset = {}, 0
    for name, count in counts.items():
        splits[name] = rows[offset:offset + count]
        offset += count
    for i, r in enumerate(splits["offline"]):
        r["router_split"] = "val" if i % 5 == 0 else "train"
    splits["test"] = raw["dev"]
    manifest = {"benchmark": "BoolQ", "task": "boolq", "test_source": "official_validation",
        "seed": seed, "source_sha256": hashes, "source_revision": revision,
        "download_source": "local_jsonl" if source_dir else "google/boolq",
        "source_repository": "https://github.com/google-research-datasets/boolean-questions",
        "counts": {k: len(v) for k, v in splits.items()},
        "split_hashes": {k: digest(v) for k, v in splits.items()}}
    manifest["id"] = digest(manifest)
    for name, values in splits.items():
        for r in values:
            append_jsonl(output / f"{name}.jsonl", r)
    write_json(output / "manifest.json", manifest)
    held = {"format": 1, "parent_data_id": manifest["id"], "seed": seed,
            "source_sha256": hashes["train"], "rows": rows[offset:offset + dev]}
    held["id"] = digest(held)
    write_json(output / "dev_fixed.json", held)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/boolq")
    p.add_argument("--source-dir")
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--output-config", default="configs/boolq.json")
    p.add_argument("--seed", type=int, default=42)
    for name, default in dict(offline=256, probe=64, guard=64, online=200, dev=128).items():
        p.add_argument("--" + name, type=int, default=default)
    a = p.parse_args()
    if Path(a.output_config).exists():
        p.error("Output config already exists; choose a new path")
    cfg = read_json(a.config)
    prepare(a.output, a.source_dir, a.seed, a.offline, a.probe, a.guard, a.online, a.dev)
    cfg["model"]["task"] = cfg["judge"]["task"] = "boolq"
    cfg["model"]["max_length"] = max(2048, cfg["model"]["max_length"])
    # Same generation budget as GSM8K, including during retention evaluation.
    write_json(a.output_config, cfg)
    print(f"Prepared {a.output}; config: {a.output_config}")


if __name__ == "__main__":
    main()
