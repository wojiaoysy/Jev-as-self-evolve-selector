import hashlib
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests

from .io import append_jsonl, digest, read_json, read_jsonl, write_json


NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"


def extract_answer(text):
    """Parse final numeric answers without treating every number as an answer.

    Priority: last explicit ####/boxed marker, last numeric-only bold span on
    the final nonempty line, then a number ending that line (possibly after
    'The answer is'). An invalid explicit marker fails closed, with no fallback
    to earlier reasoning. Supports decimal numbers and valid thousands commas;
    fractions, expressions, scientific notation and trailing units are rejected.
    """
    text = text.replace("\u2212", "-").strip()
    explicit = list(re.finditer(r"####|\\boxed\s*\{", text))
    if explicit:
        marker = explicit[-1]
        tail = text[marker.end():]
        if marker.group().startswith("####"):
            candidate = tail.split("\n", 1)[0].strip()
        else:
            closing = tail.find("}")
            if closing < 0:
                return None
            candidate = tail[:closing].strip()
        candidate = candidate.rstrip(".。!").strip()
        if candidate.startswith("**") and candidate.endswith("**"):
            candidate = candidate[2:-2].strip()
        if candidate.startswith("$") and candidate.endswith("$"):
            candidate = candidate[1:-1].strip()
        if not re.fullmatch(NUMBER, candidate):
            return None
    else:
        last_line = text.splitlines()[-1].strip() if text else ""
        bold = list(re.finditer(r"\*\*\s*(" + NUMBER + r")\s*\*\*", last_line))
        if bold:
            candidate = bold[-1].group(1)
        else:
            match = re.search(r"(?<![\w.,/+-])(" + NUMBER + r")[ \t]*[.。!！]?$", last_line)
            if not match:
                return None
            candidate = match.group(1)
    try:
        return Decimal(candidate.replace(",", ""))
    except InvalidOperation:
        return None


def correct(prediction, gold):
    value, target = extract_answer(prediction), extract_answer(gold)
    return value is not None and target is not None and value == target


def clean_solution(solution):
    return re.sub(r"<<.*?>>", "", solution)


def prepare_data(output, seed=42, offline=256, probe=64, guard=64, online=128,
                 source_revision="master", source_dir=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Data directory is not empty; use a new directory")
    if min(offline, probe, guard, online) < 1 or offline < 4:
        raise ValueError("All splits must be nonempty and offline must contain at least four examples")
    raw, source_hash = {}, {}
    for split in ("train", "test"):
        if source_dir:
            payload = (Path(source_dir) / f"{split}.jsonl").read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/openai/grade-school-math/{source_revision}/grade_school_math/data/{split}.jsonl"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            payload = response.content
        source_hash[split] = hashlib.sha256(payload).hexdigest()
        raw[split] = []
        for index, line in enumerate(payload.decode("utf-8").splitlines()):
            value = json.loads(line)
            if not value.get("question") or extract_answer(value.get("answer", "")) is None:
                raise ValueError(f"Malformed {split} row {index}")
            raw[split].append({"id": f"gsm8k-{split}-{index}",
                               "question": value["question"], "answer": clean_solution(value["answer"])})
    test_questions = {r["question"].strip() for r in raw["test"]}
    seen, unique_train = set(), []
    for row in raw["train"]:
        question = row["question"].strip()
        if question not in seen and question not in test_questions:
            unique_train.append(row)
            seen.add(question)
    random.Random(seed).shuffle(unique_train)
    if offline + probe + guard + online > len(unique_train):
        raise ValueError("Requested splits exceed available deduplicated training examples")
    counts = {"offline": offline, "probe": probe, "guard": guard, "online": online}
    offset, split_records = 0, {}
    for name, count in counts.items():
        split_records[name] = unique_train[offset:offset + count]
        offset += count
    # Episode IDs determine the router split, before any labels are observed.
    # Interleave validation IDs so a short prefix smoke run contains both splits.
    val_ids = {r["id"] for r in split_records["offline"][::5]}
    for row in split_records["offline"]:
        row["router_split"] = "val" if row["id"] in val_ids else "train"
    split_records["test"] = raw["test"]
    for name, rows in split_records.items():
        for row in rows:
            append_jsonl(output / f"{name}.jsonl", row)
    manifest = {"benchmark": "GSM8K-main", "seed": seed, "source_revision": source_revision,
                "source_sha256": source_hash, "source_repository": "https://github.com/openai/grade-school-math",
                "deduplicated_train_count": len(unique_train),
                "counts": {k: len(v) for k, v in split_records.items()},
                "split_hashes": {k: digest(v) for k, v in split_records.items()}}
    manifest["id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def load_split(directory, name):
    manifest = read_json(Path(directory) / "manifest.json")
    rows = read_jsonl(Path(directory) / f"{name}.jsonl")
    if digest(rows) != manifest["split_hashes"][name]:
        raise ValueError(f"Data split was changed after manifest creation: {name}")
    return rows
