"""Task-specific prompts and deterministic answer scoring."""
import re
from .data import extract_answer

SYSTEMS = {
    "gsm8k": "Solve the math problem. Show concise, justified steps. End with '#### ' followed by the final numeric answer.",
    "boolq": "Read the passage and answer the question using the passage. Output only '#### yes' or '#### no'.",
}


def parse(text, task="gsm8k"):
    if task == "gsm8k":
        return extract_answer(text)
    if task != "boolq":
        raise ValueError(f"Unknown task: {task}")
    text = text.strip().lower()
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    else:
        text = text.splitlines()[-1] if text else ""
    text = text.rstrip(".!。 ").strip()
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    match = re.fullmatch(r"(?:the answer is[: ]+)?(yes|no)", text)
    return match.group(1) if match else None


def score(text, row):
    task = row.get("task", "gsm8k")
    value, target = parse(text, task), parse(row["answer"], task)
    return value is not None and target is not None and value == target
