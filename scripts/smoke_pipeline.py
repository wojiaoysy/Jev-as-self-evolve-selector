"""Offline CLI integration test using local random Qwen weights and synthetic fixtures.

Exercises checkpoint identity, partial collection/resume, router fitting, all policies,
online resume equivalence, final evaluation, and incompatible-config rejection.
Never downloads a pretrained model or calls a judge service.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from jev_evolve.data import prepare_data
from jev_evolve.io import load_torch, read_json, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(42)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="jev-pipeline-") as tmp:
        folder = Path(tmp)
        model_dir = folder / "tiny-qwen"
        backend = Tokenizer(WordLevel({w: i for i, w in enumerate(
            ["[UNK]", "[PAD]", "[EOS]", "2", "3", "5", "####", "What", "is", "?", "+", "=", "assistant"]
        )}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        tokenizer.save_pretrained(model_dir)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=13, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=256, eos_token_id=2, pad_token_id=1, attention_dropout=0.0))
        model.save_pretrained(model_dir, safe_serialization=True)
        source = folder / "source"
        source.mkdir()
        for split, n in (("train", 20), ("test", 2)):
            (source / f"{split}.jsonl").write_text("\n".join(json.dumps(
                {"question": f"{split} example {i}: What is 2 + 3 ?", "answer": "2 + 3 = 5\n#### 5"}
            ) for i in range(n)), encoding="utf-8")
        data = folder / "data"
        prepare_data(data, offline=6, probe=2, guard=2, online=2, source_dir=source)
        cfg = read_json(root / "configs/autodl_4090.json")
        cfg["model"].update(name=str(model_dir), device="cpu", dtype="float32", attention="eager",
                             max_length=128, max_new_tokens=4, target_module="model.layers.0.self_attn.q_proj")
        cfg["adapter"] = {"K": 2, "rank": 2}
        cfg["judge"].update(provider="synthetic", cache_dir=str(folder / "cache"), max_steps=2, token_slots=2)
        cfg["intervention"].update(probe_size=2, steps=1)
        cfg["router"].update(hidden=8, epochs=3, batch_size=2, top_k=1, threshold=0.0)
        cfg["online"].update(min_gain=0.0)
        config_path = folder / "config.json"
        write_json(config_path, cfg)
        initial, records, router = folder / "initial.pt", folder / "records", folder / "router.pt"

        def cli(*args, failure=False):
            result = subprocess.run([sys.executable, "-m", "jev_evolve.cli", *map(str, args)],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            if (result.returncode == 0) == failure:
                raise RuntimeError(result.stdout + result.stderr)
            print("OK:", args[0], flush=True)
            return result

        common = ["--config", config_path]
        cli("init", *common, "--output", initial)
        collect = ["collect", *common, "--data", data, "--adapter", initial, "--output", records]
        cli(*collect, "--max-episodes", 2)
        partial = {p.name: p.read_bytes() for p in (records / "episodes").glob("*.json")}
        cli(*collect)
        assert len(list((records / "episodes").glob("*.json"))) == 6
        assert all((records / "episodes" / name).read_bytes() == content for name, content in partial.items())
        cli("train-router", *common, "--records", records, "--output", router)
        online = ["adapt", *common, "--data", data, "--adapter", initial]
        resumed = folder / "random-resumed"
        cli(*online, "--policy", "random", "--output", resumed, "--max-episodes", 1)
        first = load_torch(resumed / "adapter.pt")["adapter"]["R"].clone()
        cli(*online, "--policy", "random", "--output", resumed, "--max-episodes", 1)
        assert torch.equal(first, load_torch(resumed / "adapter.pt")["adapter"]["R"])
        cli(*online, "--policy", "random", "--output", resumed)
        reports = []
        for policy in ("random", "all", "router"):
            output = folder / policy
            extra = ["--router", router] if policy == "router" else []
            cli(*online, "--policy", policy, "--output", output, *extra)
            if policy == "random":
                a, b = load_torch(resumed / "adapter.pt"), load_torch(output / "adapter.pt")
                assert torch.equal(a["adapter"]["R"], b["adapter"]["R"]), "Resume differs from uninterrupted run"
            report = folder / f"eval-{policy}.json"
            cli("evaluate", *common, "--data", data, "--adapter", output / "adapter.pt", "--label", policy, "--output", report)
            reports.append(report)
        base_report = folder / "eval-base.json"
        cli("evaluate", *common, "--data", data, "--adapter", initial, "--label", "base", "--output", base_report)
        cli("compare", "--reports", base_report, *reports, "--output", folder / "comparison.json")
        cfg["intervention"]["lr"] *= 2
        write_json(config_path, cfg)
        error = cli(*online, "--router", router, "--policy", "router", "--output", folder / "wrong", failure=True)
        assert "contract" in error.stderr
        print("PASS: offline CLI pipeline; synthetic fixtures only; no benchmark performance claim.")


if __name__ == "__main__":
    main()
