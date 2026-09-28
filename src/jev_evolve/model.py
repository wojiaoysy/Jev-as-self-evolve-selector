import re

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .adapter import attach_adapter
from .tasks import SYSTEMS, parse, score
from .io import digest, load_torch, save_torch, tensor_digest


SYSTEM = "Solve the math problem. Show concise, justified steps. End with '#### ' followed by the final numeric answer."


def load_agent(settings, adapter_settings, checkpoint_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run preflight before starting")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[settings["dtype"]]
    if settings["device"] == "cpu" and dtype != torch.float32:
        raise ValueError("Use float32 for CPU runs")
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["revision"], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(settings["name"], revision=settings["revision"],
                                                torch_dtype=dtype, attn_implementation=settings["attention"],
                                                trust_remote_code=False, use_safetensors=True)
    if model.config.model_type != "qwen2":
        raise ValueError("This release supports Qwen2/Qwen2.5 causal LMs only")
    if getattr(model.config, "attention_dropout", 0) != 0:
        raise ValueError("Causal interventions require attention_dropout=0")
    print("Hashing frozen base weights for checkpoint identity...", flush=True)
    base_hash = tensor_digest(model.state_dict().items())
    model.to(settings["device"])
    payload = load_torch(checkpoint_path) if checkpoint_path else None
    identity = {"base_hash": base_hash, "target_module": settings["target_module"],
                "K": adapter_settings["K"], "rank": adapter_settings["rank"]}
    if payload and any(payload["identity"][key] != value for key, value in identity.items()):
        raise ValueError("Adapter checkpoint does not match base model, target module or subspace dimensions")
    adapter_payload = None if payload is None else payload["adapter"]
    is_pdq = adapter_payload is not None and adapter_payload.get("kind") == "pdq"
    adapter = attach_adapter(model, settings["target_module"], adapter_settings["K"], adapter_settings["rank"],
                             adapter_payload["base_adapter"] if is_pdq else adapter_payload)
    identity["basis_id"] = adapter.basis_id()
    identity["id"] = digest(identity)
    if payload and payload["identity"] != identity:
        raise ValueError("Adapter basis fingerprint mismatch")
    if is_pdq:
        from .pdq import PDQLinear
        adapter = PDQLinear(adapter, adapter_payload["feature_dim"])
        adapter.restore_payload(adapter_payload)
        parent, child = settings["target_module"].rsplit(".", 1)
        setattr(model.get_submodule(parent), child, adapter)
    if settings.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    agent = QwenAgent(model, tokenizer, settings)
    if is_pdq:
        from .pdq import configure_agent
        configure_agent(agent, adapter, payload["metadata"]["pdq_judge"], payload["metadata"]["pdq_resolved_judge"],
                        payload["metadata"]["pdq_schema_id"])
        adapter.pdq_variant = payload["metadata"].get("pdq_variant", "real")
        adapter.pdq_shuffle_seed = payload["metadata"].get("pdq_shuffle_seed", 42)
    return agent, adapter, identity


def save_adapter(path, adapter, identity, metadata):
    if hasattr(adapter, "pdq_judge_settings"):
        metadata = {**metadata, "pdq_judge": adapter.pdq_judge_settings,
                    "pdq_resolved_judge": adapter.pdq_resolved_judge,
                    "pdq_schema_id": adapter.pdq_schema_id,
                    "pdq_variant": getattr(adapter, "pdq_variant", "real"),
                    "pdq_shuffle_seed": getattr(adapter, "pdq_shuffle_seed", 42)}
    save_torch(path, {"format": 1, "adapter": adapter.export(), "identity": identity, "metadata": metadata})


class QwenAgent:
    def __init__(self, model, tokenizer, settings):
        self.model, self.tokenizer, self.settings = model, tokenizer, settings
        self.device = next(model.parameters()).device
        self._encoded = {}

    def prompt_ids(self, question, task="gsm8k"):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEMS[task]}, {"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True, return_dict=False)

    def encode(self, row):
        key = digest([row.get("task", "gsm8k"), row["question"], row["answer"], self.settings["max_length"]])
        if key not in self._encoded:
            prompt = self.prompt_ids(row["question"], row.get("task", "gsm8k"))
            completion = self.tokenizer.encode(row["answer"], add_special_tokens=False) + [self.tokenizer.eos_token_id]
            if len(prompt) + len(completion) > self.settings["max_length"]:
                raise ValueError(f"{row.get('id', 'example')} needs {len(prompt) + len(completion)} tokens; "
                                 "raise model.max_length. No supervised answer is silently truncated.")
            ids = torch.tensor([prompt + completion], dtype=torch.long)
            labels = ids.clone()
            labels[:, :len(prompt)] = -100
            self._encoded[key] = (ids, labels)
        return tuple(t.to(self.device) for t in self._encoded[key])

    def loss(self, row):
        ids, labels = self.encode(row)
        # Compute logits only for supervised completion tokens, in small chunks.
        # Avoid materializing [sequence_length, 151936] fp32 logits for the entire prompt.
        hidden = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                  return_dict=True).last_hidden_state[:, :-1, :]
        target = labels[:, 1:]
        keep = target != -100
        h, y = hidden[keep], target[keep]
        if y.numel() == 0:
            raise ValueError("No completion tokens remain")

        def head_loss(features, targets):
            return F.cross_entropy(self.model.lm_head(features).float(), targets, reduction="sum")

        total = torch.zeros((), device=self.device)
        for start in range(0, len(y), 128):
            features, targets = h[start:start + 128], y[start:start + 128]
            if torch.is_grad_enabled() and features.requires_grad:
                total = total + checkpoint(head_loss, features, targets, use_reentrant=False)
            else:
                total = total + head_loss(features, targets)
        return total / len(y)

    @torch.no_grad()
    def run(self, row, token_slots=8):
        if hasattr(self, "pdq_judge"):
            from .pdq import conditioned_run
            return conditioned_run(self, row, token_slots)
        return self.raw_run(row, token_slots)

    @torch.no_grad()
    def raw_run(self, row, token_slots=8):
        self.model.eval()
        ids = self.prompt_ids(row["question"], row.get("task", "gsm8k"))
        if len(ids) >= self.settings["max_length"]:
            raise ValueError(f"Prompt for {row['id']} exceeds model.max_length")
        inputs = torch.tensor([ids], device=self.device)
        output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     max_new_tokens=self.settings["max_new_tokens"], do_sample=False,
                                     num_beams=1, use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=self.tokenizer.eos_token_id)
        tokens = output[0, len(ids):].tolist()
        response = self.tokenizer.decode(tokens, skip_special_tokens=True)
        steps = [s.strip() for s in re.split(r"\n+", response) if s.strip()]
        positions = sorted({round(i * (len(tokens) - 1) / max(1, token_slots - 1))
                            for i in range(min(token_slots, len(tokens)))})
        samples = [{"token_index": i, "token": self.tokenizer.decode([tokens[i]]),
                    "context": self.tokenizer.decode(tokens[max(0, i - 8):i + 9])} for i in positions]
        # row['answer'] is deliberately excluded from this trajectory.
        return {"question": row["question"], "response": response, "steps": steps,
                "token_samples": samples, "tools": [], "memory": []}

    @torch.no_grad()
    def reward(self, rows, metric="neg_nll"):
        if not rows:
            raise ValueError("Cannot evaluate an empty split")
        self.model.eval()
        if metric == "neg_nll":
            return -sum(float(self.loss(row)) for row in rows) / len(rows)
        if metric == "accuracy":
            return sum(score(self.run(row)["response"], row) for row in rows) / len(rows)
        raise ValueError("reward must be neg_nll or accuracy")

    @torch.no_grad()
    def evaluate(self, rows):
        predictions = []
        for i, row in enumerate(rows):
            state = self.run(row)
            predictions.append({"id": row["id"], "response": state["response"],
                                "correct": score(state["response"], row),
                                "parsed": parse(state["response"], row.get("task", "gsm8k")) is not None})
            if (i + 1) % 20 == 0:
                print(f"Evaluated {i + 1}/{len(rows)}", flush=True)
        return predictions
