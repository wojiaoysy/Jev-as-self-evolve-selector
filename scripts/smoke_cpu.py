"""No downloads, no API key: real tiny Qwen forward/backward + interventions + router."""
import json
import argparse

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from jev_evolve.adapter import attach_adapter
from jev_evolve.intervention import adapt_with_guard, measure_interventions
from jev_evolve.judge import FeatureSchema, synthetic_response
from jev_evolve.model import QwenAgent
from jev_evolve.router import train_router, select_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    vocab = {word: i for i, word in enumerate(["[UNK]", "[PAD]", "[EOS]", "2", "3", "5", "+", "=", "####", "What", "is", "?", "assistant", "user", "system"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                        attention_dropout=0.0, eos_token_id=2, pad_token_id=1)
    model = Qwen2ForCausalLM(config)
    if args.device == "cuda":
        model.to(device="cuda", dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter = attach_adapter(model, "model.layers.0.self_attn.q_proj", 2, 2)
    agent = QwenAgent(model, tokenizer, {"max_length": 128, "max_new_tokens": 4})
    row = {"id": "smoke", "question": "What is 2 + 3 ?", "answer": "2 + 3 = 5 #### 5"}
    options = {"optimizer": "adamw", "lr": 0.01, "steps": 2, "grad_clip": 1.0, "max_update_norm": 1.0}
    original = adapter.export()["R"].clone()
    # Verify chunked completion-only CE agrees with the model's standard loss.
    model.eval()
    ids, labels = agent.encode(row)
    with torch.no_grad():
        reference = model(input_ids=ids, labels=labels, use_cache=False).loss
        torch.testing.assert_close(agent.loss(row), reference, atol=2e-5, rtol=2e-5)
    result = measure_interventions(model, adapter, lambda: agent.loss(row), lambda: agent.reward([row]), options)
    assert torch.equal(original, adapter.export()["R"]), "Intervention leaked a model update"
    assert all(torch.isfinite(torch.tensor(result["delta_reward"])))
    trajectory = agent.run(row, token_slots=2)
    assert "answer" not in trajectory
    schema = FeatureSchema(2, 2)
    active = schema.active_questions(trajectory)
    features = schema.vector(synthetic_response(active, trajectory), active)
    records = [{"jev_features": features, "delta_reward": result["delta_reward"]} for _ in range(6)]
    rs = {"hidden": 8, "epochs": 3, "batch_size": 2, "lr": 0.01, "weight_decay": 0,
          "rank_weight": 0.2, "sparsity_weight": 0.001, "gain_floor": 1e-8, "top_k": 1, "threshold": 0.5}
    router, history = train_router(records[:4], records[4:], rs)
    alpha = router(torch.tensor(features).unsqueeze(0))[0]
    selected = select_topk(alpha, 1, 0.0)
    accepted = adapt_with_guard(model, adapter, selected, lambda: agent.loss(row), lambda: agent.reward([row]),
                                lambda: agent.reward([row]), options, min_gain=0, guard_tolerance=0)
    print(json.dumps({"status": "PASS", "device": args.device, "judge": "SYNTHETIC-NOT-JEV", "delta_reward": result["delta_reward"],
                      "feature_dimension": len(features), "route": selected, "update": accepted,
                      "router_val_loss": history[-1]["loss"]}, indent=2))


if __name__ == "__main__":
    main()
