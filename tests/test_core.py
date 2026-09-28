import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from jev_evolve.adapter import OrthogonalSubspaceLinear, attach_adapter
from jev_evolve.data import correct, extract_answer, load_split, prepare_data
from jev_evolve.intervention import adapt_with_guard, measure_interventions, train_selected
from jev_evolve.io import load_torch, read_json, save_torch, write_json
from jev_evolve.judge import FeatureSchema, JevClient, synthetic_response
from jev_evolve.router import delta_to_target, pairwise_ranking_loss, select_topk, train_router


OPTIONS = {"optimizer": "sgd", "lr": 0.2, "steps": 2, "grad_clip": 1.0, "max_update_norm": 1.0}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.adapter = OrthogonalSubspaceLinear(nn.Linear(12, 10), 3, 2)
        self.model = nn.Sequential(self.adapter)
        self.x = torch.randn(4, 12)
        self.y = torch.randn(4, 10)

    def loss(self):
        return (self.model(self.x) - self.y).square().mean()

    def test_zero_init_and_matrix_order(self):
        self.assertTrue(torch.equal(self.adapter(self.x), self.adapter.base(self.x)))
        with torch.no_grad():
            for p in self.adapter.R:
                p.normal_()
        weight = self.adapter.base.weight + sum(self.adapter.U[i] @ self.adapter.R[i] @ self.adapter.V[i].T for i in range(3))
        expected = self.x @ weight.T + self.adapter.base.bias
        torch.testing.assert_close(self.adapter(self.x), expected)

    def test_subspaces_frobenius_orthogonality(self):
        a = self.adapter.U[0] @ torch.randn(2, 2) @ self.adapter.V[0].T
        b = self.adapter.U[1] @ torch.randn(2, 2) @ self.adapter.V[1].T
        self.assertLess(float((a * b).sum().abs()), 1e-5)

    def test_train_only_selected_and_keep_previous_contributions(self):
        with torch.no_grad():
            self.adapter.R[0].fill_(0.2)
        before = [p.clone() for p in self.adapter.R]
        base = self.adapter.base.weight.clone()
        output = self.adapter(self.x).detach()
        self.adapter.select([1])
        torch.testing.assert_close(output, self.adapter(self.x))
        train_selected(self.model, self.adapter, [1], self.loss, OPTIONS)
        torch.testing.assert_close(before[0], self.adapter.R[0], rtol=0, atol=0)
        torch.testing.assert_close(before[2], self.adapter.R[2], rtol=0, atol=0)
        self.assertFalse(torch.equal(before[1], self.adapter.R[1]))
        self.assertTrue(torch.equal(base, self.adapter.base.weight))
        self.assertIsNone(self.adapter.R[0].grad)

    def test_candidate_order_invariance_and_total_rollback(self):
        self.model.train()
        before = copy.deepcopy(self.model.state_dict())
        first = measure_interventions(self.model, self.adapter, self.loss, lambda: -self.loss(), OPTIONS)
        second = measure_interventions(self.model, self.adapter, self.loss, lambda: -self.loss(), OPTIONS, order=[2, 0, 1])
        self.assertEqual(first["delta_reward"], second["delta_reward"])
        for k, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[k]), k)
        self.assertTrue(self.model.training)
        self.assertFalse(any(p.requires_grad for p in self.model.parameters()))

    def test_exception_restores_rng_buffers_and_weights(self):
        self.model.register_buffer("counter", torch.zeros(1))
        state = torch.get_rng_state().clone()
        python_state = random.getstate()
        np_state = np.random.get_state()
        values = self.adapter.export()["R"].clone()

        def broken():
            self.model.counter.add_(1)
            random.random()
            np.random.rand()
            torch.rand(10)
            with torch.no_grad():
                self.adapter.R[0].fill_(99)
            raise RuntimeError("intentional")

        with self.assertRaisesRegex(RuntimeError, "intentional"):
            measure_interventions(self.model, self.adapter, broken, lambda: -self.loss(), OPTIONS)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(values, self.adapter.export()["R"]))
        self.assertEqual(float(self.model.counter), 0)

    def test_guard_rejection_and_acceptance(self):
        values = self.adapter.export()["R"].clone()
        reject = adapt_with_guard(self.model, self.adapter, [0], self.loss, lambda: -self.loss(),
                                  lambda: -sum(float(p.square().sum()) for p in self.adapter.R), OPTIONS,
                                  min_gain=0, guard_tolerance=0)
        self.assertFalse(reject["accepted"])
        self.assertTrue(torch.equal(values, self.adapter.export()["R"]))
        accept = adapt_with_guard(self.model, self.adapter, [0], self.loss, lambda: -self.loss(), lambda: 0,
                                  OPTIONS, min_gain=0, guard_tolerance=0)
        self.assertTrue(accept["accepted"])
        self.assertFalse(torch.equal(values, self.adapter.export()["R"]))

    def test_no_route_does_not_call_training_or_evaluation(self):
        def bad():
            self.fail("No route must not run callbacks")
        result = adapt_with_guard(self.model, self.adapter, [], bad, bad, bad, OPTIONS)
        self.assertEqual(result["reason"], "no_positive_route")

    def test_norm_budget(self):
        options = {**OPTIONS, "lr": 100, "max_update_norm": 0.01}
        train_selected(self.model, self.adapter, [0, 1], self.loss, options)
        self.assertLessEqual(float(self.adapter.export()["R"].norm()), 0.010001)

    def test_nonfinite_reward_rolls_back(self):
        before = self.adapter.export()["R"].clone()
        with self.assertRaises(FloatingPointError):
            measure_interventions(self.model, self.adapter, self.loss, lambda: float("nan"), OPTIONS)
        self.assertTrue(torch.equal(before, self.adapter.export()["R"]))

    def test_checkpoint_roundtrip_and_independent_parameters(self):
        with torch.no_grad():
            self.adapter.R[1].fill_(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "adapter.pt"
            save_torch(file, self.adapter.export())
            payload = load_torch(file)
            restored = OrthogonalSubspaceLinear(copy.deepcopy(self.adapter.base), 3, 2,
                                                (payload["U"], payload["V"]))
            restored.import_R(payload["R"])
            torch.testing.assert_close(restored(self.x), self.adapter(self.x), rtol=0, atol=0)
            self.assertEqual(restored.basis_id(), self.adapter.basis_id())
        self.assertEqual(len({p.data_ptr() for p in self.adapter.R}), 3)

    def test_invalid_dimensions_and_unfrozen_base_rejected(self):
        with self.assertRaises(ValueError):
            OrthogonalSubspaceLinear(nn.Linear(3, 3), 2, 2)
        self.adapter.base.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            measure_interventions(self.model, self.adapter, self.loss, lambda: 0, OPTIONS)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.schema = FeatureSchema(2, 1)
        self.state = {"question": "2+3?", "response": "#### 5", "steps": ["#### 5"], "token_samples": []}
        self.active = self.schema.active_questions(self.state)
        self.raw = synthetic_response(self.active)

    def test_probability_order_and_missing_masks(self):
        self.raw["answers"]["reasoning"]["probabilities"] = {"4": 0.7, "0": 0.3}
        self.raw["answers"]["reasoning"]["score"] = 2.8
        vector = self.schema.vector(self.raw, self.active)
        values = dict(zip(self.schema.names, vector))
        self.assertEqual(values["reasoning.p_4"], 0.7)
        self.assertEqual(values["reasoning.p_1"], 0)
        self.assertEqual(values["tool_effective.present"], 0)
        self.assertEqual(values["step_0_valid.present"], 1)
        self.assertEqual(values["step_1_valid.present"], 0)
        self.assertEqual(len(vector), len(self.schema.names))

    def test_missing_response_and_nonfinite_probabilities_fail(self):
        del self.raw["answers"]["success"]
        with self.assertRaisesRegex(ValueError, "omitted"):
            self.schema.vector(self.raw, self.active)
        self.raw = synthetic_response(self.active)
        self.raw["answers"]["success"]["noul"] = float("nan")
        with self.assertRaises(ValueError):
            self.schema.vector(self.raw, self.active)

    def test_zero_padding_is_not_used_for_malformed_score(self):
        self.raw["answers"]["reasoning"]["probabilities"] = [0, 0, 0, 0, 1]
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.schema.vector(self.raw, self.active)

    def test_schema_changes_when_question_count_changes(self):
        self.assertNotEqual(self.schema.id, FeatureSchema(3, 1).id)

    def test_raw_cache_keeps_ground_truth_out_of_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            judge = JevClient({"provider": "synthetic", "model": "test", "endpoint": "https://api.typesafe.ai/v1/systemone",
                               "max_steps": 2, "token_slots": 1, "cache_dir": tmp})
            state = {**self.state, "answer": "SECRET-GOLD", "reference": "SECRET-GOLD"}
            first, _ = judge.evaluate(state)
            second, _ = judge.evaluate(state)
            self.assertEqual(first, second)
            cache = next(Path(tmp).glob("*.json"))
            self.assertNotIn("SECRET-GOLD", cache.read_text())

    def test_http_auth_failure_does_not_retry_or_leak_key(self):
        judge = JevClient({"provider": "typesafe", "model": "test", "endpoint": "https://api.typesafe.ai/v1/systemone",
                           "max_steps": 0, "token_slots": 0, "timeout": 1, "retries": 3, "cache_dir": "unused"})
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "secret"}), patch("requests.post") as request:
            request.return_value.status_code = 401
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                judge._request({})
            self.assertEqual(request.call_count, 1)


class RouterAndDataTests(unittest.TestCase):
    def test_targets_per_episode_and_no_positive(self):
        d = torch.tensor([[-0.03, 0.15, 0.01], [-2., -1., 0.], [1., 2., 3.]])
        target = delta_to_target(d)
        torch.testing.assert_close(target[0], torch.tensor([0., 1., 1 / 15]))
        self.assertTrue(torch.equal(target[1], torch.zeros(3)))
        torch.testing.assert_close(target[2], torch.tensor([1 / 3, 2 / 3, 1.]))

    def test_tied_ranking_loss_can_backpropagate(self):
        logits = torch.randn(2, 3, requires_grad=True)
        loss = pairwise_ranking_loss(logits, torch.zeros(2, 3))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ranking_direction_and_abstention(self):
        delta = torch.tensor([[1., 2., 3.]])
        self.assertLess(float(pairwise_ranking_loss(delta, delta)), float(pairwise_ranking_loss(-delta, delta)))
        self.assertEqual(select_topk(torch.tensor([0., 0.1, 0.2]), 2, 0.5), [])
        self.assertEqual(select_topk(torch.tensor([0.1, 0.9, 0.8]), 2, 0.5), [1, 2])

    def test_numerical_answer_parser(self):
        self.assertTrue(correct("Work\n#### 1,234.00", "#### 1234"))
        self.assertTrue(correct("#### -0.5", "#### -.50"))
        for malformed in ("5", "#### 1/2", "#### 1e3", "#### 1,23", "#### 5 apples"):
            self.assertIsNone(extract_answer(malformed), malformed)

    def test_data_split_disjoint_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            for split, count in (("train", 30), ("test", 4)):
                (source / f"{split}.jsonl").write_text("\n".join(
                    __import__("json").dumps({"question": f"{split} {i}?", "answer": f"#### {i}"}) for i in range(count)), encoding="utf-8")
            out = Path(tmp) / "dataset"
            prepare_data(out, offline=10, probe=3, guard=3, online=3, source_dir=source)
            sets = [{r["id"] for r in load_split(out, name)} for name in ("offline", "probe", "guard", "online", "test")]
            self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
            with (out / "test.jsonl").open("a") as stream:
                stream.write('{"id":"tampered"}\n')
            with self.assertRaisesRegex(ValueError, "changed"):
                load_split(out, "test")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
