# Jev-Guided Subspace Adaptation

An experimental study of whether Jev can guide useful parameter updates in a llm agent.

This repository explores two approaches: learning a router that selects low-rank subspaces, and learning a factorization that maps Jev features to a weight residual. The saved experiments do not establish a consistent capability improvement from either approach. In particular, repeatedly accumulating the learned residual causes substantial degradation under the tested configuration.

The long-term motivation is agent self-evolution. The current implementation is a supervised parameter-adaptation research prototype using Qwen2.5-1.5B-Instruct on GSM8K and BoolQ. It performs single-response task rollouts; it does not implement a tool-using agent environment. Router interventions and offline factor training use reference answers. Fixed-factor PDQ accumulation uses Jev features without online gradient training, but its factors were learned with supervision.


## Motivation

[Jev](https://docs.typesafe.ai/introduction) is TypeSafe's model for fast, structured judgments. It evaluates a state against typed questions and returns values or distributions. This motivates using it as a diagnostic component with relatively little evaluation overhead. This repository does not independently establish Jev's speed advantage or broad generalization, and those properties would not by themselves guarantee useful parameter updates.

The research question is: Can a diagnosis of a model's output identify how its parameters should change? Recognizing an error and selecting a beneficial weight update require different information. The experiments test a bridge between them.

## Methods

### 1. Jev features → router → selected subspace training

For one frozen linear projection, the adapter is

$$
W' = W_0 + \sum_{i=1}^{K} U_i R_i V_i^\top.
$$

The fixed bases $U_i,V_i$ are grouped singular vectors of the original weight; each trainable $R_i$ has shape $r\times r$. This is a constrained low-rank adapter with a trainable core, rather than standard LoRA with two freely trained factors.

1. Generate a draft and extract a fixed feature vector from Jev's judgments.
2. From a common model checkpoint, train each candidate subspace separately on the reference completion and measure its change in probe reward.
3. Train a router on these intervention labels, using regression, ranking, and sparsity losses.
4. At adaptation time, rank subspaces and train at most Top-k of them, subject to a score threshold. The router may abstain.

The default reward is negative completion NLL, a proxy for final task accuracy. Jev's HTTP response is a feature source; it does not supply a differentiable gradient through the API.

Router scores only control selection. All previously learned subspace contributions remain in the forward pass. Consequently, scores below 1 cannot explain small weight updates through coefficient attenuation in this implementation.

With $K=8,r=8$, the adapter contains 512 trainable scalars; Top-2 trains 128 per intervention. Only matching blocks $U_iR_iV_i^\top$ are available, not arbitrary cross-block updates. Orthogonality in weight space does not imply independent effects on model behavior.

### 2. Jev features → diagonal coefficients → learned P/Q factors

Let $e\in\mathbb{R}^m$ be the serialized Jev feature vector. The residual is

$$
\Delta W(e)=P\,\mathrm{diag}(e)\,Q
           =\sum_{j=1}^{m}e_jp_jq_j^\top,
$$

where $P\in\mathbb{R}^{d_{out}\times m}$ and $Q\in\mathbb{R}^{m\times d_{in}}$. The diagonal matrix is represented as a vector in code.

Here $m=270$ for the configured feature schema, not the router's block rank $r=8$. The features include probabilities, presence masks, confidence values, and aggregate statistics. The residual has rank at most $\min(m,d_{out},d_{in})$; it need not have rank exactly $m$. For the selected 1536-by-1536 projection, P/Q contain 829,440 trainable parameters.

P/Q are trained offline on the router's training episode IDs and selected by loss on its validation IDs. Once fixed, they define a dictionary of at most $m$ rank-one matrices: varying Jev features changes coefficients within that dictionary. It cannot introduce a new direction outside its span, and the actual feature range can constrain the attainable updates further.

Three execution modes are implemented:

| Mode | Behavior |
| --- | --- |
| `dynamic` | Freeze P/Q; generate a draft, obtain Jev features, and use an input-conditioned residual for the final response. No persistent online weight learning occurs. |
| `accumulate` | Freeze P/Q and commit each residual: $W_t=W_0+P\,\mathrm{diag}(\sum_{s\leq t}\eta e_s)Q$. Evaluation reads the committed weights without learning from evaluation examples. |
| `continual` | Continue supervised online training of P/Q using fresh Jev features; not represented in the result tables below. |

The CLI currently defaults to `accumulate`; pass the mode explicitly. Accumulation accepts each finite residual directly. It does not use the router's gradient-update norm cap or probe/guard rollback.

## Saved experimental results

These are existing local results, copied into [docs/results](docs/results) for publication; they were not rerun while preparing this README. Values are mean ± sample standard deviation across seeds 42, 43, and 44, expressed in percent. They are not confidence intervals.

The BoolQ development set and GSM8K retention set each contain 128 examples held out from adaptation splits. These are development/retention results, not full official test scores. One changed answer corresponds to 0.78125 percentage points within one seed.

Shared configuration: Qwen2.5-1.5B-Instruct, layer 14 `q_proj` (zero-based index), BF16 backbone, FP32 adapter arithmetic, $K=8,r=8$, Top-2 routing, maximum sequence length 2048, generation budget 384, and Jev `jev-1.13.0`. Router/random/all use two AdamW steps per selected episode at learning rate 0.01. Offline PDQ defaults to three epochs and learning rate 0.0001. All listed suites use `direct` updates, without probe/guard acceptance checks.

### Fixed P/Q with accumulated updates: 80 online episodes

Source: [accumulation summary](docs/results/boolq_accumulate80/final_dev_summary.json), [suite settings](docs/results/boolq_accumulate80/suite.json). Accumulation scale is 1.0.

| Policy | BoolQ accuracy (%) | GSM8K retention accuracy (%) |
| --- | ---: | ---: |
| Frozen base | 78.12 ± 0.00 | 51.56 ± 0.00 |
| Jev router | 77.86 ± 0.45 | 49.48 ± 1.63 |
| Random Top-2 | 78.12 ± 0.00 | 47.92 ± 2.39 |
| All 8 subspaces | 77.34 ± 0.78 | 51.30 ± 2.96 |
| Jev PDQ | 52.60 ± 1.19 | 1.04 ± 0.45 |
| Jev router, forced Top-2 | 78.12 ± 0.00 | 51.04 ± 3.16 |
| PDQ, shuffled features | 66.93 ± 5.20 | 4.95 ± 1.19 |
| PDQ, identity diagonal | 48.96 ± 3.61 | 0.78 ± 0.00 |

Accumulating Jev-conditioned residuals produces a large deterioration on both tasks. Shuffling the diagonal or replacing it with the identity does not solve accumulation instability. This does not isolate whether direction quality, scale, repeated addition, or their interaction is responsible.

### Fixed P/Q with dynamic conditioning: 200 episodes processed

Source: [dynamic summary](docs/results/boolq_five_dynamic/final_dev_summary.json), [suite settings](docs/results/boolq_five_dynamic/suite.json).

| Policy | BoolQ accuracy (%) | GSM8K retention accuracy (%) |
| --- | ---: | ---: |
| Frozen base | 78.12 ± 0.00 | 49.22 ± 0.00 |
| Jev router | 77.60 ± 0.45 | 50.52 ± 1.19 |
| Random Top-2 | 78.39 ± 0.45 | 49.48 ± 3.16 |
| All 8 subspaces | 74.22 ± 2.07 | 51.82 ± 3.69 |
| Jev PDQ | 77.60 ± 0.45 | 45.05 ± 3.93 |

For dynamic PDQ, the episode count is a monitoring index, not a count of permanent PDQ updates. Its fixed factors are already trained before episode zero. Other adaptation policies can change their persistent parameters along the same stream.

The [80-episode dynamic ablations](docs/results/boolq_ablations80/final_dev_summary.json) additionally contain forced Top-2 routing, shuffled features, and an identity diagonal. Identity PDQ reaches 79.43% BoolQ accuracy versus 78.13% for the frozen base, while GSM8K retention is 42.45% versus 50.78%. That tradeoff does not establish a benefit from Jev conditioning. PDQ controls reuse the same factors trained with real features; they are substitution ablations, not independently trained, matched baselines.

Across these tables, the router has no consistent target-task advantage.

### Evaluation limitations

- The frozen-base GSM8K retention means differ across saved suites (49.22%, 50.78%, and 51.56%) despite matching declared retention data and evaluation settings. One suite also varies across seeds. This remains an unresolved reproducibility issue; small retention differences and comparisons across suites should be treated cautiously.
- [An earlier full GSM8K evaluation](docs/results/gsm8k_legacy_summary.json) scored the frozen base at 1.59%, with 1,285 parsing failures out of 1,319 examples. These scores measure the saved generation/parsing pipeline and cannot cleanly diagnose reasoning ability. The paired difference intervals in that comparison cross zero.
- Three seeds and small development sets are insufficient to establish broad agent generalization. The tables provide descriptive outcomes, not a causal explanation or a significance claim.
- Update budgets differ: random always selects two blocks, thresholded routing may select fewer, `all` trains eight, and P/Q have a much larger offline parameter budget. Dynamic PDQ also requires a draft, a Jev call, and a second generation per evaluated input.

## Interpretation and next experiments

**Supported observation:** under the recorded settings, neither proposed Jev mechanism demonstrates a consistent capability gain, and unscaled residual accumulation is especially harmful.

**Plausible explanation for PDQ:** factors learned on a limited dataset may fail to contain useful directions for new tasks or later model states. Jev can only reweight the learned dictionary. However, fixed factors alone do not prove poor generalization; the dictionary could transfer if its directions were useful. The current results do not isolate this cause.

**Additional concern for accumulation:** P/Q are trained to produce a useful single conditioned residual, while deployment repeatedly adds such residuals. Nonzero-mean features can repeatedly reinforce shared components, creating a training/deployment mismatch and growing update magnitude. This is a hypothesis to test with norm and alignment measurements, not a demonstrated diagnosis from accuracy alone.

**Plausible explanations for routing:** a restricted single-layer update space, weak or noisy intervention labels, NLL/accuracy mismatch, abstention, and drift from the fixed checkpoint used to train the router. Small router scores can suppress updates through thresholding, but do not attenuate selected updates in the current code. Low rank itself does not imply weak behavioral effects.

Useful follow-up experiments:

1. Measure router coverage and compare forced Top-2, random Top-2, and a fixed Top-2 policy with matched steps and update norms. Recompute intervention rankings at later model checkpoints to measure routing drift.
2. Measure $\|\Delta W\|_F/\|W_0\|_F$, changes in logits/loss, and alignment with a supervised descent direction. Separate insufficient update magnitude from an unhelpful direction.
3. Compare PDQ with norm-matched shuffled, identity, and constant-feature controls, and train corresponding control factors independently. Current identity substitution also changes update scale.
4. Sweep accumulation scale, compare centered or signed features, and train for the actual cumulative deployment objective. Evaluate whether rollback or a trust-region constraint helps.
5. Test larger or adaptive dictionaries, multiple target layers, and held-out task families. Keep an independent final test set and report forgetting, parsing failures, latency, and API cost.

`scripts/diagnose_pdq.py` computes offline feature variation, residual norms, angles, and effective ranks from stored records/checkpoints. These diagnostics can reveal a nearly constant residual dictionary response, but do not by themselves establish alignment with the best task update.

## Setup and reproduction

The project targets Python 3.10, PyTorch 2.1.2 with CUDA 11.8, and an RTX 4090 24 GB environment. Other hardware/software combinations are not established by this README. `requirements.txt` pins the remaining core dependencies; `pip install -e .` alone does not install them.

On the matching AutoDL image with PyTorch already installed:

```bash
bash scripts/setup_autodl.sh
source .venv/bin/activate
python -m pip install 'matplotlib>=3.7,<3.9'
python scripts/smoke_cpu.py
python -m jev_evolve.cli --help
```

The CPU smoke check uses a tiny random model and synthetic features. It needs neither a model download nor a Jev key, and verifies plumbing rather than accuracy.

For a real run, provide `TYPESAFE_API_KEY`. In Bash, this avoids placing its value in the command history:

```bash
read -rsp 'TypeSafe API key: ' TYPESAFE_API_KEY; echo
export TYPESAFE_API_KEY
```

The client uses the [TypeSafe HTTP endpoint](https://docs.typesafe.ai/api). Configurations pin `jev-1.13.0`; availability depends on the service. Changing the judge version, feature schema, data, or intervention configuration requires compatible new records and checkpoints.

### Prepare a clean checkout

Use fresh output paths if data or checkpoints already exist. The scripts enforce dataset/checkpoint contracts and do not overwrite incompatible experiments. `configs/local.json` and `configs/boolq.json` contain the original machine's absolute model path; the following creates portable run configurations instead.

```bash
python -m jev_evolve.cli download-model \
  --output models/qwen2.5-1.5b-instruct

python - <<'PYCONFIG'
import json
from pathlib import Path
cfg = json.loads(Path('configs/autodl_4090.json').read_text())
cfg['model']['name'] = 'models/qwen2.5-1.5b-instruct'
with open('configs/gsm8k_run.json', 'x') as f:
    json.dump(cfg, f, indent=2)
PYCONFIG

python -m jev_evolve.cli prepare-data --output data/gsm8k
python -m jev_evolve.cli prepare-dev \
  --data data/gsm8k --output data/gsm8k_retention128.json --size 128 --seed 2026
python scripts/prepare_boolq.py \
  --config configs/gsm8k_run.json --output-config configs/boolq_run.json \
  --output data/boolq --seed 42 --offline 256 --probe 64 --guard 64 --online 200 --dev 128
python -m jev_evolve.cli init \
  --config configs/boolq_run.json --output runs/boolq_initial.pt
```

BoolQ's official validation set becomes the project's `test` split; its monitoring development set is separately sampled from unused training examples. GSM8K and BoolQ source hashes and split manifests are recorded. Downloads resolve the currently requested upstream revision: compare hashes against archived metadata before claiming an exact reproduction of an earlier run.

### Run the two main protocols

The suite trains a router and P/Q separately for each seed, then runs policies sequentially. It makes real API calls and may take substantial time. The first command reproduces the dynamic protocol; the second reuses those offline checkpoints to test accumulation and ablations.

```bash
python scripts/run_learning_curves.py \
  --config configs/boolq_run.json --data data/boolq \
  --adapter runs/boolq_initial.pt --dev data/boolq/dev_fixed.json \
  --retention-data data/gsm8k --retention-dev data/gsm8k_retention128.json \
  --output runs/reproduce_dynamic200 --seeds 42 43 44 \
  --episodes 200 --eval-every 20 --with-pdq --pdq-mode dynamic --update-mode direct

python scripts/run_learning_curves.py \
  --config configs/boolq_run.json --data data/boolq \
  --adapter runs/boolq_initial.pt --dev data/boolq/dev_fixed.json \
  --retention-data data/gsm8k --retention-dev data/gsm8k_retention128.json \
  --output runs/reproduce_accumulate80 --seeds 42 43 44 \
  --episodes 80 --eval-every 20 --ablations --pdq-mode accumulate \
  --pdq-accumulation-scale 1.0 --update-mode direct \
  --reuse-offline runs/reproduce_dynamic200
```

For the 80-episode dynamic ablations, use the second command with `--pdq-mode dynamic`, omit `--pdq-accumulation-scale`, and choose a new output directory. Identical commands can resume saved progress; changed settings require a new directory.

`collect`, `train-router`, `train-pdq`, `adapt`, `evaluate`, and `compare` are also available individually. `adapt --update-mode guarded` enables probe/guard acceptance for supported non-PDQ policies. The older [AutoDL walkthrough](docs/AUTODL_zh.md) and [protocol notes](docs/EXPERIMENT_zh.md) describe the original router workflow; specify modes explicitly when following them, since the current CLI defaults to direct updates.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/jev_evolve/adapter.py` | Fixed orthogonal bases and trainable core matrices |
| `src/jev_evolve/router.py` | Intervention-label learning and subspace selection |
| `src/jev_evolve/pdq.py` | Learned P/Q factors, conditioning, accumulation, and controls |
| `src/jev_evolve/judge.py` | Typed Jev requests, feature validation, and response caching |
| `src/jev_evolve/intervention.py` | Common-start interventions, direct updates, and guarded rollback |
| `src/jev_evolve/cli.py` | Data, training, adaptation, and evaluation commands |
| `scripts/run_learning_curves.py` | Multi-seed suite runner |
| `scripts/diagnose_pdq.py` | Offline geometry diagnostics |
| `docs/results/` | Published aggregate results, per-seed curves, and suite metadata |
| `tests/` | Engineering checks for the implementation |

Run core checks with `python -m unittest discover -s tests -v`. Generated datasets, model weights, full runs, caches, and virtual environments are excluded from Git. The published snapshots omit per-example responses and model checkpoints, so they support inspection of reported aggregates rather than a complete independent audit of every prediction.

The repository currently has no project license file. Third-party model and dataset terms apply separately; choose a project license before presenting the code as licensed open source.
