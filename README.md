# CoRe: Construct-versus-Recall Routing

System for the [AKBC Shared Task 2026](https://lm-kbc.github.io/challenge2026/)
on knowledge base construction from language models. Given a subject entity and
a relation, predict the object entities using only the model's parametric
knowledge: no retrieval, no fine-tuning, and at most 32B parameters in a single
model repository.

CoRe routes each relation to chain-of-thought or direct answering depending on
whether its answer must be **constructed** or merely **recalled**. On the
validation split it reaches macro-F1 **0.548**, against 0.511 for a
CoT-everywhere policy and 0.510 for a direct-everywhere policy under the same
model.

This is a fork of [lm-kbc/dataset2026](https://github.com/lm-kbc/dataset2026).
The data, the official `evaluate.py`, and the task definition are the
organizers'; the original task README is preserved as
[README_upstream_task.md](README_upstream_task.md). Everything under `models/`
and `configs/` beyond the stock baseline is ours.

## Quick start

```bash
conda create -n lm-kbc-2026 python=3.11 -y && conda activate lm-kbc-2026
pip install -r requirements.txt

export HF_HOME=/workspace/.hf_home        # keep 60GB of weights off the container disk
hf download Qwen/Qwen3.5-27B

python3 baseline.py -c configs/core.yaml -i data/val.jsonl
python3 evaluate.py -p output/core.jsonl -g data/val.jsonl -o output/scores.csv
```

Needs one 80GB GPU in bf16. On smaller cards set `use_quantization: true` and
`batch_size: 1` in the config, at some cost in accuracy.

To iterate on a single relation instead of all 478 rows:

```bash
./run_relation.sh personHasCityOfDeath configs/core.yaml
```

Scores from a single-relation run are **not** comparable to that relation's row
in a full run. See [REPRODUCE.md](REPRODUCE.md).

## Results

Validation macro-F1, Qwen3.5-27B unless noted:

| Configuration | Overall |
|---|---|
| Gemma-4-31B-it, CoT everywhere | 0.522 |
| CoT everywhere | 0.511 |
| Direct everywhere | 0.510 |
| CoRe routing | 0.542 |
| CoRe + rewritten `company` prompt (`core.yaml`) | **0.548** |

Per relation, CoT and direct answering win on opposite halves of the task:

| Relation | CoT | Direct |
|---|---|---|
| awardWonBy | **0.123** | 0.078 |
| hasArea | **0.610** | 0.480 |
| hasCapacity | **0.180** | 0.160 |
| countryLandBordersCountry | 0.939 | **0.978** |
| personHasCityOfDeath | 0.380 | **0.500** |
| companyTradesAtStockExchange | 0.624 | 0.624 |

Relations whose answer has to be assembled (a long recipient list, a two-step
"is this company listed, and if so where" decision) gain from reasoning space.
Single-valued numeric recall is hurt by it: extra deliberation gives the model
room to talk itself out of a memory it already had right.

The largest single gain in the whole project came from changing the base model
rather than from any prompting change. Under an identical pipeline,
Gemma-4-31B-it answered 37/100 `hasArea` queries within tolerance against
61/100 for Qwen3.5-27B. What limits this task at 32B is which facts the model
memorised, not how well it reasons.

## How it works

**Rationalized few-shot.** For CoT relations, each exemplar's question is paired
not with its gold answer directly but with a reasoning trace the model generates
itself, given the answer as already known. This demonstrates reasoning *style*
without hand-writing chains of thought per relation. Rationales are cached by
`(subject, relation)` and reused across queries and runs.

**Deterministic exemplar selection.** Exemplars are drawn by a hash of
`(seed, subject, relation)`, so a given row always sees the same 5 exemplars
regardless of what else is in the batch or what order rows are processed in.
This matters: a shared random stream made exemplar choice depend on processing
order, which silently confounded otherwise-controlled comparisons.

**Never-abstain on numerics.** `hasArea` and `hasCapacity` gold is never empty,
so an empty prediction always scores zero while a wrong guess scores the same as
abstaining. Unparseable answers fall back to the training-split median.

**Relation-aware parsing.** Numeric relations are parsed as a single number,
with units, thousands separators and scale words handled; string relations are
comma-split with a prose filter that rejects sentence fragments, calibrated so
that none of the 63k gold aliases would be rejected.

## Repository layout

```
configs/            one YAML per reported configuration
models/
  hf_causal_model.py    the system: routing, rationalization, parsing
  user_config.py        registry mapping config keys to classes
  abstract_model.py     upstream interface
prompt_templates/   per-relation question templates
data/               official train/val/test splits
baseline.py         runner
evaluate.py         official scorer, plus a -r flag to score one relation
run_relation.sh     single-relation convenience wrapper
REPRODUCE.md        config-to-result map, and what did not work
```

## A warning about measurement

Under batched greedy decoding, left-padding length depends on which other rows
share a decoding batch, which depends on which relations are active in a run.
Moving one relation between reasoning paths shifted **other, untouched**
relations' scores by up to 5 points. Two consequences:

- Relation-level scores are comparable across runs only when batch composition
  is held fixed. Changing `cot_relations`, or adding a
  `max_new_tokens_per_relation` override, changes it.
- Differences below roughly 3 points should be read as directional, not
  established.

We twice concluded an intervention had helped when it had not, by comparing a
single-relation run against a full-run baseline. [REPRODUCE.md](REPRODUCE.md)
lists the approaches that turned out to be dead ends, including
self-consistency sampling, cloze prompting, and three exemplar-selection
strategies.

## Task reference

Six relations, roughly 100 rows per split each (68 for
`countryLandBordersCountry`, 10 for `awardWonBy`). Three of them permit an empty
answer set; two are numeric and scored within 5% relative tolerance; string
relations are matched against multi-alias gold sets after normalization.

Full relation definitions, dataset statistics, submission format, and the
official leaderboard links are in
[README_upstream_task.md](README_upstream_task.md). Read the relation
definitions before changing any prompt template: the gold sets follow them
closely, and several of our early errors came from prompting for a reasonable
but different interpretation.
