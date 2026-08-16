# Reproducing the CoRe results

Every configuration below corresponds to a number reported in the paper. Run
each with `baseline.py`, then score with `evaluate.py`.

```bash
export HF_HOME=/workspace/.hf_home          # keep model weights off the small container disk
python3 baseline.py -c configs/core.yaml -i data/val.jsonl
python3 evaluate.py -p output/core.jsonl -g data/val.jsonl -o output/core-scores.csv
```

## Configurations

| Config | Model | Routing | Val macro-F1 |
|---|---|---|---|
| `gemma-baseline.yaml` | Gemma-4-31B-it | all CoT | 0.522 |
| `ablation-all-cot.yaml` | Qwen3.5-27B | all CoT | 0.511 |
| `ablation-all-direct.yaml` | Qwen3.5-27B | all direct | 0.510 |
| `core-base.yaml` | Qwen3.5-27B | CoRe | 0.542 |
| `core.yaml` | Qwen3.5-27B | CoRe + rewritten `company` prompt | **0.548** |
| `core-company-cot.yaml` | Qwen3.5-27B | as above, `company` moved to CoT | 0.546 |

`core_base.yaml` is the best validation configuration. `core.yaml` is
the one measured on the test split (0.5408 at a 2000-token CoT budget); the
token-budget sweep in the paper varies `max_new_tokens` in that file over
300 / 500 / 2000.

Note that `core_base.yaml` and `core.yaml` differ only in whether
`companyTradesAtStockExchange` sits in `cot_relations`.

## Single-relation runs

Iterating on one relation is much cheaper than a full 478-row run:

```bash
./run_relation.sh personHasCityOfDeath configs/core.yaml
```

Scores from a single-relation run are **not** comparable to that relation's row
in a full run: batch composition differs, and we measured gaps of 3 to 5 points
under otherwise identical configurations. Use single-relation runs to screen
ideas, and confirm anything you intend to keep with a full run.

## Key config fields

| Field | Meaning |
|---|---|
| `cot_relations` | Relations answered with a reasoning trace ending in `Final Answer:`. Everything else is answered directly. |
| `numeric_relations` | Parsed as a single number rather than a comma-separated entity list. |
| `never_abstain_relations` | Fall back to the training-split median when no answer parses. Gold is never empty for these, so abstaining always scores zero. |
| `max_new_tokens` | Token budget on the CoT path. |
| `max_new_tokens_per_relation` | Per-relation override. Splits the CoT rows into separate batch groups, which shifts padding and therefore other relations' scores. Use with care. |
| `direct_max_new_tokens` | Token budget on the direct path. |
| `seed` | Exemplar selection is a hash of (seed, subject, relation), so it is reproducible and independent of processing order. |

## Approaches that did not work

These were implemented, measured, and removed. They are listed so the same
ground is not re-covered:

- **Self-consistency on numeric relations.** Median of 5 samples at
  temperature 0.7 broke 4 previously-correct `hasArea` rows and rescued 1.
  Sampled completions cluster around a single misremembered value rather than
  scattering around the truth, so resampling cannot fix errors that are not
  variance-driven.
- **Cloze / sentence-completion prompting.** Bypassing the chat template put
  the instruction-tuned model out of distribution: `companyTradesAtStockExchange`
  fell from 0.624 to 0.133.
- **Exemplar selection strategies** (nearest-name, precision-matched,
  capped-empty). All within measurement noise. `hasArea` in particular shows
  no correlation between exemplar magnitude and prediction bias, so the model
  is not anchoring on exemplars for that relation.
- **Prompt ensembling.** Implemented following Biester et al. but never run
  end to end; removed rather than shipped untested.
- **A confabulation guard for `personHasCityOfDeath`.** Forbidding
  birthplace/capital substitutions raised abstentions from 63 to 79 against a
  gold-empty rate of 39, costing 3 points.
