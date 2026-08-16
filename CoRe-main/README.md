# AKBC Shared Task 2026: Knowledge Base Construction from Language Models (5th Edition)

This repository hosts data for the [AKBC Shared Task](https://lm-kbc.github.io/challenge2026/) at [AKBC](https://www.akbc.ws/2026/) / [EMNLP 2026](https://2026.emnlp.org/) in Budapest.

This repository contains:

- The [dataset](data/) for the shared task
- [Evaluation script](evaluate.py)
- [Baseline code](models/)
- Instructions for submitting your predictions

## Table of contents

1. [News](#news)
2. [Challenge overview](#challenge-overview)
3. [Dataset](#dataset)
4. [Relation definitions](#relation-definitions)
5. [Evaluation metrics](#evaluation-metrics)
6. [Getting started](#getting-started)
    - [Setup](#setup)
    - [Baselines](#baselines)
    - [How to structure your prediction file](#how-to-structure-your-prediction-file)
    - [Submit your predictions](#submit-your-predictions)

## News

- **July 2026**: Ground-truth quality release + evaluator update. **Splits and `test.jsonl` are unchanged** — only gold answer sets were corrected (val: 21 rows, train: 25 rows, plus the private test key; ~160 further rows per split gained additional aliases only). Details:
    - The ground truth now reflects the state of the world **as of 1 July 2026**: city-of-death entries for people who died 2022–2026, current stock-exchange listings (delistings such as RPS Group and Daimler/Mercedes-Benz removed; missing listings such as Bharti Airtel's NSE listing added), and award winners through the most recent editions.
    - Award rows cleaned and completed: winning *works* (books, albums) were replaced by their authors/artists; recipients of similarly-named but distinct awards were removed (e.g. the 1945 Medal of Freedom vs the *Presidential* Medal of Freedom); rescinded awards are excluded; most award rows are now verified-complete winner lists.
    - Border corrections (e.g. Djibouti +Somalia, Denmark +Canada) and further hectare→km² unit fixes for `hasArea`.
    - Alias sets additionally include Wikipedia sitelink titles and Latin-script labels from all languages, making matching robust to name-order variants (e.g. Hungarian "Family-name First-name" ≡ Western "First-name Family-name", "Soong Mei-ling" ≡ "Soong May-ling").
    - `evaluate.py` normalization improved: apostrophe-like marks (`'`, `’`, `ʻ`) are dropped, **all** Unicode punctuation acts as a separator (previously ASCII-only), and case-folding is applied after Unicode decomposition (`ß` → `ss`). `O'Brien` / `O’Brien` / `OBrien` and `Kaua'i` / `Kauaʻi` / `Kauai` now all match. **Please re-download `evaluate.py`** — the change only converts former misses into matches.
- **May 2026**: Dataset cleanup release. Changes participants should be aware of:
    - Object alias sets are now richer per entity (full Latin-language label + aliases pulled from Wikidata), giving the evaluator more matching surface area. Non-Latin scripts, social-media handles (`@...`), ordinal abbreviations (`POTUS 45`), Wikipedia list articles, and orphaned name suffixes (`Jr.`/`Sr.`) have been filtered out.
    - Subjects that previously contained raw Q-IDs (`"Q5847811 in Lima"`) are replaced with the resolved Wikidata label (`"Estadio Caballeros del Deporte in Lima"`).
    - The val row for `"Ireland" (hasArea)` is now labeled `"Island of Ireland"` to disambiguate it from the Republic of Ireland used elsewhere in the dataset.
    - Five `hasArea` values were stored in the wrong unit (hectares or square miles) and have been corrected to km²: Isle of Bute, South Uist, Nantucket, Molokai, Bequia.
    - One duplicate row (`United Kingdom`, `countryLandBordersCountry`) was removed from `test.jsonl`; the test set now has 477 rows instead of 478.
    - Six relations now have explicit definitions ([§ Relation definitions](#relation-definitions)) clarifying scope (e.g. land-only borders, total country area, city-granularity death location, distinct predecessor/successor awards).
- **April 2026**: Release of dataset, baseline, and evaluation script.

## Challenge overview

Pretrained language models (LMs) contain a substantial amount of factual knowledge. Turning that knowledge into reliable knowledge base entries, however, is much harder than answering a single factual question. In this shared task, we invite participants to build knowledge bases from LMs for given subjects and relations. In crucial difference to existing probing benchmarks like LAMA ([Petroni et al., 2019](https://arxiv.org/pdf/1909.01066.pdf)), we make no simplifying assumptions on relation cardinalities, i.e., a subject-entity can stand in relation with zero, one, or many object-entities.

Unlike earlier editions, this version does **not require entity disambiguation**. Predictions are evaluated as **strings** using normalization and alias matching.

> Formally, given the input subject-entity (s) and relation (r), the task is to
> predict all the correct
> object-entities ({o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>k</sub>}) using LM
> probing.

## Dataset

Number of unique subject-entities in the data splits.

<table>
<thead>
    <tr>
        <th>Relation</th>
        <th>Train</th>
        <th>Val</th>
        <th>Test</th>
        <th>Special features</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td>countryLandBordersCountry</td>
        <td>67</td>
        <td>68</td>
        <td>67</td>
        <td>Null values possible; <em>land</em> borders only</td>
    </tr>
    <tr>
        <td>personHasCityOfDeath</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
    <tr>
        <td>hasCapacity</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Object is numeric</td>
    </tr>
    <tr>
        <td>awardWonBy</td>
        <td>10</td>
        <td>10</td>
        <td>10</td>
        <td>Many objects per subject</td>
    </tr>
    <tr>
        <td>companyTradesAtStockExchange</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
        <tr>
        <td>hasArea</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Object is numeric (square km)</td>
    </tr>
</tbody>
</table>

## Relation definitions

These are the precise scopes used when constructing the ground truth. Models will be evaluated against these definitions, so participants should target them rather than a generic interpretation. The ground truth reflects the state of the world **as of 1 July 2026** (deaths, stock-exchange listings, award winners, and borders up to that date).

- **`countryLandBordersCountry`** — countries (or comparable territories) that share a **land** border with the subject. Maritime borders (e.g. Russia–Japan, Samoa–USA) are **excluded**. Island countries without a land border have an empty answer set. Includes only currently-recognised states; deprecated/disputed border statements on Wikidata are not considered. Borders through a country's integral overseas territory count (e.g. Suriname–France via French Guiana, Spain–Morocco via Ceuta and Melilla), while borders via non-integral dependencies do not (e.g. Cyprus–United Kingdom via the Sovereign Base Areas). Enclave borders count (e.g. Vatican City–Italy).

- **`personHasCityOfDeath`** — the **city** where the person died. Granularity is the city (or the most specific publicly known locality), not the country or region. If the person is still living or no locality is known, the answer is empty.

- **`hasCapacity`** — the **maximum spectator capacity** of the venue, expressed as an integer **number of people**. For stadiums and arenas this corresponds to Wikidata's `P1083` (maximum capacity). When multiple capacities exist (seated vs total, before/after renovation), the **highest published capacity** is used.

- **`awardWonBy`** — entities that have received the specific award identified by the subject. Winners are recorded as the **recipient entities** (people, groups, organizations, projects) — not the winning works. Predecessor or successor awards (e.g. *Medal of Freedom* vs *Presidential Medal of Freedom*) are **distinct** and not bundled, and rescinded awards are excluded. Some awards have hundreds of recipients; participants should expect large object sets. For a small number of awards with very large or open-ended recipient sets (e.g. product-design awards, honorary doctorates), the gold set is necessarily partial.

- **`companyTradesAtStockExchange`** — the stock exchange(s) on which the company's shares are publicly traded. Multiple listings are possible. Subsidiaries that are not separately listed have an empty answer set.

- **`hasArea`** — the surface area of the subject geographic entity, in **square kilometres** (km²). For countries, the **total area** (land + inland water) is used, matching the Wikidata preferred-rank value for `P2046`. Areas reported on Wikidata in hectares, square miles, etc. are converted to km².

## Evaluation metrics

We evaluate predictions using **macro precision, recall, and F1-score**.

For **string relations**, predicted strings are normalized (case-folded, diacritics removed, apostrophe-like marks dropped, punctuation of any script treated as whitespace) and matched against the ground-truth label and its known aliases via maximum bipartite matching — each gold entity credits at most one prediction and vice versa, independent of prediction order. Predictions are deduplicated by normalized string; note that predicting several surface forms of the *same* entity (e.g. `["NYC", "New York City"]`) counts as separate predictions and lowers precision.
For **numeric relations** (`hasCapacity`, `hasArea`), a prediction is correct if it falls within **5% relative tolerance** of the ground-truth value.

See the evaluation script ([evaluate.py](evaluate.py)) for details.

```bash
python evaluate.py \
  -g data/val.jsonl \
  -p your_predictions.jsonl
```

Parameters: ``-g`` (the ground truth file), ``-p`` (the prediction file).

## Getting started

### Setup

1. Clone this repository:

    ```bash
    mkdir lm-kbc-2026
    cd lm-kbc-2026
    git clone https://github.com/lm-kbc/dataset2026.git
    cd dataset2026
    ```

2. Create a virtual environment and install the requirements:

    ```bash
    conda create -n lm-kbc-2026 python=3.11
    ```

    ```bash
    conda activate lm-kbc-2026
    pip install -r requirements.txt
    ```

3. Write your own solution and generate predictions (format described
   in [How to structure your prediction file](#how-to-structure-your-prediction-file)).
4. Evaluate your predictions using the evaluation script
   (see [Evaluation metrics](#evaluation-metrics)).
5. Submit your predictions
   (see [Submit your predictions](#submit-your-predictions)).

### Baselines

#### Baseline: Qwen3.5-9B

Config
file: [configs/baseline-qwen-3.5-9b.yaml](configs/baseline-qwen-3.5-9b.yaml)

```bash
python baseline.py -c configs/baseline-qwen-3.5-9b.yaml -i data/val.jsonl
python evaluate.py -p output/baseline-qwen-3.5-9b.jsonl -g data/val.jsonl
```

Results (validation, Participant lm-kbc, Submission ID 850875):

```text
                              macro-p  macro-r  macro-f1  micro-p  micro-r  micro-f1  avg. #preds  #empty preds
awardWonBy                      0.247    0.078     0.101    0.279    0.046     0.079       24.000             0
companyTradesAtStockExchange    0.369    0.725     0.354    0.368    0.551     0.441        1.170             0
countryLandBordersCountry       0.697    0.911     0.665    0.859    0.883     0.871        2.706             0
hasArea                         0.290    0.290     0.290    0.290    0.290     0.290        1.000             0
hasCapacity                     0.180    0.180     0.180    0.180    0.180     0.180        1.000             0
personHasCityOfDeath            0.210    0.600     0.210    0.210    0.344     0.261        1.000             0
*** All Relations ***           0.324    0.507     0.313    0.400    0.170     0.239        1.759             0
```

### How to structure your prediction file

Your prediction file should be in the jsonl format.
Each line of a valid prediction file contains a JSON object which must
contain at least 3 fields to be used by the evaluation script:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntities``: the predicted object entity strings (list of strings)

This is an example of how to write a prediction file:

```python
import json

# Dummy predictions
predictions = [
    {
        "SubjectEntity": "Dominican republic",
        "Relation": "countryLandBordersCountry",
        "ObjectEntities": ["Haiti"]
    },
    {
        "SubjectEntity": "Jiaxing Stadium in Jiaxing",
        "Relation": "hasCapacity",
        "ObjectEntities": ["35000"]
    },
    {
        "SubjectEntity": "Mauritius",
        "Relation": "countryLandBordersCountry",
        "ObjectEntities": []
    }

]

fp = "./path/to/your/prediction/file.jsonl"

with open(fp, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")
```

### Submit your predictions

Submit your system paper via [OpenReview](https://openreview.net/group?id=EMNLP/2026/Workshop/LM-KBC_Shared_Task).

For the validation leaderboard, submit your predictions to [Codabench (validation)](https://www.codabench.org/competitions/16267/).

The test leaderboard will be released together with the test data.
