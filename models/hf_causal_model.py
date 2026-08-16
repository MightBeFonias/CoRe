"""Generic HuggingFace causal-LM baseline for the LM-KBC 2026 shared task.

Two prompting modes, selected per-relation:

  - CoT relations (listed in config `cot_relations`): rationalized
    chain-of-thought few-shot. For each query, sample `few_shot` random
    training rows with the same relation, ask the model to produce a short
    reasoning trace connecting each sampled question to its already-known
    gold answer ("rationalization"), then use those as few-shot
    demonstrations before asking the real question. The final answer is
    extracted from a "Final Answer: ..." line. Rationales are cached by
    (SubjectEntity, Relation) so re-sampling the same exemplar for a
    different query does not re-generate it.

  - Direct relations (everything else): plain few-shot with the gold
    answer shown directly, no reasoning trace, no rationalization step.
    Cheaper and faster — use this for relations where CoT doesn't help.

Rows are batched separately by mode for throughput; CoT rows additionally
go through a rationalization pass before their final-answer pass.
"""

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from models.abstract_model import AbstractModel

NO_ANSWER = "None"
ExemplarKey = Tuple[str, str]  # (SubjectEntity, Relation)

# `torch_dtype` was renamed to `dtype` in transformers 4.56.
_TF_VERSION = tuple(
    int(part) for part in transformers_version.split(".")[:2] if part.isdigit()
)
DTYPE_KWARG = "dtype" if _TF_VERSION >= (4, 56) else "torch_dtype"


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HFCausalLMModel(AbstractModel):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.max_new_tokens = int(config.get("max_new_tokens", 300))
        self.direct_max_new_tokens = int(config.get("direct_max_new_tokens", 64))
        self.rationale_max_new_tokens = int(config.get("rationale_max_new_tokens", 150))
        self.batch_size = int(config.get("batch_size", 1))
        self.enable_thinking = bool(config.get("enable_thinking", False))
        self.few_shot = int(config.get("few_shot", 5))
        # Per-relation exemplar counts, e.g.
        #   few_shot_per_relation: {personHasCityOfDeath: 10, hasCapacity: 8}
        # Relations differ in what they need: a relation whose answers are
        # often empty needs enough exemplars to convey that base rate, while
        # a relation with long answers pays a real context cost per exemplar.
        self.few_shot_per_relation: Dict[str, int] = {
            k: int(v) for k, v in (config.get("few_shot_per_relation", {}) or {}).items()
        }
        if self.few_shot_per_relation:
            logger.info(
                f"Per-relation few_shot: {self.few_shot_per_relation} "
                f"(default {self.few_shot})"
            )
        self.base_seed = int(config.get("seed", 42))

        # Relations that get rationalized CoT few-shot. Anything not listed
        # here uses the plain direct-answer path instead. Empty/unset means
        # no relation uses CoT.
        self.cot_relations = set(config.get("cot_relations", []))
        if self.cot_relations:
            logger.info(f"CoT relations: {sorted(self.cot_relations)}")
        else:
            logger.info("No cot_relations configured — all relations use direct answers.")

        # Relations whose answer is a single number. These get number-aware
        # parsing (no comma-splitting, unit/filler-word stripping) instead of
        # the comma-separated entity-list parsing used for string relations.
        self.numeric_relations = set(
            config.get("numeric_relations", ["hasArea", "hasCapacity"])
        )

        # Relations where the gold answer is never empty, so predicting None
        # is strictly dominated: an empty prediction scores 0 just like a
        # wrong guess, while any guess has a chance of landing inside the
        # scorer's tolerance. For these, an unparseable/empty answer falls
        # back to a prior (the median of that relation's training values)
        # rather than being emitted as an empty set.
        self.never_abstain_relations = set(
            config.get("never_abstain_relations", ["hasArea", "hasCapacity"])
        )
        if self.never_abstain_relations:
            logger.info(f"Never-abstain relations: {sorted(self.never_abstain_relations)}")

        if self.cot_relations and self.max_new_tokens < 200:
            logger.warning(
                f"cot_relations is set but max_new_tokens={self.max_new_tokens} "
                "— the reasoning trace may get cut off before the 'Final "
                "Answer:' line. Consider raising it to 300+."
            )

        llm_path = config["llm_path"]
        logger.info(f"Loading tokenizer: {llm_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        # Decoder-only batched generation requires left padding.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"device_map": "auto", DTYPE_KWARG: torch.bfloat16}
        if config.get("use_quantization", False):
            from transformers import BitsAndBytesConfig
            logger.info("Loading in 4-bit (bitsandbytes NF4)")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(f"Loading model: {llm_path}")
        self.model = AutoModelForCausalLM.from_pretrained(llm_path, **model_kwargs)
        self.model.eval()

        # Prompt templates: Relation -> PromptTemplate
        templates = pd.read_csv(config["prompt_templates_file"])
        self.templates = dict(
            zip(templates["Relation"], templates["PromptTemplate"])
        )

        # Training rows grouped by relation. Used as the few-shot exemplar
        # pool and as the source of the numeric fallback prior, so it is
        # loaded regardless of few_shot.
        self.examples_by_relation: Dict[str, List[Dict]] = {}
        for row in _read_jsonl(config["train_data_file"]):
            self.examples_by_relation.setdefault(row["Relation"], []).append(row)

        # (SubjectEntity, Relation) -> generated rationale text. Persisted to
        # disk so a crash (e.g. the OOM this run hit mid-answering) doesn't
        # throw away rationalization work already paid for on restart.
        self.rationale_cache_path = config.get(
            "rationale_cache_path",
            f"output/.rationale_cache__{config.get('model', 'model')}.json",
        )
        self.rationale_cache: Dict[ExemplarKey, str] = self._load_rationale_cache()

        # Per-relation answer token budgets. In CoT mode the reasoning trace
        # and the "Final Answer:" line share one budget, so relations that
        # must emit a long list need far more than the default or the answer
        # line is never reached and the parser sees only reasoning.
        self.max_new_tokens_per_relation: Dict[str, int] = dict(
            config.get("max_new_tokens_per_relation", {}) or {}
        )
        if self.max_new_tokens_per_relation:
            logger.info(
                f"Per-relation token budgets: {self.max_new_tokens_per_relation}"
            )

        # Self-consistency: sample k completions and aggregate, per relation.
        # For single-valued numeric relations the median of k samples is more
        # robust than one greedy pass, because greedy commits to a single
        # recalled figure with no way to detect that it is off.
        self.self_consistency: Dict[str, int] = dict(
            config.get("self_consistency", {}) or {}
        )
        self.self_consistency_temperature = float(
            config.get("self_consistency_temperature", 0.7)
        )
        self.self_consistency_top_p = float(config.get("self_consistency_top_p", 0.95))
        if self.self_consistency:
            logger.info(f"Self-consistency samples per relation: {self.self_consistency}")

        # Per-relation generate() kwarg overrides, e.g.
        #   generation_overrides:
        #     awardWonBy: {no_repeat_ngram_size: 4}
        # awardWonBy's greedy decoding degenerates into a repetition loop
        # (582 name slots but only 48 unique), so blocking repeated n-grams
        # is the direct fix for that specific pathology.
        self.generation_overrides: Dict[str, Dict] = dict(
            config.get("generation_overrides", {}) or {}
        )
        if self.generation_overrides:
            logger.info(f"Per-relation generation overrides: {self.generation_overrides}")

        # Diagnostics: how often a generation never produced an answer marker
        # (the signature of truncation), and optionally the raw text itself.
        self.truncation_stats: Dict[str, Dict[str, int]] = {}
        self.raw_generations_path = config.get("raw_generations_path")
        self.raw_generations = [] if self.raw_generations_path else None

        # Fallback value per never-abstain numeric relation: the median of
        # that relation's training values. Used only when the model produces
        # nothing parseable — a median guess sometimes lands inside the
        # scorer's relative tolerance, whereas an empty answer never can.
        self.numeric_prior: Dict[str, str] = {}
        for relation in self.never_abstain_relations & self.numeric_relations:
            values = []
            for row in self.examples_by_relation.get(relation, []):
                for entity in (row.get("ObjectEntities") or []):
                    surface = entity[0] if isinstance(entity, list) and entity else entity
                    parsed = self._to_number(str(surface)) if surface is not None else None
                    if parsed is not None:
                        values.append(parsed)
            if values:
                values.sort()
                median = values[len(values) // 2]
                self.numeric_prior[relation] = self._format_number(median)
                logger.info(
                    f"Numeric fallback prior for {relation}: "
                    f"{self.numeric_prior[relation]} (median of {len(values)} train values)"
                )

    # ---------- shared helpers ----------

    def _question(self, subject: str, relation: str) -> str:
        template = self.templates.get(
            relation, "What is the {subject_entity} of this relation?"
        )
        return template.format(subject_entity=subject)

    @staticmethod
    def _gold_answer(row: Dict) -> str:
        """First alias of each gold entity, comma-separated."""
        objects = row.get("ObjectEntities") or []
        surface = []
        for entity in objects:
            if isinstance(entity, list):
                if entity:
                    surface.append(str(entity[0]))
            elif entity is not None:
                surface.append(str(entity))
        return ", ".join(surface) if surface else NO_ANSWER

    def _render(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Chat template does not accept enable_thinking (non-Qwen3 models).
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _generate_batch(
        self, prompts: List[str], max_new_tokens: int, desc: str, n_samples: int = 1,
        extra_kwargs: Dict = None,
    ) -> List[List[str]]:
        """Decode a list of rendered prompts, batched by self.batch_size.

        Returns one list of `n_samples` completions per prompt. With
        n_samples == 1 decoding is greedy and each inner list has one item.
        With n_samples > 1 it samples, for self-consistency aggregation.

        Two defenses against the OOM crash seen in practice on a 31B model
        with tight VRAM headroom:
          - Explicit tensor deletion + periodic `torch.cuda.empty_cache()`.
            HF `generate()` with variable-length batches (few-shot context
            length differs per query) leaves the caching allocator holding
            differently-sized blocks it can't reuse across iterations; over
            enough batches this fragments the allocator until an allocation
            that should fit fails anyway. Clearing the cache periodically
            keeps that from compounding across a long run.
          - If a batch OOMs anyway, split it in half and retry recursively
            (down to batch size 1) instead of letting the whole run die.
            Sampling multiplies activation memory by n_samples, so this
            matters more, not less, once self-consistency is on.
        """
        # Sampling k completions per prompt multiplies memory, so shrink the
        # prompt-batch to keep total sequences in flight roughly constant.
        effective_bs = max(1, self.batch_size // n_samples) if n_samples > 1 else self.batch_size
        outputs: List[List[str]] = []
        for start in tqdm(range(0, len(prompts), effective_bs), desc=desc):
            batch = prompts[start: start + effective_bs]
            outputs.extend(
                self._generate_one_batch(batch, max_new_tokens, n_samples, extra_kwargs)
            )
            if torch.cuda.is_available() and (start // effective_bs) % 5 == 0:
                torch.cuda.empty_cache()
        return outputs

    def _generate_one_batch(
        self, batch: List[str], max_new_tokens: int, n_samples: int = 1,
        extra_kwargs: Dict = None,
    ) -> List[List[str]]:
        try:
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if n_samples > 1:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=self.self_consistency_temperature,
                    top_p=self.self_consistency_top_p,
                    num_return_sequences=n_samples,
                )
            else:
                gen_kwargs["do_sample"] = False
            if extra_kwargs:
                gen_kwargs.update(extra_kwargs)

            with torch.inference_mode():
                generated = self.model.generate(**encoded, **gen_kwargs)

            new_tokens = generated[:, encoded["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            del encoded, generated, new_tokens
            # HF returns n_samples consecutive rows per input prompt.
            return [
                decoded[i * n_samples: (i + 1) * n_samples] for i in range(len(batch))
            ]

        except RuntimeError as e:
            is_oom = "out of memory" in str(e).lower()
            if not is_oom:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch) == 1:
                # Nothing smaller to fall back to — this single prompt is
                # genuinely too long for available VRAM. Emit an empty
                # completion rather than killing the whole run; the
                # never-abstain fallback (if configured) will cover it.
                logger.warning(
                    "OOM on a single-item batch — emitting empty completion "
                    f"for prompt starting: {batch[0][:80]!r}"
                )
                return [[""]]
            mid = len(batch) // 2
            logger.warning(
                f"OOM on batch of {len(batch)} — retrying as two batches of "
                f"{mid} and {len(batch) - mid}."
            )
            return (
                self._generate_one_batch(batch[:mid], max_new_tokens, n_samples, extra_kwargs)
                + self._generate_one_batch(batch[mid:], max_new_tokens, n_samples, extra_kwargs)
            )

    # ---------- step 1: sample exemplars ----------

    def _row_rng(self, subject: str, relation: str) -> random.Random:
        """A PRNG seeded deterministically from (base_seed, subject, relation).

        Using one shared PRNG for every row means the exemplars drawn for a
        given row depend on how many other rows were sampled before it —
        i.e. on processing order/subset, which changes whenever the code is
        refactored (e.g. splitting CoT vs direct rows) even with the same
        seed. Deriving a fresh, independent PRNG per row makes each row's
        sample depend only on that row's own identity, so results are
        reproducible regardless of what order or subset of rows run.
        """
        key = f"{self.base_seed}:{relation}:{subject}"
        seed_val = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
        return random.Random(seed_val)

    def _few_shot_for(self, relation: str) -> int:
        return int(self.few_shot_per_relation.get(relation, self.few_shot))

    def _sample_exemplars(self, subject: str, relation: str) -> List[Dict]:
        pool = self.examples_by_relation.get(relation, [])
        k = self._few_shot_for(relation)
        if not pool or k <= 0:
            return []
        if len(pool) <= k:
            return list(pool)

        return self._row_rng(subject, relation).sample(pool, k)

    # ---------- step 2: rationalize exemplars ----------

    def _build_rationalize_messages(self, example: Dict) -> List[Dict[str, str]]:
        question = self._question(example["SubjectEntity"], example["Relation"])
        gold = self._gold_answer(example)
        system = (
            "You are a careful research assistant. You will be given a "
            "factual question together with its correct answer. In 2-4 "
            "concise sentences, explain the reasoning that leads to this "
            "answer, as if working it out from scratch. Do not mention that "
            "the answer was provided to you, and do not restate the "
            "question verbatim."
        )
        user = f"Question: {question}\nCorrect answer: {gold}\n\nExplain the reasoning:"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _clean_rationale(text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        text = " ".join(text.strip().split())
        return text or "Recalling known facts about the subject."

    def _load_rationale_cache(self) -> Dict[ExemplarKey, str]:
        if not self.rationale_cache_path:
            return {}
        try:
            with open(self.rationale_cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cache = {tuple(item["key"]): item["rationale"] for item in raw}
            if cache:
                logger.info(
                    f"Loaded {len(cache)} cached rationale(s) from "
                    f"{self.rationale_cache_path}"
                )
            return cache
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                f"Could not read rationale cache at {self.rationale_cache_path} "
                f"({e}) — starting with an empty cache."
            )
            return {}

    def _save_rationale_cache(self) -> None:
        if not self.rationale_cache_path:
            return
        path = Path(self.rationale_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"key": list(key), "rationale": text}
            for key, text in self.rationale_cache.items()
        ]
        # Write to a temp file then rename, so a crash mid-write never
        # leaves a truncated/corrupt cache file behind.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp_path.replace(path)

    def _rationalize_needed_exemplars(self, all_rows: List[Dict]) -> List[List[Dict]]:
        """Sample exemplars for every input row, generate+cache any
        rationale not already cached, and return the sampled exemplar list
        per row (same order as all_rows)."""
        exemplars_per_row: List[List[Dict]] = []
        to_generate: Dict[ExemplarKey, Dict] = {}

        for row in all_rows:
            sampled = self._sample_exemplars(row["SubjectEntity"], row["Relation"])
            exemplars_per_row.append(sampled)
            for ex in sampled:
                key = (ex["SubjectEntity"], ex["Relation"])
                if key not in self.rationale_cache and key not in to_generate:
                    to_generate[key] = ex

        if to_generate:
            keys = list(to_generate.keys())
            prompts = [
                self._render(self._build_rationalize_messages(to_generate[k]))
                for k in keys
            ]
            logger.info(
                f"Generating {len(prompts)} rationale(s) for training exemplars "
                f"(cache had {len(self.rationale_cache)} already)."
            )
            raw = self._generate_batch(
                prompts, self.rationale_max_new_tokens, desc="rationalizing exemplars"
            )
            for key, samples in zip(keys, raw):
                self.rationale_cache[key] = self._clean_rationale(samples[0])
            self._save_rationale_cache()
            if self.rationale_cache_path:
                logger.info(f"Saved rationale cache to {self.rationale_cache_path}")

        return exemplars_per_row

    # ---------- step 3: build final prompt (CoT mode) ----------

    def _answer_policy(self, relation: str) -> str:
        """Trailing instruction about whether abstaining is allowed.

        For never-abstain relations the gold is never empty, so 'None' scores
        exactly as badly as a wrong guess while forfeiting any chance of
        landing inside the scorer's tolerance. Telling the model to always
        commit to a value is therefore free upside.
        """
        if relation in self.never_abstain_relations:
            return (
                " Always commit to a specific value, even if you are unsure — "
                "give your single best estimate rather than refusing. Never "
                "answer None."
            )
        return f" If there is no answer, write 'Final Answer: {NO_ANSWER}'."

    def _build_final_messages(
        self, subject: str, relation: str, exemplars: List[Dict]
    ) -> List[Dict[str, str]]:
        system = (
            "You are a knowledge base. For the following question, first "
            "reason step by step about what you know, then on its own final "
            "line write 'Final Answer: ' followed by only the answer "
            "entities separated by commas. No explanation on that last "
            "line, no numbering, no full sentences. For quantities, give "
            "the bare number only."
        ) + self._answer_policy(relation)
        messages = [{"role": "system", "content": system}]
        for ex in exemplars:
            key = (ex["SubjectEntity"], ex["Relation"])
            rationale = self.rationale_cache.get(key, "")
            gold = self._gold_answer(ex)
            messages.append({
                "role": "user",
                "content": self._question(ex["SubjectEntity"], relation),
            })
            messages.append({
                "role": "assistant",
                "content": f"{rationale}\nFinal Answer: {gold}".strip(),
            })
        messages.append({
            "role": "user",
            "content": self._question(subject, relation),
        })
        return messages

    # ---------- step 3b: build final prompt (direct mode, no reasoning) ----------

    def _build_direct_messages(
        self, subject: str, relation: str, exemplars: List[Dict]
    ) -> List[Dict[str, str]]:
        system = (
            "You are a knowledge base. Answer each question with only the "
            "answer entities, separated by commas. Give no explanation, no "
            "numbering and no full sentences. For quantities, reply with "
            "the bare number only."
        )
        if relation in self.never_abstain_relations:
            system += (
                " Always commit to a specific value, even if you are unsure — "
                "give your single best estimate rather than refusing. Never "
                "answer None."
            )
        else:
            system += f" If there is no answer, reply exactly {NO_ANSWER}."
        messages = [{"role": "system", "content": system}]
        for ex in exemplars:
            messages.append({
                "role": "user",
                "content": self._question(ex["SubjectEntity"], relation),
            })
            messages.append({
                "role": "assistant",
                "content": self._gold_answer(ex),
            })
        messages.append({
            "role": "user",
            "content": self._question(subject, relation),
        })
        return messages

    # ---------- step 4: parse final answer ----------

    # Some models (Gemma 4) wrap reasoning in channel tags like
    # "<|channel|>thought\n...\n<|channel|>" instead of a plain </think>.
    # Match loosely across bracket-order variants seen in the wild.
    _CHANNEL_THOUGHT_RE = re.compile(
        r"<\|?channel\|?>\s*thought.*?<\|?channel\|?>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    @classmethod
    def _strip_reasoning_tags(cls, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        text = cls._CHANNEL_THOUGHT_RE.sub("", text)
        return text

    # Entity names are short. Across all gold aliases in train+val, 99% are
    # under 7 words / 42 characters. When the model writes a sentence instead
    # of a list ("Russia is the largest country ... moving clockwise"), the
    # comma-split turns each clause into a bogus entity that destroys
    # precision. These bounds sit far above any real name and far below any
    # prose clause, so filtering costs no true positives.
    MAX_ENTITY_WORDS = 12
    MAX_ENTITY_CHARS = 80
    # A clause split off a sentence almost always begins with a lowercase
    # function word ("these neighbors are...", "and Lithuania (though...").
    # Real entity names never do: gold aliases that start lowercase are
    # spelling variants like "sir Alexandre Fleming", never connectives.
    # Note this deliberately avoids keying on sentence-final periods, since
    # 634 gold aliases contain "Jr.", "St.", "Inc." and similar.
    _PROSE_STARTERS = frozenset({
        "these", "those", "this", "that", "and", "but", "or", "it", "its",
        "they", "their", "them", "he", "she", "his", "her", "which", "who",
        "was", "were", "is", "are", "has", "have", "had", "also", "further",
        "finally", "starting", "moving", "including", "such", "there",
        "however", "although", "though", "while", "with", "within", "from",
    })

    @classmethod
    def _looks_like_prose(cls, value: str) -> bool:
        if len(value.split()) > cls.MAX_ENTITY_WORDS:
            return True
        if len(value) > cls.MAX_ENTITY_CHARS:
            return True
        first = value.split()[0] if value.split() else ""
        return first[:1].islower() and first.strip(".,;:").casefold() in cls._PROSE_STARTERS

    @classmethod
    def _parse(cls, text: str) -> List[str]:
        text = cls._strip_reasoning_tags(text)
        text = text.strip().split("\n")[0].strip()
        if not text or text.strip(" .").casefold() in {"none", "n/a", "unknown", ""}:
            return []
        answers, seen = [], set()
        for part in text.split(","):
            part = part.strip().strip(".").strip()
            if not part or part.casefold() == NO_ANSWER.casefold():
                continue
            if cls._looks_like_prose(part):
                continue
            if part.casefold() in seen:
                continue
            seen.add(part.casefold())
            answers.append(part)
        return answers

    # A number, optionally with thousands separators and a decimal part,
    # optionally followed by a spelled-out scale word. Scale words match only
    # as whole words so the "k" in "km2" is never read as a kilo multiplier.
    # The lookbehind rejects digits glued to letters, so the "2" in "km2",
    # "m2" or "mi2" is treated as part of the unit rather than as a value.
    _NUMBER_RE = re.compile(
        r"(?<![A-Za-z])(-?\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand)?\b",
        re.IGNORECASE,
    )
    _SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}

    @classmethod
    def _to_number(cls, text: str):
        """Extract the first number from a string, tolerating thousands
        separators, surrounding units and filler words."""
        match = cls._NUMBER_RE.search(text)
        if not match:
            return None
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            return None
        scale = match.group(2)
        if scale:
            value *= cls._SCALES[scale.lower()]
        return value

    @staticmethod
    def _format_number(value: float) -> str:
        """Render a number the way the scorer expects: bare digits, no
        thousands separators, no trailing .0 on whole numbers."""
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:g}"

    @classmethod
    def _extract_answer_segment(cls, text: str) -> Tuple[str, bool]:
        """Return (segment after the final-answer marker, marker_was_found)."""
        text = cls._strip_reasoning_tags(text)
        lowered = text.casefold()
        for candidate in ("final answer:", "answer:"):
            idx = lowered.rfind(candidate)
            if idx != -1:
                return text[idx + len(candidate):], True
        return text, False

    @classmethod
    def _parse_numeric(cls, text: str) -> List[str]:
        """Parse a single-valued numeric answer.

        Deliberately does NOT split on commas: for these relations a comma is
        a thousands separator, so splitting would turn "75,005" into two bogus
        predictions and destroy precision.
        """
        segment, had_marker = cls._extract_answer_segment(text)
        if had_marker:
            # The answer follows the marker, so take the first number there.
            value = cls._to_number(segment)
        else:
            # No marker: the answer is most likely the last number stated,
            # after any reasoning, so scan from the end.
            matches = list(cls._NUMBER_RE.finditer(segment))
            value = cls._to_number(matches[-1].group(0)) if matches else None
        if value is None:
            return []
        return [cls._format_number(value)]

    def _parse_answer(self, text: str, relation: str) -> List[str]:
        """Relation-aware parsing, plus the never-abstain fallback."""
        if relation in self.numeric_relations:
            answers = self._parse_numeric(text)
        else:
            segment, _ = self._extract_answer_segment(text)
            answers = self._parse(segment)

        if not answers and relation in self.never_abstain_relations:
            prior = self.numeric_prior.get(relation)
            if prior is not None:
                return [prior]
        return answers

    # ---------- orchestration ----------

    def _budget_for(self, relation: str, default: int) -> int:
        """Token budget for a relation's answer generation.

        In CoT mode the reasoning trace and the 'Final Answer:' line share one
        budget, so a relation that must emit a long list (or reason at length)
        needs a bigger allowance or the answer line is never reached.
        """
        return int(self.max_new_tokens_per_relation.get(relation, default))

    def _samples_for(self, relation: str) -> int:
        """How many completions to sample for self-consistency (1 = greedy)."""
        return max(1, int(self.self_consistency.get(relation, 1)))

    def _aggregate_samples(self, sample_answers: List[List[str]], relation: str) -> List[str]:
        """Combine k sampled answers into one.

        For single-valued numeric relations the median is the right
        aggregator: it is robust to one wild outlier in a way that the mean
        is not, and unlike majority-vote it does not require the samples to
        agree exactly (they rarely do for a continuous quantity).
        """
        if relation in self.numeric_relations:
            values = []
            for answers in sample_answers:
                if answers:
                    v = self._to_number(answers[0])
                    if v is not None:
                        values.append(v)
            if not values:
                return []
            values.sort()
            median = values[len(values) // 2] if len(values) % 2 == 1 else (
                (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
            )
            return [self._format_number(median)]

        # Non-numeric: keep entities appearing in at least half the samples.
        counts: Dict[str, int] = {}
        first_seen: Dict[str, str] = {}
        order: List[str] = []
        for answers in sample_answers:
            for a in dict.fromkeys(x.casefold() for x in answers):
                if a not in counts:
                    counts[a] = 0
                    order.append(a)
                counts[a] += 1
        for answers in sample_answers:
            for a in answers:
                first_seen.setdefault(a.casefold(), a)
        threshold = len(sample_answers) // 2 + 1  # strict majority
        return [first_seen[k] for k in order if counts[k] >= threshold]

    def _run_group(
        self, indices: List[int], inputs: List[Dict], prompt_variants: List[List[str]],
        default_budget: int, desc: str, results: List,
    ) -> None:
        """Generate for rows grouped by token budget, then parse and combine.

        `prompt_variants[i]` holds one or more prompts for row `indices[i]`.
        """
        # (position, variant index) pairs; one variant per row.
        jobs: List[Tuple[int, int]] = [
            (pos, v) for pos in range(len(indices)) for v in range(len(prompt_variants[pos]))
        ]
        by_key: Dict[Tuple, List[int]] = {}
        for job_i, (pos, _) in enumerate(jobs):
            relation = inputs[indices[pos]]["Relation"]
            overrides = self.generation_overrides.get(relation, {})
            key = (
                self._budget_for(relation, default_budget),
                self._samples_for(relation),
                tuple(sorted(overrides.items())),
            )
            by_key.setdefault(key, []).append(job_i)

        # position -> answers from each variant, filled as groups complete
        collected: Dict[int, List[List[str]]] = {pos: [] for pos in range(len(indices))}
        first_raw: Dict[int, str] = {}

        for (budget, n_samples, override_items), job_ids in sorted(
            by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))
        ):
            overrides = dict(override_items)
            group_prompts = [prompt_variants[jobs[j][0]][jobs[j][1]] for j in job_ids]
            relations = sorted({inputs[indices[jobs[j][0]]]["Relation"] for j in job_ids})
            n_ens = max(len(prompt_variants[jobs[j][0]]) for j in job_ids)
            suffix = f", {n_samples}x sampled" if n_samples > 1 else ""
            if n_ens > 1:
                suffix += f", {n_ens} prompt variants"
            if overrides:
                suffix += f", {','.join(overrides)}"
            label = f"{desc} [{budget}tok{suffix}: {','.join(r[:18] for r in relations)}]"
            raw = self._generate_batch(
                group_prompts, budget, desc=label, n_samples=n_samples,
                extra_kwargs=overrides or None,
            )
            for j, samples in zip(job_ids, raw):
                pos, variant = jobs[j]
                idx = indices[pos]
                relation = inputs[idx]["Relation"]
                if n_samples > 1:
                    parsed = [self._parse_answer(t, relation) for t in samples]
                    answer = self._aggregate_samples(parsed, relation)
                else:
                    answer = self._parse_answer(samples[0], relation)
                collected[pos].append(answer)
                if variant == 0:
                    first_raw[pos] = samples[0]

        for pos, idx in enumerate(indices):
            relation = inputs[idx]["Relation"]
            variants = collected[pos]
            results[idx] = variants[0] if variants else []
            self._record_generation(
                inputs[idx], first_raw.get(pos, ""),
                self._budget_for(relation, default_budget),
            )

    def _record_generation(self, row: Dict, text: str, budget: int) -> None:
        """Track whether the answer marker was reached, for truncation stats.

        Only meaningful for CoT rows: the direct-mode prompt never asks for a
        'Final Answer:' line, so a missing marker there is expected and says
        nothing about truncation.
        """
        _, had_marker = self._extract_answer_segment(text)
        is_cot = row["Relation"] in self.cot_relations
        if is_cot:
            stats = self.truncation_stats.setdefault(
                row["Relation"], {"total": 0, "no_marker": 0}
            )
            stats["total"] += 1
            if not had_marker:
                stats["no_marker"] += 1
        if self.raw_generations is not None:
            self.raw_generations.append({
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "mode": "cot" if is_cot else "direct",
                "budget": budget,
                "had_answer_marker": had_marker,
                "raw": text,
            })

    def _report_truncation(self) -> None:
        for relation, s in sorted(self.truncation_stats.items()):
            if s["no_marker"]:
                pct = s["no_marker"] / s["total"] * 100
                level = logger.warning if pct >= 10 else logger.info
                level(
                    f"{relation} (CoT): {s['no_marker']}/{s['total']} ({pct:.0f}%) "
                    "generations never reached a 'Final Answer:' line — these "
                    "were likely truncated by the token budget."
                )
        if self.raw_generations is not None and self.raw_generations_path:
            path = Path(self.raw_generations_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for rec in self.raw_generations:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(self.raw_generations)} raw generations to {path}")

    def generate_predictions(self, inputs: List[Dict[str, str]]) -> List[List[str]]:
        n = len(inputs)
        results: List[List[str]] = [None] * n  # type: ignore[list-item]

        cot_indices = [i for i, row in enumerate(inputs) if row["Relation"] in self.cot_relations]
        direct_indices = [i for i in range(n) if i not in set(cot_indices)]

        # --- CoT relations: rationalize exemplars, then answer with reasoning ---
        if cot_indices:
            cot_rows = [inputs[i] for i in cot_indices]
            logger.info(f"{len(cot_rows)} row(s) using rationalized CoT.")
            needs_exemplars = any(
                self._few_shot_for(row["Relation"]) > 0 for row in cot_rows
            )
            exemplars_per_row = (
                self._rationalize_needed_exemplars(cot_rows) if needs_exemplars
                else [[] for _ in cot_rows]
            )
            cot_prompts = [
                [self._render(self._build_final_messages(
                    row["SubjectEntity"], row["Relation"], exemplars))]
                for row, exemplars in zip(cot_rows, exemplars_per_row)
            ]
            self._run_group(
                cot_indices, inputs, cot_prompts,
                self.max_new_tokens, "answering (CoT)", results,
            )

        # --- everything else: plain few-shot, direct answer ---
        if direct_indices:
            direct_rows = [inputs[i] for i in direct_indices]
            logger.info(f"{len(direct_rows)} row(s) using direct answers.")
            direct_prompts = [
                [self._render(self._build_direct_messages(
                    row["SubjectEntity"], row["Relation"],
                    self._sample_exemplars(row["SubjectEntity"], row["Relation"])))]
                for row in direct_rows
            ]
            self._run_group(
                direct_indices, inputs, direct_prompts,
                self.direct_max_new_tokens, "answering (direct)", results,
            )

        self._report_truncation()
        assert all(r is not None for r in results), "some rows were never processed"
        return results
