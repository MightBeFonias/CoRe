import argparse
import json
from pathlib import Path

import yaml
from loguru import logger

from models.user_config import Models


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run an LM baseline over a LM-KBC 2026 split")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to the YAML config file")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input jsonl file")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to the output jsonl file "
                             "(default: output/<config-name>.jsonl, or "
                             "output/<config-name>__<relation>.jsonl when -r is used)")
    parser.add_argument("-r", "--relation", type=str, default=None,
                        help="Only run these relations (comma-separated). Use this to "
                             "iterate on one relation without paying for the whole "
                             "dataset, e.g. -r personHasCityOfDeath")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: {config}")

    rows = read_jsonl(args.input)
    logger.info(f"Loaded {len(rows)} rows from {args.input}")

    relations = None
    if args.relation:
        relations = [r.strip() for r in args.relation.split(",") if r.strip()]
        available = sorted({row["Relation"] for row in rows})
        unknown = [r for r in relations if r not in available]
        if unknown:
            raise SystemExit(
                f"Relation(s) {unknown} not present in {args.input}. "
                f"Available: {available}"
            )
        rows = [row for row in rows if row["Relation"] in relations]
        logger.info(f"Filtered to {len(rows)} row(s) for relation(s): {relations}")

    if args.output:
        output_path = Path(args.output)
    else:
        stem = Path(args.config).stem
        if relations:
            stem = f"{stem}__{'_'.join(relations)}"
        output_path = Path("output") / f"{stem}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_cls = Models.get_model(config["model"])
    model = model_cls(config)

    predictions = model.generate_predictions(rows)

    with open(output_path, "w", encoding="utf-8") as f:
        for row, preds in zip(rows, predictions):
            f.write(json.dumps({
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "ObjectEntities": preds,
            }, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
