#!/usr/bin/env bash
# Run inference + evaluation for a single relation. Much cheaper than the full
# 478-row run when iterating on one relation's prompt or policy.
#
#   ./run_relation.sh personHasCityOfDeath
#   ./run_relation.sh hasArea configs/other-config.yaml
set -euo pipefail

RELATION="${1:?usage: ./run_relation.sh <relation> [config] [input]}"
CONFIG="${2:-configs/gemma4-31b-cot5.yaml}"
INPUT="${3:-data/val.jsonl}"

STEM="$(basename "${CONFIG%.*}")__${RELATION}"
PRED="output/${STEM}.jsonl"
SCORES="output/${STEM}__scores.csv"

echo "=== relation: ${RELATION} | config: ${CONFIG} ==="
python3 baseline.py -c "$CONFIG" -i "$INPUT" -r "$RELATION" -o "$PRED"
echo
python3 evaluate.py -p "$PRED" -g "$INPUT" -r "$RELATION" -o "$SCORES"
echo
echo "predictions: $PRED"
echo "scores:      $SCORES"
