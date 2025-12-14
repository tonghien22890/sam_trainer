#!/usr/bin/env python3
"""
Compare Style Learner variants (Ranker vs legacy classifier).

Usage:
    python model_build/scripts/two_layer/compare_style_learners.py \
        --data model_build/simple_sam.jsonl \
        --new-model model_build/models/style_learner_sam.pkl \
        --old-model model_build/models/style_learner_sam_classifier.pkl \
        --max-records 2000
"""

import argparse
import json
import os
from typing import Dict, List, Any, Tuple

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # model_build
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.two_layer.style_learner as ranker_module  # type: ignore
import scripts.two_layer.style_learner_bu as legacy_module  # type: ignore


DEFAULT_FRAMEWORK = {
    "unbeatable_sequence": [],
    "framework_strength": 0.0,
    "core_combos": [],
    "protected_ranks": [],
    "protected_windows": [],
    "recommended_moves": [],
    "alternative_sequences": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Ranker StyleLearner vs legacy classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="model_build/simple_sam.jsonl",
        help="Path to training/eval data JSONL.",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default="sam",
        choices=["sam", "tlmn"],
        help="Filter records by game_type.",
    )
    parser.add_argument(
        "--new-model",
        type=str,
        required=True,
        help="Path to Ranker model (.pkl).",
    )
    parser.add_argument(
        "--old-model",
        type=str,
        required=True,
        help="Path to legacy classifier model (.pkl).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit number of records for quicker comparison.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print mismatch samples for debugging.",
    )
    return parser.parse_args()


def load_records(path: str, game_type: str, limit: int | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("game_type") != game_type:
                continue
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def ensure_framework(record: Dict[str, Any]) -> Dict[str, Any]:
    fw = record.get("framework")
    if not fw:
        return DEFAULT_FRAMEWORK.copy()
    # Ensure required keys exist
    merged = DEFAULT_FRAMEWORK.copy()
    merged.update(fw)
    merged.setdefault("alternative_sequences", [])
    return merged


def get_legal_moves(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = record.get("meta", {})
    moves = meta.get("legal_moves") or meta.get("legal_stage2") or []
    # Ensure we return deep copies to avoid mutation issues
    return [dict(m) for m in moves if isinstance(m, dict)]


def get_ground_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("action") or {}).get("stage2", {}) or {}


def moves_equal(move1: Dict[str, Any], move2: Dict[str, Any]) -> bool:
    if not move1 or not move2:
        return False
    cards1 = sorted(move1.get("cards", []))
    cards2 = sorted(move2.get("cards", []))
    return (
        move1.get("type") == move2.get("type")
        and cards1 == cards2
        and move1.get("combo_type") == move2.get("combo_type")
        and move1.get("rank_value") == move2.get("rank_value")
    )


def evaluate_model(
    learner,
    records: List[Dict[str, Any]],
    name: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    hits = 0
    total = 0
    hard_cases: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []

    for idx, record in enumerate(records):
        legal_moves = get_legal_moves(record)
        if not legal_moves:
            continue
        framework = ensure_framework(record)
        ground_truth = get_ground_truth(record)
        if not ground_truth:
            continue

        prediction = learner.predict_with_framework(
            record,
            [dict(m) for m in legal_moves],  # defensive copy
            framework,
        )

        total += 1
        if moves_equal(prediction, ground_truth):
            hits += 1
        elif verbose and len(hard_cases) < 5:
            hard_cases.append((idx, ground_truth, prediction))

    accuracy = hits / total if total else 0.0
    if verbose and hard_cases:
        print(f"\n[{name}] Sample mismatches:")
        for idx, gt, pred in hard_cases:
            print(f"- Record #{idx}: expected {gt}, predicted {pred}")

    return {"accuracy": accuracy, "samples": total}


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if not os.path.exists(args.new_model):
        raise FileNotFoundError(f"New model file not found: {args.new_model}")
    if not os.path.exists(args.old_model):
        raise FileNotFoundError(f"Old model file not found: {args.old_model}")

    print(f"[Compare] Loading records from {args.data} (game_type={args.game_type})...")
    records = load_records(args.data, args.game_type, args.max_records)
    print(f"[Compare] Loaded {len(records)} records.")

    ranker = ranker_module.StyleLearner()
    ranker.load(args.new_model)

    legacy = legacy_module.StyleLearner()
    legacy.load(args.old_model)

    ranker_metrics = evaluate_model(ranker, records, "Ranker", verbose=args.verbose)
    legacy_metrics = evaluate_model(legacy, records, "Classifier", verbose=args.verbose)

    print("\n=== Comparison Results ===")
    print(f"Ranker accuracy   : {ranker_metrics['accuracy']:.4f} over {ranker_metrics['samples']} samples")
    print(f"Classifier accuracy: {legacy_metrics['accuracy']:.4f} over {legacy_metrics['samples']} samples")
    delta = ranker_metrics["accuracy"] - legacy_metrics["accuracy"]
    print(f"Delta (Ranker - Classifier): {delta:+.4f}")


if __name__ == "__main__":
    main()

