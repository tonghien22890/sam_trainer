from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import os
import json

ComboTypeToId = {
    "single": 0,
    "pair": 1,
    "triple": 2,
    "four_kind": 3,
    "straight": 4,
    "double_seq": 5,
}


def _ensure_two_stage_action(rec: Dict[str, Any]) -> Dict[str, Any]:
    action = rec.get("action", {})
    if "stage1" in action and "stage2" in action:
        return action
    # legacy flat → two-stage
    move_type = action.get("type", "pass")
    combo_type = action.get("combo_type")
    rank_value = action.get("rank_value")
    cards = action.get("cards", [])
    if move_type == "play_cards" and combo_type:
        stage1 = {"type": "combo_type", "value": combo_type}
    else:
        stage1 = {"type": "pass", "value": "pass"}
    stage2 = {
        "type": move_type if move_type in ("play_cards", "pass") else "pass",
        "cards": cards,
        "combo_type": combo_type if move_type == "play_cards" else None,
        "rank_value": rank_value if move_type == "play_cards" else None,
    }
    return {"stage1": stage1, "stage2": stage2}


def _extract_legal_moves(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = rec.get("meta", {})
    legal = meta.get("legal_moves") or meta.get("legal_stage2") or []
    return legal


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# --------- Classic ML encoding for Phase 4 simplified pipeline ---------

def encode_state_features(record: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
    """DEPRECATED: Old baseline encoder (52-card one-hot).

    This module is kept for reference only. The current pipeline uses per-candidate
    rank-only modeling in `scripts/optimized_general_model_v3.py` and should be
    preferred over this function.

    Returns: (X, legal_moves, chosen_index)
    """
    hand = record.get("hand", [])
    last_move = record.get("last_move") or None
    meta = record.get("meta", {})
    legal_moves = _extract_legal_moves(record)

    # One-hot hand 52 (deprecated; suit-dependent). Kept for compatibility.
    hand_oh = np.zeros(52, dtype=np.float32)
    for cid in hand:
        if 0 <= cid < 52:
            hand_oh[cid] = 1.0

    # last_move one-hot combo type + rank_value
    ct_oh = np.zeros(len(ComboTypeToId), dtype=np.float32)
    rank_val = -1.0
    if last_move and last_move.get("combo_type"):
        ct_id = ComboTypeToId.get(last_move.get("combo_type"), None)
        if ct_id is not None and ct_id >= 0:
            ct_oh[ct_id] = 1.0
        try:
            rank_val = float(last_move.get("rank_value", -1))
        except Exception:
            rank_val = -1.0

    # players_left_count and cards_left_sum
    players_left = record.get("players_left", [])
    cards_left = record.get("cards_left", [])
    players_left_count = float(len(players_left))
    try:
        cards_left_sum = float(sum(cards_left))
    except Exception:
        cards_left_sum = 0.0

    features = np.concatenate([
        hand_oh,
        ct_oh,
        np.array([rank_val, players_left_count, cards_left_sum], dtype=np.float32),
    ], axis=0)

    # Chosen label: index into legal moves matching action.stage2.cards
    action = _ensure_two_stage_action(record)
    chosen_cards = action.get("stage2", {}).get("cards", [])
    try:
        chosen_index = next(i for i, m in enumerate(legal_moves) if m.get("cards", []) == chosen_cards)
    except StopIteration:
        chosen_index = -1

    return features, legal_moves, chosen_index


def build_dataset_from_jsonl(path: str) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
    """Load JSONL and build (X, y, candidates_per_sample).

    X: np.ndarray shape (N, F)
    y: np.ndarray shape (N,) with indices into candidates (or -1 if unknown)
    candidates: list of legal_moves lists per sample
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    cand_list: List[List[Dict[str, Any]]] = []

    for rec in load_jsonl(path):
        # Skip session_start records - they don't have game data
        if rec.get("type") == "session_start":
            continue
            
        x, legal, y = encode_state_features(rec)
        X_list.append(x)
        y_list.append(y)
        cand_list.append(legal)

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, 60), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, cand_list


def export_normalized_dataset(out_dir: str, X: np.ndarray, y: np.ndarray, candidates: List[List[Dict[str, Any]]]) -> None:
    """Export normalized dataset artifacts for traceability.

    Writes:
      - X.npy: float32 feature matrix
      - y.npy: int64 labels (index into candidates per sample)
      - candidates.jsonl: one line per sample with { "candidates": [...] }
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    cand_path = os.path.join(out_dir, "candidates.jsonl")
    with open(cand_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps({"candidates": c}, ensure_ascii=False) + "\n")


