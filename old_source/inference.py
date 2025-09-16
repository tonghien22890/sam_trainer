"""DEPRECATED: Old inference entrypoint (suit-dependent baseline).

Kept for reference. Prefer rank-only per-candidate pipeline in
`model_build/scripts/optimized_general_model_v3.py`.
"""

from __future__ import annotations

import argparse
import joblib
from typing import Dict, Any

from data_loader import encode_state_features


def predict(model_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
    clf = joblib.load(model_path)
    X, legal_moves, _ = encode_state_features(record)
    y_pred = clf.predict(X.reshape(1, -1))[0]
    if 0 <= y_pred < len(legal_moves):
        return legal_moves[y_pred]
    # Fallback: first legal or pass
    for m in legal_moves:
        if m.get("type") == "play_cards":
            return m
    return {"type": "pass", "cards": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    # This CLI expects a pre-encoded JSON dict via stdin or a file; kept simple here
    print("Inference module ready. Use predict(model_path, record_dict) programmatically.")

    args = ap.parse_args()
    _ = args


if __name__ == "__main__":
    main()


