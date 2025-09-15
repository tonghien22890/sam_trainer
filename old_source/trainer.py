from __future__ import annotations

import argparse
import math
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

from data_loader import build_dataset_from_jsonl, export_normalized_dataset
from model_architecture import make_estimator


def train(data_path: str, out_path: str, model_kind: str = "random_forest", export_dir: str = "runs/phase4_export") -> None:
    X, y, candidates = build_dataset_from_jsonl(data_path)
    # Export normalized artifacts for traceability
    export_normalized_dataset(export_dir, X, y, candidates)
    # Remove samples without a chosen label
    valid = y >= 0
    X = X[valid]
    y = y[valid]

    # Handle small datasets - use simple split without stratification if too small
    if len(X) < 20:
        # For small datasets, use simple split without stratification
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"⚠️  Small dataset ({len(X)} samples) - using simple split without stratification")
    else:
        # For larger datasets, try stratification but fallback to simple split if it fails
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print(f"✅ Using stratified split")
        except ValueError as e:
            print(f"⚠️  Stratification failed: {e}")
            print(f"⚠️  Falling back to simple split")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = make_estimator(model_kind)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"val accuracy: {acc:.4f}")

    joblib.dump(clf, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", type=str, help="Path to training data JSONL file")
    ap.add_argument("--out", type=str, default="runs/phase4_rf.pkl")
    ap.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "decision_tree"])
    ap.add_argument("--export", action="store_true", help="Export normalized dataset")
    ap.add_argument("--export-dir", type=str, default="runs/phase4_export", help="Export directory")
    args = ap.parse_args()
    
    if args.export:
        print(f"📊 Loading and exporting dataset from {args.data_path}")
        try:
            X, y, candidates = build_dataset_from_jsonl(args.data_path)
            print(f"📊 Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
            export_normalized_dataset(args.export_dir, X, y, candidates)
            print(f"✅ Exported dataset to {args.export_dir}")
            print(f"   - X.shape: {X.shape}")
            print(f"   - y.shape: {y.shape}")
            print(f"   - Valid samples: {(y >= 0).sum()}/{len(y)}")
        except Exception as e:
            print(f"❌ Error processing dataset: {e}")
            import traceback
            traceback.print_exc()
        return
    
    train(args.data_path, args.out, args.model, args.export_dir)


if __name__ == "__main__":
    main()


