"""
Training script for Optimized General Model V3
Stage 1: Decision Tree
Stage 2: XGBoost
"""

import json
import os
import sys
from optimized_general_model_v3 import OptimizedGeneralModelV3

def load_training_data(filepath: str):
    """Load training data from JSONL file"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    return records

def main():
    print("=" * 60)
    print("OPTIMIZED GENERAL MODEL V3 TRAINING")
    print("=" * 60)
    
    # 1. Load training data
    print("\n1. Loading training data...")
    data_file = "data/synthetic_training_data.jsonl"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found")
        print("Please run generate_test_data.py first")
        return
    
    records = load_training_data(data_file)
    print(f"Loaded {len(records)} training records")
    
    # 2. Initialize model
    print("\n2. Initializing model...")
    model = OptimizedGeneralModelV3()
    
    # 3. Train Stage 1
    print("\n3. Training Stage 1 (Decision Tree - Combo Type Selection)...")
    stage1_accuracy, stage1_report = model.train_stage1(records)
    print(f"Stage 1 Training Accuracy: {stage1_accuracy:.4f}")
    print(f"Stage 1 Classification Report:")
    print(f"  Precision (macro avg): {stage1_report.get('macro avg', {}).get('precision', 0):.3f}")
    print(f"  Recall (macro avg): {stage1_report.get('macro avg', {}).get('recall', 0):.3f}")
    print(f"  F1-score (macro avg): {stage1_report.get('macro avg', {}).get('f1-score', 0):.3f}")
    
    # 4. Train Stage 2
    print("\n4. Training Stage 2 (XGBoost - Card Selection)...")
    stage2_accuracy, stage2_report = model.train_stage2(records)
    print(f"Stage 2 Training Accuracy: {stage2_accuracy:.4f}")
    print(f"Stage 2 Classification Report:")
    print(f"  Precision (macro avg): {stage2_report.get('macro avg', {}).get('precision', 0):.3f}")
    print(f"  Recall (macro avg): {stage2_report.get('macro avg', {}).get('recall', 0):.3f}")
    print(f"  F1-score (macro avg): {stage2_report.get('macro avg', {}).get('f1-score', 0):.3f}")
    
    # 5. Save model
    print("\n5. Saving model...")
    model_path = "models/optimized_general_model_v3.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Stage 1 (Decision Tree): {stage1_accuracy:.4f}")
    print(f"Stage 2 (XGBoost): {stage2_accuracy:.4f}")

if __name__ == "__main__":
    main()


