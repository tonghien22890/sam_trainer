"""
Data formatting and training script for Optimized General Model V3
Per-Candidate Approach: XGBoost model ranks all legal moves
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

def export_formatted_data(records, output_file: str):
    """Export formatted training data to JSONL file"""
    # Filter records with action field
    action_records = [r for r in records if 'action' in r]
    
    print(f"Exporting {len(action_records)} formatted records to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in action_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return action_records

def main():
    print("=" * 60)
    print("OPTIMIZED GENERAL MODEL V3 - DATA FORMAT & TRAINING")
    print("=" * 60)
    
    # 1. Load training data
    print("\n1. Loading training data...")
    # Try formatted data first, then synthetic data, fallback to original data
    data_file = "formatted_training_data.jsonl"  # Use formatted data first
    if not os.path.exists(data_file):
        data_file = "simple_synthetic_training_data.jsonl"  # Use synthetic data
        if not os.path.exists(data_file):
            data_file = "../training_data.jsonl"  # Fallback to original data
            if not os.path.exists(data_file):
                print(f"Error: No training data found")
                print("Please ensure formatted_training_data.jsonl, simple_synthetic_training_data.jsonl, or training_data.jsonl exists")
                return
    
    print(f"Using data file: {data_file}")
    
    records = load_training_data(data_file)
    print(f"Loaded {len(records)} total records")
    
    # 2. Export formatted data
    print("\n2. Exporting formatted data...")
    formatted_file = "formatted_training_data.jsonl"
    action_records = export_formatted_data(records, formatted_file)
    print(f"Exported {len(action_records)} formatted records to {formatted_file}")
    
    # 3. Initialize model
    print("\n3. Initializing model...")
    model = OptimizedGeneralModelV3()
    
    # 4. Train Per-Candidate Model
    print("\n4. Training Per-Candidate Model...")
    candidate_accuracy = model.train_candidate_model(action_records, model_type="xgb")
    print(f"Per-Candidate Training Accuracy: {candidate_accuracy:.4f}")
    
    # Show hybrid approach info
    hybrid_info = model.get_hybrid_info()
    print(f"Hybrid Approach Info:")
    print(f"  Training Data Size: {hybrid_info['training_data_size']}")
    print(f"  Approach: {hybrid_info['approach']}")
    print(f"  Threshold: {hybrid_info['threshold']}")
    print(f"  Description: {hybrid_info['description']}")
    
    # Evaluate per-candidate model
    eval_results = model.evaluate_candidate_model(action_records)
    eval_top3 = model.evaluate_candidate_model_topk(action_records, k=3)
    
    print(f"Per-Candidate Evaluation:")
    print(f"  Turn Accuracy: {eval_results.get('turn_accuracy', 0):.3f}")
    print(f"  Top-3 Accuracy: {eval_top3.get('turn_topk_accuracy', 0):.3f}")
    print(f"  Number of turns: {eval_results.get('num_turns', 0)}")
    print(f"  Number of samples: {eval_results.get('num_samples', 0)}")
    
    # 5. Save model
    print("\n5. Saving model...")
    model_path = "models/optimized_general_model_v3.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Per-Candidate Model (XGBoost): {candidate_accuracy:.4f}")
    print(f"Turn Accuracy: {eval_results.get('turn_accuracy', 0):.3f}")
    print(f"Formatted data exported to: {formatted_file}")

if __name__ == "__main__":
    main()


