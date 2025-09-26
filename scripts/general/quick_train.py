#!/usr/bin/env python3

import sys
sys.path.append('.')

from optimized_general_model_v3 import OptimizedGeneralModelV3
import json
import os

def main():
    print("=== QUICK TRAINING ===")
    
    # Load training data
    data_file = "formatted_training_data.jsonl"
    print(f"Loading data from {data_file}")
    
    records = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records")
    
    # Initialize model
    model = OptimizedGeneralModelV3()
    print("Model initialized")
    
    # Train model
    print("Training model...")
    accuracy = model.train_candidate_model(records, model_type="xgb")
    print(f"Training accuracy: {accuracy:.4f}")
    
    # Evaluate
    eval_results = model.evaluate_candidate_model(records)
    print(f"Turn accuracy: {eval_results.get('turn_accuracy', 0):.3f}")
    
    # Save model
    model_path = "models/optimized_general_model_v3.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
