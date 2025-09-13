"""
Retrain Hybrid Conservative Model với combo types đúng cho Sâm
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from hybrid_conservative_model import HybridConservativeModel

def main():
    print("🔄 Retraining Hybrid Conservative Model với Sâm combo types...")
    
    # Create new model instance
    model = HybridConservativeModel()
    
    # Load and prepare data
    records = model.load_data('data/sam_training_data.jsonl')
    X, y = model.prepare_dataset(records)
    
    # Train model
    model.train_model(X, y)
    
    # Test model
    results = model.test_hybrid('data/sam_training_data.jsonl')
    
    # Save model
    model_path = 'models/hybrid_conservative_bao_sam_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Retrained Model saved to {model_path}")
    
    print(f"\n🎯 RETRAINED MODEL RESULTS:")
    print(f"   Accuracy: {results['accuracy']:.3f}")
    print(f"   Precision: {results['precision']:.3f}")
    print(f"   False Positives: {results['false_positives']} ⚠️")
    print(f"   Rulebase Blocked: {results['rulebase_blocked']}")

if __name__ == "__main__":
    main()
