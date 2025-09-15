"""
Train Enhanced Pipeline Model với features đơn giản hơn
"""

import json
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_pipeline_model import EnhancedPipelineModel


def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSONL file"""
    
    records = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid JSON line: {e}")
                continue
    
    return records


def main():
    """Main training function"""
    
    # Configuration
    data_path = "data/sam_improved_training_data.jsonl"
    model_path = "models/enhanced_pipeline_model.pkl"
    
    print("🔄 Training Enhanced Pipeline Gameplay Model...")
    
    # Load training data
    print(f"📊 Loading data from {data_path}...")
    records = load_training_data(data_path)
    print(f"✅ Loaded {len(records)} records")
    
    if len(records) == 0:
        print("❌ No training data found!")
        return
    
    # Create enhanced pipeline model
    model = EnhancedPipelineModel()
    
    # Train Stage 1
    print("\n🤖 Training Stage 1 (Combo Type Selection)...")
    stage1_accuracy, stage1_report = model.train_stage1(records)
    print(f"✅ Stage 1 Accuracy: {stage1_accuracy:.4f}")
    
    # Train Stage 2
    print("\n🤖 Training Stage 2 (Card Selection)...")
    stage2_accuracy, stage2_report = model.train_stage2(records)
    print(f"✅ Stage 2 Accuracy: {stage2_accuracy:.4f}")
    
    # Print detailed reports
    print("\n📈 Stage 1 Classification Report:")
    for combo_type, metrics in stage1_report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"   {combo_type}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")
    
    print("\n📈 Stage 2 Classification Report:")
    for move_index, metrics in stage2_report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"   Move {move_index}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Test a few predictions
    print("\n🧪 Testing predictions...")
    for i in range(min(3, len(records))):
        record = records[i]
        try:
            prediction = model.predict(record)
            actual = record.get("action", {}).get("stage2", {})
            
            print(f"\nTest {i+1}:")
            print(f"   Hand: {len(record.get('hand', []))} cards")
            print(f"   Cards left: {record.get('cards_left', [])}")
            print(f"   Last move: {record.get('last_move', {}).get('combo_type', 'None')}")
            print(f"   Predicted: {prediction.get('type', 'unknown')} - {prediction.get('combo_type', 'N/A')}")
            print(f"   Actual: {actual.get('type', 'unknown')} - {actual.get('combo_type', 'N/A')}")
            
            # Check if prediction matches actual
            predicted_cards = set(prediction.get('cards', []))
            actual_cards = set(actual.get('cards', []))
            match = predicted_cards == actual_cards
            print(f"   Match: {'✅' if match else '❌'}")
            
        except Exception as e:
            print(f"   Error in prediction: {e}")
    
    print(f"\n🎯 Enhanced Pipeline Model Training Complete!")
    print(f"   Stage 1 Accuracy: {stage1_accuracy:.4f}")
    print(f"   Stage 2 Accuracy: {stage2_accuracy:.4f}")
    
    # Compare with previous model
    print(f"\n📊 COMPARISON WITH PREVIOUS MODEL:")
    print(f"   Previous Stage 1: 86.2%")
    print(f"   Previous Stage 2: 79.8%")
    print(f"   Enhanced Stage 1: {stage1_accuracy:.1%}")
    print(f"   Enhanced Stage 2: {stage2_accuracy:.1%}")
    
    improvement_stage1 = stage1_accuracy - 0.862
    improvement_stage2 = stage2_accuracy - 0.798
    print(f"   Stage 1 Improvement: {improvement_stage1:+.1%}")
    print(f"   Stage 2 Improvement: {improvement_stage2:+.1%}")


if __name__ == "__main__":
    main()
