#!/usr/bin/env python3
"""
Test script để kiểm tra chức năng output combo sequence cho Báo Sâm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from hybrid_conservative_model import HybridConservativeModel
import json

def test_combo_sequence_output():
    """Test model với chức năng output combo sequence"""
    
    print("🔄 Testing Combo Sequence Output Functionality...")
    
    # Load model
    try:
        model = joblib.load('models/hybrid_conservative_bao_sam_model.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test cases với different combo sequences
    test_cases = [
        {
            "name": "Bài mạnh - Tứ Quý + Straight + Triple",
            "sequence": [
                {"cards": [0, 13, 26, 39], "combo_type": "quad", "rank_value": 0},
                {"cards": [8, 9, 10, 11, 12], "combo_type": "straight", "rank_value": 8},
                {"cards": [45], "combo_type": "single", "rank_value": 6}
            ],
            "expected_declare": True
        },
        {
            "name": "Bài yếu - Toàn single và pair",
            "sequence": [
                {"cards": [5], "combo_type": "single", "rank_value": 2},
                {"cards": [15, 16], "combo_type": "pair", "rank_value": 3},
                {"cards": [25], "combo_type": "single", "rank_value": 4},
                {"cards": [35, 36], "combo_type": "pair", "rank_value": 5},
                {"cards": [45], "combo_type": "single", "rank_value": 6},
                {"cards": [55, 56], "combo_type": "pair", "rank_value": 7},
                {"cards": [65], "combo_type": "single", "rank_value": 8}
            ],
            "expected_declare": False
        },
        {
            "name": "Bài mixed - Straight + Triple + Pair",
            "sequence": [
                {"cards": [6, 7, 8, 9, 10], "combo_type": "straight", "rank_value": 6},
                {"cards": [20, 33, 46], "combo_type": "triple", "rank_value": 7},
                {"cards": [12, 25], "combo_type": "pair", "rank_value": 12}
            ],
            "expected_declare": True
        }
    ]
    
    print(f"📋 Testing {len(test_cases)} cases...")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {case['name']}")
        
        # Create record
        record = {
            'sammove_sequence': case['sequence'],
            'hand': [card for combo in case['sequence'] for card in combo['cards']]
        }
        
        # Test prediction
        result = model.predict_hybrid(record)
        
        print(f"   Expected Declare: {case['expected_declare']}")
        print(f"   Actual Declare: {result['should_declare']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Reason: {result['reason']}")
        
        # Check combo sequence output
        if result['should_declare'] and result['optimal_combo_sequence']:
            print(f"   🎯 OPTIMAL COMBO SEQUENCE:")
            for j, combo in enumerate(result['optimal_combo_sequence']):
                strength = combo.get('strength', 0)
                position = combo.get('position', j)
                print(f"      {position+1}. {combo['combo_type']} (rank: {combo['rank_value']}, strength: {strength:.3f})")
            
            # Verify sequence has 10 cards total
            total_cards = sum(len(combo['cards']) for combo in result['optimal_combo_sequence'])
            print(f"   📊 Total cards: {total_cards}/10")
            
            if total_cards != 10:
                print(f"   ⚠️  WARNING: Sequence should have exactly 10 cards!")
        else:
            print(f"   ❌ No combo sequence generated (not declaring)")
    
    print("\n✅ Combo sequence output test completed!")

if __name__ == "__main__":
    test_combo_sequence_output()
