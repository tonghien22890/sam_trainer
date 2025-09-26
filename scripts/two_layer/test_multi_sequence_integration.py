#!/usr/bin/env python3
"""
Test multi-sequence integration with StyleLearner
"""

import sys
import os
sys.path.insert(0, '.')

from model_build.scripts.two_layer.framework_generator import FrameworkGenerator
from model_build.scripts.two_layer.style_learner import StyleLearner

def test_multi_sequence_features():
    """Test multi-sequence feature extraction"""
    print("🧪 Testing multi-sequence feature extraction...")
    
    # Initialize components
    fg = FrameworkGenerator()
    sl = StyleLearner()
    
    # Test hand
    hand = [0, 1, 2, 13, 14, 15, 26, 27, 28, 39, 40, 41, 12]  # 3,4,5,6,7,8,9,10,J,Q,K,A,2
    print(f"Test hand: {hand}")
    
    # Generate framework with top 3 sequences
    framework = fg.generate_framework(hand)
    print(f"Framework generated with {len(framework.get('alternative_sequences', []))} alternative sequences")
    
    # Test feature extraction
    test_move = {
        'type': 'play_cards',
        'cards': [12],  # Play 2
        'combo_type': 'single',
        'rank_value': 12
    }
    
    # Extract all features
    original_features = sl.extract_original_features(test_move, {'hand': hand, 'meta': {'legal_moves': [test_move]}})
    framework_features = sl.extract_framework_features(test_move, framework)
    multi_sequence_features = sl.extract_multi_sequence_features(test_move, framework)
    
    print(f"Original features: {len(original_features)} dims")
    print(f"Framework features: {len(framework_features)} dims")
    print(f"Multi-sequence features: {len(multi_sequence_features)} dims")
    print(f"Total features: {len(original_features) + len(framework_features) + len(multi_sequence_features)} dims")
    
    # Show multi-sequence feature breakdown
    print(f"\nMulti-sequence features breakdown:")
    for i in range(3):
        start_idx = i * 5
        end_idx = start_idx + 5
        seq_features = multi_sequence_features[start_idx:end_idx]
        print(f"  Sequence {i+1}: {seq_features}")
    
    return True

def test_framework_with_alternatives():
    """Test framework generation with alternatives"""
    print("\n🧪 Testing framework with alternatives...")
    
    fg = FrameworkGenerator()
    
    # Test different hand types
    test_cases = [
        {
            'name': 'Strong hand with quads',
            'hand': [0, 13, 26, 39, 1, 14, 27, 40, 2, 15, 28, 41, 3]  # 3,3,3,3,4,4,4,4,5,5,5,5,6
        },
        {
            'name': 'Hand with pairs and straights',
            'hand': [0, 13, 1, 14, 2, 15, 3, 16, 4, 17, 5, 18, 6]  # 3,3,4,4,5,5,6,6,7,7,8,8,9
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['name']}")
        print(f"Hand: {test_case['hand']}")
        
        framework = fg.generate_framework(test_case['hand'])
        
        # Show framework info
        print(f"  Framework strength: {framework.get('framework_strength', 0):.3f}")
        print(f"  Coverage score: {framework.get('coverage_score', 0):.3f}")
        print(f"  End rule compliance: {framework.get('end_rule_compliance', True)}")
        print(f"  Core combos: {len(framework.get('core_combos', []))}")
        print(f"  Alternative sequences: {len(framework.get('alternative_sequences', []))}")
        
        # Show alternative sequences
        alt_sequences = framework.get('alternative_sequences', [])
        for i, alt_seq in enumerate(alt_sequences):
            print(f"    Alt {i+1}: strength={alt_seq.get('total_strength', 0):.3f}, "
                  f"coverage={alt_seq.get('coverage_score', 0):.3f}, "
                  f"combos={len(alt_seq.get('sequence', []))}")

if __name__ == "__main__":
    print("🚀 Testing multi-sequence integration...")
    
    # Test multi-sequence features
    success1 = test_multi_sequence_features()
    
    # Test framework with alternatives
    test_framework_with_alternatives()
    
    if success1:
        print("\n✅ Multi-sequence integration test completed!")
    else:
        print("\n❌ Multi-sequence integration test failed!")
