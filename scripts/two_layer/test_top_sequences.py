#!/usr/bin/env python3
"""
Test top sequences from SequenceEvaluator
"""

import sys
sys.path.insert(0, '.')

from model_build.scripts.two_layer.framework_generator import FrameworkGenerator

def test_top_sequences():
    print("🧪 Testing top 3 sequences from SequenceEvaluator...")
    
    # Initialize FrameworkGenerator
    fg = FrameworkGenerator()
    
    # Test hand
    hand = [0, 1, 2, 13, 14, 15, 26, 27, 28, 39, 40, 41, 12]  # 3,4,5,6,7,8,9,10,J,Q,K,A,2
    print(f"Test hand: {hand}")
    print(f"Hand size: {len(hand)}")
    
    # Get top 3 sequences
    print("\n🔍 Getting top 3 sequences...")
    top_sequences = fg.sequence_evaluator.evaluate_top_sequences(hand, k=3)
    print(f"Found {len(top_sequences)} sequences")
    
    # Show detailed results
    for i, seq in enumerate(top_sequences):
        print(f"\n📋 Sequence {i+1}:")
        print(f"  Total Strength: {seq['total_strength']:.3f}")
        print(f"  Coverage Score: {seq['coverage_score']:.3f}")
        print(f"  End Rule Compliance: {seq['end_rule_compliance']}")
        print(f"  Combo Count: {seq['combo_count']}")
        print(f"  Avg Combo Strength: {seq['avg_combo_strength']:.3f}")
        print(f"  Used Cards: {sorted(seq['used_cards'])}")
        
        print(f"  Combos:")
        for j, combo in enumerate(seq['sequence']):
            print(f"    {j+1}. {combo['type']} rank={combo['rank_value']} strength={combo['strength']:.3f} cards={combo['cards']}")
    
    # Test with different hand
    print("\n" + "="*50)
    print("🧪 Testing with different hand...")
    
    hand2 = [0, 13, 1, 14, 2, 15, 3, 16, 4, 17, 5, 18, 6]  # 3,3,4,4,5,5,6,6,7,7,8,8,9
    print(f"Test hand 2: {hand2}")
    
    top_sequences2 = fg.sequence_evaluator.evaluate_top_sequences(hand2, k=3)
    print(f"Found {len(top_sequences2)} sequences")
    
    for i, seq in enumerate(top_sequences2):
        print(f"\n📋 Sequence {i+1}:")
        print(f"  Total Strength: {seq['total_strength']:.3f}")
        print(f"  Coverage Score: {seq['coverage_score']:.3f}")
        print(f"  End Rule Compliance: {seq['end_rule_compliance']}")
        print(f"  Combo Count: {seq['combo_count']}")
        
        print(f"  Combos:")
        for j, combo in enumerate(seq['sequence']):
            print(f"    {j+1}. {combo['type']} rank={combo['rank_value']} strength={combo['strength']:.3f} cards={combo['cards']}")

if __name__ == "__main__":
    test_top_sequences()
