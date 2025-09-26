#!/usr/bin/env python3
"""
Training script for Style Learner (Layer 2)
"""

import json
import os
import sys
from typing import List, Dict, Any

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_build_dir = os.path.dirname(os.path.dirname(current_dir))  # model_build/
project_root = os.path.dirname(model_build_dir)  # AI-Sam/

# Add to path
if model_build_dir not in sys.path:
    sys.path.insert(0, model_build_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from style_learner import StyleLearner
from framework_generator import FrameworkGenerator


def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSONL file"""
    records = []
    
    if not os.path.exists(data_path):
        print(f"⚠️ Training data not found at {data_path}")
        return records
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing line {line_num}: {e}")
                    continue
        
        print(f"✅ Loaded {len(records)} training records from {data_path}")
        return records
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return []


def prepare_training_data_with_framework(records: List[Dict[str, Any]], 
                                       use_hand_variations: bool = True,
                                       variations_per_hand: int = 3) -> List[Dict[str, Any]]:
    """Prepare training data với framework từ Layer 1 + Hand Variations"""
    print("🎯 Preparing training data with framework...")
    print(f"🎯 Hand variations: {'Enabled' if use_hand_variations else 'Disabled'}")
    print(f"🎯 Variations per hand: {variations_per_hand}")
    
    framework_generator = FrameworkGenerator()
    prepared_records = []
    
    for i, record in enumerate(records):
        if i % 100 == 0:
            print(f"🎯 Processing record {i}/{len(records)}")
            
        hand = record.get('hand', [])
        if not hand:
            continue
            
        # Original record
        original_framework = framework_generator.generate_framework(hand)
        record_copy = record.copy()
        record_copy['framework'] = original_framework
        prepared_records.append(record_copy)
        
        # Generate hand variations for more diverse training
        if use_hand_variations and len(hand) >= 6:  # Only for hands with enough cards
            variations = generate_hand_variations(hand, variations_per_hand)
            
            for j, variation in enumerate(variations):
                try:
                    # Generate framework for variation
                    variation_framework = framework_generator.generate_framework(variation)
                    
                    # Create new record with variation hand but same legal_moves and action
                    variation_record = record.copy()
                    variation_record['hand'] = variation
                    variation_record['framework'] = variation_framework
                    variation_record['is_variation'] = True
                    variation_record['variation_id'] = j
                    variation_record['original_hand'] = hand
                    
                    prepared_records.append(variation_record)
                    
                except Exception as e:
                    print(f"⚠️ Error processing variation {j} for record {i}: {e}")
                    continue
    
    print(f"✅ Prepared {len(prepared_records)} records with framework")
    print(f"✅ Original records: {len(records)}")
    print(f"✅ Total records (with variations): {len(prepared_records)}")
    return prepared_records


def generate_hand_variations(hand: List[int], num_variations: int = 3) -> List[List[int]]:
    """Generate hand variations using multiple strategies"""
    import random
    
    variations = []
    hand_copy = hand.copy()
    
    # Strategy 1: Random swaps (1-2 cards)
    for i in range(min(2, num_variations)):
        variation = hand_copy.copy()
        
        # Swap 1-2 cards randomly
        num_swaps = random.randint(1, min(2, len(hand) // 3))
        
        for _ in range(num_swaps):
            # Pick random positions
            pos1, pos2 = random.sample(range(len(variation)), 2)
            variation[pos1], variation[pos2] = variation[pos2], variation[pos1]
        
        variations.append(variation)
    
    # Strategy 2: Rank-based variations (if we have enough variations left)
    if len(variations) < num_variations and len(hand) >= 8:
        variation = hand_copy.copy()
        
        # Group cards by rank (0-12)
        rank_groups = {}
        for card in variation:
            rank = card % 13
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)
        
        # Swap cards between different ranks
        available_ranks = [r for r in rank_groups if len(rank_groups[r]) > 0]
        if len(available_ranks) >= 2:
            rank1, rank2 = random.sample(available_ranks, 2)
            card1 = rank_groups[rank1][0]
            card2 = rank_groups[rank2][0]
            
            # Find positions and swap
            pos1 = variation.index(card1)
            pos2 = variation.index(card2)
            variation[pos1], variation[pos2] = variation[pos2], variation[pos1]
            
            variations.append(variation)
    
    # Strategy 3: Suit-based variations (if we have enough variations left)
    if len(variations) < num_variations and len(hand) >= 6:
        variation = hand_copy.copy()
        
        # Group cards by suit (0-3)
        suit_groups = {}
        for card in variation:
            suit = card // 13
            if suit not in suit_groups:
                suit_groups[suit] = []
            suit_groups[suit].append(card)
        
        # Swap cards between different suits
        available_suits = [s for s in suit_groups if len(suit_groups[s]) > 0]
        if len(available_suits) >= 2:
            suit1, suit2 = random.sample(available_suits, 2)
            card1 = suit_groups[suit1][0]
            card2 = suit_groups[suit2][0]
            
            # Find positions and swap
            pos1 = variation.index(card1)
            pos2 = variation.index(card2)
            variation[pos1], variation[pos2] = variation[pos2], variation[pos1]
            
            variations.append(variation)
    
    # Fill remaining variations with random swaps
    while len(variations) < num_variations:
        variation = hand_copy.copy()
        
        # Random swap
        if len(variation) >= 2:
            pos1, pos2 = random.sample(range(len(variation)), 2)
            variation[pos1], variation[pos2] = variation[pos2], variation[pos1]
        
        variations.append(variation)
    
    return variations[:num_variations]


def main():
    """Main training function"""
    print("🎯 Training Style Learner (Layer 2)...")
    
    # Data paths
    data_path = os.path.join(model_build_dir, "simple_synthetic_training_data_with_sequence.jsonl")
    model_path = os.path.join(model_build_dir, "models", "style_learner_model.pkl")
    
    # Load training data
    training_data = load_training_data(data_path)
    if not training_data:
        print("❌ No training data available")
        return
    
    # Prepare data với framework + hand variations
    print("🎯 Configuring training parameters...")
    # Disable hand variations for data integrity (re-enable after recomputing legal_moves for variations)
    use_variations = False
    variations_per_hand = 0
    
    prepared_data = prepare_training_data_with_framework(
        training_data, 
        use_hand_variations=use_variations,
        variations_per_hand=variations_per_hand
    )
    
    # Initialize StyleLearner
    style_learner = StyleLearner()
    
    # Train with multi-sequence features
    print("🎯 Starting training with multi-sequence features...")
    print("🎯 Using FrameworkGenerator (Layer 1) + StyleLearner (Layer 2)")
    training_results = style_learner.train(prepared_data)
    
    print("✅ Training completed!")
    print(f"Training results: {training_results}")
    
    # Save model
    style_learner.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Test prediction với original và variation data
    print("\n🎯 Testing prediction with multi-sequence features...")
    if prepared_data:
        # Test original record
        original_records = [r for r in prepared_data if not r.get('is_variation', False)]
        if original_records:
            test_record = original_records[0]
            hand = test_record.get('hand', [])
            legal_moves = test_record.get('meta', {}).get('legal_moves', [])
            framework = test_record.get('framework', {})
            
            print(f"🎯 Testing original record: hand={hand}")
            print(f"🎯 Framework has {len(framework.get('alternative_sequences', []))} alternative sequences")
            
            if hand and legal_moves:
                predicted_move = style_learner.predict_with_framework(
                    test_record, legal_moves, framework
                )
                print(f"Original prediction: {predicted_move}")
        
        # Test variation record
        variation_records = [r for r in prepared_data if r.get('is_variation', False)]
        if variation_records:
            test_variation = variation_records[0]
            variation_hand = test_variation.get('hand', [])
            original_hand = test_variation.get('original_hand', [])
            variation_framework = test_variation.get('framework', {})
            
            print(f"🎯 Testing variation record:")
            print(f"  Original hand: {original_hand}")
            print(f"  Variation hand: {variation_hand}")
            print(f"  Variation framework has {len(variation_framework.get('alternative_sequences', []))} alternative sequences")
            
            if variation_hand and test_variation.get('meta', {}).get('legal_moves'):
                predicted_move = style_learner.predict_with_framework(
                    test_variation, test_variation.get('meta', {}).get('legal_moves'), variation_framework
                )
                print(f"Variation prediction: {predicted_move}")
    
    # Statistics
    print(f"\n📊 Training Statistics:")
    print(f"  Original records: {len([r for r in prepared_data if not r.get('is_variation', False)])}")
    print(f"  Variation records: {len([r for r in prepared_data if r.get('is_variation', False)])}")
    print(f"  Total records: {len(prepared_data)}")
    print(f"  Data augmentation ratio: {len(prepared_data) / len(training_data):.2f}x")


if __name__ == "__main__":
    main()
