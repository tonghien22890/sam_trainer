#!/usr/bin/env python3
"""
General Sequence Model - For general gameplay framework generation
Extends base classes for general gameplay (any hand size, flexible validation)
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

import os
import sys

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from base_sequence_models import BaseSequenceGenerator
from ai_common.core.combo_analyzer import ComboAnalyzer

logger = logging.getLogger(__name__)


class GeneralSequenceGenerator(BaseSequenceGenerator):
    """General Sequence Generator - For framework generation in general gameplay"""
    
    def __init__(self):
        super().__init__("general")
        logger.info("GeneralSequenceGenerator initialized")
    
    def validate_hand_for_game_type(self, possible_combos: List[Dict[str, Any]]) -> tuple:
        """General gameplay: skip strict validation, just analyze"""
        # Skip strict validation for general gameplay
        _, reason, strength_profile = self.rule_engine.validate_hand(possible_combos)
        logger.info(f"General context - skipping strict validation: {reason}")
        return True, "general_context_validation_skipped", strength_profile
    
    def generate_sequence(self, hand: List[int], player_count: int = 4) -> Dict[str, Any]:
        """Generate sequence for general gameplay (flexible hand size)
        
        Args:
            hand: List of card IDs (any size)
            player_count: Number of players
        
        Returns:
            Dict with sequence information for framework generation
        """
        logger.info(f"Generating general sequence for hand: {hand}")
        
        # Step 1: Analyze hand
        possible_combos = self.analyze_hand(hand)
        
        hand_data = {
            'hand': hand,
            'player_count': player_count,
            'possible_combos': possible_combos
        }
        
        # Step 2: For general context, skip strict validation but still analyze
        is_valid, reason, strength_profile = self.validate_hand_for_game_type(possible_combos)
        
        # Step 3: For general context, create mock validation result
        validation_result = self.create_mock_validation_result()
        logger.info(f"General context - using mock validation result")
        
        # Step 4: For general context, create mock user patterns
        user_patterns = self.create_mock_user_patterns()
        logger.info(f"General context - using mock user patterns")
        
        # Step 5: Build sequence from patterns
        sequence = self.build_sequence_from_patterns(possible_combos, user_patterns)
        
        # Step 6: Order by power (strongest first)
        ordered_sequence = sorted(sequence, key=lambda combo: -ComboAnalyzer.calculate_combo_strength(combo))
        
        # Step 7: For general context, calculate simple probability based on combo strengths
        unbeatable_prob = self.calculate_simple_unbeatable_probability(ordered_sequence)
        logger.info(f"General context - calculated simple unbeatable probability: {unbeatable_prob:.3f}")
        
        # Step 8: For general context, use default values
        user_threshold = 0.5
        should_declare = False  # Never declare Báo Sâm in general context
        
        # Calculate model confidence
        model_confidence = validation_result.get('confidence', 0.5)
        
        # Calculate sequence stats
        sequence_stats = self.calculate_sequence_stats(ordered_sequence, user_patterns)
        
        result = {
            'should_declare_bao_sam': bool(should_declare),
            'unbeatable_probability': float(unbeatable_prob),
            'user_threshold': float(user_threshold),
            'model_confidence': float(model_confidence),
            'reason': f'general_context_{unbeatable_prob:.2f}',
            'unbeatable_sequence': ordered_sequence,
            'sequence_stats': sequence_stats,
            # Additional general-specific fields
            'framework_strength': float(unbeatable_prob),
            'sequence_length': len(ordered_sequence),
            'avg_combo_strength': float(np.mean([ComboAnalyzer.calculate_combo_strength(combo) for combo in ordered_sequence])) if ordered_sequence else 0.0,
            'hand_size': len(hand),
            'player_count': player_count,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"General sequence generated - Decision: {should_declare}, Prob: {unbeatable_prob:.3f}, Threshold: {user_threshold:.3f}")
        return result
    
    def calculate_sequence_stats(self, sequence: List[Dict[str, Any]], user_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sequence stats - simplified version"""
        if not sequence:
            return {}
        
        strengths = [ComboAnalyzer.calculate_combo_strength(combo) for combo in sequence]
        return {
            'total_combos': len(sequence),
            'avg_strength': float(np.mean(strengths)),
            'max_strength': float(max(strengths)),
            'min_strength': float(min(strengths)),
            'combo_types': len(set(combo.get('combo_type', 'unknown') for combo in sequence))
        }
    
    def get_combo_recommendations(self, hand: List[int]) -> Dict[str, Any]:
        """Get combo recommendations for general gameplay
        
        Args:
            hand: List of card IDs
        
        Returns:
            Dict with combo recommendations and analysis
        """
        logger.info(f"Getting combo recommendations for hand: {hand}")
        
        # Analyze hand
        possible_combos = self.analyze_hand(hand)
        
        # Categorize combos by type
        combo_types = {}
        for combo in possible_combos:
            combo_type = combo.get('combo_type', 'unknown')
            if combo_type not in combo_types:
                combo_types[combo_type] = []
            
            strength = ComboAnalyzer.calculate_combo_strength(combo)
            combo_with_strength = combo.copy()
            combo_with_strength['strength'] = strength
            combo_types[combo_type].append(combo_with_strength)
        
        # Sort each type by strength
        for combo_type in combo_types:
            combo_types[combo_type].sort(key=lambda x: -x['strength'])
        
        # Generate recommendations
        recommendations = {
            'combo_types': combo_types,
            'total_combos': len(possible_combos),
            'unique_types': len(combo_types),
            'strongest_combo': max(possible_combos, key=lambda x: ComboAnalyzer.calculate_combo_strength(x)) if possible_combos else None,
            'hand_analysis': {
                'hand_size': len(hand),
                'has_strong_combos': any(ComboAnalyzer.calculate_combo_strength(combo) > 0.5 for combo in possible_combos),
                'combo_diversity': len(combo_types) / 5.0  # Normalize by max possible types
            }
        }
        
        logger.info(f"Combo recommendations generated - {len(possible_combos)} total combos, {len(combo_types)} types")
        return recommendations
    
    def validate_hand_for_general_play(self, hand: List[int]) -> Dict[str, Any]:
        """Validate hand for general gameplay (less strict than Báo Sâm)
        
        Args:
            hand: List of card IDs
        
        Returns:
            Dict with validation results
        """
        logger.info(f"Validating hand for general play: {hand}")
        
        # Basic validation
        if not hand:
            return {
                'is_valid': False,
                'reason': 'empty_hand',
                'confidence': 0.0
            }
        
        if len(hand) < 1:
            return {
                'is_valid': False,
                'reason': 'insufficient_cards',
                'confidence': 0.0
            }
        
        # Analyze combos
        possible_combos = self.analyze_hand(hand)
        
        if not possible_combos:
            return {
                'is_valid': False,
                'reason': 'no_valid_combos',
                'confidence': 0.0
            }
        
        # Check for reasonable combo distribution
        combo_types = set(combo.get('combo_type', 'unknown') for combo in possible_combos)
        combo_diversity = len(combo_types)
        
        # Calculate confidence based on combo quality and diversity
        strengths = [ComboAnalyzer.calculate_combo_strength(combo) for combo in possible_combos]
        avg_strength = np.mean(strengths) if strengths else 0.0
        
        # Higher confidence for hands with good combo diversity and strength
        confidence = min(1.0, (avg_strength * 0.5 + combo_diversity * 0.1))
        
        return {
            'is_valid': True,
            'reason': 'general_play_ready',
            'confidence': float(confidence),
            'combo_count': len(possible_combos),
            'combo_diversity': combo_diversity,
            'avg_strength': float(avg_strength)
        }


def main():
    """Main function for testing General Sequence Model"""
    logger.info("Starting General Sequence Model test")
    
    # Initialize generator
    generator = GeneralSequenceGenerator()
    
    # Test with sample hands of different sizes
    test_hands = [
        [3, 16, 29, 42, 7, 20],  # 6 cards
        [3, 16, 29, 42, 7, 20, 33, 46],  # 8 cards
        [3, 16, 29, 42, 7, 20, 33, 46, 11, 24],  # 10 cards
        [3, 16, 29, 42, 7, 20, 33, 46, 11, 24, 37, 50, 1],  # 13 cards
    ]
    
    for i, test_hand in enumerate(test_hands):
        print(f"\n{'='*80}")
        print(f"GENERAL SEQUENCE MODEL - TEST {i+1}")
        print(f"{'='*80}")
        print(f"Hand ({len(test_hand)} cards): {test_hand}")
        
        # Generate sequence
        result = generator.generate_sequence(test_hand, player_count=4)
        
        print(f"Framework strength: {result['framework_strength']:.3f}")
        print(f"Sequence length: {result['sequence_length']}")
        print(f"Avg combo strength: {result['avg_combo_strength']:.3f}")
        
        # Calculate max combo strength from sequence
        if result['unbeatable_sequence']:
            max_strength = max(ComboAnalyzer.calculate_combo_strength(combo) for combo in result['unbeatable_sequence'])
            print(f"Max combo strength: {max_strength:.3f}")
        else:
            print(f"Max combo strength: 0.000")
        
        if result['unbeatable_sequence']:
            print("\nFramework combos (top 5):")
            for j, combo in enumerate(result['unbeatable_sequence'][:5]):
                print(f"  {j+1}. {combo['combo_type']} rank={combo['rank_value']} strength={ComboAnalyzer.calculate_combo_strength(combo):.3f}")
        
        # Get recommendations
        recommendations = generator.get_combo_recommendations(test_hand)
        print(f"\nCombo analysis:")
        print(f"  Total combos: {recommendations['total_combos']}")
        print(f"  Unique types: {recommendations['unique_types']}")
        print(f"  Combo diversity: {recommendations['hand_analysis']['combo_diversity']:.3f}")
        
        # Validate hand
        validation = generator.validate_hand_for_general_play(test_hand)
        print(f"\nValidation:")
        print(f"  Valid: {validation['is_valid']}")
        print(f"  Reason: {validation['reason']}")
        print(f"  Confidence: {validation['confidence']:.3f}")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
