#!/usr/bin/env python3
"""
Synthetic Data Generator for Unbeatable Sequence Model
Generate training data with realistic user behavior patterns
"""

import json
import random
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic training data for all model phases"""
    
    def __init__(self):
        self.combo_types = ['single', 'pair', 'triple', 'straight', 'quad']
        self.user_profiles = {
            'conservative': {'threshold_base': 0.85, 'variance': 0.05, 'power_preference': 0.8},
            'balanced': {'threshold_base': 0.75, 'variance': 0.08, 'power_preference': 0.6},
            'aggressive': {'threshold_base': 0.65, 'variance': 0.12, 'power_preference': 0.4},
        }
        logger.info("SyntheticDataGenerator initialized")
    
    def generate_hand(self) -> List[int]:
        """Generate random 10-card hand"""
        # Create deck
        deck = list(range(52))
        random.shuffle(deck)
        return deck[:10]
    
    def generate_combo(self, combo_type: str, rank: int) -> Dict[str, Any]:
        """Generate a combo of specified type and rank"""
        cards = []
        
        if combo_type == 'single':
            cards = [rank]
        elif combo_type == 'pair':
            cards = [rank, rank + 13]
        elif combo_type == 'triple':
            cards = [rank, rank + 13, rank + 26]
        elif combo_type == 'quad':
            cards = [rank, rank + 13, rank + 26, rank + 39]
        elif combo_type == 'straight':
            # Generate 5-card straight
            length = random.choice([3, 4, 5, 6])
            if rank + length > 12:  # Avoid going over King
                rank = 12 - length
            cards = list(range(rank, rank + length))
        
        return {
            'combo_type': combo_type,
            'rank_value': rank,
            'cards': cards
        }
    
    def generate_realistic_combos(self, target_cards: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic combo combinations that use exactly 10 cards"""
        combos = []
        cards_used = 0
        
        # Strategy: Generate one strong combo, then fill with weaker ones
        strong_combo_type = random.choice(['quad', 'triple', 'straight'])
        
        if strong_combo_type == 'quad':
            rank = random.randint(8, 12)  # High rank quad
            combo = self.generate_combo('quad', rank)
            combos.append(combo)
            cards_used += 4
            
            # Fill remaining 6 cards
            remaining = target_cards - cards_used
            if remaining >= 3:
                # Add a triple
                rank2 = random.randint(0, 7)
                combo2 = self.generate_combo('triple', rank2)
                combos.append(combo2)
                cards_used += 3
                remaining -= 3
            
            # Fill with singles/pairs
            while cards_used < target_cards:
                if target_cards - cards_used >= 2 and random.random() < 0.5:
                    rank3 = random.randint(0, 12)
                    combo3 = self.generate_combo('pair', rank3)
                    combos.append(combo3)
                    cards_used += 2
                else:
                    rank3 = random.randint(0, 12)
                    combo3 = self.generate_combo('single', rank3)
                    combos.append(combo3)
                    cards_used += 1
        
        elif strong_combo_type == 'triple':
            # Two triples + filler
            rank1 = random.randint(8, 12)  # High rank
            combo1 = self.generate_combo('triple', rank1)
            combos.append(combo1)
            cards_used += 3
            
            rank2 = random.randint(0, 7)  # Lower rank
            combo2 = self.generate_combo('triple', rank2)
            combos.append(combo2)
            cards_used += 3
            
            # Fill remaining 4 cards
            remaining = target_cards - cards_used
            while cards_used < target_cards:
                if remaining >= 2 and random.random() < 0.6:
                    rank3 = random.randint(0, 12)
                    combo3 = self.generate_combo('pair', rank3)
                    combos.append(combo3)
                    cards_used += 2
                    remaining -= 2
                else:
                    rank3 = random.randint(0, 12)
                    combo3 = self.generate_combo('single', rank3)
                    combos.append(combo3)
                    cards_used += 1
                    remaining -= 1
        
        elif strong_combo_type == 'straight':
            # Long straight + filler
            length = random.choice([5, 6, 7])
            rank = random.randint(0, 12 - length)
            combo = {
                'combo_type': 'straight',
                'rank_value': rank + length - 1,  # Highest card
                'cards': list(range(rank, rank + length))
            }
            combos.append(combo)
            cards_used += length
            
            # Fill remaining
            remaining = target_cards - cards_used
            while cards_used < target_cards:
                if remaining >= 3 and random.random() < 0.4:
                    rank2 = random.randint(0, 12)
                    combo2 = self.generate_combo('triple', rank2)
                    combos.append(combo2)
                    cards_used += 3
                    remaining -= 3
                elif remaining >= 2 and random.random() < 0.6:
                    rank2 = random.randint(0, 12)
                    combo2 = self.generate_combo('pair', rank2)
                    combos.append(combo2)
                    cards_used += 2
                    remaining -= 2
                else:
                    rank2 = random.randint(0, 12)
                    combo2 = self.generate_combo('single', rank2)
                    combos.append(combo2)
                    cards_used += 1
                    remaining -= 1
        
        return combos
    
    def calculate_combo_strength(self, combo: Dict[str, Any]) -> float:
        """Calculate combo strength (same as main model)"""
        combo_type = combo['combo_type']
        rank_value = combo.get('rank_value', 0)
        cards = combo.get('cards', [])

        is_two = (rank_value == 12)
        is_ace = (rank_value == 11)
        is_face = rank_value in (8, 9, 10)

        # Straights
        if combo_type == 'straight':
            length = len(cards)
            if length >= 10:
                return 1.0
            if length >= 7:
                return 0.85 + (length - 7) * 0.02
            elif length == 6:
                return 0.6 + (rank_value / 11.0) * 0.05
            elif length == 5:
                return 0.4 + (rank_value / 11.0) * 0.05
            else:
                return 0.3 + (length - 3) * 0.05 + (rank_value / 11.0) * 0.02

        # Singles
        if combo_type == 'single':
            if is_two:
                return 1.0
            if is_ace:
                return 0.3
            return 0.1

        # Pairs
        if combo_type == 'pair':
            if is_two:
                return 1.0
            if is_ace:
                return 0.8
            return 0.2 + (min(rank_value, 7) / 7.0) * 0.1

        # Triples
        if combo_type == 'triple':
            if is_two:
                return 1.0
            if is_ace:
                return 0.9
            if is_face:
                return 0.8
            if rank_value >= 4:
                return 0.5
            return 0.3 + (rank_value / 4.0) * 0.05

        # Quads
        if combo_type == 'quad':
            if is_two:
                return 1.0
            if is_ace:
                return 0.98
            return 0.95 + (rank_value / 11.0) * 0.03

        return 0.1
    
    def simulate_user_decision(self, combos: List[Dict[str, Any]], user_profile: str, player_count: int) -> Tuple[bool, float, float]:
        """Simulate user decision to declare Báo Sâm"""
        profile = self.user_profiles[user_profile]
        
        # Calculate unbeatable probability
        strengths = [self.calculate_combo_strength(combo) for combo in combos]
        avg_strength = np.mean(strengths)
        max_strength = max(strengths)
        strong_count = sum(1 for s in strengths if s >= 0.8)
        
        unbeatable_prob = min(1.0, avg_strength * 0.4 + max_strength * 0.4 + (strong_count / len(strengths)) * 0.2)
        
        # User's threshold (varies by profile and context)
        base_threshold = profile['threshold_base']
        variance = profile['variance']
        
        # Adjust for player count (more players = higher threshold)
        player_adjustment = (player_count - 2) * 0.02
        
        # Add some randomness
        noise = np.random.normal(0, variance)
        
        user_threshold = base_threshold + player_adjustment + noise
        user_threshold = max(0.5, min(0.95, user_threshold))
        
        # Decision
        would_declare = unbeatable_prob >= user_threshold
        
        return would_declare, unbeatable_prob, user_threshold
    
    def generate_validation_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate Phase 1 validation training data"""
        logger.info(f"Generating {num_samples} validation samples...")
        
        data = []
        
        for i in range(num_samples):
            # Generate hand and combos
            hand = self.generate_hand()
            
            # 70% valid hands, 30% invalid
            if random.random() < 0.7:
                combos = self.generate_realistic_combos(10)
                is_valid = True
            else:
                # Generate invalid hand (too weak)
                combos = []
                # Only weak combos
                for _ in range(random.randint(3, 5)):
                    combo_type = random.choice(['single', 'pair'])
                    rank = random.randint(0, 6)  # Low ranks
                    combo = self.generate_combo(combo_type, rank)
                    combos.append(combo)
                is_valid = False
            
            player_count = random.choice([2, 3, 4])
            
            record = {
                'hand': hand,
                'player_count': player_count,
                'possible_combos': combos,
                'is_valid': is_valid,
                'timestamp': datetime.now().isoformat()
            }
            
            data.append(record)
        
        logger.info(f"Generated {len(data)} validation samples")
        return data
    
    def generate_pattern_data(self, num_samples: int = 2000) -> List[Dict[str, Any]]:
        """Generate Phase 2 pattern learning data"""
        logger.info(f"Generating {num_samples} pattern samples...")
        
        data = []
        
        for i in range(num_samples):
            # Choose user profile
            user_profile = random.choice(['conservative', 'balanced', 'aggressive'])
            profile_data = self.user_profiles[user_profile]
            
            hand = self.generate_hand()
            combos = self.generate_realistic_combos(10)
            player_count = random.choice([2, 3, 4])
            
            # Calculate pattern score based on profile
            strengths = [self.calculate_combo_strength(combo) for combo in combos]
            power_concentration = sum(1 for s in strengths if s >= 0.8) / len(strengths)
            combo_diversity = len(set(combo['combo_type'] for combo in combos)) / 5.0
            
            # Pattern score influenced by user profile
            pattern_score = (
                profile_data['power_preference'] * power_concentration +
                (1 - profile_data['power_preference']) * combo_diversity
            )
            pattern_score = max(0.0, min(1.0, pattern_score))
            
            record = {
                'hand': hand,
                'player_count': player_count,
                'possible_combos': combos,
                'user_profile': user_profile,
                'pattern_score': pattern_score,
                'user_patterns': {
                    'combo_patterns': {
                        'power_concentration': power_concentration,
                        'combo_diversity': combo_diversity,
                        'balance_preference': np.var(strengths)
                    },
                    'sequence_building_preference': 'power_first' if power_concentration > 0.6 else 'balanced'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            data.append(record)
        
        logger.info(f"Generated {len(data)} pattern samples")
        return data
    
    def generate_threshold_data(self, num_samples: int = 1500) -> List[Dict[str, Any]]:
        """Generate Phase 3 threshold learning data"""
        logger.info(f"Generating {num_samples} threshold samples...")
        
        data = []
        
        for i in range(num_samples):
            user_profile = random.choice(['conservative', 'balanced', 'aggressive'])
            
            hand = self.generate_hand()
            combos = self.generate_realistic_combos(10)
            player_count = random.choice([2, 3, 4])
            
            # Simulate user decision
            would_declare, unbeatable_prob, user_threshold = self.simulate_user_decision(
                combos, user_profile, player_count
            )
            
            # Generate user patterns
            strengths = [self.calculate_combo_strength(combo) for combo in combos]
            power_concentration = float(sum(1 for s in strengths if s >= 0.8) / len(strengths))
            combo_diversity = float(len(set(combo['combo_type'] for combo in combos)) / 5.0)
            
            record = {
                'hand': list(map(int, hand)),
                'player_count': int(player_count),
                'possible_combos': combos,
                'user_profile': str(user_profile),
                'unbeatable_probability': float(unbeatable_prob),
                'user_threshold': float(user_threshold),
                'would_declare': bool(would_declare),
                'user_patterns': {
                    'combo_patterns': {
                        'power_concentration': float(power_concentration),
                        'combo_diversity': float(combo_diversity),
                        'balance_preference': float(np.var(strengths))
                    },
                    'pattern_score': float(random.uniform(0.3, 0.9)),
                    'sequence_building_preference': 'power_first' if power_concentration > 0.6 else 'balanced'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            data.append(record)
        
        logger.info(f"Generated {len(data)} threshold samples")
        return data
    
    def save_data(self, data: List[Dict[str, Any]], filename: str):
        """Save data to JSONL file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} records to {filename}")
    
    def generate_all_data(self):
        """Generate all training data phases"""
        logger.info("Generating all synthetic training data...")
        
        # Phase 1: Validation data
        validation_data = self.generate_validation_data(1000)
        self.save_data(validation_data, 'data/validation_training_data.jsonl')
        
        # Phase 2: Pattern data
        pattern_data = self.generate_pattern_data(2000)
        self.save_data(pattern_data, 'data/pattern_training_data.jsonl')
        
        # Phase 3: Threshold data
        threshold_data = self.generate_threshold_data(1500)
        self.save_data(threshold_data, 'data/threshold_training_data.jsonl')
        
        logger.info("All synthetic data generated successfully!")


def main():
    """Generate synthetic training data"""
    # Create data directory
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    generator = SyntheticDataGenerator()
    generator.generate_all_data()
    
    print("Synthetic data generation completed!")
    print("Files created:")
    print("- data/validation_training_data.jsonl")
    print("- data/pattern_training_data.jsonl") 
    print("- data/threshold_training_data.jsonl")


if __name__ == "__main__":
    main()
