#!/usr/bin/env python3
"""
BaoSam Data Generator - Specialized for Báo Sâm training data
"""

import random
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import os
import sys

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_generation_dir = os.path.dirname(current_dir)
model_build_dir = os.path.dirname(os.path.dirname(data_generation_dir))
project_root = os.path.dirname(model_build_dir)

if data_generation_dir not in sys.path:
    sys.path.insert(0, data_generation_dir)
if model_build_dir not in sys.path:
    sys.path.insert(0, model_build_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from base import BaseDataGenerator, ComboGenerator, GameSimulator

logger = logging.getLogger(__name__)

class BaoSamGameSimulator(GameSimulator):
    """BaoSam-specific game simulator"""
    
    def __init__(self):
        super().__init__('bao_sam')
        self.user_profiles = {
            'conservative': {'threshold_base': 0.85, 'variance': 0.05, 'power_preference': 0.8},
            'balanced': {'threshold_base': 0.75, 'variance': 0.08, 'power_preference': 0.6},
            'aggressive': {'threshold_base': 0.65, 'variance': 0.12, 'power_preference': 0.4},
        }
    
    def simulate_user_behavior(self, combos: List[Dict[str, Any]], user_profile: str, 
                              player_count: int) -> Tuple[bool, float, float]:
        """Simulate user decision to declare Báo Sâm"""
        profile = self.user_profiles[user_profile]
        
        # Calculate unbeatable probability
        combo_gen = ComboGenerator('bao_sam')
        strengths = [combo_gen.calculate_combo_strength(combo) for combo in combos]
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

class BaoSamGenerator(BaseDataGenerator):
    """Generator cho Báo Sâm training data"""
    
    def __init__(self):
        super().__init__('bao_sam')
        self.game_simulator = BaoSamGameSimulator()
        logger.info("BaoSamGenerator initialized")
    
    def generate_training_data(self, num_sessions: int) -> List[Dict[str, Any]]:
        """Generate Báo Sâm specific training data"""
        logger.info(f"Generating {num_sessions} Báo Sâm training sessions...")
        
        all_records = []
        
        for i in range(num_sessions):
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_sessions} Báo Sâm sessions...")
            
            try:
                # Generate Báo Sâm specific data
                session_records = self.generate_bao_sam_session()
                all_records.extend(session_records)
            except Exception as e:
                logger.warning(f"Error generating Báo Sâm session {i + 1}: {e}")
                continue
        
        logger.info(f"Generated {len(all_records)} total Báo Sâm records")
        return all_records
    
    def generate_bao_sam_session(self) -> List[Dict[str, Any]]:
        """Generate a single Báo Sâm session"""
        # Generate hand with 10 cards
        hand = self.generate_hand(10)
        
        # Generate realistic combos for Báo Sâm
        combos = self.combo_generator.generate_realistic_combos(10)
        
        # Choose user profile
        user_profile = random.choice(['conservative', 'balanced', 'aggressive'])
        player_count = random.choice([2, 3, 4])
        
        # Simulate user decision
        would_declare, unbeatable_prob, user_threshold = self.game_simulator.simulate_user_behavior(
            combos, user_profile, player_count
        )
        
        # Calculate pattern features
        strengths = [self.combo_generator.calculate_combo_strength(combo) for combo in combos]
        power_concentration = sum(1 for s in strengths if s >= 0.8) / len(strengths)
        combo_diversity = len(set(combo['combo_type'] for combo in combos)) / 5.0
        
        # Create Báo Sâm specific record
        record = {
            'hand': hand,
            'player_count': player_count,
            'possible_combos': combos,
            'user_profile': user_profile,
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
        
        return [record]  # Báo Sâm is single-decision, not turn-based
    
    def generate_validation_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate Phase 1 validation training data"""
        logger.info(f"Generating {num_samples} validation samples...")
        
        data = []
        
        for i in range(num_samples):
            # Generate hand and combos
            hand = self.generate_hand(10)
            
            # 70% valid hands, 30% invalid
            if random.random() < 0.7:
                combos = self.combo_generator.generate_realistic_combos(10)
                is_valid = True
            else:
                # Generate invalid hand (too weak)
                combos = []
                # Only weak combos
                for _ in range(random.randint(3, 5)):
                    combo_type = random.choice(['single', 'pair'])
                    rank = random.randint(0, 6)  # Low ranks
                    combo = self.combo_generator._generate_combo(combo_type, rank)
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
            profile_data = self.game_simulator.user_profiles[user_profile]
            
            hand = self.generate_hand(10)
            combos = self.combo_generator.generate_realistic_combos(10)
            player_count = random.choice([2, 3, 4])
            
            # Calculate pattern score based on profile
            strengths = [self.combo_generator.calculate_combo_strength(combo) for combo in combos]
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
            
            hand = self.generate_hand(10)
            combos = self.combo_generator.generate_realistic_combos(10)
            player_count = random.choice([2, 3, 4])
            
            # Simulate user decision
            would_declare, unbeatable_prob, user_threshold = self.game_simulator.simulate_user_behavior(
                combos, user_profile, player_count
            )
            
            # Generate user patterns
            strengths = [self.combo_generator.calculate_combo_strength(combo) for combo in combos]
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
    
    def generate_all_phases(self, validation_samples: int = 1000, 
                           pattern_samples: int = 2000, 
                           threshold_samples: int = 1500):
        """Generate all phases of Báo Sâm training data"""
        logger.info("Generating all Báo Sâm training data phases...")
        
        # Phase 1: Validation data
        validation_data = self.generate_validation_data(validation_samples)
        self.save_data(validation_data, 'data/bao_sam_validation_training_data.jsonl')
        
        # Phase 2: Pattern data
        pattern_data = self.generate_pattern_data(pattern_samples)
        self.save_data(pattern_data, 'data/bao_sam_pattern_training_data.jsonl')
        
        # Phase 3: Threshold data
        threshold_data = self.generate_threshold_data(threshold_samples)
        self.save_data(threshold_data, 'data/bao_sam_threshold_training_data.jsonl')
        
        logger.info("All Báo Sâm synthetic data generated successfully!")
        
        return {
            'validation': validation_data,
            'pattern': pattern_data,
            'threshold': threshold_data
        }
