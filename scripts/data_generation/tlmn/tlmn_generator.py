#!/usr/bin/env python3
"""
TLMN Data Generator - Specialized for TLMN training data
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

class TLMNGameSimulator(GameSimulator):
    """TLMN-specific game simulator"""
    
    def __init__(self):
        super().__init__('tlmn')
    
    def find_legal_moves(self, hand: List[int], last_move: Dict) -> List[Dict[str, Any]]:
        """Find legal moves for TLMN gameplay"""
        combo_gen = ComboGenerator('tlmn')
        all_combos = combo_gen.find_combos_in_hand(hand)
        
        # Filter based on last_move (if any)
        if last_move:
            legal_moves = []
            for combo in all_combos:
                if self._is_legal_move_tlmn(combo, last_move):
                    legal_moves.append(combo)
            return legal_moves if legal_moves else [{'type': 'pass', 'cards': [], 'combo_type': 'pass', 'rank_value': -1}]
        
        return all_combos
    
    def _is_legal_move_tlmn(self, move: Dict, last_move: Dict) -> bool:
        """Check if move is legal in TLMN"""
        if move.get('type') == 'pass':
            return True
        
        move_combo_type = move.get('combo_type', '')
        move_rank = move.get('rank_value', 0)
        last_combo_type = last_move.get('combo_type', '')
        last_rank = last_move.get('rank_value', 0)
        
        # TLMN rules: must play same combo type with higher rank
        if move_combo_type == last_combo_type:
            return self._is_higher_rank_tlmn(move_rank, last_rank)
        
        return False
    
    def _is_higher_rank_tlmn(self, rank1: int, rank2: int) -> bool:
        """Check if rank1 is higher than rank2 in TLMN (A-2-3-...-K)"""
        # TLMN ranking: A(0) < 2(1) < 3(2) < ... < K(12)
        return rank1 > rank2
    
    def choose_move(self, legal_moves: List[Dict], hand: List[int], 
                   last_move: Dict, game_progress: float) -> Dict[str, Any]:
        """Choose a move using TLMN strategy"""
        
        # Filter out bad moves
        good_moves = []
        for move in legal_moves:
            if self._is_good_move_tlmn(move, hand, game_progress):
                good_moves.append(move)
        
        if not good_moves:
            # Fallback to pass or random move
            pass_moves = [m for m in legal_moves if m.get("type") == "pass"]
            if pass_moves:
                return pass_moves[0]
            else:
                return random.choice(legal_moves)
        
        # Score moves using TLMN strategy
        scored_moves = []
        for move in good_moves:
            score = self._score_move_tlmn(move, hand, last_move, game_progress)
            scored_moves.append((move, score))
        
        # Sort by score and pick (with some randomness)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        if len(scored_moves) > 1 and random.random() < 0.25:
            # 25% chance to pick 2nd best
            return scored_moves[1][0]
        else:
            return scored_moves[0][0]
    
    def _is_good_move_tlmn(self, move: Dict[str, Any], hand: List[int], game_progress: float) -> bool:
        """Check if move is good in TLMN"""
        combo_type = move.get("combo_type", "")
        rank_value = move.get("rank_value", 0)
        
        # In TLMN, avoid playing high cards too early
        if game_progress < 0.3:  # Early game
            if rank_value >= 8:  # 8, 9, 10, J, Q, K
                # Only play high cards if you have many of them
                high_card_count = sum(1 for card in hand if card % 13 >= 8)
                if high_card_count < 4:
                    return False
        
        return True
    
    def _score_move_tlmn(self, move: Dict, hand: List[int], last_move: Dict, game_progress: float) -> float:
        """Score move using TLMN strategy"""
        score = 1.0
        
        combo_type = move.get("combo_type", "")
        rank_value = move.get("rank_value", 0)
        
        # TLMN rank preference (different from Sam)
        rank_multiplier = 1.0
        if game_progress < 0.5:
            # In TLMN, lower ranks are often better to play first
            if rank_value <= 3:  # A, 2, 3, 4
                rank_multiplier = 1.2
            elif rank_value <= 7:  # 5, 6, 7, 8
                rank_multiplier = 1.0
            elif rank_value >= 9:  # 10, J, Q, K
                rank_multiplier = 0.8
        
        score *= rank_multiplier
        
        # TLMN combo type preference (different from Sam)
        if combo_type == "four_kind":
            score *= 6.0  # Very strong in TLMN
        elif combo_type == "straight":
            score *= 3.0
        elif combo_type == "triple":
            score *= 2.5
        elif combo_type == "pair":
            score *= 2.0
        elif combo_type == "single":
            score *= 1.0  # Singles are more common in TLMN
        
        # TLMN-specific strategy: prefer playing lower cards first
        if rank_value <= 5 and game_progress < 0.6:
            score *= 1.5
        
        # Avoid playing too many high cards early
        if rank_value >= 8 and game_progress < 0.4:
            score *= 0.6
        
        return score

class TLMNGenerator(BaseDataGenerator):
    """Generator cho TLMN training data"""
    
    def __init__(self):
        super().__init__('tlmn')
        self.game_simulator = TLMNGameSimulator()
        logger.info("TLMNGenerator initialized")
    
    def generate_training_data(self, num_sessions: int) -> List[Dict[str, Any]]:
        """Generate TLMN specific training data"""
        logger.info(f"Generating {num_sessions} TLMN training sessions...")
        
        all_records = []
        
        for i in range(num_sessions):
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_sessions} TLMN sessions...")
            
            try:
                session_records = self.generate_tlmn_session(4)
                all_records.extend(session_records)
            except Exception as e:
                logger.warning(f"Error generating TLMN session {i + 1}: {e}")
                continue
        
        logger.info(f"Generated {len(all_records)} total TLMN records")
        return all_records
    
    def generate_tlmn_session(self, num_players: int = 4) -> List[Dict[str, Any]]:
        """Generate a complete TLMN game session"""
        
        # Deal cards - TLMN uses 13 cards per player
        hands = self.deal_cards(num_players, cards_per_player=13)
        
        records = []
        current_player = 0
        last_move = None
        turn_count = 0
        max_turns = 52  # More turns for TLMN
        
        # Simulate game progress
        while turn_count < max_turns and all(len(hand) > 0 for hand in hands):
            
            # Get current player's hand
            hand = hands[current_player]
            
            # Calculate game progress
            total_cards = 52
            played_cards = 52 - sum(len(h) for h in hands)
            game_progress = played_cards / total_cards
            
            # Find legal moves
            legal_moves = self.game_simulator.find_legal_moves(hand, last_move)
            
            # Choose move
            chosen_move = self.game_simulator.choose_move(legal_moves, hand, last_move, game_progress)
            
            # Create record
            record = self.create_base_record(
                current_player, hand, last_move, legal_moves, chosen_move, 
                hands, game_progress, turn_count
            )
            
            # Add TLMN-specific features
            record.update(self.get_game_specific_features(hand, legal_moves, chosen_move, game_progress))
            
            records.append(record)
            
            # Execute move
            if chosen_move.get("type") == "play_cards":
                move_cards = chosen_move.get("cards", [])
                # Remove cards from hand
                for card in move_cards:
                    if card in hand:
                        hand.remove(card)
                last_move = chosen_move
            else:
                # Pass - don't change last_move
                pass
            
            # Next player
            current_player = (current_player + 1) % num_players
            turn_count += 1
            
        return records
    
    def get_game_specific_features(self, hand: List[int], legal_moves: List[Dict], 
                                 chosen_move: Dict, game_progress: float) -> Dict[str, Any]:
        """Get TLMN-specific features"""
        
        # Analyze hand composition
        hand_ranks = [card % 13 for card in hand]
        rank_counts = {}
        for rank in hand_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Count high cards (8, 9, 10, J, Q, K)
        high_cards = sum(1 for rank in hand_ranks if rank >= 8)
        
        # Count low cards (A, 2, 3, 4, 5)
        low_cards = sum(1 for rank in hand_ranks if rank <= 4)
        
        # Calculate hand balance
        hand_balance = abs(high_cards - low_cards) / len(hand) if hand else 0
        
        # Find potential combos
        potential_combos = self.combo_generator.find_combos_in_hand(hand)
        
        # Count combo types
        combo_type_counts = {}
        for combo in potential_combos:
            combo_type = combo.get('combo_type', 'unknown')
            combo_type_counts[combo_type] = combo_type_counts.get(combo_type, 0) + 1
        
        return {
            'tlmn_features': {
                'hand_size': len(hand),
                'high_cards_count': high_cards,
                'low_cards_count': low_cards,
                'hand_balance': hand_balance,
                'potential_combos_count': len(potential_combos),
                'combo_type_distribution': combo_type_counts,
                'has_four_kind': 'four_kind' in combo_type_counts,
                'has_straight': 'straight' in combo_type_counts,
                'has_triple': 'triple' in combo_type_counts,
                'has_pair': 'pair' in combo_type_counts,
                'singles_count': combo_type_counts.get('single', 0)
            }
        }
    
    def generate_tlmn_pattern_data(self, num_samples: int = 1500) -> List[Dict[str, Any]]:
        """Generate TLMN pattern analysis data"""
        logger.info(f"Generating {num_samples} TLMN pattern samples...")
        
        data = []
        
        for i in range(num_samples):
            # Generate hand with 13 cards
            hand = self.generate_hand(13)
            
            # Find all possible combos
            combos = self.combo_generator.find_combos_in_hand(hand)
            
            # Analyze patterns
            combo_types = [combo.get('combo_type', 'unknown') for combo in combos]
            combo_type_counts = {}
            for combo_type in combo_types:
                combo_type_counts[combo_type] = combo_type_counts.get(combo_type, 0) + 1
            
            # Calculate pattern metrics
            hand_ranks = [card % 13 for card in hand]
            rank_counts = {}
            for rank in hand_ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            # Pattern analysis
            has_four_kind = 'four_kind' in combo_type_counts
            has_straight = 'straight' in combo_type_counts
            has_triple = 'triple' in combo_type_counts
            has_pair = 'pair' in combo_type_counts
            singles_count = combo_type_counts.get('single', 0)
            
            # Calculate pattern strength
            pattern_strength = 0.0
            if has_four_kind:
                pattern_strength += 0.4
            if has_straight:
                pattern_strength += 0.3
            if has_triple:
                pattern_strength += 0.2
            if has_pair:
                pattern_strength += 0.1
            pattern_strength += min(0.1, singles_count * 0.01)
            
            record = {
                'hand': hand,
                'possible_combos': combos,
                'combo_type_distribution': combo_type_counts,
                'pattern_analysis': {
                    'has_four_kind': has_four_kind,
                    'has_straight': has_straight,
                    'has_triple': has_triple,
                    'has_pair': has_pair,
                    'singles_count': singles_count,
                    'pattern_strength': pattern_strength,
                    'hand_balance': abs(sum(1 for r in hand_ranks if r >= 8) - sum(1 for r in hand_ranks if r <= 4)) / len(hand)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            data.append(record)
        
        logger.info(f"Generated {len(data)} TLMN pattern samples")
        return data
