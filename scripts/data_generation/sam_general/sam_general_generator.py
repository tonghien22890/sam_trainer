#!/usr/bin/env python3
"""
Sam General Data Generator - Specialized for Sam general gameplay training data
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

# Import FrameworkGenerator for sequence context
try:
    from scripts.two_layer.framework_generator import FrameworkGenerator
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

logger = logging.getLogger(__name__)

class SamGeneralGameSimulator(GameSimulator):
    """Sam General-specific game simulator"""
    
    def __init__(self):
        super().__init__('sam_general')
    
    def find_legal_moves(self, hand: List[int], last_move: Dict) -> List[Dict[str, Any]]:
        """Find legal moves for Sam general gameplay"""
        combo_gen = ComboGenerator('sam_general')
        all_combos = combo_gen.find_combos_in_hand(hand)
        
        # Filter based on last_move (if any)
        if last_move:
            legal_moves = []
            for combo in all_combos:
                if self._is_legal_move(combo, last_move):
                    legal_moves.append(combo)
            return legal_moves if legal_moves else [{'type': 'pass', 'cards': [], 'combo_type': 'pass', 'rank_value': -1}]
        
        return all_combos
    
    def _is_legal_move(self, move: Dict, last_move: Dict) -> bool:
        """Check if move is legal based on last move"""
        if move.get('type') == 'pass':
            return True
        
        move_combo_type = move.get('combo_type', '')
        move_rank = move.get('rank_value', 0)
        last_combo_type = last_move.get('combo_type', '')
        last_rank = last_move.get('rank_value', 0)
        
        # Same combo type, higher rank
        if move_combo_type == last_combo_type and move_rank > last_rank:
            return True
        
        # Different combo type, check hierarchy
        combo_hierarchy = {'single': 1, 'pair': 2, 'triple': 3, 'straight': 4, 'four_kind': 5}
        if move_combo_type in combo_hierarchy and last_combo_type in combo_hierarchy:
            if combo_hierarchy[move_combo_type] > combo_hierarchy[last_combo_type]:
                return True
        
        return False
    
    def choose_move(self, legal_moves: List[Dict], hand: List[int], 
                   last_move: Dict, game_progress: float) -> Dict[str, Any]:
        """Choose a smart move using Sam general strategy"""
        
        # Filter out bad moves
        good_moves = []
        for move in legal_moves:
            if self._is_good_move(move, hand, game_progress):
                good_moves.append(move)
        
        if not good_moves:
            # Fallback to pass or random move
            pass_moves = [m for m in legal_moves if m.get("type") == "pass"]
            if pass_moves:
                return pass_moves[0]
            else:
                return random.choice(legal_moves)
        
        # Score moves using Sam general strategy
        scored_moves = []
        for move in good_moves:
            score = self._score_move(move, hand, last_move, game_progress)
            scored_moves.append((move, score))
        
        # Sort by score and pick (with some randomness)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        if len(scored_moves) > 1 and random.random() < 0.3:
            # 30% chance to pick 2nd best
            return scored_moves[1][0]
        else:
            return scored_moves[0][0]
    
    def _is_good_move(self, move: Dict[str, Any], hand: List[int], game_progress: float) -> bool:
        """Check if move is good (not foolish)"""
        combo_type = move.get("combo_type", "")
        rank_value = move.get("rank_value", 0)
        
        # Avoid playing weak cards (3,4,5,6) in strong combos early
        if game_progress < 0.3:  # Early game
            if rank_value <= 3:  # 3, 4, 5, 6 (weakest ranks)
                # Only play weak cards as singles, not in strong combos
                if combo_type in ["triple", "four_kind"]:
                    return False
        
        return True
    
    def _score_move(self, move: Dict, hand: List[int], last_move: Dict, game_progress: float) -> float:
        """Score move using Sam general strategy"""
        score = 1.0
        
        combo_type = move.get("combo_type", "")
        rank_value = move.get("rank_value", 0)
        
        # Rank preference (following Sam rules)
        rank_multiplier = 1.0
        if game_progress < 0.5:
            if rank_value == 12:  # 2 - strongest
                rank_multiplier = 1.5
            elif rank_value >= 8 and rank_value <= 10:  # J, Q, K
                rank_multiplier = 1.3
            elif rank_value == 11:  # A
                rank_multiplier = 1.1
            elif rank_value <= 3:  # 3, 4, 5, 6 - weakest
                rank_multiplier = 0.7
        
        score *= rank_multiplier
        
        # Combo type preference
        if combo_type == "four_kind":
            score *= 5.0
        elif combo_type == "straight":
            score *= 4.0
        elif combo_type == "triple":
            score *= 3.5
        elif combo_type == "pair":
            score *= 3.0
        elif combo_type == "single":
            score *= 0.2
        
        # Special logic for 2 cards
        if rank_value == 12:  # 2 cards
            if self._should_use_2_card(move, hand, game_progress, last_move):
                score *= 3.0
            else:
                score *= 0.05
        
        # Avoid weak cards unless late game
        if rank_value <= 3 and game_progress < 0.7:
            score *= 0.2
        
        return score
    
    def _should_use_2_card(self, move: Dict, hand: List[int], game_progress: float, last_move: Dict) -> bool:
        """Determine if 2 card should be used based on Sam strategy"""
        rank_value = move.get('rank_value', 0)
        combo_type = move.get('combo_type', '')
        
        if rank_value != 12:  # Not a 2 card
            return True
        
        # 2 cards should be used when:
        # 1. Very late game
        if game_progress > 0.9:
            return True
        
        # 2. Can break opponent's strong combos
        if last_move:
            last_combo_type = last_move.get('combo_type', '')
            last_rank = last_move.get('rank_value', 0)
            
            if combo_type == 'single' and last_rank < 12:
                return True
            elif combo_type == 'pair' and last_combo_type == 'pair' and last_rank < 12:
                return True
            elif combo_type == 'triple' and last_combo_type == 'triple' and last_rank < 12:
                return True
        
        # 3. Hand is getting very small
        if len(hand) <= 2:
            return True
        
        # Otherwise, keep 2 cards for later
        return False

class SamGeneralGenerator(BaseDataGenerator):
    """Generator cho Sam General training data"""
    
    def __init__(self):
        super().__init__('sam_general')
        self.game_simulator = SamGeneralGameSimulator()
        
        # Initialize framework generator if available
        if FRAMEWORK_AVAILABLE:
            self.framework_generator = FrameworkGenerator()
        else:
            self.framework_generator = None
            logger.warning("FrameworkGenerator not available - using fallback")
        
        logger.info("SamGeneralGenerator initialized")
    
    def generate_training_data(self, num_sessions: int) -> List[Dict[str, Any]]:
        """Generate Sam General specific training data"""
        logger.info(f"Generating {num_sessions} Sam General training sessions...")
        
        all_records = []
        
        for i in range(num_sessions):
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_sessions} Sam General sessions...")
            
            try:
                session_records = self.generate_sam_general_session(4)
                all_records.extend(session_records)
            except Exception as e:
                logger.warning(f"Error generating Sam General session {i + 1}: {e}")
                continue
        
        logger.info(f"Generated {len(all_records)} total Sam General records")
        return all_records
    
    def generate_sam_general_session(self, num_players: int = 4) -> List[Dict[str, Any]]:
        """Generate a complete Sam General game session"""
        
        # Deal cards with variable hand sizes
        hands = self.deal_cards(num_players, cards_per_player=random.randint(5, 12))
        
        # Initialize sequence contexts for each player
        sequence_contexts = {}
        for i in range(num_players):
            sequence_contexts[i] = {
                'played_moves': [],
                'remaining_combos': [],
                'current_position': 0,
                'sequence_progress': 0.0,
                'framework': {}
            }
        
        records = []
        current_player = 0
        last_move = None
        turn_count = 0
        max_turns = 40
        
        # Simulate game progress
        while turn_count < max_turns and all(len(hand) > 0 for hand in hands):
            
            # Get current player's hand
            hand = hands[current_player]
            
            # Generate framework for current hand (if not already generated)
            if not sequence_contexts[current_player]['framework']:
                framework = self._generate_framework_for_hand(hand)
                sequence_contexts[current_player]['framework'] = framework
                sequence_contexts[current_player]['remaining_combos'] = framework.get('core_combos', [])
            
            # Calculate game progress
            total_cards = 52
            played_cards = 52 - sum(len(h) for h in hands)
            game_progress = played_cards / total_cards
            
            # Find legal moves
            legal_moves = self.game_simulator.find_legal_moves(hand, last_move)
            
            # Choose move
            chosen_move = self.game_simulator.choose_move(legal_moves, hand, last_move, game_progress)
            
            # Create record with sequence context
            record = self.create_base_record(
                current_player, hand, last_move, legal_moves, chosen_move, 
                hands, game_progress, turn_count
            )
            
            # Add sequence context and framework
            record['sequence_context'] = sequence_contexts[current_player]
            record['framework'] = sequence_contexts[current_player].get('framework', {})
            
            records.append(record)
            
            # Update sequence context
            self._update_sequence_context(current_player, hand, chosen_move, sequence_contexts)
            
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
    
    def _generate_framework_for_hand(self, hand: List[int]) -> Dict[str, Any]:
        """Generate framework for hand using FrameworkGenerator"""
        if self.framework_generator is not None:
            try:
                framework = self.framework_generator.generate_framework(hand)
                return framework
            except Exception as e:
                logger.warning(f"Error generating framework for hand {hand}: {e}")
                return self._get_empty_framework()
        else:
            return self._get_empty_framework()
    
    def _get_empty_framework(self) -> Dict[str, Any]:
        """Get empty framework when FrameworkGenerator is not available"""
        return {
            'unbeatable_sequence': [],
            'framework_strength': 0.0,
            'core_combos': [],
            'protected_ranks': [],
            'protected_windows': [],
            'recommended_moves': []
        }
    
    def _update_sequence_context(self, player_id: int, hand: List[int], 
                               chosen_move: Dict[str, Any], sequence_contexts: Dict[int, Dict[str, Any]]):
        """Update sequence context after a move"""
        if player_id not in sequence_contexts:
            sequence_contexts[player_id] = {
                'played_moves': [],
                'remaining_combos': [],
                'current_position': 0,
                'sequence_progress': 0.0,
                'framework': {}
            }
        
        # Add move to played moves
        if chosen_move.get("type") == "play_cards":
            sequence_contexts[player_id]['played_moves'].append(chosen_move)
            sequence_contexts[player_id]['current_position'] += 1
            
            # Update remaining combos (simplified)
            move_cards = set(chosen_move.get("cards", []))
            remaining_combos = []
            for combo in sequence_contexts[player_id]['framework'].get('core_combos', []):
                combo_cards = set(combo.get('cards', []))
                if not move_cards.issubset(combo_cards):
                    remaining_combos.append(combo)
            sequence_contexts[player_id]['remaining_combos'] = remaining_combos
            
            # Update sequence progress
            total_combos = len(sequence_contexts[player_id]['framework'].get('core_combos', []))
            if total_combos > 0:
                sequence_contexts[player_id]['sequence_progress'] = (
                    sequence_contexts[player_id]['current_position'] / total_combos
                )
