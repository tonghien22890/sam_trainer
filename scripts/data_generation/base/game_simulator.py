#!/usr/bin/env python3
"""
Game Simulator - Common game simulation logic for all game types
"""

import random
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class GameSimulator:
    """Logic simulate game chung cho tất cả game types"""
    
    def __init__(self, game_type: str):
        self.game_type = game_type
        logger.info(f"GameSimulator initialized for {game_type}")
    
    def simulate_game_session(self, hands: List[List[int]], max_turns: int = 40) -> List[Dict[str, Any]]:
        """Simulate a complete game session"""
        records = []
        current_player = 0
        last_move = None
        turn_count = 0
        
        # Simulate game progress
        while turn_count < max_turns and all(len(hand) > 0 for hand in hands):
            
            # Get current player's hand
            hand = hands[current_player]
            
            # Calculate game progress
            total_cards = sum(len(h) for h in hands)
            game_progress = 1.0 - (total_cards / (len(hands) * self.get_default_cards_per_player()))
            
            # Find legal moves (override in subclasses)
            legal_moves = self.find_legal_moves(hand, last_move)
            
            # Choose move (override in subclasses)
            chosen_move = self.choose_move(legal_moves, hand, last_move, game_progress)
            
            # Create record
            record = self.create_record(
                current_player, hand, last_move, legal_moves, chosen_move, 
                hands, game_progress, turn_count
            )
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
            current_player = (current_player + 1) % len(hands)
            turn_count += 1
            
        return records
    
    def get_default_cards_per_player(self) -> int:
        """Get default cards per player for this game type"""
        defaults = {
            'bao_sam': 10,
            'sam_general': 10,
            'tlmn': 13
        }
        return defaults.get(self.game_type, 10)
    
    def find_legal_moves(self, hand: List[int], last_move: Dict) -> List[Dict[str, Any]]:
        """Find legal moves for current hand - override in subclasses"""
        # Default implementation - return all possible combos
        from .combo_generator import ComboGenerator
        combo_gen = ComboGenerator(self.game_type)
        return combo_gen.find_combos_in_hand(hand)
    
    def choose_move(self, legal_moves: List[Dict], hand: List[int], 
                   last_move: Dict, game_progress: float) -> Dict[str, Any]:
        """Choose a move from legal moves - override in subclasses"""
        # Default implementation - random choice
        return random.choice(legal_moves)
    
    def create_record(self, player_id: int, hand: List[int], last_move: Dict,
                     legal_moves: List[Dict], chosen_move: Dict, all_hands: List[List[int]],
                     game_progress: float, turn_count: int) -> Dict[str, Any]:
        """Create game record - override in subclasses"""
        
        # Calculate cards left per player
        cards_left = [len(hand) for hand in all_hands]
        
        # Create base record
        record = {
            "game_id": f"{self.game_type}_game_{random.randint(1000, 9999)}",
            "game_type": self.game_type.title(),
            "players_count": len(all_hands),
            "round_id": 0,
            "turn_id": turn_count,
            "current_player_id": player_id,
            "player_id": player_id,
            "hand": hand,
            "last_move": last_move,
            "players_left": [i for i in range(len(all_hands)) if cards_left[i] > 0],
            "cards_left": cards_left,
            "is_finished": False,
            "winner_id": None,
            "meta": {
                "agentType": f"synthetic_{self.game_type}",
                "legal_moves": legal_moves
            },
            "action": {
                "stage1": {"type": "pass", "value": "pass"},
                "stage2": chosen_move
            },
            "hand_count": len(hand),
            "timestamp": datetime.now().isoformat(),
            "synthetic": True,
            "game_progress": game_progress
        }
        
        return record
    
    def simulate_user_behavior(self, combos: List[Dict[str, Any]], user_profile: str, 
                              player_count: int) -> Tuple[bool, float, float]:
        """Simulate user behavior based on profile - override in subclasses"""
        # Default implementation
        return False, 0.5, 0.75
    
    def calculate_game_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate game metrics from records"""
        if not records:
            return {}
        
        total_moves = len(records)
        combo_counts = {}
        game_duration = records[-1].get('turn_id', 0) if records else 0
        
        for record in records:
            action = record.get('action', {})
            stage2 = action.get('stage2', {})
            combo_type = stage2.get('combo_type', 'unknown')
            combo_counts[combo_type] = combo_counts.get(combo_type, 0) + 1
        
        return {
            'total_moves': total_moves,
            'game_duration': game_duration,
            'combo_distribution': combo_counts,
            'avg_moves_per_player': total_moves / len(set(r.get('player_id', 0) for r in records))
        }
