#!/usr/bin/env python3
"""
Base Data Generator - Common functionality for all game types
"""

import json
import random
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import os
import sys

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_build_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
project_root = os.path.dirname(model_build_dir)

if model_build_dir not in sys.path:
    sys.path.insert(0, model_build_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .combo_generator import ComboGenerator
from .game_simulator import GameSimulator

logger = logging.getLogger(__name__)

class BaseDataGenerator:
    """Base class for all data generators"""
    
    def __init__(self, game_type: str):
        self.game_type = game_type
        self.combo_generator = ComboGenerator(game_type)
        self.game_simulator = GameSimulator(game_type)
        logger.info(f"BaseDataGenerator initialized for {game_type}")
    
    def generate_card_deck(self) -> List[int]:
        """Generate a deck of cards (0-51)"""
        return list(range(52))
    
    def deal_cards(self, num_players: int = 4, cards_per_player: int = None) -> List[List[int]]:
        """Deal cards to players - override in subclasses"""
        if cards_per_player is None:
            cards_per_player = self.get_default_cards_per_player()
        
        deck = self.generate_card_deck()
        random.shuffle(deck)
        
        hands = []
        for i in range(num_players):
            start_idx = i * cards_per_player
            end_idx = start_idx + cards_per_player
            hands.append(deck[start_idx:end_idx])
            
        return hands
    
    def get_default_cards_per_player(self) -> int:
        """Get default cards per player for this game type"""
        defaults = {
            'bao_sam': 10,
            'sam_general': 10,
            'tlmn': 13
        }
        return defaults.get(self.game_type, 10)
    
    def get_card_rank(self, card_id: int) -> int:
        """Get rank of card (0-12: A, 2, 3, ..., K)"""
        return card_id % 13
    
    def get_card_suit(self, card_id: int) -> int:
        """Get suit of card (0-3: Spades, Hearts, Diamonds, Clubs)"""
        return card_id // 13
    
    def find_combos_in_hand(self, hand: List[int]) -> List[Dict[str, Any]]:
        """Find all possible combos in a hand using ComboGenerator"""
        return self.combo_generator.find_combos_in_hand(hand)
    
    def generate_training_data(self, num_sessions: int) -> List[Dict[str, Any]]:
        """Generate training data - override in subclasses"""
        raise NotImplementedError("Subclasses must implement generate_training_data")
    
    def save_data(self, records: List[Dict[str, Any]], filename: str):
        """Save data to JSONL file"""
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
        logger.info(f"Saved {len(records)} records to {filename}")
    
    def get_game_specific_features(self, hand: List[int], legal_moves: List[Dict], 
                                 chosen_move: Dict, game_progress: float) -> Dict[str, Any]:
        """Get game-specific features - override in subclasses"""
        return {}
    
    def create_base_record(self, player_id: int, hand: List[int], last_move: Dict,
                          legal_moves: List[Dict], chosen_move: Dict, all_hands: List[List[int]],
                          game_progress: float, turn_count: int) -> Dict[str, Any]:
        """Create base record structure"""
        
        # Calculate cards left per player
        cards_left = [len(hand) for hand in all_hands]
        
        # Get game-specific features
        game_features = self.get_game_specific_features(hand, legal_moves, chosen_move, game_progress)
        
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
            "game_progress": game_progress,
            # Game-specific features
            **game_features
        }
        
        return record
