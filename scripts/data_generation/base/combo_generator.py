#!/usr/bin/env python3
"""
Combo Generator - Common combo generation logic for all game types
"""

import random
import logging
from typing import Dict, List, Any
from ai_common.core.combo_analyzer import ComboAnalyzer

logger = logging.getLogger(__name__)

class ComboGenerator:
    """Logic tạo combo chung cho tất cả game types"""
    
    def __init__(self, game_type: str):
        self.game_type = game_type
        self.combo_analyzer = ComboAnalyzer()
        logger.info(f"ComboGenerator initialized for {game_type}")
    
    def find_combos_in_hand(self, hand: List[int]) -> List[Dict[str, Any]]:
        """Find all possible combos in a hand"""
        combos = []
        
        # Group cards by rank
        rank_groups = {}
        for card in hand:
            rank = card % 13
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)
        
        # Find singles, pairs, triples, four_kinds
        for rank, cards in rank_groups.items():
            if len(cards) >= 1:
                combos.append({
                    "type": "play_cards",
                    "cards": [cards[0]],
                    "combo_type": "single",
                    "rank_value": rank
                })
            if len(cards) >= 2:
                combos.append({
                    "type": "play_cards", 
                    "cards": cards[:2],
                    "combo_type": "pair",
                    "rank_value": rank
                })
            if len(cards) >= 3:
                combos.append({
                    "type": "play_cards",
                    "cards": cards[:3], 
                    "combo_type": "triple",
                    "rank_value": rank
                })
            if len(cards) >= 4:
                combos.append({
                    "type": "play_cards",
                    "cards": cards[:4],
                    "combo_type": "four_kind", 
                    "rank_value": rank
                })
        
        # Find straights (game-specific logic)
        if self.game_type in ['bao_sam', 'sam_general']:
            combos.extend(self._find_straights_sam(rank_groups))
        elif self.game_type == 'tlmn':
            combos.extend(self._find_straights_tlmn(rank_groups))
        
        # Always add pass option
        combos.append({
            "type": "pass",
            "cards": [],
            "combo_type": "pass",
            "rank_value": -1
        })
        
        return combos
    
    def _find_straights_sam(self, rank_groups: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """Find straights for Sam games (3-A, no 2)"""
        combos = []
        sorted_ranks = sorted(rank_groups.keys())
        
        # Remove rank 12 (2) for Sam games
        sam_ranks = [r for r in sorted_ranks if r != 12]
        
        for i in range(len(sam_ranks) - 4):
            if sam_ranks[i+4] - sam_ranks[i] == 4:  # 5 consecutive ranks
                straight_cards = []
                for rank in sam_ranks[i:i+5]:
                    straight_cards.append(rank_groups[rank][0])
                combos.append({
                    "type": "play_cards",
                    "cards": straight_cards,
                    "combo_type": "straight", 
                    "rank_value": sam_ranks[i]
                })
        
        return combos
    
    def _find_straights_tlmn(self, rank_groups: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """Find straights for TLMN (A-2-3-...-K, including 2)"""
        combos = []
        sorted_ranks = sorted(rank_groups.keys())
        
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i+4] - sorted_ranks[i] == 4:  # 5 consecutive ranks
                straight_cards = []
                for rank in sorted_ranks[i:i+5]:
                    straight_cards.append(rank_groups[rank][0])
                combos.append({
                    "type": "play_cards",
                    "cards": straight_cards,
                    "combo_type": "straight", 
                    "rank_value": sorted_ranks[i]
                })
        
        return combos
    
    def calculate_combo_strength(self, combo: Dict[str, Any]) -> float:
        """Calculate combo strength using ComboAnalyzer"""
        return self.combo_analyzer.calculate_combo_strength(combo)
    
    def generate_realistic_combos(self, target_cards: int) -> List[Dict[str, Any]]:
        """Generate realistic combo combinations based on game type"""
        if self.game_type == 'bao_sam':
            return self._generate_bao_sam_combos(target_cards)
        elif self.game_type == 'sam_general':
            return self._generate_sam_general_combos(target_cards)
        elif self.game_type == 'tlmn':
            return self._generate_tlmn_combos(target_cards)
        else:
            return self._generate_default_combos(target_cards)
    
    def _generate_bao_sam_combos(self, target_cards: int) -> List[Dict[str, Any]]:
        """Generate combos for Báo Sâm (10 cards, strong combinations)"""
        combos = []
        cards_used = 0
        
        # Strategy: Generate one strong combo, then fill with weaker ones
        strong_combo_type = random.choice(['four_kind', 'triple', 'straight'])
        
        if strong_combo_type == 'four_kind':
            rank = random.randint(8, 12)  # High rank quad
            combo = self._generate_combo('four_kind', rank)
            combos.append(combo)
            cards_used += 4
            
            # Fill remaining 6 cards
            remaining = target_cards - cards_used
            if remaining >= 3:
                # Add a triple
                rank2 = random.randint(0, 7)
                combo2 = self._generate_combo('triple', rank2)
                combos.append(combo2)
                cards_used += 3
                remaining -= 3
            
            # Fill with singles/pairs
            while cards_used < target_cards:
                if target_cards - cards_used >= 2 and random.random() < 0.5:
                    rank3 = random.randint(0, 12)
                    combo3 = self._generate_combo('pair', rank3)
                    combos.append(combo3)
                    cards_used += 2
                else:
                    rank3 = random.randint(0, 12)
                    combo3 = self._generate_combo('single', rank3)
                    combos.append(combo3)
                    cards_used += 1
        
        elif strong_combo_type == 'triple':
            # Two triples + filler
            rank1 = random.randint(8, 12)  # High rank
            combo1 = self._generate_combo('triple', rank1)
            combos.append(combo1)
            cards_used += 3
            
            rank2 = random.randint(0, 7)  # Lower rank
            combo2 = self._generate_combo('triple', rank2)
            combos.append(combo2)
            cards_used += 3
            
            # Fill remaining 4 cards
            while cards_used < target_cards:
                if target_cards - cards_used >= 2 and random.random() < 0.6:
                    rank3 = random.randint(0, 12)
                    combo3 = self._generate_combo('pair', rank3)
                    combos.append(combo3)
                    cards_used += 2
                else:
                    rank3 = random.randint(0, 12)
                    combo3 = self._generate_combo('single', rank3)
                    combos.append(combo3)
                    cards_used += 1
        
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
                    combo2 = self._generate_combo('triple', rank2)
                    combos.append(combo2)
                    cards_used += 3
                    remaining -= 3
                elif remaining >= 2 and random.random() < 0.6:
                    rank2 = random.randint(0, 12)
                    combo2 = self._generate_combo('pair', rank2)
                    combos.append(combo2)
                    cards_used += 2
                    remaining -= 2
                else:
                    rank2 = random.randint(0, 12)
                    combo2 = self._generate_combo('single', rank2)
                    combos.append(combo2)
                    cards_used += 1
                    remaining -= 1
        
        return combos
    
    def _generate_sam_general_combos(self, target_cards: int) -> List[Dict[str, Any]]:
        """Generate combos for Sam General (variable cards, balanced)"""
        combos = []
        cards_used = 0
        
        # More balanced approach for general gameplay
        while cards_used < target_cards:
            remaining = target_cards - cards_used
            
            if remaining >= 4 and random.random() < 0.1:  # 10% chance for quad
                rank = random.randint(0, 12)
                combo = self._generate_combo('four_kind', rank)
                combos.append(combo)
                cards_used += 4
            elif remaining >= 3 and random.random() < 0.3:  # 30% chance for triple
                rank = random.randint(0, 12)
                combo = self._generate_combo('triple', rank)
                combos.append(combo)
                cards_used += 3
            elif remaining >= 2 and random.random() < 0.6:  # 60% chance for pair
                rank = random.randint(0, 12)
                combo = self._generate_combo('pair', rank)
                combos.append(combo)
                cards_used += 2
            else:  # Single
                rank = random.randint(0, 12)
                combo = self._generate_combo('single', rank)
                combos.append(combo)
                cards_used += 1
        
        return combos
    
    def _generate_tlmn_combos(self, target_cards: int) -> List[Dict[str, Any]]:
        """Generate combos for TLMN (13 cards, TLMN-specific patterns)"""
        combos = []
        cards_used = 0
        
        # TLMN-specific strategy (more singles, fewer large combos)
        while cards_used < target_cards:
            remaining = target_cards - cards_used
            
            if remaining >= 4 and random.random() < 0.05:  # 5% chance for quad
                rank = random.randint(0, 12)
                combo = self._generate_combo('four_kind', rank)
                combos.append(combo)
                cards_used += 4
            elif remaining >= 3 and random.random() < 0.15:  # 15% chance for triple
                rank = random.randint(0, 12)
                combo = self._generate_combo('triple', rank)
                combos.append(combo)
                cards_used += 3
            elif remaining >= 2 and random.random() < 0.4:  # 40% chance for pair
                rank = random.randint(0, 12)
                combo = self._generate_combo('pair', rank)
                combos.append(combo)
                cards_used += 2
            else:  # Single (most common in TLMN)
                rank = random.randint(0, 12)
                combo = self._generate_combo('single', rank)
                combos.append(combo)
                cards_used += 1
        
        return combos
    
    def _generate_default_combos(self, target_cards: int) -> List[Dict[str, Any]]:
        """Generate default combos for unknown game types"""
        return self._generate_sam_general_combos(target_cards)
    
    def _generate_combo(self, combo_type: str, rank: int) -> Dict[str, Any]:
        """Generate a combo of specified type and rank"""
        cards = []
        
        if combo_type == 'single':
            cards = [rank]
        elif combo_type == 'pair':
            cards = [rank, rank + 13]
        elif combo_type == 'triple':
            cards = [rank, rank + 13, rank + 26]
        elif combo_type == 'four_kind':
            cards = [rank, rank + 13, rank + 26, rank + 39]
        
        return {
            'combo_type': combo_type,
            'rank_value': rank,
            'cards': cards
        }
