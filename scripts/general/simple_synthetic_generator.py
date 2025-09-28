"""
Simple Synthetic Training Data Generator for OptimizedGeneralModelV3

Creates realistic game sessions without complex game engine dependencies.
Focuses on generating good training data with smart strategies.
Now includes sequence context and framework generation for Two-Layer Architecture.
"""

import json
import copy
import random
from typing import Dict, List, Any
from datetime import datetime
import os
import sys

# Add model_build to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
model_build_dir = os.path.dirname(current_dir)  # model_build/
project_root = os.path.dirname(model_build_dir)  # AI-Sam/

if model_build_dir not in sys.path:
    sys.path.insert(0, model_build_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import FrameworkGenerator for sequence context
try:
    from scripts.two_layer.framework_generator import FrameworkGenerator
    FRAMEWORK_AVAILABLE = True
except ImportError:
    print("⚠️ FrameworkGenerator not available - using fallback")
    FRAMEWORK_AVAILABLE = False

# Toggle framework usage (keep enabled by default)
FORCE_DISABLE_FRAMEWORK = False


class SimpleSyntheticGenerator:
    """Simple synthetic data generator for training data with sequence context"""
    
    # Game-specific constants
    CARDS_PER_PLAYER = {
        'sam': 10,   # Sam: 10 cards per player
        'tlmn': 13   # TLMN: 13 cards per player
    }
    
    def __init__(self, game_type: str = 'sam'):
        # Normalize game type to lowercase tokens used across pipeline
        self.game_type = (game_type or 'sam').strip().lower()
        self.combo_types = ["single", "pair", "triple", "four_kind", "straight", "double_seq", "pass"]
        
        # Initialize framework generator if available
        if not FORCE_DISABLE_FRAMEWORK:
            if FRAMEWORK_AVAILABLE:
                self.framework_generator = FrameworkGenerator()
            else:
                self.framework_generator = None
                print("⚠️ [SimpleSyntheticGenerator] FrameworkGenerator not available - using fallback")
        else:
            self.framework_generator = None
        
    def generate_card_deck(self) -> List[int]:
        """Generate a deck of cards (0-51)"""
        return list(range(52))
    
    def deal_cards(self, num_players: int = 4, cards_per_player: int = None) -> List[List[int]]:
        """Deal cards to players"""
        # Set cards per player based on game type if not specified
        if cards_per_player is None:
            cards_per_player = self.CARDS_PER_PLAYER.get(self.game_type, 10)
        
        deck = self.generate_card_deck()
        random.shuffle(deck)
        
        hands = []
        for i in range(num_players):
            start_idx = i * cards_per_player
            end_idx = start_idx + cards_per_player
            hands.append(deck[start_idx:end_idx])
            
        return hands
    
    def get_card_rank(self, card_id: int) -> int:
        """Get rank of card (0-12: A, 2, 3, ..., K)"""
        return card_id % 13
    
    def get_card_suit(self, card_id: int) -> int:
        """Get suit of card (0-3: Spades, Hearts, Diamonds, Clubs)"""
        return card_id // 13
    
    def find_combos_in_hand(self, hand: List[int]) -> List[Dict[str, Any]]:
        """Find all possible combos in a hand"""
        combos = []
        
        # Check if hand is empty - only return pass option
        if not hand or len(hand) == 0:
            combos.append({
                "type": "pass",
                "cards": [],
                "combo_type": "pass",
                "rank_value": -1
            })
            return combos
        
        # Group cards by rank
        rank_groups = {}
        for card in hand:
            rank = self.get_card_rank(card)
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
        
        # Find straights
        sorted_ranks = sorted(rank_groups.keys())
        if self.game_type == 'tlmn':
            # TLMN straight: length >= 3, exclude rank 12 (2)
            tlmn_ranks = [r for r in sorted_ranks if r != 12]
            n = len(tlmn_ranks)
            j = 0
            while j < n:
                start = j
                while j + 1 < n and tlmn_ranks[j + 1] == tlmn_ranks[j] + 1:
                    j += 1
                window = tlmn_ranks[start:j + 1]
                if len(window) >= 3:
                    for L in range(3, len(window) + 1):
                        for s in range(0, len(window) - L + 1):
                            seq = window[s:s + L]
                            straight_cards = [rank_groups[r][0] for r in seq]
                            combos.append({
                                "type": "play_cards",
                                "cards": straight_cards,
                                "combo_type": "straight",
                                "rank_value": seq[0]
                            })
                j += 1
        else:
            # Sam simplified: exactly length-5 windows
            for i in range(len(sorted_ranks) - 4):
                if sorted_ranks[i+4] - sorted_ranks[i] == 4:
                    straight_cards = [rank_groups[rank][0] for rank in sorted_ranks[i:i+5]]
                    combos.append({
                        "type": "play_cards",
                        "cards": straight_cards,
                        "combo_type": "straight",
                        "rank_value": sorted_ranks[i]
                    })

        # TLMN: Detect three/four consecutive pairs (đôi thông)
        if self.game_type == 'tlmn':
            pair_ranks = sorted([r for r, cards in rank_groups.items() if len(cards) >= 2 and r != 12])
            n = len(pair_ranks)
            i = 0
            while i < n:
                start = i
                while i + 1 < n and pair_ranks[i + 1] == pair_ranks[i] + 1:
                    i += 1
                window = pair_ranks[start:i + 1]
                # Build 3 and 4 consecutive pairs from this window
                for need in (3, 4):
                    if len(window) >= need:
                        for s in range(0, len(window) - need + 1):
                            seq = window[s:s + need]
                            cards = []
                            for r in seq:
                                cards.extend(rank_groups[r][:2])
                            combos.append({
                                "type": "play_cards",
                                "cards": cards,
                                "combo_type": "double_seq",
                                "rank_value": seq[0]
                            })
                i += 1
        
        # Always add pass option
        combos.append({
            "type": "pass",
            "cards": [],
            "combo_type": "pass",
            "rank_value": -1
        })
        
        return combos
    
    def _generate_framework_for_hand(self, hand: List[int]) -> Dict[str, Any]:
        """Generate framework for hand using FrameworkGenerator"""
        if FORCE_DISABLE_FRAMEWORK:
            # Skip framework generation entirely
            return self._get_empty_framework()
        if self.framework_generator is not None:
            try:
                framework = self.framework_generator.generate_framework(hand)
                return framework
            except Exception as e:
                print(f"⚠️ Error generating framework for hand {hand}: {e}")
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
    
    def is_good_move(self, move: Dict[str, Any], hand: List[int], game_progress: float, framework: Dict[str, Any] = None, last_move: Dict[str, Any] = None) -> bool:
        """Check if move is good (not foolish)"""
        
        combo_type = move.get("combo_type", "")
        rank_value = move.get("rank_value", 0)
        cards = move.get("cards", [])
        move_ranks = [c % 13 for c in cards]
        # Treat any combo containing a 2 card (ids 12,25,38,51 -> rank%13==12) as absolute-strong
        has_two_card = any((c % 13) == 12 for c in cards)
        is_two_combo = has_two_card and combo_type in ["single", "pair", "triple", "four_kind"]
        is_ace_high_straight = (combo_type == "straight" and 11 in move_ranks and len(cards) >= 5)

        def _in_core(c: Dict[str, Any]) -> bool:
            if not framework:
                return False
            core = framework.get('core_combos', []) or []
            mc = set(c.get('cards', []))
            for cb in core:
                if mc.issubset(set(cb.get('cards', []))):
                    return True
            return False

        def _can_block(curr_move: Dict[str, Any], prev_move: Dict[str, Any]) -> bool:
            if not prev_move or prev_move.get('type') != 'play_cards':
                return True  # opening move or no constraint
            if curr_move.get('combo_type') != prev_move.get('combo_type'):
                return False
            # Simple rule: higher rank_value beats
            return curr_move.get('rank_value', -1) > prev_move.get('rank_value', -1)
        
        # TLMN Strategy: Early game should prioritize singles/pairs, be more flexible with combos
        if game_progress < 0.4:  # Early game
            if rank_value <= 3:  # 3, 4, 5, 6 (weakest ranks)
                # Only play weak cards as singles, not in strong combos
                if combo_type in ["triple", "four_kind"]:
                    return False

            # More flexible approach: Allow combos early if they don't break strong structures
            strong_combo_types = ["triple", "four_kind", "double_seq", "straight"]
            if combo_type in strong_combo_types:
                if self._would_break_strong_combo(move, hand):
                    return False
                # More lenient: Allow combos even without strict framework planning
                # Only block if framework explicitly says no
                if framework:
                    rec = framework.get('recommended_moves', []) or []
                    move_cards = set(move.get('cards', []))
                    # Only block if framework has explicit negative recommendations
                    negative_recs = framework.get('negative_moves', []) or []
                    if any(set(r) == move_cards for r in negative_recs):
                        return False

            # Absolute-strong cases: 2-cards combos and Ace-high straights
            # More lenient: Allow if not explicitly breaking strong combos
            if is_two_combo or is_ace_high_straight:
                if self._would_break_strong_combo(move, hand):
                    return False
                # Allow if framework doesn't explicitly forbid
                if framework:
                    negative_recs = framework.get('negative_moves', []) or []
                    move_cards = set(cards)
                    if any(set(r) == move_cards for r in negative_recs):
                        return False

            # Positive allowance: Prioritize singles/pairs early
            if combo_type in ["single", "pair"]:
                if not self._would_break_strong_combo(move, hand):
                    # Strong preference for singles/pairs early
                    return True
                    
        # Allow triples and four_kinds - they are good combos!
        # Only block if it's clearly foolish (e.g., weak ranks in strong combos early)
        return True
    
    def _would_break_strong_combo(self, move: Dict[str, Any], hand: List[int]) -> bool:
        """Check if this move would break a strong combo structure"""
        if not hand or not move.get("cards"):
            return False
            
        move_cards = move.get("cards", [])
        move_combo_type = move.get("combo_type", "")
        
        # Count ranks in hand
        rank_counts = {}
        for card in hand:
            rank = card % 13
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Check if move would break strong structures
        for card in move_cards:
            rank = card % 13
            count_before = rank_counts.get(rank, 0)
            
            # Check if we have strong combos that would be broken
            if count_before >= 4:  # Quad
                if move_combo_type != "four_kind":
                    return True  # Breaking a quad
            elif count_before == 3:  # Triple
                # Allow breaking triple for four_kind, but not for single
                if move_combo_type == "single":
                    return True  # Breaking a triple for single
            elif count_before == 2:  # Pair
                # Allow breaking pair for triple/four_kind, penalty for single but not extreme
                if move_combo_type == "single":
                    # Only penalty if the single is very weak (3,4,5,6)
                    move_rank = move_cards[0] % 13 if move_cards else -1
                    if move_rank <= 3:  # 3,4,5,6 - allow breaking weak singles
                        return False  # Don't penalty breaking pair for weak singles
                    else:
                        return True  # Penalty breaking pair for stronger singles
        
        # Check for straight breaking - only break for good reasons
        if self._has_long_straight(hand) and move_combo_type == "single":
            # Only break straight for single if it's a very weak card (3, 4, 5, 6) or 2
            move_rank = move_cards[0] % 13
            if move_rank <= 3 or move_rank == 12:  # 3, 4, 5, 6, or 2 - allow breaking
                return False  # Don't break penalty for weak cards or 2
            else:
                return True  # Break penalty for stronger cards (except 2)
            
        return False
    
    def _should_use_2_card(self, move: Dict[str, Any], hand: List[int], game_progress: float, last_move: Dict = None) -> bool:
        """Determine if 2 card should be used based on Sam strategy"""
        rank_value = move.get('rank_value', 0)
        combo_type = move.get('combo_type', '')
        
        if rank_value != 12:  # Not a 2 card
            return True
        
        # 2 cards should be used when:
        # 1. Very late game (game_progress > 0.9) - more strict
        if game_progress > 0.9:
            return True
        
        # 2. Can break opponent's strong combos (if last_move exists)
        if last_move:
            last_combo_type = last_move.get('combo_type', '')
            last_rank = last_move.get('rank_value', 0)
            
            # 2 can block most combos (except higher 2s)
            if combo_type == 'single' and last_rank < 12:  # Single 2 can block most single combos
                return True
            elif combo_type == 'pair' and last_combo_type == 'pair' and last_rank < 12:  # Pair 2 can block most pairs
                return True
            elif combo_type == 'triple' and last_combo_type == 'triple' and last_rank < 12:  # Triple 2 can block most triples
                return True
        
        # 3. Hand is getting very small (few cards left) - more strict
        if len(hand) <= 2:
            return True
        
        # 4. Can break own weak combos to form stronger ones
        if self._can_break_for_upgrade(move, hand):
            return True
        
        # Otherwise, keep 2 cards for later (blocking strategy) - VERY STRICT
        return False
    
    def _can_break_for_upgrade(self, move: Dict[str, Any], hand: List[int]) -> bool:
        """Check if using 2 can break weak combos to form stronger ones"""
        move_cards = move.get('cards', [])
        move_combo_type = move.get('combo_type', '')
        
        # Count ranks in hand
        rank_counts = {}
        for card in hand:
            rank = card % 13
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Check if breaking a weak combo can form a stronger one
        for card in move_cards:
            rank = card % 13
            count_before = rank_counts.get(rank, 0)
            
            # If we have 3+ cards of same rank, using 2 to break for quad is good
            if count_before >= 3 and move_combo_type == 'single':
                return True
            # If we have 2 cards of same rank, using 2 to break for triple is good
            elif count_before == 2 and move_combo_type == 'single':
                return True
        
        return False
    
    def _has_long_straight(self, hand: List[int]) -> bool:
        """Check if hand has a long straight (3+ cards) - Sam rules: 2 cannot be in straight"""
        ranks = [card % 13 for card in hand]
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Check for consecutive ranks (0-11 only, rank 12=2 cannot be in straight)
        consecutive = 0
        max_consecutive = 0
        for rank in range(12):  # Only ranks 0-11 (3-A), exclude rank 12 (2)
            if rank_counts.get(rank, 0) > 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        # In Sam, straight 3+ cards should be protected
        return max_consecutive >= 3
    
    def choose_smart_move(self, legal_moves: List[Dict], hand: List[int], 
                         last_move: Dict, game_progress: float, framework: Dict[str, Any] = None) -> Dict[str, Any]:
        """Choose a smart move from legal moves"""
        
        # Filter out bad moves
        good_moves = []
        for move in legal_moves:
            if self.is_good_move(move, hand, game_progress, framework=framework, last_move=last_move):
                good_moves.append(move)
                
        if not good_moves:
            # Fallback to pass or random move
            pass_moves = [m for m in legal_moves if m.get("type") == "pass"]
            if pass_moves:
                return pass_moves[0]
            else:
                return random.choice(legal_moves)
        
        # Score moves (tempo- and framework-aware)
        scored_moves = []
        for move in good_moves:
            score = 1.0
            
            combo_type = move.get("combo_type", "")
            rank_value = move.get("rank_value", 0)
            
            # STRONG preference for high cards early (following Sam rules)
            # But combo type preference should be MORE important than rank preference
            rank_multiplier = 1.0
            if game_progress < 0.5:
                if rank_value == 12:  # 2 - strongest
                    rank_multiplier = 1.5  # Moderate preference (not too strong)
                elif rank_value >= 8 and rank_value <= 10:  # J, Q, K
                    rank_multiplier = 1.3  # Moderate preference
                elif rank_value == 11:  # A
                    rank_multiplier = 1.1  # Slight preference
                elif rank_value <= 3:  # 3, 4, 5, 6 - weakest
                    rank_multiplier = 0.7  # Moderate penalty
                elif rank_value <= 7:  # 7, 8, 9, 10
                    rank_multiplier = 1.0  # No penalty
            
            score *= rank_multiplier
                
            # TLMN Strategy: Prioritize singles/pairs early, save combos for later
            if self._would_break_strong_combo(move, hand) and rank_value != 12:  # Don't penalty 2 cards
                # Stronger penalty early
                score *= (0.05 if game_progress < 0.4 else 0.1)
            elif combo_type == "single":
                # Singles get HIGH priority early game (TLMN strategy)
                if game_progress < 0.6:
                    score *= 2.5  # High preference for singles early
                else:
                    score *= 1.5  # Moderate preference later
            elif combo_type == "pair":
                # Pairs get good priority early-mid game
                if game_progress < 0.7:
                    score *= 2.0  # Good preference for pairs early-mid
                else:
                    score *= 1.2  # Lower preference late game
            elif combo_type == "triple":
                # Triples: moderate early, strong late
                if game_progress < 0.5:
                    score *= 0.8  # Lower preference early (save for later)
                else:
                    score *= 2.0  # Strong preference late game
            elif combo_type == "straight":
                # Straights: save for mid-late game
                if game_progress < 0.6:
                    score *= 0.6  # Lower preference early (save for later)
                else:
                    score *= 2.5  # Strong preference mid-late game
            elif combo_type == "four_kind":
                # Four_kind: save for late game
                if game_progress < 0.7:
                    score *= 0.5  # Lower preference early (save for later)
                else:
                    score *= 3.0  # Very strong preference late game
            
            # Special logic for 2 cards - only use for blocking/breaking combos
            if rank_value == 12:  # 2 cards
                if self._should_use_2_card(move, hand, game_progress, last_move):
                    score *= 3.0  # Very strong bonus for 2 cards when needed
                else:
                    score *= 0.05  # EXTREME penalty for 2 cards when not needed (20x penalty)
                
            # Avoid weak cards (3, 4, 5, 6) unless late game
            if rank_value <= 3 and game_progress < 0.7:
                score *= 0.2  # Strong penalty for weak cards

            # TLMN Tempo adjustments: Encourage singles/pairs early, combos later
            is_strong = combo_type in ["straight", "triple", "four_kind", "double_seq"]
            is_simple = combo_type in ["single", "pair"]
            
            if is_simple and game_progress < 0.6:
                # Encourage singles/pairs early game
                score *= 1.3
            elif is_strong and game_progress < 0.5:
                # Discourage strong combos very early (save for later)
                score *= 0.4
            elif is_strong and game_progress < 0.7:
                # Moderate preference for combos mid-game
                score *= 0.8
            elif is_strong and len(hand) >= 7:
                # Don't preserve combos when hand is large
                score *= 0.9
            if (game_progress > 0.7) or (len(hand) <= 4):
                # Late game: finish with strong combos
                if is_strong:
                    score *= 1.5
                elif is_simple:
                    score *= 1.2

            # Framework plan alignment (if provided)
            if framework:
                rec_moves = framework.get('recommended_moves', []) or []
                move_cards = set(move.get('cards', []))
                if any(set(rec) == move_cards for rec in rec_moves):
                    # Slightly stronger in early to prefer planned steps rather than raw strength
                    score *= (1.3 if game_progress < 0.4 else 1.2)

                # Encourage shedding singles not in any core combo during early-mid game
                if combo_type in ["single", "pair"] and game_progress <= 0.6:
                    in_core = False
                    for combo in framework.get('core_combos', []) or []:
                        if move_cards.issubset(set(combo.get('cards', []))):
                            in_core = True
                            break
                    if not in_core:
                        score *= 1.25
                
            scored_moves.append((move, score))
        
        # Sort by score and pick (with some randomness)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        rand_pick_second = 0.1 if game_progress < 0.4 else 0.3
        if len(scored_moves) > 1 and random.random() < rand_pick_second:
            # 30% chance to pick 2nd best
            return scored_moves[1][0]
        else:
            return scored_moves[0][0]
    
    def generate_game_session(self, num_players: int = 4) -> List[Dict[str, Any]]:
        """Generate a complete game session with sequence context"""
        
        # Deal cards
        hands = self.deal_cards(num_players)
        
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
        max_turns = 40  # Safety limit
        
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
            legal_moves = self.find_combos_in_hand(hand)
            
            # Choose move (pass framework for plan-aware scoring)
            chosen_move = self.choose_smart_move(
                legal_moves,
                hand,
                last_move,
                game_progress,
                framework=sequence_contexts[current_player].get('framework', {})
            )
            
            # Create record with sequence context
            record = self._create_record(
                current_player, hand, last_move, legal_moves, chosen_move, 
                hands, game_progress, turn_count, sequence_contexts[current_player]
            )
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
    
    def _create_record(self, player_id: int, hand: List[int], last_move: Dict,
                      legal_moves: List[Dict], chosen_move: Dict, all_hands: List[List[int]],
                      game_progress: float, turn_count: int, sequence_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create training record in correct format with sequence context"""
        
        # Calculate cards left per player
        cards_left = [len(hand) for hand in all_hands]
        
        # Create record
        record = {
            "game_id": f"synthetic_game_{random.randint(1000, 9999)}",
            "game_type": ("tlmn" if self.game_type == 'tlmn' else "sam"),
            "players_count": len(all_hands),
            "round_id": 0,
            "turn_id": turn_count,
            "current_player_id": player_id,
            "player_id": player_id,
            # IMPORTANT: store copies to avoid later mutation corrupting saved data
            "hand": list(hand),
            "last_move": copy.deepcopy(last_move),
            "players_left": [i for i in range(len(all_hands)) if cards_left[i] > 0],
            "cards_left": cards_left,
            "is_finished": False,
            "winner_id": None,
            "meta": {
                "agentType": "synthetic_smart",
                "legal_moves": [copy.deepcopy(m) for m in legal_moves]
            },
            "action": {
                "stage1": {"type": "pass", "value": "pass"},
                "stage2": copy.deepcopy(chosen_move)
            },
            "hand_count": len(hand),
            "timestamp": datetime.now().isoformat(),
            "synthetic": True,
            "game_progress": game_progress,
            # NEW: Sequence context and framework
            "sequence_context": copy.deepcopy(sequence_context),
            "framework": copy.deepcopy(sequence_context.get('framework', {}))
        }
        
        return record
    
    def generate_training_data(self, num_sessions: int = 100) -> List[Dict[str, Any]]:
        """Generate complete training dataset with sequence context"""
        
        all_records = []
        
        print(f"Generating {num_sessions} game sessions with sequence context...")
        
        for i in range(num_sessions):
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_sessions} sessions...")
                
            try:
                session_records = self.generate_game_session(4)
                all_records.extend(session_records)
            except Exception as e:
                print(f"Error generating session {i + 1}: {e}")
                continue
                
        print(f"Generated {len(all_records)} total records with sequence context")
        return all_records
    
    def _clean_for_json(self, obj):
        """Recursively convert sets to lists for JSON serialization"""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        else:
            return obj
    
    def save_data(self, records: List[Dict[str, Any]], filename: str):
        """Save data to JSONL file"""
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for record in records:
                # Clean record to remove sets and other non-JSON serializable objects
                clean_record = self._clean_for_json(record)
                f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
                
        print(f"Saved {len(records)} records to {filename}")


def main():
    """Main function to generate training data with sequence context"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic training data with sequence context')
    parser.add_argument('--game_type', type=str, default='sam', choices=['sam', 'tlmn'], help='Game type for data generation')
    parser.add_argument('--sessions', type=int, default=200, help='Number of game sessions to simulate')
    parser.add_argument('--output', type=str, default='simple_synthetic_training_data_with_sequence.jsonl', help='Output JSONL path')
    args = parser.parse_args()

    print("=" * 60)
    print("SIMPLE SYNTHETIC TRAINING DATA GENERATOR")
    print("WITH SEQUENCE CONTEXT & FRAMEWORK GENERATION")
    print(f"GAME TYPE: {args.game_type}")
    print("=" * 60)
    
    # Initialize generator
    generator = SimpleSyntheticGenerator(game_type=args.game_type)
    
    # Generate data
    records = generator.generate_training_data(num_sessions=args.sessions)
    
    # Save data
    output_file = args.output
    generator.save_data(records, output_file)
    
    # Show statistics
    print("\n" + "=" * 60)
    print("GENERATION STATISTICS")
    print("=" * 60)
    print(f"Total records: {len(records)}")
    
    # Count by combo type
    combo_counts = {}
    legal_has_double_seq = 0
    framework_double_seq = 0
    foolish_moves = 0
    records_with_framework = 0
    
    for record in records:
        action = record.get("action", {})
        stage2 = action.get("stage2", {})
        combo_type = stage2.get("combo_type", "unknown")
        rank_value = stage2.get("rank_value", 0)
        game_progress = record.get("game_progress", 0)
        
        combo_counts[combo_type] = combo_counts.get(combo_type, 0) + 1
        
        # Track availability of double_seq in legal moves (TLMN consecutive pairs)
        legal_moves_rec = record.get("meta", {}).get("legal_moves", []) or []
        if any(m.get("combo_type") == "double_seq" for m in legal_moves_rec):
            legal_has_double_seq += 1

        # Track framework double_seq presence
        fw = record.get("framework", {}) or {}
        for cc in fw.get("core_combos", []) or []:
            if (cc.get("type") == "double_seq"):
                framework_double_seq += 1

        # Check for foolish moves
        if combo_type in ["triple", "four_kind"] and rank_value == 1 and game_progress < 0.3:
            foolish_moves += 1
            
        # Check for framework availability
        if record.get("framework") and record["framework"].get("core_combos"):
            records_with_framework += 1
            
    print("\nCombo type distribution:")
    for combo_type, count in sorted(combo_counts.items()):
        percentage = (count / len(records)) * 100
        print(f"  {combo_type}: {count} ({percentage:.1f}%)")
    print(f"\nRecords with legal double_seq available: {legal_has_double_seq}/{len(records)} ({(legal_has_double_seq/len(records))*100:.1f}%)")
    print(f"Framework double_seq occurrences in core_combos: {framework_double_seq}")
        
    print(f"\nFoolish moves detected: {foolish_moves}/{len(records)} ({foolish_moves/len(records)*100:.1f}%)")
    print(f"Records with framework: {records_with_framework}/{len(records)} ({records_with_framework/len(records)*100:.1f}%)")
    print(f"Data saved to: {output_file}")
    print("Ready for training with Two-Layer Architecture!")


if __name__ == "__main__":
    main()
