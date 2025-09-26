"""
Optimized General Gameplay Model V3 - Per-Candidate Approach
Single Stage: Move Ranking with 27-dim features per candidate
"""

import numpy as np
import joblib
from typing import Dict, List, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb


class OptimizedGeneralModelV3:
    """
    Per-Candidate XGBoost model that ranks all legal moves and selects the best one.
    
    Features: 27 dims per candidate (11 general + 16 combo-specific)
    - General: legal_moves_combo_counts, cards_left, hand_count
    - Combo-specific: combo_type_onehot, hybrid_rank, combo_length, breaks_combo_flag, 
      individual_strength, combo_type_multiplier, enhanced_breaks_penalty, 
      combo_efficiency_score, combo_preference_bonus, combo_preservation_bonus
    """
    
    def __init__(self):
        self.candidate_model = None  # Per-candidate model (ranking)
        self.combo_types = ["single", "pair", "triple", "four_kind", "straight", "double_seq", "pass"]
        self.combo_type_to_id = {ct: i for i, ct in enumerate(self.combo_types)}
        self.training_data_size = 0  # Track training data size for hybrid approach
        self.use_rank_category = True  # Default to rank_category for small datasets

    # -------------------------------
    # Rank-only helpers (0-12)
    # -------------------------------
    def _to_ranks(self, cards: List[int]) -> List[int]:
        """Convert absolute card ids (0-51) to rank-only ids (0-12)."""
        if not isinstance(cards, list):
            return []
        return [int(c) % 13 for c in cards]
        
    def calculate_combo_strength_relative(self, legal_moves: List[Dict[str, Any]]) -> float:
        """
        Tính sức mạnh tương đối của các combos cho Stage 1 (rank-only, suit-agnostic)
        Mỗi combo type có cách tính rank khác nhau
        """
        combo_strengths = []
        
        for move in legal_moves:
            if move.get("type") == "play_cards":
                combo_type = move.get("combo_type")
                rank_value = move.get("rank_value", 0)
                cards = move.get("cards", [])
                ranks = self._to_ranks(cards)
                
                # Calculate strength based on combo type - MORE AGGRESSIVE DIFFERENTIATION
                if combo_type == "single":
                    # Single: 2, A, Phần còn lại (đánh từ bé đến lớn)
                    if rank_value == 1:  # 2
                        strength = 8.0  # Tăng từ 3.0 → 8.0
                    elif rank_value == 0:  # A
                        strength = 6.0  # Tăng từ 2.0 → 6.0
                    else:  # Phần còn lại
                        strength = 1.0 + (rank_value - 2) / 5.0  # 3-K: 1.0-3.0 (tăng range)
                        
                elif combo_type == "pair":
                    # Pair: 2, A, Mặt người (J,Q,K), Phần còn lại
                    if rank_value == 1:  # 2
                        strength = 12.0  # Tăng từ 4.0 → 12.0
                    elif rank_value == 0:  # A
                        strength = 10.0  # Tăng từ 3.0 → 10.0
                    elif rank_value >= 10:  # J, Q, K (mặt người)
                        strength = 7.0   # Tăng từ 2.5 → 7.0
                    else:  # Phần còn lại
                        strength = 4.0 + (rank_value - 2) / 2.0  # 3-10: 4.0-8.0 (tăng range)
                        
                elif combo_type == "triple":
                    # Triple: 2, A, >= 7, Phần còn lại
                    if rank_value == 1:  # 2
                        strength = 15.0  # Tăng từ 5.0 → 15.0
                    elif rank_value == 0:  # A
                        strength = 13.0  # Tăng từ 4.0 → 13.0
                    elif rank_value >= 6:  # >= 7 (7,8,9,10,J,Q,K)
                        strength = 10.0  # Tăng từ 3.5 → 10.0
                    else:  # Phần còn lại (3,4,5,6)
                        strength = 7.0 + (rank_value - 2) / 2.0  # 3-6: 7.0-9.5 (tăng range)
                        
                elif combo_type == "four_kind":
                    # Four_kind: A và phần còn lại (2 thì thắng luôn)
                    if rank_value == 1:  # 2 - thắng luôn
                        strength = 25.0  # Tăng từ 10.0 → 25.0
                    elif rank_value == 0:  # A
                        strength = 22.0  # Tăng từ 9.0 → 22.0
                    else:  # Phần còn lại
                        strength = 18.0 + (rank_value - 2) / 2.0  # 3-K: 18.0-23.0 (tăng range)
                        
                elif combo_type == "straight":
                    # Straight: Dây chạm A thì tối đa sức mạnh
                    has_ace = any(r == 0 for r in ranks)  # rank-only Ace
                    length = len(ranks)
                    
                    if has_ace:
                        strength = 16.0 + length / 2.0  # A straight: 17.0-18.0 (tăng từ 7.5-8.0)
                    else:
                        strength = 12.0 + length / 2.0 + (rank_value / 13.0) * 2.0  # Other: 13.0-15.0 (tăng từ 6.5-7.0)
                        
                elif combo_type == "double_seq":
                    # Double_seq: Cực mạnh, vượt trội
                    length = len(ranks)
                    strength = 20.0 + length / 2.0  # 21.0-22.0 (tăng từ 9.5-10.0)
                    
                else:
                    strength = 0.0
                    
                combo_strengths.append(strength)
        
        # Return average strength (0-1 normalized, rounded to 3 decimal places)
        max_possible_strength = 10.0  # 2 four_kind
        normalized_strengths = [s / max_possible_strength for s in combo_strengths]
        avg_strength = sum(normalized_strengths) / len(normalized_strengths) if normalized_strengths else 0.0
        return float(round(avg_strength, 3))
    
    def calculate_combo_strength_ranking(self, legal_moves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tính ranking strength cho từng move trong legal_moves cho Stage 2
        Chỉ cần ranking cơ bản theo rank_value (0-12) vì đã xác định combo rồi
        """
        move_rankings = []
        
        for move in legal_moves:
            if move.get("type") == "play_cards":
                combo_type = move.get("combo_type")
                rank_value = move.get("rank_value", 0)
                cards = move.get("cards", [])
                
                # Chỉ cần ranking cơ bản theo rank_value (0-12)
                # A=0, 2=1, 3=2, ..., K=12
                strength = rank_value
                
                move_rankings.append({
                    "move": move,
                    "strength": strength,
                    "combo_type": combo_type,
                    "rank_value": rank_value,
                    "cards": cards
                })
        
        # Sort by strength (descending - rank cao hơn mạnh hơn)
        move_rankings.sort(key=lambda x: x["strength"], reverse=True)
        
        return move_rankings
    
    def extract_legal_moves_combo_counts(self, legal_moves: List[Dict[str, Any]]) -> List[int]:
        """Extract combo counts from legal_moves"""
        combo_counts = {"single": 0, "pair": 0, "triple": 0, "four_kind": 0, "straight": 0, "double_seq": 0}
        
        for move in legal_moves:
            if move.get("type") == "play_cards":
                combo_type = move.get("combo_type")
                if combo_type in combo_counts:
                    combo_counts[combo_type] += 1
        
        return [
            combo_counts["single"],
            combo_counts["pair"],
            combo_counts["triple"],
            combo_counts["four_kind"],
            combo_counts["straight"],
            combo_counts["double_seq"]
        ]
    
    
    

    # ================================
    # Per-candidate Stage 1 (Ranking)
    # ================================
    def _compute_rank_category(self, combo_type: str, rank_value: int, cards: List[int]) -> int:
        """Return compact rank category per spec for each combo type.
        Categories are encoded as small integers.
        """
        if combo_type == "single":
            # {2, A, rest}
            if rank_value == 1:
                return 2  # 2
            if rank_value == 0:
                return 1  # A
            return 0      # rest
        if combo_type == "pair":
            # {2, A, face(JQK), rest}
            if rank_value == 1:
                return 3  # 2
            if rank_value == 0:
                return 2  # A
            if rank_value >= 10:
                return 1  # face
            return 0      # rest
        if combo_type == "triple":
            # {2, A, >=7, rest}
            if rank_value == 1:
                return 3  # 2
            if rank_value == 0:
                return 2  # A
            if rank_value >= 6:  # rank_value 6 == 7
                return 1
            return 0
        if combo_type == "four_kind":
            # {2, A, rest}
            if rank_value == 1:
                return 2
            if rank_value == 0:
                return 1
            return 0
        if combo_type in ("straight", "double_seq"):
            # Encode by length using rank-only sequence
            length = len(self._to_ranks(cards))
            return min(12, max(0, length))
        # pass or unknown
        return 0

    def _compute_hybrid_rank_feature(self, combo_type: str, rank_value: int, cards: List[int]) -> float:
        """Hybrid rank feature: use rank_category for small datasets, rank_value for large datasets.
        
        Args:
            combo_type: Type of combo
            rank_value: Raw rank value (0-12)
            cards: List of card ids
            
        Returns:
            Rank feature value (normalized)
        """
        # For singles, always prefer continuous rank_value so 3..K are distinguished.
        if combo_type == "single":
            rank_feature = float(rank_value) / 12.0
        else:
            # Threshold: use rank_category if training data < 1000 samples
            if self.training_data_size < 1000:
                # Use rank_category for small datasets
                rank_feature = float(self._compute_rank_category(combo_type, rank_value, cards))
            else:
                # Use actual rank_value for large datasets (normalized to 0-1)
                rank_feature = float(rank_value) / 12.0
        
        return rank_feature

    def _compute_combo_length(self, combo_type: str, cards: List[int]) -> int:
        if combo_type in ("straight", "double_seq"):
            return len(self._to_ranks(cards))
        return 0

    def _breaks_combo_flag(self, hand: List[int], move_cards: List[int]) -> int:
        """Return severity if this move breaks stronger structures (0/1/2).
        Heuristics:
          - Quad split → severity 2 (heavy)
          - Double_seq lost → severity 2 (heavy)
          - Triple split → severity 1 (normal)
          - Straight length reduced (when before >=5) → severity 1 (normal)
        """
        if not hand or not move_cards:
            return 0
        rank_counts = {}
        for c in hand:
            r = c % 13
            rank_counts[r] = rank_counts.get(r, 0) + 1
        used_by_rank = {}
        for c in move_cards:
            r = c % 13
            used_by_rank[r] = used_by_rank.get(r, 0) + 1
        severity = 0
        for r, used in used_by_rank.items():
            count_before = rank_counts.get(r, 0)
            if count_before >= 4:
                # Splitting a quad by playing <4 cards
                if used < 4:
                    severity = max(severity, 2)
            if count_before == 3:
                # Splitting a triple by playing <3 cards
                if used < 3:
                    severity = max(severity, 1)

        # Straight/double_seq potentials: compare before vs after removing move_cards
        def longest_straight_length(counts: Dict[int, int]) -> int:
            # Detect longest straight allowing Ace-high wrap (.., Q(11), K(12), A(0))
            order = list(range(1, 13)) + [0]  # 2..K,A
            max_len = 0
            cur = 0
            for r in order:
                if counts.get(r, 0) > 0:
                    cur += 1
                    max_len = max(max_len, cur)
                else:
                    cur = 0
            return max_len

        def exists_double_sequence(counts: Dict[int, int]) -> bool:
            # Need at least two consecutive ranks with count>=2 (length >= 2 pairs => 4 cards)
            order = list(range(1, 13)) + [0]
            cur = 0
            for r in order:
                if counts.get(r, 0) >= 2:
                    cur += 1
                    if cur >= 2:
                        return True
                else:
                    cur = 0
            return False

        before_straight = longest_straight_length(rank_counts)
        before_double = exists_double_sequence(rank_counts)

        # simulate removal
        after_counts = dict(rank_counts)
        for c in move_cards:
            r = c % 13
            if after_counts.get(r, 0) > 0:
                after_counts[r] -= 1

        after_straight = longest_straight_length(after_counts)
        after_double = exists_double_sequence(after_counts)

        # Breaking straights: heavier penalty for long straights
        # Penalize breaking straights at shorter lengths too
        if before_straight >= 6 and after_straight < before_straight:
            severity = max(severity, 2)
        elif before_straight >= 5 and after_straight < before_straight:
            severity = max(severity, 2)
        elif before_straight >= 4 and after_straight < before_straight:
            severity = max(severity, 1)
        if before_double and not after_double:
            severity = max(severity, 2)
        return severity

    def _compute_individual_move_strength(self, combo_type: str, rank_value: int, cards: List[int]) -> float:
        """Compute individual move strength (normalized 0-1) for this specific move.
        This replaces the average strength approach with per-move strength calculation.
        """
        if combo_type == "pass":
            return 0.0
            
        ranks = self._to_ranks(cards)
        
        # Use the same strength calculation logic as calculate_combo_strength_relative
        # Preserve relative order for singles: 2 > A > others, but keep all small so combos dominate.
        # Early-game suppression is handled in efficiency/preservation features.
        if combo_type == "single":
            # Keep singles clearly weaker than pairs, preserve 2 > A > others
            if rank_value == 1:  # 2 - strongest single
                strength = 3.5
            elif rank_value == 0:  # A - second strongest single
                strength = 3.0
            else:
                # 3..K mapped to 1.0 .. 2.5 (linear)
                step = max(0, rank_value - 2)
                strength = 1.0 + 0.125 * float(step)
                
        elif combo_type == "pair":
            if rank_value == 1:  # 2
                strength = 12.0
            elif rank_value == 0:  # A
                strength = 10.0
            elif rank_value >= 10:  # J, Q, K (mặt người)
                strength = 7.0
            else:  # Phần còn lại
                strength = 4.0 + (rank_value - 2) / 2.0  # 3-10: 4.0-8.0
                
        elif combo_type == "triple":
            if rank_value == 1:  # 2
                strength = 15.0
            elif rank_value == 0:  # A
                strength = 13.0
            elif rank_value >= 6:  # >= 7 (7,8,9,10,J,Q,K)
                strength = 10.0
            else:  # Phần còn lại (3,4,5,6)
                strength = 7.0 + (rank_value - 2) / 2.0  # 3-6: 7.0-9.5
                
        elif combo_type == "four_kind":
            if rank_value == 1:  # 2 - thắng luôn
                strength = 25.0
            elif rank_value == 0:  # A
                strength = 22.0
            else:  # Phần còn lại
                strength = 18.0 + (rank_value - 2) / 2.0  # 3-K: 18.0-23.0
                
        elif combo_type == "straight":
            has_ace = any(r == 0 for r in ranks)
            length = len(ranks)
            
            if has_ace:
                strength = 16.0 + length / 2.0  # A straight: 17.0-18.0
            else:
                strength = 12.0 + length / 2.0 + (rank_value / 13.0) * 2.0  # Other: 13.0-15.0
                
        elif combo_type == "double_seq":
            length = len(ranks)
            strength = 20.0 + length / 2.0  # 21.0-22.0
            
        else:
            strength = 0.0
        
        # Normalize to 0-1 range (max possible strength is 25.0 for 2 four_kind)
        return float(round(strength / 25.0, 3))

    def _compute_combo_type_strength_multiplier(self, combo_type: str) -> float:
        """Compute combo type strength multiplier to encourage stronger combos.
        Higher values = stronger combo type preference.
        MUCH STRONGER WEIGHTS to override single bias.
        """
        multipliers = {
            "single": 1.0,      # Base strength
            "pair": 10.0,       # 10x stronger than single (was 5.0)
            "triple": 25.0,     # 25x stronger than single (was 10.0)
            "four_kind": 50.0,  # 50x stronger than single (was 20.0)
            "straight": 45.0,   # Increase to prioritize straights
            "double_seq": 80.0, # Increase to prioritize double sequences
            "pass": 0.0         # No strength
        }
        return multipliers.get(combo_type, 0.0)

    def _compute_enhanced_breaks_penalty(self, hand: List[int], move_cards: List[int], combo_type: str) -> float:
        """Enhanced penalty for breaking combos with MUCH STRONGER penalties.
        Returns penalty value (higher = worse move).
        """
        if not hand or not move_cards or combo_type == "pass":
            return 0.0
            
        # Get basic breaks flag
        basic_penalty = self._breaks_combo_flag(hand, move_cards)

        if basic_penalty == 0:
            return 0.0

        # Derive additional context: straight/double presence before removal
        rank_counts: Dict[int, int] = {}
        for c in hand:
            r = c % 13
            rank_counts[r] = rank_counts.get(r, 0) + 1

        def longest_straight_length(counts: Dict[int, int]) -> int:
            max_len = 0
            cur = 0
            for r in range(13):
                if counts.get(r, 0) > 0:
                    cur += 1
                    max_len = max(max_len, cur)
                else:
                    cur = 0
            return max_len

        def exists_double_sequence(counts: Dict[int, int]) -> bool:
            cur = 0
            for r in range(13):
                if counts.get(r, 0) >= 2:
                    cur += 1
                    if cur >= 2:
                        return True
                else:
                    cur = 0
            return False

        before_straight = longest_straight_length(rank_counts)
        before_double = exists_double_sequence(rank_counts)

        # Start with stronger base for severity
        penalty = 0.0
        if basic_penalty == 1:
            penalty = 1.8
        elif basic_penalty == 2:
            penalty = 3.0

        # Extra if breaking via single
        if combo_type == "single":
            penalty += 1.5

        # Extra if there is a long straight potential in hand
        if before_straight >= 6:
            penalty += 1.5
        elif before_straight >= 5:
            penalty += 1.0

        # Extra if a double sequence potential exists
        if before_double:
            penalty += 1.0

        return float(min(6.0, penalty))

    def _compute_combo_efficiency_score(self, combo_type: str, rank_value: int, cards: List[int], cards_left: List[int] = None) -> float:
        """Compute combo efficiency score to encourage playing stronger combos.
        Uses cards_left to adjust efficiency based on game phase.
        Higher values = more efficient use of cards.
        MUCH STRONGER EFFICIENCY SCORES to override single bias.
        """
        if combo_type == "pass":
            return 0.0
            
        # Base efficiency by combo type - reinforce combos >> singles
        base_efficiency = {
            "single": 0.08,
            "pair": 0.55,
            "triple": 0.8,
            "four_kind": 1.0,
            "straight": 0.85,
            "double_seq": 0.97
        }
        
        efficiency = base_efficiency.get(combo_type, 0.0)
        
        # Adjust by rank value (lower ranks = more efficient early game)
        if combo_type in ["single", "pair", "triple", "four_kind"]:
            if rank_value <= 3:  # 3, 4, 5, 6
                efficiency *= 1.2  # Bonus for low cards
            elif rank_value >= 10:  # J, Q, K
                efficiency *= 0.8  # Penalty for high cards
            elif rank_value in [0, 1]:  # A, 2
                efficiency *= 0.6  # Strong penalty for power cards
        
        # Adjust by combo length for straights
        if combo_type in ["straight", "double_seq"]:
            length = len(self._to_ranks(cards))
            if length >= 5:
                efficiency *= 1.1  # Bonus for long straights
            elif length <= 3:
                efficiency *= 0.9  # Penalty for short straights
        
        # Apply game phase adjustment using cards_left
        if cards_left:
            total_cards_left = len(cards_left)
            if total_cards_left > 40:  # Early game
                # EXTREME penalty for singles in early game, especially A/2
                if combo_type == "single":
                    if rank_value in [0, 1]:
                        efficiency *= 0.02
                    else:
                        efficiency *= 0.05
                elif combo_type == "pair" and rank_value in [0, 1]:  # high pairs early
                    efficiency *= 0.2
                elif combo_type in ["triple", "four_kind"] and rank_value in [0, 1]:
                    efficiency *= 0.15
                elif combo_type == "four_kind":
                    efficiency *= 0.2
            elif total_cards_left > 20:  # Mid game
                # Rank-weighted penalty for singles in mid game: higher ranks get penalized more
                if combo_type == "single":
                    # rank_value: A=0, 2=1, 3=2, ..., K=12
                    # Map 3..K to factor ~0.45..0.15; A/2 get heavier 0.10
                    if rank_value in [0, 1]:
                        rank_factor = 0.10
                    else:
                        # Decrease linearly with rank_value
                        rank_factor = max(0.15, 0.55 - 0.03 * float(max(2, rank_value)))
                    efficiency *= rank_factor
                elif combo_type in ["pair", "triple", "four_kind"] and rank_value in [0, 1]:  # A, 2
                    efficiency *= 0.35  # Slightly stronger than before
            # Late game: no additional penalty
        
        return float(round(min(1.0, efficiency), 3))

    def _compute_combo_preference_bonus(self, combo_type: str) -> float:
        """Compute combo preference bonus - ULTRA STRONG feature to override single bias.
        This is the most aggressive feature to encourage combos over singles.
        """
        if combo_type == "pass":
            return 0.0
            
        # ULTRA STRONG BONUS for combos vs singles - INCREASED VALUES
        bonuses = {
            "single": 0.0,
            "pair": 2.2,
            "triple": 5.5,
            "four_kind": 10.0,
            "straight": 8.0,
            "double_seq": 16.0
        }
        return bonuses.get(combo_type, 0.0)

    def _compute_combo_preservation_bonus(self, hand: List[int], move_cards: List[int], combo_type: str, cards_left: List[int] = None) -> float:
        """Compute combo preservation bonus - encourages keeping combos intact.
        Uses cards_left to detect game phase and adjust penalties accordingly.
        Returns bonus value (higher = better move).
        """
        if not hand or not move_cards or combo_type == "pass":
            return 0.0
            
        # Count potential combos in hand
        rank_counts = {}
        for c in hand:
            r = c % 13
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        # Calculate preservation bonus based on what we're playing
        if combo_type == "single":
            # Check if this single is part of a potential combo
            move_rank = move_cards[0] % 13
            rank_count = rank_counts.get(move_rank, 0)

            # Detect straight membership and length containing rank r
            def straight_length_for_rank(counts: Dict[int, int], r: int) -> int:
                order = list(range(1, 13)) + [0]
                pos = {rv: i for i, rv in enumerate(order)}
                if r not in pos:
                    return 0
                best = 0
                for start_idx in range(max(0, pos[r] - 4), min(pos[r], len(order) - 5) + 1):
                    cur = 0
                    for j in range(13):
                        idx = start_idx + j
                        if idx >= len(order):
                            break
                        rr = order[idx]
                        if counts.get(rr, 0) > 0:
                            cur += 1
                            best = max(best, cur)
                        else:
                            cur = 0
                return best

            seq_len = straight_length_for_rank(rank_counts, move_rank)

            # Base penalties
            if rank_count >= 4:  # Breaking a quad
                base_penalty = -1.2
            elif rank_count >= 3:  # Breaking a triple
                base_penalty = -1.0
            elif rank_count >= 2:  # Breaking a pair
                base_penalty = -0.7
            else:
                base_penalty = 0.0

            # Extra penalty if this rank belongs to an existing straight window
            if seq_len >= 3:
                base_penalty -= 0.4 + 0.2 * (seq_len - 3)  # -0.4 for 3, -0.6 for 4, -0.8 for 5, ...

            if base_penalty == 0.0:
                return 0.0

            # Apply game phase multiplier using cards_left
            if cards_left:
                total_cards_left = len(cards_left)
                if total_cards_left > 40:  # Early game - VERY STRONG penalty
                    penalty_multiplier = 18.0
                elif total_cards_left > 20:  # Mid game - strong penalty
                    penalty_multiplier = 9.0
                else:  # Late game - normal penalty
                    penalty_multiplier = 3.0

                return base_penalty * penalty_multiplier
            else:
                return base_penalty
                
        elif combo_type == "pair":
            # Bonus for using pair (preserves other combos). If pair uses ranks that belong to a potential straight,
            # we discourage it via a negative adjustment, especially early game.
            def straight_length_for_rank(counts: Dict[int, int], r: int) -> int:
                order = list(range(1, 13)) + [0]
                pos = {rv: i for i, rv in enumerate(order)}
                if r not in pos:
                    return 0
                best = 0
                for start_idx in range(max(0, pos[r] - 4), min(pos[r], len(order) - 5) + 1):
                    cur = 0
                    for j in range(13):
                        idx = start_idx + j
                        if idx >= len(order):
                            break
                        rr = order[idx]
                        if counts.get(rr, 0) > 0:
                            cur += 1
                            best = max(best, cur)
                        else:
                            cur = 0
                return best

            r = move_cards[0] % 13 if move_cards else -1
            base = 0.3
            if r >= 0:
                seq_len = straight_length_for_rank(rank_counts, r)
                if seq_len >= 3:
                    base -= 0.5 + 0.2 * (seq_len - 3)  # stronger discouragement for longer sequences
                if cards_left:
                    total_cards_left = len(cards_left)
                    if total_cards_left > 40:
                        base -= 1.2
                    elif total_cards_left > 20:
                        base -= 0.6
            return base
            
        elif combo_type == "triple":
            # Bonus for using triple (preserves other combos)
            return 0.6
            
        elif combo_type == "four_kind":
            # Bonus for using quad (preserves other combos)
            return 1.0
            
        elif combo_type in ["straight", "double_seq"]:
            # Bonus for using sequences (preserves other combos). Add extra bonus if a long straight exists.
            # Detect longest straight pre-move to encourage playing the straight instead of breaking it later.
            def longest_straight_length(counts: Dict[int, int]) -> int:
                order = list(range(1, 13)) + [0]
                max_len = 0
                cur = 0
                for r in order:
                    if counts.get(r, 0) > 0:
                        cur += 1
                        max_len = max(max_len, cur)
                    else:
                        cur = 0
                return max_len

            length = longest_straight_length(rank_counts)
            base = 0.8
            if length >= 6:
                base += 0.6
            elif length >= 5:
                base += 0.3
            return base
            
        return 0.0

    def extract_candidate_features(self, record: Dict[str, Any], move: Dict[str, Any]) -> np.ndarray:
        """Extract per-candidate features for move ranking.
        General (11 dims) + combo-specific (15 dims):
        - combo_type one-hot (7)
        - hybrid_rank_feature (1) - rank_category for small datasets, rank_value for large datasets
        - combo_length (1)
        - breaks_combo_flag (1)
        - individual_move_strength (1) - NEW: strength of this specific move (replaces combo_strength_relative)
        - combo_type_strength_multiplier (1) - NEW: relative strength between combo types
        - enhanced_breaks_penalty (1) - NEW: stronger penalty for breaking combos
        - combo_efficiency_score (1) - NEW: encourages stronger combos
        - combo_preference_bonus (1) - NEW: STRONGEST feature to override single bias
        - combo_preservation_bonus (1) - NEW: encourages keeping combos intact
        Total dims: 11 + 16 = 27
        """
        # General features (reuse Stage 1 general pipeline)
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        combo_counts = self.extract_legal_moves_combo_counts(legal_moves)
        cards_left = record.get("cards_left", [])
        while len(cards_left) < 4:
            cards_left.append(0)
        cards_left = cards_left[:4]
        hand = record.get("hand", [])
        hand_count = len(hand)
        # REMOVED: combo_strength_relative - replaced by individual_move_strength

        general_features = np.array([
            *combo_counts,                # 6
            *cards_left,                  # 4
            float(hand_count),            # 1
            # REMOVED: combo_strength_relative (1) - now 11 dims total
        ], dtype=np.float32)

        # Combo-specific features (rank-only where relevant)
        combo_type = move.get("combo_type", "pass") or "pass"
        rank_value = int(move.get("rank_value", -1)) if move.get("rank_value") is not None else -1
        cards = move.get("cards", []) or []

        one_hot = np.zeros(len(self.combo_types), dtype=np.float32)
        ct_id = self.combo_type_to_id.get(combo_type, self.combo_type_to_id["pass"])
        one_hot[ct_id] = 1.0

        # Use hybrid rank feature
        hybrid_rank = self._compute_hybrid_rank_feature(combo_type, rank_value, cards)
        combo_length = float(self._compute_combo_length(combo_type, cards))
        breaks_flag = float(self._breaks_combo_flag(hand, cards))
        
        # NEW FEATURES
        individual_strength = self._compute_individual_move_strength(combo_type, rank_value, cards)
        combo_type_multiplier = self._compute_combo_type_strength_multiplier(combo_type)
        enhanced_breaks_penalty = self._compute_enhanced_breaks_penalty(hand, cards, combo_type)
        combo_efficiency = self._compute_combo_efficiency_score(combo_type, rank_value, cards, cards_left)  # Pass cards_left
        combo_preference_bonus = self._compute_combo_preference_bonus(combo_type)  # NEW
        combo_preservation_bonus = self._compute_combo_preservation_bonus(hand, cards, combo_type, cards_left)  # Pass cards_left

        combo_specific = np.concatenate([
            one_hot,                              # 7
            np.array([hybrid_rank], np.float32),  # 1 - hybrid rank feature
            np.array([combo_length], np.float32), # 1
            np.array([breaks_flag], np.float32),  # 1
            np.array([individual_strength], np.float32),      # 1 - NEW
            np.array([combo_type_multiplier], np.float32),    # 1 - NEW
            np.array([enhanced_breaks_penalty], np.float32),  # 1 - NEW
            np.array([combo_efficiency], np.float32),         # 1 - NEW
            np.array([combo_preference_bonus], np.float32),   # 1 - NEW
            np.array([combo_preservation_bonus], np.float32)  # 1 - NEW
        ])

        return np.concatenate([general_features, combo_specific]).astype(np.float32)

    def get_hybrid_info(self) -> Dict[str, Any]:
        """Get information about current hybrid approach configuration."""
        return {
            'training_data_size': self.training_data_size,
            'use_rank_category': self.training_data_size < 1000,
            'threshold': 1000,
            'approach': 'rank_category' if self.training_data_size < 1000 else 'rank_value',
            'description': 'Using rank_category for small datasets (<1000), rank_value for large datasets (≥1000)'
        }

    def build_candidate_dataset(self, records: List[Dict[str, Any]]):
        """Build dataset of per-candidate samples.
        Returns:
          X: np.ndarray [N, F]
          y: np.ndarray [N] (0/1)
          groups: List[List[int]] mapping each turn to indices in X for per-turn evaluation
        """
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        groups: List[List[int]] = []

        for record in records:
            legal_moves = record.get("meta", {}).get("legal_moves", []) or []
            if not legal_moves:
                continue
            turn_indices = []

            # Determine chosen move from action
            action = record.get("action", {})
            stage2 = action.get("stage2", {})
            chosen_type = stage2.get("type")
            chosen_cards = stage2.get("cards", []) if chosen_type == "play_cards" else []
            chose_pass = (chosen_type == "pass")
            
            # Get chosen combo info for rank-based comparison
            chosen_combo_type = stage2.get("combo_type") if chosen_type == "play_cards" else None
            chosen_rank_value = stage2.get("rank_value") if chosen_type == "play_cards" else None

            for move in legal_moves:
                if move.get("type") not in ("play_cards", "pass"):
                    continue
                feat = self.extract_candidate_features(record, move)
                X_list.append(feat)
                idx = len(X_list) - 1
                turn_indices.append(idx)

                # Label 1 if this move was chosen (rank-based comparison)
                if move.get("type") == "pass":
                    y_list.append(1 if chose_pass else 0)
                else:
                    # Compare by combo_type and rank_value instead of exact cards
                    move_combo_type = move.get("combo_type")
                    move_rank_value = move.get("rank_value")
                    is_chosen = (chosen_type == "play_cards" and 
                               move_combo_type == chosen_combo_type and 
                               move_rank_value == chosen_rank_value)
                    y_list.append(1 if is_chosen else 0)

            if turn_indices:
                groups.append(turn_indices)

        if not X_list:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        return X, y, groups

    def train_candidate_model(self, records: List[Dict[str, Any]], model_type: str = "xgb"):
        """Train per-candidate model.
        model_type: "dt" | "rf" | "xgb"
        Returns sample-level training accuracy.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier as SkDT

        # Update training data size for hybrid approach
        self.training_data_size = len(records)
        
        X, y, _ = self.build_candidate_dataset(records)
        if X.shape[0] == 0:
            return 0.0

        if model_type == "dt":
            model = SkDT(max_depth=16, min_samples_split=10, min_samples_leaf=5, criterion='entropy', random_state=42)
        elif model_type == "xgb":
            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            )

        model.fit(X, y)
        self.candidate_model = model

        y_pred = model.predict(X)
        acc = (y_pred == y).mean().item()
        return float(acc)

    def evaluate_candidate_model(self, records: List[Dict[str, Any]]):
        """Evaluate per-candidate model with per-turn top-1 accuracy."""
        X, y, groups = self.build_candidate_dataset(records)
        if X.shape[0] == 0 or not groups or self.candidate_model is None:
            return {"turn_accuracy": 0.0, "num_turns": 0, "num_samples": int(X.shape[0])}

        probs = self.candidate_model.predict_proba(X)[:, 1]
        correct = 0
        for turn_indices in groups:
            # Find max prob in this turn
            best_idx = max(turn_indices, key=lambda i: probs[i])
            if y[best_idx] == 1:
                correct += 1
        turn_accuracy = correct / len(groups) if groups else 0.0
        return {"turn_accuracy": float(turn_accuracy), "num_turns": len(groups), "num_samples": int(X.shape[0])}

    def evaluate_candidate_model_topk(self, records: List[Dict[str, Any]], k: int = 3):
        """Evaluate per-candidate model with per-turn top-k accuracy."""
        X, y, groups = self.build_candidate_dataset(records)
        if X.shape[0] == 0 or not groups or self.candidate_model is None:
            return {"turn_topk_accuracy": 0.0, "k": k, "num_turns": 0, "num_samples": int(X.shape[0])}

        probs = self.candidate_model.predict_proba(X)[:, 1]
        correct = 0
        for turn_indices in groups:
            # top-k indices by probability
            sorted_indices = sorted(turn_indices, key=lambda i: probs[i], reverse=True)
            topk = sorted_indices[:max(1, k)]
            if any(y[i] == 1 for i in topk):
                correct += 1
        turn_topk_accuracy = correct / len(groups) if groups else 0.0
        return {"turn_topk_accuracy": float(turn_topk_accuracy), "k": k, "num_turns": len(groups), "num_samples": int(X.shape[0])}

    def compare_candidate_models(self, records: List[Dict[str, Any]]):
        """Train/evaluate DT, RF, XGB for per-candidate model and return metrics."""
        results = {}
        for mt in ["dt", "rf", "xgb"]:
            acc = self.train_candidate_model(records, model_type=mt)
            eval_res = self.evaluate_candidate_model(records)
            eval_top3 = self.evaluate_candidate_model_topk(records, k=3)
            results[mt] = {
                "sample_acc": acc,
                "turn_accuracy": eval_res.get("turn_accuracy", 0.0),
                "turn_top3": eval_top3.get("turn_topk_accuracy", 0.0),
                "num_turns": eval_res.get("num_turns", 0),
                "num_samples": eval_res.get("num_samples", 0),
            }
        return results
    
    
    
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predict move using per-candidate approach"""
        
        if self.candidate_model is None:
            raise ValueError("Per-candidate model not trained yet")
        
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        if not legal_moves:
            return {"type": "pass", "cards": []}
        
        # Use per-candidate model to rank all legal moves
        move_scores = []
        
        for move in legal_moves:
            # Extract features for this candidate move
            features = self.extract_candidate_features(record, move)
            
            # Get probability score from per-candidate model
            prob = self.candidate_model.predict_proba(features.reshape(1, -1))[0, 1]
            
            move_scores.append({
                'move': move,
                'score': prob,
                'combo_type': move.get('combo_type', 'pass'),
                'rank_value': move.get('rank_value', -1)
            })
        
        # Sort by score (descending - higher score = better move)
        move_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the best move
        return move_scores[0]['move']
    
    def save(self, filepath: str):
        """Save trained model"""
        model_data = {
            'candidate_model': self.candidate_model,
            'combo_types': self.combo_types,
            'combo_type_to_id': self.combo_type_to_id,
            'training_data_size': self.training_data_size,
            'use_rank_category': self.use_rank_category
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.candidate_model = model_data['candidate_model']
        self.combo_types = model_data['combo_types']
        self.combo_type_to_id = model_data['combo_type_to_id']
        self.training_data_size = model_data.get('training_data_size', 0)
        self.use_rank_category = model_data.get('use_rank_category', True)


