from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ #
# Feature flags để dễ bật/tắt các logic tuỳ biến chiến thuật
# ------------------------------------------------------------------ #
ENABLE_BLOCK_LEAD_GATING = True  # Phân vùng nhẹ chiến thuật khi block vs khi mở vòng
ENABLE_LEAD_QUALITY_FEATURES = True  # Đánh giá chất lượng lá/combo dùng để giật cái


class FrameworkAwareFeatureBuilder:
    """
    Utility that mirrors the feature engineering logic from StyleLearner.

    The RL policy consumes one feature vector per legal move. Each vector
    concatenates:
        - 24 context dims (22 original + 1 blocking + 1 can_beat)
        - 9 framework-aware dims (8 original + 1 sequence_order_penalty)
        - 12 multi-sequence dims
        Total: 45 dims
    """

    def __init__(self) -> None:
        self.combo_types = [
            "single",
            "pair",
            "triple",
            "four_kind",
            "straight",
            "double_seq",
            "pass",
        ]
        self.combo_type_to_id = {ct: i for i, ct in enumerate(self.combo_types)}
        # Base dim: 23 context (22 original + 1 blocking + 1 can_beat)
        #         + 9 framework (8 original + 1 seq_order_penalty)
        #         + 12 multi-sequence + 1 = 45
        # Các flag có thể thêm dims:
        # - ENABLE_BLOCK_LEAD_GATING: +2 (efficiency_block, efficiency_lead)
        # - ENABLE_LEAD_QUALITY_FEATURES: +2 (lead_candidate_score, lead_waste_penalty)
        base_dim = 45
        extra_dims = 0
        if ENABLE_BLOCK_LEAD_GATING:
            extra_dims += 2
        if ENABLE_LEAD_QUALITY_FEATURES:
            extra_dims += 2
        self.feature_dim = base_dim + extra_dims

        # ------------------------------------------------------------------ #
        # NOTE: Original feature scales (trước khi giảm để PPO dễ khám phá hơn)
        # Lưu lại để có thể rollback nếu cần tune lại theo phiên bản cũ.
        #
        # Context features:
        #   blocking_scale_orig   = 12.0
        #   can_beat_scale_orig   = 15.0
        #
        # Framework-aware features:
        #   priority_scale_orig       = 15.0
        #   breaking_scale_orig       = 15.0
        #   strength_scale_orig       = 8.0
        #   position_scale_orig       = 12.0
        #   combo_type_scale_orig     = 3.0
        #   rank_pref_scale_orig      = 4.0
        #   timing_scale_orig         = 3.0
        #   compliance_scale_orig     = 16.0
        #   seq_order_penalty_scale_orig = 20.0
        #
        # Các giá trị hiện tại đã được giảm xuống ≤ 5 (hoặc giữ nguyên với
        # combo_type / rank_pref / timing) để giảm mức ép khuôn của framework.
        # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def build_feature_matrix(
        self,
        game_record: Dict[str, Any],
        legal_moves: List[Dict[str, Any]],
        framework: Dict[str, Any],
    ) -> List[List[float]]:
        """
        Build feature vectors for all legal moves.

        Args:
            game_record: Current game snapshot (from BaseGame.get_game_record)
            legal_moves: List of move dicts
            framework: Output of FrameworkGenerator
        """
        # Store legal_moves in game_record.meta for feature extraction
        if "meta" not in game_record:
            game_record["meta"] = {}
        game_record["meta"]["legal_moves"] = legal_moves
        
        matrix: List[List[float]] = []
        for move in legal_moves:
            matrix.append(self._compose_features(move, game_record, framework))

        if not matrix:
            matrix.append(
                self._compose_features(
                    {"type": "pass", "cards": [], "combo_type": "pass", "rank_value": -1},
                    game_record,
                    framework,
                )
            )
        return matrix

    # ------------------------------------------------------------------ #
    # Feature composition
    # ------------------------------------------------------------------ #
    def _compose_features(
        self,
        move: Dict[str, Any],
        game_record: Dict[str, Any],
        framework: Dict[str, Any],
    ) -> List[float]:
        original = self._extract_original_features(move, game_record)
        framework_features = self._extract_framework_features(move, framework, game_record)
        multi_seq = self._extract_multi_sequence_features(move, framework)
        return original + framework_features + multi_seq

    def _extract_original_features(
        self,
        move: Dict[str, Any],
        game_record: Dict[str, Any],
    ) -> List[float]:
        """Context-only features (22 dims) với scale/chuẩn hoá cân bằng hơn."""
        features: List[float] = []
        legal_moves = (
            game_record.get("meta", {}).get("legal_moves", []) if game_record else []
        )
        # ------------------------------------------------------------------ #
        # 1) Thống kê loại combo trong legal moves (6 dims, chuẩn hoá theo
        #    tổng số legal moves để tránh scale quá lớn khi có nhiều lựa chọn).
        # ------------------------------------------------------------------ #
        combo_counts = [0] * 6
        for candidate in legal_moves:
            combo_type = candidate.get("combo_type", "pass")
            idx = self.combo_type_to_id.get(combo_type, -1)
            if combo_type != "pass" and 0 <= idx < 6:
                combo_counts[idx] += 1
        total_moves = max(1, len(legal_moves))
        combo_counts = [c / float(total_moves) for c in combo_counts]
        features.extend(combo_counts)

        # ------------------------------------------------------------------ #
        # 2) Số lá còn lại của từng đối thủ (4 dims), chuẩn hoá về [0, 1]
        #    với giả định tối đa ~13 lá mỗi người.
        # ------------------------------------------------------------------ #
        cards_left = list(game_record.get("cards_left", []))
        while len(cards_left) < 4:
            cards_left.append(0)
        norm_cards_left = [min(13.0, float(c)) / 13.0 for c in cards_left[:4]]
        features.extend(norm_cards_left)

        # ------------------------------------------------------------------ #
        # 3) Số lá trên tay (1 dim), chuẩn hoá về [0, 1].
        # ------------------------------------------------------------------ #
        hand = game_record.get("hand", []) or []
        features.append(min(13.0, float(len(hand))) / 13.0)

        combo_type = move.get("combo_type", "pass")
        onehot = [0.0] * len(self.combo_types)
        if combo_type in self.combo_type_to_id:
            onehot[self.combo_type_to_id[combo_type]] = 1.0
        features.extend(onehot)

        rank_value = move.get("rank_value", 0) or 0
        features.append(float(rank_value) / 12.0)

        cards = move.get("cards", []) or []
        features.append(len(cards) / 10.0)

        if combo_type == "pass" or not hand:
            efficiency = 0.0
        else:
            efficiency = len(cards) / max(1.0, float(len(hand)))
        features.append(efficiency)

        phase = self._infer_game_phase(cards_left, len(hand))
        urgency_map = {"early": 0.1, "mid": 0.5, "late": 1.0}
        features.append(urgency_map.get(phase, 0.1))

        # Blocking feature: 1.0 if opponent played before (need to block), 0.0 if we start
        is_blocking = self._is_blocking(game_record)
        # Dùng lại scale gốc để tín hiệu block đủ mạnh
        blocking_scale = 12.0
        features.append(is_blocking * blocking_scale)

        # Can beat last_move feature: 1.0 if move can beat last_move, 0.0 if cannot (or no last_move)
        can_beat = self._can_beat_last_move(move, game_record)
        # Dùng lại scale gốc để ưu tiên rõ nước chặn được
        can_beat_scale = 15.0
        features.append(can_beat * can_beat_scale)

        # ------------------------------------------------------------------ #
        # Optional: phân vùng nhẹ chiến thuật theo block / lead
        # - Khi block: efficiency_block = efficiency, efficiency_lead = 0
        # - Khi lead:  efficiency_block = 0,           efficiency_lead = efficiency
        # Nhờ đó policy có thể học hai cách dùng efficiency khác nhau cho 2 trạng thái.
        # ------------------------------------------------------------------ #
        if ENABLE_BLOCK_LEAD_GATING:
            efficiency_block = efficiency * is_blocking
            efficiency_lead = efficiency * (1.0 - is_blocking)
            features.append(efficiency_block)
            features.append(efficiency_lead)

        return features

    def _extract_framework_features(
        self, move: Dict[str, Any], framework: Dict[str, Any], game_record: Dict[str, Any]
    ) -> List[float]:
        """Framework-aware features (8 dims) dùng lại scale gốc (phiên bản cũ)."""
        features: List[float] = []
        # Dùng lại đúng các scale gốc đã note ở phần header:
        #   priority_scale       = 15.0
        #   breaking_scale       = 15.0
        #   strength_scale       = 8.0
        #   position_scale       = 12.0
        #   combo_type_scale     = 3.0
        #   rank_pref_scale      = 4.0
        #   timing_scale         = 3.0
        #   compliance_scale     = 16.0
        priority_scale = 5.0
        breaking_scale = 15.0
        strength_scale = 8.0
        position_scale = 12.0
        combo_type_scale = 3.0
        rank_pref_scale = 4.0
        timing_scale = 3.0
        compliance_scale = 16.0

        features.append(self._framework_priority_score(move, framework) * priority_scale)
        features.append(
            -self._framework_breaking_severity(move, framework) * breaking_scale
        )
        features.append(framework.get("framework_strength", 0.0) * strength_scale)
        features.append(self._framework_position(move, framework) * position_scale)
        features.append(
            self._combo_type_preference(move, framework) * combo_type_scale
        )
        features.append(self._rank_preference(move, framework) * rank_pref_scale)
        features.append(self._timing_preference(move, framework) * timing_scale)
        features.append(self._sequence_compliance(move, framework) * compliance_scale)
        
        # Tính urgency trước để điều chỉnh seq_order_penalty
        min_opp = None
        lead_urgency = 0.0
        if ENABLE_LEAD_QUALITY_FEATURES:
            cards_left = game_record.get("cards_left", []) or []
            current_player_id = game_record.get("current_player_id", 0)
            if cards_left and len(cards_left) > 1:
                opp_counts = [
                    c for idx, c in enumerate(cards_left) 
                    if idx != current_player_id and c > 0
                ]
                if opp_counts:
                    min_opp = min(opp_counts)
            
            # lead_urgency: mức độ cấp thiết để giật cái (0.0-1.0)
            if min_opp is not None:
                if min_opp <= 2:
                    lead_urgency = 1.0  # Rất cấp thiết: đối thủ sắp thắng
                elif min_opp <= 4:
                    lead_urgency = 0.7  # Cấp thiết: đối thủ gần thắng
                elif min_opp <= 6:
                    lead_urgency = 0.4  # Trung bình: cần chú ý
                else:
                    lead_urgency = 0.1  # Thấp: không cấp thiết
        
        # Sequence order penalty: penalty when playing move out of sequence order
        seq_order_penalty = self._sequence_order_penalty(move, framework)
        seq_order_penalty_scale = 20.0
        
        # Giảm penalty khi urgency cao, đặc biệt cho single 2 để giật cái
        if ENABLE_LEAD_QUALITY_FEATURES and lead_urgency > 0:
            combo_type = move.get("combo_type", "pass")
            rank_value = move.get("rank_value", 0) or 0
            is_single_two = combo_type == "single" and rank_value >= 12
            
            if is_single_two and lead_urgency >= 0.7:
                # Single 2 khi urgency cao: giảm penalty mạnh để cho phép giật cái
                penalty_reduction = lead_urgency * 0.8  # Giảm 56-80% khi urgency cao
                seq_order_penalty = seq_order_penalty * (1.0 - penalty_reduction)
            elif lead_urgency >= 0.7:
                # Các combo khác khi urgency cao: giảm penalty nhẹ
                penalty_reduction = lead_urgency * 0.4  # Giảm 28-40%
                seq_order_penalty = seq_order_penalty * (1.0 - penalty_reduction)
        
        features.append(seq_order_penalty * seq_order_penalty_scale)

        # ------------------------------------------------------------------ #
        # Optional: lead quality features (đánh giá lá/combo dùng để giật cái)
        # ------------------------------------------------------------------ #
        if ENABLE_LEAD_QUALITY_FEATURES:
            combo_type = move.get("combo_type", "pass")
            rank_value = move.get("rank_value", 0) or 0  # 12 thường là lá 2
            strength = framework.get("framework_strength", 0.0)

            # Đánh dấu combo chứa 2 mạnh (đôi 2, three 2, four of 2) – nên ưu tiên để block
            is_two_combo = combo_type in {"pair", "triple", "four_kind"} and rank_value >= 12
            is_single_two = combo_type == "single" and rank_value >= 12

            # Mức độ phá sequence: seq_order_penalty < 0 khi đánh sai thứ tự/đốt combo mạnh sớm
            break_amount = -min(0.0, seq_order_penalty)  # 0 nếu đúng khung, >0 nếu phá khung

            # lead_candidate_score: combo phù hợp để giật cái
            # Kết hợp cả chất lượng combo và urgency
            lead_candidate_score = 0.0
            if is_single_two:
                # Single 2: ứng viên đẹp để giật cái, urgency cao thì càng tốt
                lead_candidate_score = 0.6 + (lead_urgency * 0.4)  # 0.6-1.0
            elif strength >= 0.9 and break_amount > 0.0 and not is_two_combo:
                # Bộ rất mạnh, phá khung có chủ đích để mở đường xả
                lead_candidate_score = 0.5 + (lead_urgency * 0.3)  # 0.5-0.8
            elif combo_type in {"single", "pair"} and rank_value <= 3:
                # Combo yếu (3, 4, 5, 6) phù hợp để giật cái khi urgency cao
                lead_candidate_score = lead_urgency * 0.6  # 0.0-0.6

            # lead_waste_penalty: combo đốt tài nguyên block tốt (đôi/triple/four 2)
            # Penalty giảm khi urgency cao (cần giật cái gấp)
            lead_waste_penalty = 1.0 if is_two_combo else 0.0
            if lead_waste_penalty > 0 and lead_urgency >= 0.7:
                # Khi rất cấp thiết, giảm penalty cho đôi/triple/four 2
                lead_waste_penalty *= (1.0 - lead_urgency * 0.5)  # Giảm 35-50%

            # Scale tăng lên để có tác động rõ rệt hơn, cân bằng với các feature framework khác
            lead_candidate_scale = 15.0  # Tăng từ 12.0 để đảm bảo single 2 được ưu tiên
            lead_waste_scale = 12.0

            features.append(lead_candidate_score * lead_candidate_scale)
            features.append(-lead_waste_penalty * lead_waste_scale)

        return features

    def _extract_multi_sequence_features(
        self, move: Dict[str, Any], framework: Dict[str, Any]
    ) -> List[float]:
        """4 dims per sequence * 3 sequences = 12 dims."""
        features: List[float] = []
        all_sequences: List[List[Dict[str, Any]]] = [framework.get("core_combos", [])]
        alt_sequences = framework.get("alternative_sequences", []) or []
        for alt in alt_sequences:
            all_sequences.append(alt.get("sequence", []))

        for i in range(3):
            if i < len(all_sequences):
                seq = all_sequences[i]
                seq_framework = {
                    "core_combos": seq,
                    "framework_strength": (
                        framework.get("framework_strength", 0.0)
                        if i == 0
                        else alt_sequences[i - 1].get("total_strength", 0.0)
                        if i - 1 < len(alt_sequences)
                        else 0.0
                    ),
                    "recommended_moves": [
                        combo.get("cards", []) for combo in seq if combo.get("cards")
                    ],
                }
            else:
                seq_framework = {
                    "core_combos": [],
                    "framework_strength": 0.0,
                    "recommended_moves": [],
                }

            features.append(self._framework_priority_score(move, seq_framework) * 2.0)
            features.append(
                -self._framework_breaking_severity(move, seq_framework) * 2.0
            )
            features.append(self._framework_position(move, seq_framework) * 2.0)
            features.append(self._sequence_compliance(move, seq_framework) * 2.0)
        return features

    # ------------------------------------------------------------------ #
    # Framework helpers (borrowed from StyleLearner)
    # ------------------------------------------------------------------ #
    def _framework_priority_score(
        self, move: Dict[str, Any], framework: Dict[str, Any]
    ) -> float:
        move_cards = set(move.get("cards", []))
        for combo in framework.get("core_combos", []) or []:
            combo_cards = set(combo.get("cards", []))
            if move_cards.issubset(combo_cards):
                return combo.get("strength", 0.0)
        return 0.0

    def _framework_breaking_severity(
        self, move: Dict[str, Any], framework: Dict[str, Any]
    ) -> float:
        move_cards = set(move.get("cards", []))
        max_severity = 0.0
        for combo in framework.get("core_combos", []) or []:
            combo_cards = set(combo.get("cards", []))
            combo_type = combo.get("type", "")
            if not move_cards.intersection(combo_cards):
                continue
            if move_cards != combo_cards:
                if combo_type in {"four_kind", "double_seq", "straight"}:
                    max_severity = max(max_severity, 2.0)
                elif combo_type in {"triple"}:
                    max_severity = max(max_severity, 1.5)
                elif combo_type in {"pair"}:
                    max_severity = max(max_severity, 1.0)
        return max_severity

    def _sequence_compliance(
        self, move: Dict[str, Any], framework: Dict[str, Any]
    ) -> float:
        move_cards = set(move.get("cards", []))
        recommended_moves = framework.get("recommended_moves", []) or []
        if not recommended_moves:
            return 0.0

        for idx, rec_move in enumerate(recommended_moves):
            if set(rec_move) == move_cards:
                base = 1.0 - (idx / max(1, len(recommended_moves) - 1))
                return base
        for idx, rec_move in enumerate(recommended_moves):
            if move_cards.issubset(set(rec_move)):
                base = 1.0 - (idx / max(1, len(recommended_moves) - 1))
                return base * 0.5
        return 0.0

    def _framework_position(
        self, move: Dict[str, Any], framework: Dict[str, Any]
    ) -> float:
        move_cards = set(move.get("cards", []))
        combos = framework.get("core_combos", []) or []
        if not combos:
            return 0.0
        for idx, combo in enumerate(combos):
            if move_cards.issubset(set(combo.get("cards", []))):
                if len(combos) == 1:
                    return 1.0
                return 1.0 - (idx / (len(combos) - 1))
        return 0.0

    def _combo_type_preference(
        self, move: Dict[str, Any], framework: Dict[str, Any]
    ) -> float:
        move_type = move.get("combo_type", "pass")
        combos = framework.get("core_combos", []) or []
        if not combos:
            return 0.0
        counts: Dict[str, int] = {}
        for combo in combos:
            c_type = combo.get("type", "")
            counts[c_type] = counts.get(c_type, 0) + 1
        return counts.get(move_type, 0) / len(combos)

    def _rank_preference(
        self, move: Dict[str, Any], framework: Dict[str, Any]
    ) -> float:
        move_rank = move.get("rank_value", 0)
        combos = framework.get("core_combos", []) or []
        if not combos:
            return 0.0
        counts: Dict[int, int] = {}
        for combo in combos:
            rank = combo.get("rank_value", 0)
            counts[rank] = counts.get(rank, 0) + 1
        return counts.get(move_rank, 0) / len(combos)

    def _timing_preference(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        # Placeholder heuristic; can be replaced by smarter logic later.
        return 0.5

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def _infer_game_phase(
        self, cards_left: Optional[List[int]], hand_count: int
    ) -> str:
        total_cards_left = sum(cards_left or [])
        total_on_table = total_cards_left + hand_count
        if total_on_table >= 20:
            return "early"
        if total_on_table >= 8:
            return "mid"
        return "late"
    
    def _is_blocking(self, game_record: Dict[str, Any]) -> float:
        """
        Check if agent is blocking (someone played before).
        
        Returns:
            1.0 if last_move exists (need to block or pass)
            0.0 if last_move is None (new round, agent starts)
        """
        last_move = game_record.get("last_move")
        if last_move is None:
            # New round -> agent starts
            return 0.0
        # Someone played before -> agent needs to block or pass
        return 1.0
    
    def _can_beat_last_move(self, move: Dict[str, Any], game_record: Dict[str, Any]) -> float:
        """
        Check if move can beat last_move.
        Uses legal_moves to determine if any blocking moves are available.
        
        Returns:
            1.0 if move can beat last_move AND there are blocking moves available
            0.0 if cannot beat (pass, or no blocking moves available, or no last_move)
        """
        last_move = game_record.get("last_move")
        if last_move is None:
            # No last_move -> not blocking
            return 0.0
        
        # Check if there are any blocking moves available (not just pass)
        legal_moves = game_record.get("meta", {}).get("legal_moves", [])
        has_blocking_moves = False
        for lm in legal_moves:
            if lm.get("combo_type", "pass") != "pass":
                has_blocking_moves = True
                break
        
        # If only pass available -> cannot beat
        if not has_blocking_moves:
            return 0.0
        
        # Check if this specific move can beat last_move
        move_type = move.get("combo_type", "pass")
        if move_type == "pass":
            # Pass cannot beat
            return 0.0
        
        return 1.0
    
    def _sequence_order_penalty(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """
        Penalty for playing move out of sequence order.
        
        Returns:
            Negative value (penalty) if playing move that should come later in sequence
            0.0 if move is in correct order or no framework available
        """
        move_cards = set(move.get("cards", []))
        combos = framework.get("core_combos", []) or []
        if not combos or len(combos) <= 1:
            return 0.0
        
        # Find position of move in sequence
        move_position = None
        for idx, combo in enumerate(combos):
            combo_cards = set(combo.get("cards", []))
            if move_cards.issubset(combo_cards):
                move_position = idx
                break
        
        if move_position is None:
            # Move not in framework -> no penalty (might be blocking move)
            return 0.0
        
        # Penalty increases with position (later in sequence = higher penalty)
        # Also penalize based on rank_value if it's a single
        move_rank = move.get("rank_value", 0) or 0
        move_type = move.get("combo_type", "pass")
        
        # Base penalty: position in sequence (0.0 for first, 1.0 for last)
        position_penalty = move_position / max(1, len(combos) - 1)
        
        # Additional penalty if it's a strong single played early
        if move_type == "single" and move_rank >= 8:  # Strong cards (8+)
            rank_penalty = (move_rank / 12.0) * 0.5  # Additional 0-0.5 penalty
        else:
            rank_penalty = 0.0
        
        # Total penalty: negative value (will be scaled by -20.0, so becomes positive penalty)
        total_penalty = -(position_penalty + rank_penalty)
        
        return total_penalty

