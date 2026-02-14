from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ #
# Feature flags để dễ bật/tắt các logic tuỳ biến chiến thuật
# ------------------------------------------------------------------ #
ENABLE_BLOCK_LEAD_GATING = True  # Phân vùng nhẹ chiến thuật khi block vs khi mở vòng
ENABLE_LEAD_QUALITY_FEATURES = True  # Đánh giá chất lượng lá/combo dùng để giật cái
ENABLE_CARD_COUNTING_FEATURES = True  # Card counting and rank power features (8 dims)


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
        # Map 3 đôi thông và 4 đôi thông về cùng index với double_seq
        self.combo_type_to_id['three_consecutive_pairs'] = self.combo_type_to_id['double_seq']
        self.combo_type_to_id['four_consecutive_pairs'] = self.combo_type_to_id['double_seq']
        # Base dim: 23 context (22 original + 1 blocking + 1 can_beat)
        #         + 8 framework (7 original + 1 seq_order_penalty, removed timing_preference)
        #         + 12 multi-sequence + 1 = 44
        # Các flag có thể thêm dims:
        # - ENABLE_BLOCK_LEAD_GATING: +2 (efficiency_block, efficiency_lead)
        # - ENABLE_LEAD_QUALITY_FEATURES: +2 (lead_candidate_score, lead_waste_penalty)
        # - ENABLE_CARD_COUNTING_FEATURES: +8 (is_lead, curr_move_rank_power, is_unbeatable,
        #                                       top1/2/3_rank_power, lose_lead_prob, next_player_danger)
        base_dim = 44
        extra_dims = 0
        if ENABLE_BLOCK_LEAD_GATING:
            extra_dims += 2
        if ENABLE_LEAD_QUALITY_FEATURES:
            extra_dims += 2
        if ENABLE_CARD_COUNTING_FEATURES:
            extra_dims += 8
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
        card_counting = self._extract_card_counting_features(move, game_record)
        return original + framework_features + multi_seq + card_counting

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
        blocking_scale = 8.0  # Reduced from 12.0
        features.append(is_blocking * blocking_scale)

        # Can beat last_move feature: 1.0 if move can beat last_move, 0.0 if cannot (or no last_move)
        can_beat = self._can_beat_last_move(move, game_record)
        can_beat_scale = 8.0  # Reduced from 10.0
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
        # All scales normalized to <= 12.0
        priority_scale = 5.0  # Keep as is
        breaking_scale = 10.0  # Reduced from 15.0, raw value normalized [0, 1]
        strength_scale = 6.0  # Reduced from 8.0
        position_scale = 8.0  # Reduced from 12.0
        combo_type_scale = 3.0  # Keep as is
        rank_pref_scale = 4.0  # Keep as is
        # timing_scale removed - _timing_preference() is placeholder
        compliance_scale = 10.0  # Reduced from 16.0

        # Tính urgency TRƯỚC để có thể điều chỉnh breaking/compliance/seq_order_penalty
        min_opp = None
        lead_urgency = 0.0
        is_single_two = False
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
            
            # Check if single 2 để áp dụng penalty reduction mạnh hơn
            combo_type = move.get("combo_type", "pass")
            rank_value = move.get("rank_value", 0) or 0
            is_single_two = combo_type == "single" and rank_value >= 12

        # Tính breaking severity và compliance trước khi áp dụng reduction
        # breaking_severity is already normalized to [0, 1]
        breaking_severity = self._framework_breaking_severity(move, framework)
        compliance_value = self._sequence_compliance(move, framework)
        
        # Áp dụng penalty reduction cho breaking và compliance khi urgency cao
        breaking_reduction = 0.0
        compliance_reduction = 0.0
        if ENABLE_LEAD_QUALITY_FEATURES and lead_urgency >= 0.7:
            if is_single_two:
                # Single 2 khi urgency cao: giảm breaking và compliance penalty mạnh (50-70%)
                breaking_reduction = lead_urgency * 0.6  # 42-70% reduction
                compliance_reduction = lead_urgency * 0.6  # 42-70% reduction
            else:
                # Các combo khác khi urgency cao: giảm penalty nhẹ hơn (28-40%)
                breaking_reduction = lead_urgency * 0.3  # 21-40% reduction
                compliance_reduction = lead_urgency * 0.3  # 21-40% reduction
        
        # Apply reduction cho breaking severity (chỉ áp dụng khi penalty > 0)
        breaking_severity = breaking_severity * (1.0 - breaking_reduction) if breaking_severity > 0 else breaking_severity
        
        # Compliance: giảm penalty khi không comply
        # Compliance value từ 0 (không comply) đến 1 (comply đầy đủ)
        # Penalty = (1 - compliance) * scale
        # Nếu giảm penalty, ta tăng compliance value
        # compliance_new = 1.0 - (1.0 - compliance_old) * (1.0 - reduction)
        if compliance_value < 1.0 and compliance_reduction > 0:
            compliance_penalty = 1.0 - compliance_value  # Phần penalty hiện tại
            compliance_penalty_reduced = compliance_penalty * (1.0 - compliance_reduction)  # Giảm penalty
            compliance_value = 1.0 - compliance_penalty_reduced  # Compliance mới sau khi giảm penalty

        features.append(self._framework_priority_score(move, framework) * priority_scale)
        # breaking_severity is normalized [0, 1], apply as penalty (negative)
        features.append(-breaking_severity * breaking_scale)
        features.append(framework.get("framework_strength", 0.0) * strength_scale)
        features.append(self._framework_position(move, framework) * position_scale)
        features.append(
            self._combo_type_preference(move, framework) * combo_type_scale
        )
        features.append(self._rank_preference(move, framework) * rank_pref_scale)
        # _timing_preference() removed - was placeholder returning constant 0.5
        features.append(compliance_value * compliance_scale)
        
        # Sequence order penalty: penalty when playing move out of sequence order
        # Returns normalized [0, 1] where 1.0 = maximum penalty
        seq_order_penalty = self._sequence_order_penalty(move, framework)
        seq_order_penalty_scale = 12.0  # Reduced from 20.0, max scale
        
        # Giảm penalty khi urgency cao, đặc biệt cho single 2 để giật cái (90-95%)
        if ENABLE_LEAD_QUALITY_FEATURES and lead_urgency >= 0.7:
            if is_single_two:
                # Single 2 khi urgency cao: giảm penalty rất mạnh để cho phép giật cái (90-95%)
                penalty_reduction = 0.9 + (lead_urgency - 0.7) * 0.17  # 0.9-0.95 (90-95% reduction)
                seq_order_penalty = seq_order_penalty * (1.0 - penalty_reduction)
            else:
                # Các combo khác khi urgency cao: giảm penalty nhẹ (28-40%)
                penalty_reduction = lead_urgency * 0.4  # Giảm 28-40%
                seq_order_penalty = seq_order_penalty * (1.0 - penalty_reduction)
        
        # Apply as penalty (negative)
        features.append(-seq_order_penalty * seq_order_penalty_scale)

        # ------------------------------------------------------------------ #
        # Optional: lead quality features (đánh giá lá/combo dùng để giật cái)
        # CHỈ tính khi đang blocking và có thể beat (tình huống giật cái thực sự)
        # ------------------------------------------------------------------ #
        if ENABLE_LEAD_QUALITY_FEATURES:
            combo_type = move.get("combo_type", "pass")
            rank_value = move.get("rank_value", 0) or 0  # 12 thường là lá 2
            strength = framework.get("framework_strength", 0.0)

            # Check xem có đang ở tình huống giật cái không (blocking + có thể beat)
            is_blocking = self._is_blocking(game_record)
            can_beat = self._can_beat_last_move(move, game_record)
            is_lead_situation = is_blocking > 0.5 and can_beat > 0.5  # Đang blocking và có thể beat

            # Đánh dấu combo chứa 2 mạnh (đôi 2, three 2, four of 2) – nên ưu tiên để block
            is_two_combo = combo_type in {"pair", "triple", "four_kind"} and rank_value >= 12
            is_single_two = combo_type == "single" and rank_value >= 12

            # Mức độ phá sequence: seq_order_penalty < 0 khi đánh sai thứ tự/đốt combo mạnh sớm
            break_amount = -min(0.0, seq_order_penalty)  # 0 nếu đúng khung, >0 nếu phá khung

            # lead_candidate_score: combo phù hợp để giật cái
            # Simplified logic: raw score based on rank strength, modulated by urgency
            # Returns normalized [0, 1] range
            lead_candidate_score = 0.0
            if is_lead_situation:  # Chỉ tính khi đang blocking và có thể beat
                rank_ratio = rank_value / 12.0  # [0, 1] based on rank strength
                # Base score: rank strength modulated by urgency (minimum 0.3 to ensure some signal)
                lead_candidate_score = rank_ratio * max(0.3, lead_urgency)
                
                # Single 2 gets extra boost
                if is_single_two:
                    lead_candidate_score = min(1.0, lead_candidate_score + 0.3)
                
                # Penalty for weak cards (rank <= 3) when urgency is high
                if rank_value <= 3 and lead_urgency >= 0.7:
                    lead_candidate_score = 0.1  # Very low score for weak cards in urgent situations

            # lead_waste_penalty: phạt khi đốt tài nguyên block tốt (đôi/triple/four 2) KHÔNG phải để giật cái
            # CHỈ phạt khi KHÔNG ở tình huống giật cái (vì khi giật cái thì OK để đốt đôi 2)
            # Khi urgency cao, penalty giảm nhẹ (vì có thể sẽ cần giật cái sớm)
            lead_waste_penalty = 0.0
            if is_two_combo and not is_lead_situation:
                # Đốt đôi/triple/four 2 khi KHÔNG phải giật cái -> penalty
                if lead_urgency >= 0.7:
                    # Urgency cao: giảm penalty một chút (vì có thể sớm cần giật cái)
                    lead_waste_penalty = 0.6  # Giảm từ 1.0 xuống 0.6
                else:
                    lead_waste_penalty = 1.0  # Penalty đầy đủ khi urgency thấp

            # Scales normalized: all <= 12.0
            lead_candidate_scale = 10.0  # Reduced from 60.0, raw value is now [0, 1]
            lead_waste_scale = 8.0  # Reduced from 12.0

            # LUÔN add cả 2 features (với giá trị 0.0 khi không áp dụng)
            # Điều này giúp network học được sự khác biệt giữa tình huống giật cái và không giật cái
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
                # Include 3-4 đôi thông as high severity combos
                if combo_type in {"four_kind", "double_seq", "straight", "three_consecutive_pairs", "four_consecutive_pairs"}:
                    max_severity = max(max_severity, 2.0)
                elif combo_type in {"triple"}:
                    max_severity = max(max_severity, 1.5)
                elif combo_type in {"pair"}:
                    max_severity = max(max_severity, 1.0)
        # Normalize to [0, 1] range (max severity is 2.0)
        return max_severity / 2.0

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
        
        # Map 3-4 đôi thông to double_seq for preference matching
        normalized_move_type = move_type
        if move_type in {"three_consecutive_pairs", "four_consecutive_pairs"}:
            normalized_move_type = "double_seq"
        
        # Check direct match first
        if normalized_move_type in counts:
            return counts[normalized_move_type] / len(combos)
        
        # Check if framework has 3-4 đôi thông when move is double_seq
        if move_type == "double_seq":
            double_seq_count = counts.get("three_consecutive_pairs", 0) + counts.get("four_consecutive_pairs", 0)
            if double_seq_count > 0:
                return double_seq_count / len(combos)
        
        return 0.0

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
        # cards_left already includes all players (including agent), so no need to add hand_count
        total_on_table = sum(cards_left or [])
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
        
        # Total penalty: negative value (will be normalized and scaled)
        total_penalty = -(position_penalty + rank_penalty)
        
        # Normalize to [0, 1] range (max penalty magnitude is 1.5)
        # Return as positive value [0, 1] where 1.0 = maximum penalty
        normalized_penalty = abs(total_penalty) / 1.5
        
        return normalized_penalty
    
    # ------------------------------------------------------------------ #
    # Card Counting Features (8 dims)
    # ------------------------------------------------------------------ #
    def _get_rank_power(self, rank: int, agent_hand: List[int], seen_ranks: List[int]) -> int:
        """
        Calculate how many cards larger than `rank` are still unknown (not in agent hand, not seen).
        
        Args:
            rank: Rank of the card (0-12, where 12=2)
            agent_hand: List of card_ids in agent's hand
            seen_ranks: List of 13 elements, each = number of cards of that rank seen (played)
        
        Returns:
            Number of larger cards remaining (unknown to agent)
        """
        if rank >= 12:  # Rank 12 (2) is highest
            return 0
        
        # Count cards per rank in agent hand
        agent_hand_ranks = [0] * 13
        for card_id in agent_hand:
            r = card_id % 13
            agent_hand_ranks[r] += 1
        
        # Calculate remaining larger cards = total - seen - in_agent_hand
        larger_cards_left = 0
        for r in range(rank + 1, 13):
            total_for_rank = 4
            seen_count = seen_ranks[r] if r < len(seen_ranks) else 0
            in_agent_hand = agent_hand_ranks[r]
            remaining = total_for_rank - seen_count - in_agent_hand
            if remaining < 0:
                # Log warning: tracking error (should never happen if logic is correct)
                import logging
                logging.warning(f"Rank {r} remaining < 0: seen={seen_count}, in_hand={in_agent_hand}")
            larger_cards_left += max(0, remaining)
        
        return larger_cards_left
    
    def _get_rank_power_normalized(self, rank: int, agent_hand: List[int], seen_ranks: List[int]) -> float:
        """
        Normalized rank power [0.0-1.0].
        
        Returns:
            0.0 = No larger cards remaining (unbeatable)
            1.0 = Maximum larger cards remaining (weakest)
        """
        if rank >= 12:
            return 0.0
        
        # Calculate max_possible (subtract cards in agent hand)
        agent_hand_ranks = [0] * 13
        for card_id in agent_hand:
            r = card_id % 13
            agent_hand_ranks[r] += 1
        
        max_possible = 0
        for r in range(rank + 1, 13):
            max_possible += (4 - agent_hand_ranks[r])  # Subtract cards in agent hand
        
        if max_possible == 0:
            return 0.0
        
        raw_power = self._get_rank_power(rank, agent_hand, seen_ranks)
        return min(1.0, raw_power / max_possible)
    
    def _get_top_3_ranks_in_hand(self, hand: List[int]) -> List[int]:
        """
        Get Top-3 ranks in hand, prioritizing:
        1. Rank 12 (2) always first
        2. Number of cards (prefer pairs, triples)
        3. Rank value (higher is better)
        
        Returns:
            List of up to 3 ranks (may be < 3 if hand has fewer unique ranks)
        """
        if not hand:
            return []
        
        rank_counts: Dict[int, int] = {}
        for card_id in hand:
            rank = card_id % 13
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Sort: (1) Rank 12 first, (2) More cards, (3) Higher rank
        # key returns (True/False, count, rank) - True > False in Python
        sorted_ranks = sorted(
            rank_counts.keys(),
            key=lambda r: (r == 12, rank_counts[r], r),
            reverse=True
        )
        
        return sorted_ranks[:3]
    
    def _calculate_lose_lead_probability(
        self,
        move_rank: int,
        agent_hand: List[int],
        seen_ranks: List[int],
        cards_left: List[int]
    ) -> float:
        """
        Calculate probability of losing lead if playing this move.
        
        Formula: remaining_larger_cards / total_remaining_cards
        
        Returns:
            [0.0-1.0] probability of losing lead
            0.0 = Safe (no larger cards remaining)
            1.0 = Very risky
        """
        # Calculate remaining larger cards
        remaining_larger = self._get_rank_power(move_rank, agent_hand, seen_ranks)
        
        if remaining_larger == 0:
            return 0.0  # No larger cards → safe
        
        # Total remaining cards on table (all players except agent)
        total_remaining = sum(cards_left) if cards_left else 0
        
        if total_remaining == 0:
            return 0.0
        
        # Simple probability
        prob = min(1.0, remaining_larger / total_remaining)
        return prob
    
    def _extract_card_counting_features(
        self, move: Dict[str, Any], game_record: Dict[str, Any]
    ) -> List[float]:
        """
        Extract card counting features (8 dims):
        1. is_lead: 1.0 if agent has lead (no last_move), 0.0 if blocking
        2. curr_move_rank_power: Normalized rank power of current move [0.0-1.0]
        3. is_unbeatable: 1.0 if rank_power == 0 (no larger cards), 0.0 otherwise
        4. top1_rank_power: Rank power of highest rank in hand
        5. top2_rank_power: Rank power of 2nd highest rank in hand
        6. top3_rank_power: Rank power of 3rd highest rank in hand
        7. lose_lead_prob: Probability of losing lead [0.0-1.0]
        8. next_player_danger: HIGH when next player has FEW cards (about to win) [0.0-1.0]
        
        All features are scaled appropriately before returning.
        """
        features: List[float] = []
        
        if not ENABLE_CARD_COUNTING_FEATURES:
            return [0.0] * 8
        
        # Get seen_ranks and agent hand from game_record
        seen_ranks = game_record.get("seen_ranks", [0] * 13)
        agent_hand = game_record.get("hand", []) or []
        cards_left = game_record.get("cards_left", []) or []
        current_player_id = game_record.get("current_player_id", 0)
        
        # Feature scales normalized to <= 12.0
        is_lead_scale = 8.0  # Reduced from 10.0
        curr_move_rank_power_scale = 8.0  # Reduced from 12.0
        is_unbeatable_scale = 12.0  # Important signal, keep high but within limit
        top1_rank_power_scale = 6.0  # Reduced from 10.0
        top2_rank_power_scale = 5.0  # Reduced from 8.0
        top3_rank_power_scale = 4.0  # Reduced from 6.0
        lose_lead_prob_scale = 8.0  # Reduced from 12.0
        next_player_danger_scale = 8.0  # Reduced from 10.0
        
        # 1. is_lead: 1.0 if agent has lead (no last_move)
        last_move = game_record.get("last_move")
        is_lead = 1.0 if last_move is None else 0.0
        features.append(is_lead * is_lead_scale)
        
        # 2. curr_move_rank_power
        combo_type = move.get("combo_type", "pass")
        move_rank = move.get("rank_value", 0) or 0
        if combo_type == "pass" or not agent_hand:
            curr_move_rank_power = 0.0
        else:
            curr_move_rank_power = self._get_rank_power_normalized(
                move_rank, agent_hand, seen_ranks
            )
        features.append(curr_move_rank_power * curr_move_rank_power_scale)
        
        # 3. is_unbeatable: 1.0 if rank_power == 0 and not pass
        is_unbeatable = 1.0 if curr_move_rank_power == 0.0 and combo_type != "pass" else 0.0
        features.append(is_unbeatable * is_unbeatable_scale)
        
        # 4-6. Top-3 rank powers
        if agent_hand:
            top_3_ranks = self._get_top_3_ranks_in_hand(agent_hand)
            top_3_powers = [
                self._get_rank_power_normalized(rank, agent_hand, seen_ranks)
                for rank in top_3_ranks
            ]
            # Padding if less than 3 ranks
            while len(top_3_powers) < 3:
                top_3_powers.append(1.0)  # Max rank power = weakest (no strong cards)
            
            features.append(top_3_powers[0] * top1_rank_power_scale)
            features.append(top_3_powers[1] * top2_rank_power_scale)
            features.append(top_3_powers[2] * top3_rank_power_scale)
        else:
            # Empty hand: default to weakest
            features.append(1.0 * top1_rank_power_scale)
            features.append(1.0 * top2_rank_power_scale)
            features.append(1.0 * top3_rank_power_scale)
        
        # 7. lose_lead_prob
        if combo_type == "pass" or not agent_hand:
            lose_lead_prob = 0.0
        else:
            lose_lead_prob = self._calculate_lose_lead_probability(
                move_rank, agent_hand, seen_ranks, cards_left
            )
        features.append(lose_lead_prob * lose_lead_prob_scale)
        
        # 8. next_player_danger: HIGH when opponent has FEW cards (about to win)
        # INVERTED logic: low cards = high danger (opponent close to winning!)
        num_players = len(cards_left) if cards_left else 4
        next_player_id = (current_player_id + 1) % num_players
        next_player_cards = cards_left[next_player_id] if next_player_id < len(cards_left) else 13
        
        if next_player_cards <= 2:
            next_player_danger = 1.0  # Maximum danger - opponent about to win!
        elif next_player_cards <= 4:
            next_player_danger = 0.8  # High danger
        elif next_player_cards <= 6:
            next_player_danger = 0.5  # Medium danger
        else:
            next_player_danger = max(0.0, 1.0 - (next_player_cards / 13.0))
        
        features.append(next_player_danger * next_player_danger_scale)
        
        return features

