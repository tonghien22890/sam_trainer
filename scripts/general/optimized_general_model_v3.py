"""
Optimized General Gameplay Model V3 - Implementation theo RANK_COMBO_DISCUSSION.md
Stage 1: Combo Type Selection (Decision Tree) - 12 dims
Stage 2: Card Selection (XGBoost) - 4 features + 2 labels
"""

import numpy as np
import joblib
from typing import Dict, List, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb


class OptimizedGeneralModelV3:
    """
    Optimized General Gameplay Model V3 với features tối ưu:
    - Stage 1: Decision Tree - 12 dims (legal_moves_combo_counts + cards_left + hand_count + combo_strength)
    - Stage 2: XGBoost - 4 features (combo_type + combo_strength_ranking + cards_left + hand_count)
    """
    
    def __init__(self):
        self.stage1_model = None
        self.stage2_model = None
        self.stage1_candidate_model = None  # Per-candidate Stage 1 model (ranking)
        self.combo_types = ["single", "pair", "triple", "four_kind", "straight", "double_seq", "pass"]
        self.combo_type_to_id = {ct: i for i, ct in enumerate(self.combo_types)}

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
                
                # Calculate strength based on combo type
                if combo_type == "single":
                    # Single: 2, A, Phần còn lại (đánh từ bé đến lớn)
                    if rank_value == 1:  # 2
                        strength = 3.0
                    elif rank_value == 0:  # A
                        strength = 2.0
                    else:  # Phần còn lại
                        strength = 1.0 + (rank_value - 2) / 10.0  # 3-K: 1.0-1.9
                        
                elif combo_type == "pair":
                    # Pair: 2, A, Mặt người (J,Q,K), Phần còn lại
                    if rank_value == 1:  # 2
                        strength = 4.0
                    elif rank_value == 0:  # A
                        strength = 3.0
                    elif rank_value >= 10:  # J, Q, K (mặt người)
                        strength = 2.5
                    else:  # Phần còn lại
                        strength = 2.0 + (rank_value - 2) / 8.0  # 3-10: 2.0-2.875
                        
                elif combo_type == "triple":
                    # Triple: 2, A, >= 7, Phần còn lại
                    if rank_value == 1:  # 2
                        strength = 5.0
                    elif rank_value == 0:  # A
                        strength = 4.0
                    elif rank_value >= 6:  # >= 7 (7,8,9,10,J,Q,K)
                        strength = 3.5
                    else:  # Phần còn lại (3,4,5,6)
                        strength = 3.0 + (rank_value - 2) / 4.0  # 3-6: 3.0-3.75
                        
                elif combo_type == "four_kind":
                    # Four_kind: A và phần còn lại (2 thì thắng luôn)
                    if rank_value == 1:  # 2 - thắng luôn
                        strength = 10.0  # Cực mạnh
                    elif rank_value == 0:  # A
                        strength = 9.0
                    else:  # Phần còn lại
                        strength = 8.0 + (rank_value - 2) / 11.0  # 3-K: 8.0-8.82
                        
                elif combo_type == "straight":
                    # Straight: Dây chạm A thì tối đa sức mạnh
                    has_ace = any(r == 0 for r in ranks)  # rank-only Ace
                    length = len(ranks)
                    
                    if has_ace:
                        strength = 7.0 + length / 10.0  # A straight: 7.5-8.0
                    else:
                        strength = 6.0 + length / 10.0 + (rank_value / 13.0) * 0.5  # Other: 6.5-7.0
                        
                elif combo_type == "double_seq":
                    # Double_seq: Cực mạnh, vượt trội
                    length = len(ranks)
                    strength = 9.0 + length / 10.0  # 9.5-10.0
                    
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
    
    def extract_stage1_features(self, record: Dict[str, Any]) -> np.ndarray:
        """Extract features for Stage 1 (combo_type selection) - 12 dims"""
        
        # Legal moves combo counts (6 dims)
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        combo_counts = self.extract_legal_moves_combo_counts(legal_moves)
        
        # Cards left (4 dims - số bài trên tay của từng người)
        cards_left = record.get("cards_left", [])
        # Pad or truncate to 4 dimensions
        while len(cards_left) < 4:
            cards_left.append(0)
        cards_left = cards_left[:4]
        
        # Hand count (1 dim - số bài trên tay của mình)
        hand = record.get("hand", [])
        hand_count = len(hand)
        
        # Combo strength relative (1 dim)
        combo_strength_relative = self.calculate_combo_strength_relative(legal_moves)
        
        # Combine all features
        features = np.concatenate([
            combo_counts,                # 6 dims
            cards_left,                  # 4 dims
            [hand_count],                # 1 dim
            [combo_strength_relative]    # 1 dim
        ])
        
        return features.astype(np.float32)
    
    def extract_stage2_features(self, record: Dict[str, Any], combo_type: str) -> Dict[str, Any]:
        """Extract features for Stage 2 (card selection) - 4 features"""
        
        # 1. Combo type (index encoding, not one-hot)
        combo_type_id = self.combo_type_to_id.get(combo_type, 6)  # 6 = "pass"
        
        # 2. Combo strength ranking (của moves thuộc combo_type)
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        filtered_moves = [move for move in legal_moves if move.get("combo_type") == combo_type]
        combo_strength_ranking = self.calculate_combo_strength_ranking(filtered_moves)
        
        # 3. Cards left (số bài còn lại trên tay của từng người)
        cards_left = record.get("cards_left", [])  # [9, 8, 3, 3]
        # Pad or truncate to 4 dimensions
        while len(cards_left) < 4:
            cards_left.append(0)
        cards_left = cards_left[:4]
        
        # 4. Hand count (số bài trên tay của mình)
        hand = record.get("hand", [])
        hand_count = len(hand)  # 8
        
        return {
            'combo_type': combo_type_id,
            'combo_strength_ranking': combo_strength_ranking,
            'cards_left': cards_left,      # Số bài trên tay của từng người
            'hand_count': hand_count       # Số bài trên tay của mình
        }
    
    def encode_stage2_features_for_model(self, features: Dict[str, Any]) -> np.ndarray:
        """Encode Stage 2 features for model training - 4 features"""
        
        # 1. Combo type (1 dim - index)
        combo_type_id = features['combo_type']
        
        # 2. Combo strength ranking - encode as list of strengths
        ranking = features['combo_strength_ranking']
        if ranking:
            # Use top 3 strengths (pad with 0 if less than 3)
            strengths = [item['strength'] for item in ranking[:3]]
            while len(strengths) < 3:
                strengths.append(0.0)
        else:
            strengths = [0.0, 0.0, 0.0]
        
        # 3. Cards left (4 dims - số bài trên tay của từng người)
        cards_left = features['cards_left']
        
        # 4. Hand count (1 dim - số bài trên tay của mình)
        hand_count = features['hand_count']
        
        # Combine all features
        encoded_features = np.concatenate([
            [combo_type_id],                  # 1 dim
            strengths,                        # 3 dims
            cards_left,                       # 4 dims
            [hand_count]                      # 1 dim
        ])
        
        return encoded_features.astype(np.float32)

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
            max_len = 0
            cur = 0
            # A(0) to K(12)
            for r in range(13):
                if counts.get(r, 0) > 0:
                    cur += 1
                    max_len = max(max_len, cur)
                else:
                    cur = 0
            return max_len

        def exists_double_sequence(counts: Dict[int, int]) -> bool:
            # Need at least two consecutive ranks with count>=2 (length >= 2 pairs => 4 cards)
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

        # simulate removal
        after_counts = dict(rank_counts)
        for c in move_cards:
            r = c % 13
            if after_counts.get(r, 0) > 0:
                after_counts[r] -= 1

        after_straight = longest_straight_length(after_counts)
        after_double = exists_double_sequence(after_counts)

        if before_straight >= 5 and after_straight < before_straight:
            severity = max(severity, 1)
        if before_double and not after_double:
            severity = max(severity, 2)
        return severity

    def extract_stage1_candidate_features(self, record: Dict[str, Any], move: Dict[str, Any]) -> np.ndarray:
        """Extract per-candidate features for Stage 1 ranking.
        General (12 dims) + combo-specific (~8-13 dims simplified):
        - combo_type one-hot (7)
        - rank_category (1)
        - combo_length (1)
        - breaks_combo_flag (1)
        Total dims: 12 + 7 + 1 + 1 + 1 = 22
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
        combo_strength_relative = self.calculate_combo_strength_relative(legal_moves)

        general_features = np.array([
            *combo_counts,                # 6
            *cards_left,                  # 4
            float(hand_count),            # 1
            float(combo_strength_relative)  # 1
        ], dtype=np.float32)

        # Combo-specific features (rank-only where relevant)
        combo_type = move.get("combo_type", "pass") or "pass"
        rank_value = int(move.get("rank_value", -1)) if move.get("rank_value") is not None else -1
        cards = move.get("cards", []) or []

        one_hot = np.zeros(len(self.combo_types), dtype=np.float32)
        ct_id = self.combo_type_to_id.get(combo_type, self.combo_type_to_id["pass"])
        one_hot[ct_id] = 1.0

        rank_category = float(self._compute_rank_category(combo_type, rank_value, cards))
        combo_length = float(self._compute_combo_length(combo_type, cards))
        breaks_flag = float(self._breaks_combo_flag(hand, cards))

        combo_specific = np.concatenate([
            one_hot,                              # 7
            np.array([rank_category], np.float32),# 1
            np.array([combo_length], np.float32), # 1
            np.array([breaks_flag], np.float32)   # 1
        ])

        return np.concatenate([general_features, combo_specific]).astype(np.float32)

    def build_stage1_candidate_dataset(self, records: List[Dict[str, Any]]):
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
                feat = self.extract_stage1_candidate_features(record, move)
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

    def train_stage1_candidates(self, records: List[Dict[str, Any]], model_type: str = "xgb"):
        """Train per-candidate Stage 1 model.
        model_type: "dt" | "rf" | "xgb"
        Returns sample-level training accuracy.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier as SkDT

        X, y, _ = self.build_stage1_candidate_dataset(records)
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
        self.stage1_candidate_model = model

        y_pred = model.predict(X)
        acc = (y_pred == y).mean().item()
        return float(acc)

    def evaluate_stage1_candidates(self, records: List[Dict[str, Any]]):
        """Evaluate per-candidate model with per-turn top-1 accuracy."""
        X, y, groups = self.build_stage1_candidate_dataset(records)
        if X.shape[0] == 0 or not groups or self.stage1_candidate_model is None:
            return {"turn_accuracy": 0.0, "num_turns": 0, "num_samples": int(X.shape[0])}

        probs = self.stage1_candidate_model.predict_proba(X)[:, 1]
        correct = 0
        for turn_indices in groups:
            # Find max prob in this turn
            best_idx = max(turn_indices, key=lambda i: probs[i])
            if y[best_idx] == 1:
                correct += 1
        turn_accuracy = correct / len(groups) if groups else 0.0
        return {"turn_accuracy": float(turn_accuracy), "num_turns": len(groups), "num_samples": int(X.shape[0])}

    def evaluate_stage1_candidates_topk(self, records: List[Dict[str, Any]], k: int = 3):
        """Evaluate per-candidate model with per-turn top-k accuracy."""
        X, y, groups = self.build_stage1_candidate_dataset(records)
        if X.shape[0] == 0 or not groups or self.stage1_candidate_model is None:
            return {"turn_topk_accuracy": 0.0, "k": k, "num_turns": 0, "num_samples": int(X.shape[0])}

        probs = self.stage1_candidate_model.predict_proba(X)[:, 1]
        correct = 0
        for turn_indices in groups:
            # top-k indices by probability
            sorted_indices = sorted(turn_indices, key=lambda i: probs[i], reverse=True)
            topk = sorted_indices[:max(1, k)]
            if any(y[i] == 1 for i in topk):
                correct += 1
        turn_topk_accuracy = correct / len(groups) if groups else 0.0
        return {"turn_topk_accuracy": float(turn_topk_accuracy), "k": k, "num_turns": len(groups), "num_samples": int(X.shape[0])}

    def compare_stage1_candidate_models(self, records: List[Dict[str, Any]]):
        """Train/evaluate DT, RF, XGB for per-candidate Stage 1 and return metrics."""
        results = {}
        for mt in ["dt", "rf", "xgb"]:
            acc = self.train_stage1_candidates(records, model_type=mt)
            eval_res = self.evaluate_stage1_candidates(records)
            eval_top3 = self.evaluate_stage1_candidates_topk(records, k=3)
            results[mt] = {
                "sample_acc": acc,
                "turn_accuracy": eval_res.get("turn_accuracy", 0.0),
                "turn_top3": eval_top3.get("turn_topk_accuracy", 0.0),
                "num_turns": eval_res.get("num_turns", 0),
                "num_samples": eval_res.get("num_samples", 0),
            }
        return results
    
    def train_stage1(self, records: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Train Stage 1 model with regularization to prevent overfitting"""
        
        X_list = []
        y_list = []
        
        for record in records:
            # Only train when last_move is pass (conditional approach)
            last_move = record.get("last_move")
            if last_move and last_move.get("combo_type"):
                continue  # Skip when có combo trước đó
            
            # Extract features
            features = self.extract_stage1_features(record)
            X_list.append(features)
            
            # Extract label (combo_type from stage1)
            action = record.get("action", {})
            stage1 = action.get("stage1", {})
            combo_type = stage1.get("value", "pass")
            
            # Convert combo_type to label
            if combo_type in self.combo_type_to_id:
                label = self.combo_type_to_id[combo_type]
            else:
                label = 6  # "pass" is index 6
            
            y_list.append(label)
        
        if len(X_list) == 0:
            return 0.0, {}
        
        X = np.stack(X_list, axis=0)
        y = np.array(y_list)
        
        # Train model with regularization to prevent overfitting
        self.stage1_model = DecisionTreeClassifier(
            max_depth=12,           # Tăng depth để học phức tạp hơn
            min_samples_split=15,   # Tăng để yêu cầu nhiều samples hơn
            min_samples_leaf=8,     # Tăng để tránh overfitting
            criterion='entropy',
            random_state=42
        )
        self.stage1_model.fit(X, y)
        
        # Evaluate
        y_pred = self.stage1_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Get classification report
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        return accuracy, report
    
    def train_stage2(self, records: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Train Stage 2 model with XGBoost"""
        
        X_list = []
        y_list = []
        
        for record in records:
            # Only train on non-pass moves
            action = record.get("action", {})
            stage1 = action.get("stage1", {})
            stage2 = action.get("stage2", {})
            
            combo_type = stage1.get("value", "pass")
            if combo_type == "pass":
                continue
            
            # Extract features
            features = self.extract_stage2_features(record, combo_type)
            encoded_features = self.encode_stage2_features_for_model(features)
            X_list.append(encoded_features)
            
            # Extract label (index in legal_moves) - rank-based comparison
            legal_moves = record.get("meta", {}).get("legal_moves", [])
            chosen_combo_type = stage2.get("combo_type")
            chosen_rank_value = stage2.get("rank_value")
            
            # Find index of chosen move by combo_type and rank_value
            label = -1
            for i, move in enumerate(legal_moves):
                if (move.get("combo_type") == chosen_combo_type and 
                    move.get("rank_value") == chosen_rank_value):
                    label = i
                    break
            
            if label == -1:
                continue  # Skip if not found
            
            y_list.append(label)
        
        if len(X_list) == 0:
            return 0.0, {}
        
        X = np.stack(X_list, axis=0)
        y = np.array(y_list)
        
        # Train XGBoost model with regularization
        self.stage2_model = xgb.XGBClassifier(
            max_depth=6,                # Moderate depth
            learning_rate=0.1,          # Standard learning rate
            n_estimators=100,           # Number of trees
            subsample=0.8,              # Subsample ratio
            colsample_bytree=0.8,       # Feature sampling ratio
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            random_state=42,
            eval_metric='mlogloss'
        )
        self.stage2_model.fit(X, y)
        
        # Evaluate
        y_pred = self.stage2_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Get classification report
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        return accuracy, report
    
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predict move using per-candidate approach"""
        
        if self.stage1_candidate_model is None:
            raise ValueError("Per-candidate model not trained yet")
        
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        if not legal_moves:
            return {"type": "pass", "cards": []}
        
        # Use per-candidate model to rank all legal moves
        move_scores = []
        
        for move in legal_moves:
            # Extract features for this candidate move
            features = self.extract_stage1_candidate_features(record, move)
            
            # Get probability score from per-candidate model
            prob = self.stage1_candidate_model.predict_proba(features.reshape(1, -1))[0, 1]
            
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
        """Save trained models"""
        model_data = {
            'stage1_model': self.stage1_model,
            'stage2_model': self.stage2_model,
            'stage1_candidate_model': self.stage1_candidate_model,
            'combo_types': self.combo_types,
            'combo_type_to_id': self.combo_type_to_id
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        self.stage1_model = model_data['stage1_model']
        self.stage2_model = model_data['stage2_model']
        self.stage1_candidate_model = model_data.get('stage1_candidate_model')
        self.combo_types = model_data['combo_types']
        self.combo_type_to_id = model_data['combo_type_to_id']


