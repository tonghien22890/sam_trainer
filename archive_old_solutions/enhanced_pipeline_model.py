"""
Enhanced Pipeline Model - Simplified approach
Chỉ học phong cách người chơi từ training data
"""

import numpy as np
import joblib
from typing import Dict, List, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class EnhancedPipelineModel:
    """
    Enhanced Pipeline Model với features đơn giản hơn:
    - Hand one-hot (52 dims)
    - Cards left của từng người chơi (4 dims)
    - Last move combo type (6 dims)
    - Last move rank (1 dim)
    - Hand combo analysis (6 dims)
    Total: 69 dims
    """
    
    def __init__(self):
        self.stage1_model = None
        self.stage2_model = None
        self.combo_types = ["single", "pair", "triple", "four_kind", "straight", "double_seq", "pass"]
        self.combo_type_to_id = {ct: i for i, ct in enumerate(self.combo_types)}
        
    def analyze_hand_combos(self, hand: List[int]) -> Dict[str, int]:
        """Analyze combos available in hand"""
        combo_counts = {
            "single": 0,
            "pair": 0,
            "triple": 0,
            "four_kind": 0,
            "straight": 0,
            "double_seq": 0
        }
        
        # Count ranks
        rank_counts = {}
        for card_id in hand:
            if 0 <= card_id < 52:
                rank = card_id % 13
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Count combos
        for rank, count in rank_counts.items():
            if count >= 1:
                combo_counts["single"] += 1
            if count >= 2:
                combo_counts["pair"] += 1
            if count >= 3:
                combo_counts["triple"] += 1
            if count >= 4:
                combo_counts["four_kind"] += 1
        
        # Check for straights (simplified)
        ranks = sorted(rank_counts.keys())
        for i in range(len(ranks) - 4):
            if ranks[i+4] - ranks[i] == 4:  # 5 consecutive ranks
                combo_counts["straight"] += 1
                break
        
        return combo_counts
    
    def extract_stage1_features(self, record: Dict[str, Any]) -> np.ndarray:
        """Extract features for Stage 1 (combo_type selection)"""
        
        # Hand one-hot (52 dims)
        hand = record.get("hand", [])
        hand_oh = np.zeros(52, dtype=np.float32)
        for card_id in hand:
            if 0 <= card_id < 52:
                hand_oh[card_id] = 1.0
        
        # Cards left của từng người chơi (4 dims)
        cards_left = record.get("cards_left", [])
        total_cards = sum(cards_left) if cards_left else 0
        if total_cards > 0:
            cards_left_normalized = [count / total_cards for count in cards_left]
        else:
            cards_left_normalized = [0.0] * 4  # Default 4 players
        
        # Pad or truncate to 4 dimensions
        while len(cards_left_normalized) < 4:
            cards_left_normalized.append(0.0)
        cards_left_normalized = cards_left_normalized[:4]
        
        # Last move combo type (7 dims)
        last_move_combo_oh = np.zeros(7, dtype=np.float32)
        last_move = record.get("last_move")
        if last_move and last_move.get("combo_type"):
            combo_type = last_move.get("combo_type")
            if combo_type in self.combo_type_to_id:
                last_move_combo_oh[self.combo_type_to_id[combo_type]] = 1.0
        
        # Last move rank (1 dim)
        last_move_rank = -1.0
        if last_move and last_move.get("rank_value") is not None:
            try:
                last_move_rank = float(last_move.get("rank_value"))
            except:
                last_move_rank = -1.0
        
        # Hand combo analysis (6 dims)
        combo_counts = self.analyze_hand_combos(hand)
        combo_features = [
            combo_counts["single"],
            combo_counts["pair"],
            combo_counts["triple"],
            combo_counts["four_kind"],
            combo_counts["straight"],
            combo_counts["double_seq"]
        ]
        
        # Combine all features
        features = np.concatenate([
            hand_oh,                    # 52 dims
            cards_left_normalized,      # 4 dims
            last_move_combo_oh,         # 7 dims
            np.array([last_move_rank], dtype=np.float32),  # 1 dim
            combo_features              # 6 dims
        ])
        
        return features
    
    def extract_stage2_features(self, record: Dict[str, Any], chosen_combo_type: str) -> np.ndarray:
        """Extract features for Stage 2 (card selection)"""
        
        # All stage1 features
        stage1_features = self.extract_stage1_features(record)
        
        # Chosen combo type one-hot (7 dims)
        chosen_combo_oh = np.zeros(7, dtype=np.float32)
        if chosen_combo_type in self.combo_type_to_id:
            chosen_combo_oh[self.combo_type_to_id[chosen_combo_type]] = 1.0
        
        # Available moves count for this combo type (1 dim)
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        available_moves_count = sum(1 for move in legal_moves if move.get("combo_type") == chosen_combo_type)
        
        # Combine all features
        features = np.concatenate([
            stage1_features,            # 70 dims (52+4+7+1+6)
            chosen_combo_oh,            # 7 dims
            np.array([available_moves_count], dtype=np.float32),  # 1 dim
        ])
        
        return features
    
    def train_stage1(self, records: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Train Stage 1 model (combo_type selection)"""
        
        X_list = []
        y_list = []
        
        for record in records:
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
        
        X = np.stack(X_list, axis=0)
        y = np.array(y_list)
        
        # Train model with better parameters
        self.stage1_model = DecisionTreeClassifier(
            max_depth=12,           # Tăng depth để học pattern phức tạp hơn
            min_samples_split=8,    # Giảm để học được patterns nhỏ
            min_samples_leaf=4,     # Giảm để không miss patterns
            criterion='entropy',     # Entropy thường tốt hơn cho classification
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
        """Train Stage 2 model (card selection)"""
        
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
            X_list.append(features)
            
            # Extract label (index in legal_moves)
            legal_moves = record.get("meta", {}).get("legal_moves", [])
            chosen_cards = stage2.get("cards", [])
            
            # Find index of chosen move
            label = -1
            for i, move in enumerate(legal_moves):
                if move.get("cards") == chosen_cards:
                    label = i
                    break
            
            if label == -1:
                continue  # Skip if not found
            
            y_list.append(label)
        
        if len(X_list) == 0:
            return 0.0, {}
        
        X = np.stack(X_list, axis=0)
        y = np.array(y_list)
        
        # Train model with better parameters
        self.stage2_model = DecisionTreeClassifier(
            max_depth=10,           # Tăng depth để học pattern phức tạp hơn
            min_samples_split=5,    # Giảm để học được patterns nhỏ
            min_samples_leaf=2,     # Giảm để không miss patterns
            criterion='entropy',     # Entropy thường tốt hơn cho classification
            random_state=42
        )
        self.stage2_model.fit(X, y)
        
        # Evaluate
        y_pred = self.stage2_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Get classification report
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        return accuracy, report
    
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predict move using enhanced pipeline"""
        
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Models not trained yet")
        
        # Stage 1: Choose combo_type
        stage1_features = self.extract_stage1_features(record)
        combo_type_id = self.stage1_model.predict(stage1_features.reshape(1, -1))[0]
        
        # Convert combo_type_id back to string
        if combo_type_id < len(self.combo_types):
            combo_type = self.combo_types[combo_type_id]
        else:
            combo_type = "pass"
        
        # Stage 2: Choose specific cards
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        
        if combo_type == "pass":
            # Return pass move
            for move in legal_moves:
                if move.get("type") == "pass":
                    return move
            return {"type": "pass", "cards": []}
        
        # Filter legal_moves by combo_type
        filtered_moves = [move for move in legal_moves if move.get("combo_type") == combo_type]
        
        if not filtered_moves:
            # Fallback: return first legal move
            return legal_moves[0] if legal_moves else {"type": "pass", "cards": []}
        
        # Stage 2: Choose from filtered moves
        stage2_features = self.extract_stage2_features(record, combo_type)
        move_index = self.stage2_model.predict(stage2_features.reshape(1, -1))[0]
        
        # Ensure move_index is valid
        if 0 <= move_index < len(filtered_moves):
            return filtered_moves[move_index]
        else:
            # Fallback: return first filtered move
            return filtered_moves[0]
    
    def save(self, filepath: str):
        """Save trained models"""
        model_data = {
            'stage1_model': self.stage1_model,
            'stage2_model': self.stage2_model,
            'combo_types': self.combo_types,
            'combo_type_to_id': self.combo_type_to_id
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        self.stage1_model = model_data['stage1_model']
        self.stage2_model = model_data['stage2_model']
        self.combo_types = model_data['combo_types']
        self.combo_type_to_id = model_data['combo_type_to_id']
