"""
Optimized Pipeline Model - Tận dụng legal_moves và tránh overfitting
"""

import numpy as np
import joblib
from typing import Dict, List, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class OptimizedPipelineModel:
    """
    Optimized Pipeline Model với features tối ưu:
    - Hand one-hot (52 dims)
    - Cards left của từng người chơi (4 dims)
    - Last move features (binned, không specific) (4 dims)
    - Legal moves summary (6 dims)
    - Game context (2 dims)
    Total: 68 dims
    """
    
    def __init__(self):
        self.stage1_model = None
        self.stage2_model = None
        self.combo_types = ["single", "pair", "triple", "four_kind", "straight", "double_seq", "pass"]
        self.combo_type_to_id = {ct: i for i, ct in enumerate(self.combo_types)}
        
    def bin_rank(self, rank_value: int) -> int:
        """Group ranks into bins to avoid overfitting"""
        if rank_value <= 2:    # A, 2, 3
            return 0  # "low_rank"
        elif rank_value <= 5:  # 4, 5, 6
            return 1  # "mid_low_rank"
        elif rank_value <= 8:  # 7, 8, 9
            return 2  # "mid_high_rank"
        else:                  # 10, J, Q, K
            return 3  # "high_rank"
    
    def categorize_combo_strength(self, combo_type: str, rank_value: int) -> str:
        """Categorize combo strength instead of exact rank"""
        base_strength = {
            "single": 1, "pair": 2, "triple": 3,
            "straight": 4, "four_kind": 5, "double_seq": 6
        }.get(combo_type, 0)
        
        rank_contribution = self.bin_rank(rank_value)
        total_strength = base_strength * 4 + rank_contribution
        
        if total_strength >= 20:
            return "very_strong"
        elif total_strength >= 15:
            return "strong"
        elif total_strength >= 10:
            return "medium"
        elif total_strength >= 5:
            return "weak"
        else:
            return "very_weak"
    
    def extract_legal_moves_summary(self, legal_moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract summary from legal_moves instead of calculating from hand"""
        summary = {
            "combo_counts": {"single": 0, "pair": 0, "triple": 0, "four_kind": 0, "straight": 0, "double_seq": 0},
            "has_pass": False,
            "total_moves": len(legal_moves),
            "can_beat_last_move": False,
            "strongest_combo": "none",
            "weakest_combo": "none"
        }
        
        if not legal_moves:
            return summary
        
        combo_strengths = []
        
        for move in legal_moves:
            if move.get("type") == "pass":
                summary["has_pass"] = True
            elif move.get("type") == "play_cards":
                combo_type = move.get("combo_type")
                rank_value = move.get("rank_value", 0)
                
                if combo_type in summary["combo_counts"]:
                    summary["combo_counts"][combo_type] += 1
                
                strength = self.categorize_combo_strength(combo_type, rank_value)
                combo_strengths.append((combo_type, strength, rank_value))
        
        # Determine strongest and weakest combos
        if combo_strengths:
            # Sort by strength (simplified)
            combo_strengths.sort(key=lambda x: (x[1], x[2]), reverse=True)
            summary["strongest_combo"] = combo_strengths[0][0]
            summary["weakest_combo"] = combo_strengths[-1][0]
        
        return summary
    
    def extract_stage1_features(self, record: Dict[str, Any]) -> np.ndarray:
        """Extract optimized features for Stage 1 (combo_type selection)"""
        
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
            cards_left_normalized = [0.0] * 4
        
        # Pad or truncate to 4 dimensions
        while len(cards_left_normalized) < 4:
            cards_left_normalized.append(0.0)
        cards_left_normalized = cards_left_normalized[:4]
        
        # Last move features (binned, không specific) (4 dims)
        last_move_features = np.zeros(4, dtype=np.float32)
        last_move = record.get("last_move")
        if last_move and last_move.get("combo_type"):
            combo_type = last_move.get("combo_type")
            rank_value = last_move.get("rank_value", 0)
            
            # Combo type one-hot (3 dims for common types)
            if combo_type in ["single", "pair", "triple"]:
                last_move_features[["single", "pair", "triple"].index(combo_type)] = 1.0
            
            # Rank bin (1 dim)
            last_move_features[3] = self.bin_rank(rank_value) / 3.0  # Normalized 0-1
        
        # Legal moves summary (6 dims)
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        legal_summary = self.extract_legal_moves_summary(legal_moves)
        legal_features = [
            legal_summary["combo_counts"]["single"],
            legal_summary["combo_counts"]["pair"],
            legal_summary["combo_counts"]["triple"],
            legal_summary["combo_counts"]["four_kind"],
            legal_summary["combo_counts"]["straight"],
            legal_summary["combo_counts"]["double_seq"]
        ]
        
        # Game context (2 dims)
        game_context = np.zeros(2, dtype=np.float32)
        game_context[0] = 1.0 if legal_summary["has_pass"] else 0.0  # Can pass
        game_context[1] = legal_summary["total_moves"] / 20.0  # Normalized move count
        
        # Combine all features
        features = np.concatenate([
            hand_oh,                    # 52 dims
            cards_left_normalized,      # 4 dims
            last_move_features,         # 4 dims
            legal_features,             # 6 dims
            game_context                # 2 dims
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
            stage1_features,            # 68 dims (52+4+4+6+2)
            chosen_combo_oh,            # 7 dims
            np.array([available_moves_count], dtype=np.float32),  # 1 dim
        ])
        
        return features
    
    def train_stage1(self, records: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Train Stage 1 model with regularization to prevent overfitting"""
        
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
        
        # Train model with regularization to prevent overfitting
        self.stage1_model = DecisionTreeClassifier(
            max_depth=8,            # Giảm depth để tránh overfitting
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
        """Train Stage 2 model with regularization"""
        
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
        
        # Train model with regularization
        self.stage2_model = DecisionTreeClassifier(
            max_depth=6,            # Giảm depth để tránh overfitting
            min_samples_split=10,   # Tăng để yêu cầu nhiều samples hơn
            min_samples_leaf=5,     # Tăng để tránh overfitting
            criterion='entropy',
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
        """Predict move using optimized pipeline"""
        
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
