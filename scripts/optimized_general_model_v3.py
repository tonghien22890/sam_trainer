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
        self.combo_types = ["single", "pair", "triple", "four_kind", "straight", "double_seq", "pass"]
        self.combo_type_to_id = {ct: i for i, ct in enumerate(self.combo_types)}
        
    def calculate_combo_strength_relative(self, legal_moves: List[Dict[str, Any]]) -> float:
        """
        Tính sức mạnh tương đối của các combos cho Stage 1
        Mỗi combo type có cách tính rank khác nhau
        """
        combo_strengths = []
        
        for move in legal_moves:
            if move.get("type") == "play_cards":
                combo_type = move.get("combo_type")
                rank_value = move.get("rank_value", 0)
                cards = move.get("cards", [])
                
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
                    has_ace = any(card % 13 == 0 for card in cards)  # Check if has Ace
                    length = len(cards)
                    
                    if has_ace:
                        strength = 7.0 + length / 10.0  # A straight: 7.5-8.0
                    else:
                        strength = 6.0 + length / 10.0 + (rank_value / 13.0) * 0.5  # Other: 6.5-7.0
                        
                elif combo_type == "double_seq":
                    # Double_seq: Cực mạnh, vượt trội
                    length = len(cards)
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
        """Predict move using optimized pipeline V3"""
        
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Models not trained yet")
        
        # Check if có combo trước đó
        last_move = record.get("last_move")
        if last_move and last_move.get("combo_type"):
            # Bỏ qua Stage 1, chuyển thẳng sang Stage 2
            legal_moves = record.get("meta", {}).get("legal_moves", [])
            # Filter by combo_type from last_move
            combo_type = last_move.get("combo_type")
            filtered_moves = [move for move in legal_moves if move.get("combo_type") == combo_type]
            
            if not filtered_moves:
                # Fallback: return first legal move
                return legal_moves[0] if legal_moves else {"type": "pass", "cards": []}
            
            # Use basic ranking for Stage 2
            move_rankings = self.calculate_combo_strength_ranking(filtered_moves)
            return move_rankings[0]["move"] if move_rankings else filtered_moves[0]
        
        else:
            # Stage 1: Choose combo_type (when pass)
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
            
            # Stage 2: Use XGBoost to predict
            features = self.extract_stage2_features(record, combo_type)
            encoded_features = self.encode_stage2_features_for_model(features)
            
            # Predict index in filtered_moves
            predicted_index = self.stage2_model.predict(encoded_features.reshape(1, -1))[0]
            
            # Ensure index is valid
            if 0 <= predicted_index < len(filtered_moves):
                return filtered_moves[predicted_index]
            else:
                # Fallback: use basic ranking
                move_rankings = self.calculate_combo_strength_ranking(filtered_moves)
                return move_rankings[0]["move"] if move_rankings else filtered_moves[0]
    
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


