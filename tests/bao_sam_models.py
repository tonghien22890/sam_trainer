"""
Báo Sâm Model Building Module

This module implements specialized models for Báo Sâm (Sam Declaration) functionality:
1. Báo Sâm Decision Model: Predict whether to declare Báo Sâm or not
2. Báo Sâm Combo Sequence Model: Predict optimal sequence of combos to win

Data Schema:
- 1 record = 1 Báo Sâm declaration with complete combo sequence
- Only logs when someone actually declares Báo Sâm
- Tracks success/failure and combo patterns
"""

from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

@dataclass
class BaoSamRecord:
    """Data structure for Báo Sâm declaration records"""
    game_id: str
    player_id: int
    hand: List[int]
    sammove_sequence: List[Dict[str, Any]]
    result: str  # "success" or "fail"
    meta: Dict[str, Any]
    timestamp: str

class BaoSamDataLogger:
    """Specialized logger for Báo Sâm declarations"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def log_bao_sam_declaration(self, game_id: str, player_id: int, 
                               hand: List[int], sammove_sequence: List[Dict[str, Any]], 
                               result: str, meta: Dict[str, Any]) -> None:
        """Log a Báo Sâm declaration record"""
        record = {
            "game_id": game_id,
            "player_id": player_id,
            "hand": hand,
            "sammove_sequence": sammove_sequence,
            "result": result,
            "meta": meta,
            "timestamp": self._get_timestamp()
        }
        
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

class BaoSamDataLoader:
    """Data loader for Báo Sâm training data"""
    
    def __init__(self):
        self.combo_type_to_id = {
            "single": 0, "pair": 1, "triple": 2, "straight": 3, 
            "quad": 4
        }
        self.rank_values = list(range(13))  # 0-12 for card ranks
    
    def load_bao_sam_data(self, file_path: str) -> List[BaoSamRecord]:
        """Load Báo Sâm declaration data from JSONL file"""
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    record = BaoSamRecord(
                        game_id=data['game_id'],
                        player_id=data['player_id'],
                        hand=data['hand'],
                        sammove_sequence=data['sammove_sequence'],
                        result=data['result'],
                        meta=data['meta'],
                        timestamp=data.get('timestamp', '')
                    )
                    records.append(record)
        
        except Exception as e:
            print(f"Error loading Báo Sâm data: {e}")
        
        return records
    
    def encode_hand_features(self, hand: List[int]) -> np.ndarray:
        """Encode hand cards into feature vector (52 features)"""
        features = np.zeros(52, dtype=np.float32)
        for card_id in hand:
            if 0 <= card_id < 52:
                features[card_id] = 1.0
        return features
    
    def encode_combo_sequence_features(self, sammove_sequence: List[Dict[str, Any]]) -> np.ndarray:
        """Encode combo sequence into features"""
        # Features for combo sequence analysis
        num_combos = len(sammove_sequence)
        total_cards = sum(len(combo.get('cards', [])) for combo in sammove_sequence)
        
        # Combo type distribution (7 features)
        combo_type_counts = np.zeros(7, dtype=np.float32)
        for combo in sammove_sequence:
            combo_type = combo.get('combo_type', '')
            if combo_type in self.combo_type_to_id:
                combo_type_counts[self.combo_type_to_id[combo_type]] += 1
        
        # Rank distribution (13 features)
        rank_counts = np.zeros(13, dtype=np.float32)
        for combo in sammove_sequence:
            rank_value = combo.get('rank_value', 0)
            if 0 <= rank_value < 13:
                rank_counts[rank_value] += 1
        
        # Sequence statistics (4 features)
        sequence_stats = np.array([
            float(num_combos),           # Number of combos
            float(total_cards),          # Total cards used
            float(num_combos / max(total_cards, 1)),  # Efficiency ratio
            float(max(len(combo.get('cards', [])) for combo in sammove_sequence) if sammove_sequence else 0)  # Max combo size
        ], dtype=np.float32)
        
        return np.concatenate([combo_type_counts, rank_counts, sequence_stats])

class BaoSamDecisionModel:
    """Model to predict whether to declare Báo Sâm or not"""
    
    def __init__(self):
        self.model = None
        self.data_loader = BaoSamDataLoader()
    
    def train(self, data_path: str, model_type: str = "random_forest") -> Dict[str, Any]:
        """Train Báo Sâm decision model"""
        records = self.data_loader.load_bao_sam_data(data_path)
        
        if len(records) < 5:
            return {"error": f"Insufficient data: {len(records)} records, need at least 5"}
        
        # Prepare features and labels
        X = []
        y = []
        
        for record in records:
            # Features: hand cards (52) + sequence features (24) = 76 features
            hand_features = self.data_loader.encode_hand_features(record.hand)
            sequence_features = self.data_loader.encode_combo_sequence_features(record.sammove_sequence)
            
            # Combine features
            features = np.concatenate([hand_features, sequence_features])
            X.append(features)
            
            # Label: 1 for success, 0 for failure
            y.append(1 if record.result == "success" else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        if len(X) < 10:
            X_train, X_val, y_train, y_val = X, X, y, y
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == "decision_tree":
            self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return {
            "model_type": model_type,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "accuracy": accuracy,
            "feature_count": X.shape[1]
        }
    
    def predict_bao_sam_decision(self, hand: List[int], sammove_sequence: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """Predict whether to declare Báo Sâm"""
        if self.model is None:
            return False, 0.0
        
        hand_features = self.data_loader.encode_hand_features(hand)
        sequence_features = self.data_loader.encode_combo_sequence_features(sammove_sequence)
        features = np.concatenate([hand_features, sequence_features]).reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1] if hasattr(self.model, 'predict_proba') else 0.5
        
        return bool(prediction), float(probability)
    
    def save_model(self, file_path: str) -> None:
        """Save trained model"""
        if self.model is not None:
            joblib.dump(self.model, file_path)
    
    def load_model(self, file_path: str) -> None:
        """Load trained model"""
        self.model = joblib.load(file_path)

class BaoSamComboModel:
    """Model to predict optimal combo sequence for Báo Sâm"""
    
    def __init__(self):
        self.model = None
        self.data_loader = BaoSamDataLoader()
    
    def train(self, data_path: str, model_type: str = "random_forest") -> Dict[str, Any]:
        """Train combo sequence optimization model"""
        records = self.data_loader.load_bao_sam_data(data_path)
        
        if len(records) < 5:
            return {"error": f"Insufficient data: {len(records)} records, need at least 5"}
        
        # Prepare training data for sequence prediction
        X = []
        y = []
        
        for record in records:
            if record.result != "success":
                continue  # Only use successful sequences
            
            # For each combo in the sequence, predict the next combo
            for i in range(len(record.sammove_sequence) - 1):
                current_combo = record.sammove_sequence[i]
                next_combo = record.sammove_sequence[i + 1]
                
                # Features: current hand state + current combo
                remaining_hand = record.hand.copy()
                for j in range(i + 1):
                    for card_id in record.sammove_sequence[j].get('cards', []):
                        if card_id in remaining_hand:
                            remaining_hand.remove(card_id)
                
                hand_features = self.data_loader.encode_hand_features(remaining_hand)
                current_combo_features = self._encode_combo_features(current_combo)
                sequence_position = np.array([float(i)], dtype=np.float32)
                
                features = np.concatenate([hand_features, current_combo_features, sequence_position])
                X.append(features)
                
                # Label: next combo type and rank
                next_combo_type = next_combo.get('combo_type', '')
                next_combo_rank = next_combo.get('rank_value', 0)
                label = f"{next_combo_type}_{next_combo_rank}"
                y.append(label)
        
        if len(X) < 5:
            return {"error": f"Insufficient sequence data: {len(X)} samples, need at least 5"}
        
        X = np.array(X)
        
        # Create label mapping
        unique_labels = list(set(y))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        y_encoded = np.array([label_to_id[label] for label in y])
        
        # Split data
        if len(X) < 10:
            X_train, X_val, y_train, y_val = X, X, y_encoded, y_encoded
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == "decision_tree":
            self.model = DecisionTreeClassifier(max_depth=15, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return {
            "model_type": model_type,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "accuracy": accuracy,
            "unique_labels": len(unique_labels),
            "feature_count": X.shape[1]
        }
    
    def _encode_combo_features(self, combo: Dict[str, Any]) -> np.ndarray:
        """Encode a single combo into features"""
        combo_type = combo.get('combo_type', '')
        rank_value = combo.get('rank_value', 0)
        num_cards = len(combo.get('cards', []))
        
        # Combo type one-hot (7 features)
        combo_type_oh = np.zeros(7, dtype=np.float32)
        if combo_type in self.data_loader.combo_type_to_id:
            combo_type_oh[self.data_loader.combo_type_to_id[combo_type]] = 1.0
        
        # Rank value (1 feature)
        rank_feature = np.array([float(rank_value)], dtype=np.float32)
        
        # Number of cards (1 feature)
        cards_count = np.array([float(num_cards)], dtype=np.float32)
        
        return np.concatenate([combo_type_oh, rank_feature, cards_count])
    
    def predict_next_combo(self, hand: List[int], current_combo: Dict[str, Any], 
                          sequence_position: int) -> Dict[str, Any]:
        """Predict next optimal combo in sequence"""
        if self.model is None:
            return {"type": "pass", "cards": []}
        
        hand_features = self.data_loader.encode_hand_features(hand)
        current_combo_features = self._encode_combo_features(current_combo)
        position_feature = np.array([float(sequence_position)], dtype=np.float32)
        
        features = np.concatenate([hand_features, current_combo_features, position_feature]).reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        # Note: This is simplified - in practice you'd need to decode the prediction back to combo format
        
        return {"type": "single", "cards": [hand[0]], "rank_value": 0}
    
    def save_model(self, file_path: str) -> None:
        """Save trained model"""
        if self.model is not None:
            joblib.dump(self.model, file_path)
    
    def load_model(self, file_path: str) -> None:
        """Load trained model"""
        self.model = joblib.load(file_path)

def main():
    """Example usage of Báo Sâm models"""
    print("Báo Sâm Model Building Pipeline")
    print("=" * 40)
    
    # Initialize models
    decision_model = BaoSamDecisionModel()
    combo_model = BaoSamComboModel()
    
    # Train decision model
    print("Training Báo Sâm Decision Model...")
    decision_results = decision_model.train("bao_sam_data.jsonl", "random_forest")
    print(f"Decision Model Results: {decision_results}")
    
    # Train combo model
    print("\nTraining Báo Sâm Combo Sequence Model...")
    combo_results = combo_model.train("bao_sam_data.jsonl", "random_forest")
    print(f"Combo Model Results: {combo_results}")
    
    # Save models
    decision_model.save_model("runs/bao_sam_decision_model.pkl")
    combo_model.save_model("runs/bao_sam_combo_model.pkl")
    
    print("\nModels saved to runs/ directory")

if __name__ == "__main__":
    main()
