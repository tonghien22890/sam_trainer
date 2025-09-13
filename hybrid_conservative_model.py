"""
Hybrid Conservative Báo Sâm Model
Kết hợp ML model học tốt + Rulebase chặn bài yếu
"""
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridConservativeModel:
    """Hybrid model: ML học tốt + Rulebase chặn bài yếu"""
    
    def __init__(self):
        self.model = None
        # Rulebase để chặn bài quá yếu
        self.weak_hand_rules = {
            'required_total_cards': 10,      # Sequence phải đủ 10 lá
            'max_weak_combos': 2,            # Tối đa 2 combo yếu (strength < 0.5)
            'min_strong_combos': 1,          # Phải có ít nhất 1 combo mạnh (strength >= 0.7)
            'min_avg_strength': 0.6,         # Trung bình strength phải >= 0.6
            'min_high_ranks': 1,             # Phải có ít nhất 1 combo rank >= 8
        }
    
    def calculate_combo_strength(self, combo: Dict[str, Any]) -> float:
        """Calculate strength of a single combo"""
        combo_type = combo['combo_type']
        rank_value = combo['rank_value']
        
        # Base strength by combo type (Sâm rules)
        base_strength = {
            'single': 0.1, 'pair': 0.3, 'triple': 0.5,
            'straight': 0.7, 'quad': 0.9
        }.get(combo_type, 0.1)
        
        # Rank bonus (higher rank = stronger)
        rank_bonus = (rank_value / 12.0) * 0.3
        
        return base_strength + rank_bonus
    
    def is_weak_hand(self, sequence: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Rulebase: Kiểm tra xem bài có quá yếu không"""
        
        if not sequence:
            return True, "no_sequence"
        
        # Rule 0: Sequence phải đủ 10 lá
        total_cards = sum(len(combo['cards']) for combo in sequence)
        if total_cards < self.weak_hand_rules['required_total_cards']:
            return True, f"insufficient_cards_{total_cards}"
        
        # Calculate strengths and ranks
        strengths = [self.calculate_combo_strength(combo) for combo in sequence]
        rank_values = [combo['rank_value'] for combo in sequence]
        
        # Rule 1: Quá nhiều combo yếu
        weak_combos = sum(1 for s in strengths if s < 0.5)
        if weak_combos > self.weak_hand_rules['max_weak_combos']:
            return True, f"too_many_weak_combos_{weak_combos}"
        
        # Rule 2: Không có combo mạnh nào
        strong_combos = sum(1 for s in strengths if s >= 0.7)
        if strong_combos < self.weak_hand_rules['min_strong_combos']:
            return True, f"no_strong_combos_{strong_combos}"
        
        # Rule 3: Trung bình strength quá thấp
        avg_strength = np.mean(strengths)
        if avg_strength < self.weak_hand_rules['min_avg_strength']:
            return True, f"low_avg_strength_{avg_strength:.2f}"
        
        # Rule 4: Không có combo rank cao
        high_rank_combos = sum(1 for r in rank_values if r >= 8)
        if high_rank_combos < self.weak_hand_rules['min_high_ranks']:
            return True, f"no_high_ranks_{high_rank_combos}"
        
        # Rule 5: Nếu chỉ có 1 combo thì phải rất mạnh
        if len(sequence) == 1 and strengths[0] < 0.8:
            return True, f"single_combo_too_weak_{strengths[0]:.2f}"
        
        return False, "passed_rules"
    
    def extract_enhanced_features(self, record: Dict[str, Any]) -> List[float]:
        """Extract enhanced features for ML model"""
        features = []
        
        sequence = record.get('sammove_sequence', [])
        
        if not sequence:
            return [0.0] * 35  # Fill with zeros (5 combo types instead of 6)
        
        # 1. Basic sequence info
        features.append(len(sequence))
        
        # 2. Combo type pattern (one-hot encoding for first 3 combos)
        combo_types = ['single', 'pair', 'triple', 'straight', 'quad']
        
        # First combo type
        first_combo_type = sequence[0]['combo_type']
        for combo_type in combo_types:
            features.append(1.0 if combo_type == first_combo_type else 0.0)
        
        # Second combo type (if exists)
        if len(sequence) > 1:
            second_combo_type = sequence[1]['combo_type']
            for combo_type in combo_types:
                features.append(1.0 if combo_type == second_combo_type else 0.0)
        else:
            features.extend([0.0] * 5)  # No second combo
        
        # Third combo type (if exists)
        if len(sequence) > 2:
            third_combo_type = sequence[2]['combo_type']
            for combo_type in combo_types:
                features.append(1.0 if combo_type == third_combo_type else 0.0)
        else:
            features.extend([0.0] * 5)  # No third combo
        
        # 3. Rank pattern (first 3 combos)
        for i in range(min(3, len(sequence))):
            features.append(sequence[i]['rank_value'] / 12.0)  # Normalized rank
        
        # Fill remaining ranks if sequence is shorter than 3
        for i in range(len(sequence), 3):
            features.append(0.0)
        
        # 4. Combo strength pattern (first 3 combos)
        for i in range(min(3, len(sequence))):
            strength = self.calculate_combo_strength(sequence[i])
            features.append(strength)
        
        # Fill remaining strengths if sequence is shorter than 3
        for i in range(len(sequence), 3):
            features.append(0.0)
        
        # 5. Sequence statistics
        strengths = [self.calculate_combo_strength(combo) for combo in sequence]
        rank_values = [combo['rank_value'] for combo in sequence]
        
        features.extend([
            max(strengths),  # Max combo strength
            min(strengths),  # Min combo strength
            np.mean(strengths),  # Average combo strength
            max(rank_values),  # Max rank
            min(rank_values),  # Min rank
            np.mean(rank_values),  # Average rank
        ])
        
        # 6. Pattern indicators
        # Has strong start (first combo is strong)
        features.append(1.0 if strengths[0] >= 0.7 else 0.0)
        
        # Has strong finish (last combo is strong)
        features.append(1.0 if strengths[-1] >= 0.7 else 0.0)
        
        # Has ascending strength pattern
        ascending = all(strengths[i] <= strengths[i+1] for i in range(len(strengths)-1))
        features.append(1.0 if ascending else 0.0)
        
        # Has descending strength pattern
        descending = all(strengths[i] >= strengths[i+1] for i in range(len(strengths)-1))
        features.append(1.0 if descending else 0.0)
        
        # Count of strong combos (strength >= 0.7)
        strong_count = sum(1 for s in strengths if s >= 0.7)
        features.append(strong_count)
        
        # Count of high rank combos (rank >= 8)
        high_rank_count = sum(1 for r in rank_values if r >= 8)
        features.append(high_rank_count)
        
        return features
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load training data"""
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    
    def prepare_dataset(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset for training"""
        X = []
        y = []
        
        for record in records:
            try:
                features = self.extract_enhanced_features(record)
                X.append(features)
                
                # Label: 1 for success, 0 for failure
                label = 1 if record.get('result') == 'success' else 0
                y.append(label)
                
            except Exception as e:
                logger.warning(f"Error processing record: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Success rate: {np.mean(y):.3f}")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train hybrid model - ML học tốt"""
        logger.info("Training Hybrid Conservative Báo Sâm Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # ML Model - học tốt hơn
        self.model = DecisionTreeClassifier(
            max_depth=12,           # Tăng từ 6 → 12 để học tốt hơn
            min_samples_split=10,   # Giảm từ 50 → 10
            min_samples_leaf=5,     # Giảm từ 25 → 5
            criterion='entropy',     
            class_weight={0:1, 1:2}, # Giảm từ 1:10 → 1:2 để cân bằng hơn
            random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        training_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross validation
        cv_scores = cross_val_score(self.model, X, y, cv=10, scoring='accuracy')
        cv_accuracy = cv_scores.mean()
        
        logger.info(f"Training accuracy: {training_accuracy:.3f}")
        logger.info(f"CV accuracy: {cv_accuracy:.3f} ± {cv_scores.std():.3f}")
    
    def predict_hybrid(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid prediction: Rulebase + ML Model"""
        if self.model is None:
            return {
                'should_declare': False,
                'confidence': 0.0,
                'reason': 'model_not_trained',
                'timestamp': datetime.now().isoformat()
            }
        
        sequence = record.get('sammove_sequence', [])
        
        # Step 1: Rulebase - chặn bài quá yếu
        is_weak, weak_reason = self.is_weak_hand(sequence)
        if is_weak:
            return {
                'should_declare': False,
                'confidence': 0.0,
                'reason': f'weak_hand_blocked_{weak_reason}',
                'rulebase_blocked': True,
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 2: ML Model prediction
        features = self.extract_enhanced_features(record)
        X = np.array(features).reshape(1, -1)
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        # Step 3: Conservative decision (confidence >= 0.8)
        should_declare = (prediction == 1 and 
                         confidence >= 0.8 and 
                         probabilities[1] > 0.8)
        
        if not should_declare:
            reason = f'ml_low_confidence_{confidence:.3f}'
        else:
            reason = f'ml_high_confidence_{confidence:.3f}'
        
        # Step 4: Generate optimal combo sequence if declaring
        optimal_sequence = None
        if should_declare:
            optimal_sequence = self.generate_optimal_combo_sequence(sequence)
        
        return {
            'should_declare': should_declare,
            'prediction': prediction,
            'confidence': confidence,
            'reason': reason,
            'rulebase_blocked': False,
            'sequence_length': len(sequence),
            'optimal_combo_sequence': optimal_sequence,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_optimal_combo_sequence(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimal combo sequence for Báo Sâm declaration"""
        if not sequence:
            return []
        
        # Strategy: Strong combos first, weak combos last
        # This maximizes chances of winning early and reduces risk
        
        # Calculate combo strengths
        combo_strengths = []
        for i, combo in enumerate(sequence):
            strength = self.calculate_combo_strength(combo)
            combo_strengths.append((i, combo, strength))
        
        # Sort by strength (descending) - strongest first
        combo_strengths.sort(key=lambda x: x[2], reverse=True)
        
        # Reorder sequence
        optimal_sequence = []
        for _, combo, strength in combo_strengths:
            # Add position information for strategy
            combo_with_position = combo.copy()
            combo_with_position['position'] = len(optimal_sequence)
            combo_with_position['strength'] = strength
            optimal_sequence.append(combo_with_position)
        
        return optimal_sequence
    
    def calculate_combo_strength(self, combo: Dict[str, Any]) -> float:
        """Calculate strength of a single combo"""
        combo_type = combo['combo_type']
        rank_value = combo['rank_value']
        
        # Base strength by combo type
        base_strength = {
            'single': 0.1, 'pair': 0.3, 'triple': 0.5,
            'straight': 0.7, 'quad': 0.9
        }.get(combo_type, 0.1)
        
        # Rank bonus (higher rank = stronger)
        rank_bonus = (rank_value / 12.0) * 0.3
        
        # Special bonuses for high-value combos
        special_bonus = 0.0
        if combo_type == 'straight' and rank_value >= 8:
            special_bonus = 0.2  # High straight
        elif combo_type == 'quad':
            special_bonus = 0.3  # Quad is very strong
        elif combo_type == 'triple' and rank_value >= 10:
            special_bonus = 0.15  # High triple
        
        return base_strength + rank_bonus + special_bonus
    
    def test_hybrid(self, data_file: str) -> Dict[str, Any]:
        """Test hybrid model"""
        logger.info(f"Testing Hybrid Model on {data_file}...")
        
        records = self.load_data(data_file)
        X, y = self.prepare_dataset(records)
        
        # Make predictions with hybrid logic
        predictions = []
        rulebase_blocked = 0
        ml_predictions = []
        
        for i, record in enumerate(records):
            result = self.predict_hybrid(record)
            predictions.append(result['should_declare'])
            
            if result.get('rulebase_blocked', False):
                rulebase_blocked += 1
            else:
                ml_predictions.append(result['should_declare'])
        
        predictions = np.array(predictions)
        
        # Overall metrics
        accuracy = accuracy_score(y, predictions)
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        logger.info(f"🎯 HYBRID MODEL RESULTS:")
        logger.info(f"   Overall Accuracy: {accuracy:.3f}")
        logger.info(f"   Precision: {precision:.3f}")
        logger.info(f"   Recall: {recall:.3f}")
        logger.info(f"   True Positives: {tp}")
        logger.info(f"   False Positives: {fp} ⚠️")
        logger.info(f"   True Negatives: {tn}")
        logger.info(f"   False Negatives: {fn}")
        logger.info(f"   Rulebase Blocked: {rulebase_blocked}")
        
        # ML-only metrics
        if ml_predictions:
            ml_accuracy = accuracy_score(y[rulebase_blocked:], ml_predictions)
            logger.info(f"   ML-only Accuracy: {ml_accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'false_positives': fp,
            'false_negatives': fn,
            'rulebase_blocked': rulebase_blocked
        }

def main():
    """Main training and testing function"""
    model = HybridConservativeModel()
    
    # Load and prepare data
    records = model.load_data('enhanced_bao_sam_data.jsonl')
    X, y = model.prepare_dataset(records)
    
    # Train model
    model.train_model(X, y)
    
    # Test model
    results = model.test_hybrid('enhanced_bao_sam_data.jsonl')
    
    # Save model
    model_path = 'hybrid_conservative_bao_sam_model.pkl'
    joblib.dump(model, model_path)
    logger.info(f"✅ Hybrid Conservative Model saved to {model_path}")
    
    print(f"\n🎯 FINAL HYBRID RESULTS:")
    print(f"   Accuracy: {results['accuracy']:.3f}")
    print(f"   Precision: {results['precision']:.3f}")
    print(f"   False Positives: {results['false_positives']} ⚠️")
    print(f"   Rulebase Blocked: {results['rulebase_blocked']}")

if __name__ == "__main__":
    main()
