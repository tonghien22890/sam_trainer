#!/usr/bin/env python3
"""
Unbeatable Sequence Model - Complete Implementation
Build unbeatable combo sequences từ user behavior patterns
"""

import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
import os
import sys

# Add ai_common to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up: scripts/unbeatable -> scripts -> model_build -> project_root
model_build_dir = os.path.dirname(os.path.dirname(current_dir))  # model_build/
project_root = os.path.dirname(model_build_dir)  # AI-Sam/
ai_common_path = os.path.join(project_root, "ai_common")

# Add paths to sys.path
if model_build_dir not in sys.path:
    sys.path.insert(0, model_build_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if ai_common_path not in sys.path:
    sys.path.insert(0, ai_common_path)

# Import extracted utilities
from ai_common.core.combo_analyzer import ComboAnalyzer
from ai_common.rules.sam_rule_engine import SamRuleEngine
from ai_common.features.sequence_features import SequenceFeatureExtractor
from ai_common.probability.unbeatable_calculator import UnbeatableProbabilityCalculator
from ai_common.core.sequence_evaluator import SequenceEvaluator

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unbeatable_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# UnbeatableRuleEngine class removed - now using SamRuleEngine from ai_common


class SequenceValidationModel:
    """Phase 1: Learn what makes a sequence valid/invalid"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            class_weight='balanced',
            random_state=42
        )
        self.feature_names = []
        logger.info("SequenceValidationModel initialized")
    
    def extract_combo_features(self, combo: Dict[str, Any]) -> List[float]:
        """Extract combo-level features"""
        return SequenceFeatureExtractor.extract_combo_features(combo)
    
    def extract_sequence_features(self, combos: List[Dict[str, Any]]) -> List[float]:
        """Extract sequence-level features"""
        return SequenceFeatureExtractor.extract_sequence_features(combos)
    
    def extract_features(self, hand_data: Dict[str, Any]) -> List[float]:
        """Extract all features for validation"""
        return SequenceFeatureExtractor.extract_validation_features(hand_data)
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train validation model"""
        logger.info("Training SequenceValidationModel...")
        
        X = []
        y = []
        
        for record in training_data:
            try:
                features = self.extract_features(record)
                X.append(features)
                
                # Label: 1 if valid sequence, 0 if invalid
                label = 1 if record.get('is_valid', False) else 0
                y.append(label)
                
            except Exception as e:
                logger.warning(f"Error processing record: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Valid rate: {np.mean(y):.3f}")
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"Validation model trained - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        return results
    
    def predict(self, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if hand is valid for sequence building"""
        features = self.extract_features(hand_data)
        X = np.array(features).reshape(1, -1)
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return {
            'is_valid': bool(prediction),
            'confidence': float(confidence),
            'probability_valid': float(probabilities[1]) if len(probabilities) > 1 else 0.0
        }


class PatternLearningModel:
    """Phase 2: Learn user combo building patterns"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        logger.info("PatternLearningModel initialized")
    
    def extract_pattern_features(self, hand_data: Dict[str, Any]) -> List[float]:
        """Extract features for pattern learning"""
        return SequenceFeatureExtractor.extract_pattern_features(hand_data)
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train pattern learning model"""
        logger.info("Training PatternLearningModel...")
        
        X = []
        y = []
        
        for record in training_data:
            try:
                features = self.extract_pattern_features(record)
                X.append(features)
                
                # Target: pattern score (derived from user behavior)
                pattern_score = record.get('pattern_score', 0.5)
                y.append(pattern_score)
                
            except Exception as e:
                logger.warning(f"Error processing record: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Pattern training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        results = {
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
        
        logger.info(f"Pattern model trained - RMSE: {np.sqrt(mse):.3f}")
        return results
    
    def predict(self, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user combo building patterns"""
        features = self.extract_pattern_features(hand_data)
        X = np.array(features).reshape(1, -1)
        
        pattern_score = self.model.predict(X)[0]
        
        return {
            'combo_patterns': {
                'power_concentration': float(features[1]),
                'combo_diversity': float(features[0]),
                'balance_preference': float(features[2])
            },
            'pattern_score': float(pattern_score),
            'sequence_building_preference': 'power_first' if features[1] > 0.6 else 'balanced'
        }


class ThresholdLearningModel:
    """Learn user's decision threshold for Báo Sâm declarations"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        logger.info("ThresholdLearningModel initialized")
    
    def extract_threshold_features(self, hand_data: Dict[str, Any], user_patterns: Dict[str, Any]) -> List[float]:
        """Extract features for threshold learning"""
        return SequenceFeatureExtractor.extract_threshold_features(hand_data, user_patterns)
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train threshold learning model"""
        logger.info("Training ThresholdLearningModel...")
        
        X = []
        y = []
        
        for record in training_data:
            try:
                user_patterns = record.get('user_patterns', {})
                features = self.extract_threshold_features(record, user_patterns)
                X.append(features)
                
                # Target: user's actual threshold (learned from decisions)
                threshold = record.get('user_threshold', 0.75)
                y.append(threshold)
                
            except Exception as e:
                logger.warning(f"Error processing record: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Threshold training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        results = {
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
        
        logger.info(f"Threshold model trained - RMSE: {np.sqrt(mse):.3f}")
        return results
    
    def predict_user_threshold(self, hand_data: Dict[str, Any], user_patterns: Dict[str, Any]) -> float:
        """Predict user's decision threshold"""
        features = self.extract_threshold_features(hand_data, user_patterns)
        X = np.array(features).reshape(1, -1)
        
        predicted_threshold = self.model.predict(X)[0]
        
        # Clamp between reasonable bounds
        return max(0.6, min(0.95, predicted_threshold))


class UnbeatableSequenceGenerator:
    """Phase 3: Generate optimal unbeatable sequences and decide Báo Sâm"""
    
    def __init__(self):
        self.rule_engine = SamRuleEngine()
        self.validation_model = SequenceValidationModel()
        self.pattern_model = PatternLearningModel()
        self.threshold_model = ThresholdLearningModel()
        # Use SequenceEvaluator with unbeatable strength for Báo Sâm context
        self.seq_evaluator = SequenceEvaluator(
            enforce_full_coverage=False,
            strengthFn=ComboAnalyzer.calculate_unbeatable_strength
        )
        logger.info("UnbeatableSequenceGenerator initialized")
    
    def analyze_hand(self, hand: List[int]) -> List[Dict[str, Any]]:
        """Analyze hand and find possible combos"""
        return ComboAnalyzer.analyze_hand(hand)
    
    def calculate_unbeatable_probability(self, sequence: List[Dict[str, Any]]) -> float:
        """Calculate probability that sequence is unbeatable"""
        return UnbeatableProbabilityCalculator.calculate_unbeatable_probability(sequence)
    
    def build_sequence_from_patterns(self, combos: List[Dict[str, Any]], user_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build sequence based on user patterns"""
        if not combos:
            return []
        
        # For now, return all combos
        # In real implementation, you'd select optimal subset based on patterns
        return combos.copy()
    
    def calculate_model_confidence(self, validation_result: Dict[str, Any], user_patterns: Dict[str, Any]) -> float:
        """Calculate overall model confidence"""
        return UnbeatableProbabilityCalculator.calculate_model_confidence(validation_result, user_patterns)
    
    def generate_sequence(self, hand: List[int], player_count: int = 4, 
                         context: str = "bao_sam") -> Dict[str, Any]:
        """Generate unbeatable sequence and decide Báo Sâm
        
        Args:
            hand: List of card IDs
            player_count: Number of players
            context: "bao_sam" for Báo Sâm decisions, "general" for framework generation
        """
        logger.info(f"Generating sequence for hand: {hand} (context: {context})")
        
        # Step 1: Analyze hand
        possible_combos = self.analyze_hand(hand)
        
        hand_data = {
            'hand': hand,
            'player_count': player_count,
            'possible_combos': possible_combos
        }
        
        # Step 2: Rulebase validation (only for Báo Sâm context)
        if context == "bao_sam":
            is_valid, reason, strength_profile = self.rule_engine.validate_hand(possible_combos)
            
            if not is_valid:
                logger.info(f"Hand rejected by rules: {reason}")
                return {
                    'should_declare_bao_sam': bool(False),
                    'unbeatable_probability': 0.0,
                    'user_threshold': 0.0,
                    'model_confidence': 0.0,
                    'reason': reason,
                    'unbeatable_sequence': None,
                    'sequence_stats': None
                }
        else:
            # For general context, skip strict validation but still analyze
            _, reason, strength_profile = self.rule_engine.validate_hand(possible_combos)
            logger.info(f"General context - skipping strict validation: {reason}")
        
        # Step 3: ML Validation (only for Báo Sâm context)
        if context == "bao_sam":
            validation_result = self.validation_model.predict(hand_data)
            
            if not validation_result['is_valid']:
                logger.info(f"Hand rejected by ML validation: confidence={validation_result['confidence']:.3f}")
                return {
                    'should_declare_bao_sam': bool(False),
                    'unbeatable_probability': 0.0,
                    'user_threshold': 0.0,
                    'model_confidence': validation_result['confidence'],
                    'reason': 'ml_validation_failed',
                    'unbeatable_sequence': None,
                    'sequence_stats': None
                }
        else:
            # For general context, create mock validation result
            validation_result = {
                'is_valid': True,
                'confidence': 0.5,  # Medium confidence for general use
                'probability_valid': 0.5
            }
            logger.info(f"General context - using mock validation result")
        
        # Step 4: Extract user patterns (only for Báo Sâm context)
        if context == "bao_sam":
            user_patterns = self.pattern_model.predict(hand_data)
        else:
            # For general context, create mock user patterns
            user_patterns = {
                'combo_patterns': {
                    'power_concentration': 0.5,
                    'combo_diversity': 0.5,
                    'balance_preference': 0.5
                },
                'pattern_score': 0.5,
                'sequence_building_preference': 'balanced'
            }
            logger.info(f"General context - using mock user patterns")
        
        # Step 5-6: Use SequenceEvaluator to get top sequences and pick the strongest
        # Always select based on unbeatable-strength-aware scoring for bao_sam
        top_sequences = self.seq_evaluator.evaluate_top_sequences(hand, k=3, beam_size=50)
        if top_sequences:
            best_seq_entry = top_sequences[0]
            ordered_sequence = best_seq_entry.get('sequence', []) or []
        else:
            ordered_sequence = []

        # Normalize combo schema to expected keys (combo_type, cards, rank_value)
        def _normalize_combo_schema(combo: Dict[str, Any]) -> Dict[str, Any]:
            if not combo:
                return combo
            normalized = dict(combo)
            if 'combo_type' not in normalized and 'type' in normalized:
                normalized['combo_type'] = normalized['type']
            # Ensure rank_value exists; if not, derive from highest card rank
            if 'rank_value' not in normalized:
                cards = normalized.get('cards', [])
                if cards:
                    normalized['rank_value'] = max((c % 13) for c in cards)
            # For straights, always use highest rank as rank_value to detect Ace-high
            if normalized.get('combo_type') == 'straight':
                cards = normalized.get('cards', [])
                if cards:
                    normalized['rank_value'] = max((c % 13) for c in cards)
            return normalized

        ordered_sequence = [_normalize_combo_schema(c) for c in ordered_sequence]

        # Ensure strongest -> weakest ordering for bao_sam context
        if ordered_sequence:
            ordered_sequence = sorted(
                ordered_sequence,
                key=lambda combo: -ComboAnalyzer.calculate_unbeatable_strength(combo)
            )
        
        # Step 7: Calculate unbeatable probability from the selected sequence
        if context == "bao_sam":
            unbeatable_prob = self.calculate_unbeatable_probability(ordered_sequence)
        else:
            # For general context, calculate simple probability based on combo strengths
            if ordered_sequence:
                strengths = [ComboAnalyzer.calculate_combo_strength(combo) for combo in ordered_sequence]
                unbeatable_prob = min(0.8, max(0.1, sum(strengths) / len(strengths)))
            else:
                unbeatable_prob = 0.1
            logger.info(f"General context - calculated simple unbeatable probability: {unbeatable_prob:.3f}")
        
        # Step 8: Learn user's threshold preference (only for Báo Sâm context)
        if context == "bao_sam":
            user_threshold = self.threshold_model.predict_user_threshold(hand_data, user_patterns)
            should_declare = unbeatable_prob >= user_threshold
        else:
            # For general context, use default values
            user_threshold = 0.5
            should_declare = False  # Never declare Báo Sâm in general context
        
        model_confidence = self.calculate_model_confidence(validation_result, user_patterns)
        
        # Calculate sequence stats
        sequence_stats = UnbeatableProbabilityCalculator.calculate_sequence_stats(ordered_sequence, user_patterns)
        
        result = {
            'should_declare_bao_sam': bool(should_declare),
            'unbeatable_probability': float(unbeatable_prob),
            'user_threshold': float(user_threshold),
            'model_confidence': float(model_confidence),
            'reason': f'unbeatable_prob_{unbeatable_prob:.2f}_vs_threshold_{user_threshold:.2f}' if context == "bao_sam" else f'general_context_{unbeatable_prob:.2f}',
            'unbeatable_sequence': ordered_sequence,
            'sequence_stats': sequence_stats
        }
        
        logger.info(f"Sequence generated - Decision: {should_declare}, Prob: {unbeatable_prob:.3f}, Threshold: {user_threshold:.3f}")
        return result
    
    def save_models(self, model_dir: str = 'models'):
        """Save all trained models (pure sklearn objects only for compatibility)"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save only the .model attribute (pure sklearn objects)
        # This is compatible with Nuitka and doesn't require module definition
        joblib.dump(self.validation_model.model, os.path.join(model_dir, 'validation_model.pkl'))
        joblib.dump(self.pattern_model.model, os.path.join(model_dir, 'pattern_model.pkl'))
        joblib.dump(self.threshold_model.model, os.path.join(model_dir, 'threshold_model.pkl'))
        
        logger.info(f"Models saved to {model_dir} (pure sklearn objects)")
    
    def load_models(self, model_dir: str = 'models'):
        """Load pre-trained models (pure sklearn objects)"""
        try:
            # Load pure sklearn objects into .model attributes
            self.validation_model.model = joblib.load(os.path.join(model_dir, 'validation_model.pkl'))
            self.pattern_model.model = joblib.load(os.path.join(model_dir, 'pattern_model.pkl'))
            self.threshold_model.model = joblib.load(os.path.join(model_dir, 'threshold_model.pkl'))
            logger.info(f"Models loaded from {model_dir} (pure sklearn objects)")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
            raise e


def main():
    """Main function for testing"""
    logger.info("Starting Unbeatable Sequence Model test")
    
    # Initialize generator
    generator = UnbeatableSequenceGenerator()
    
    # Test with sample hand
    test_hand = [3, 16, 29, 42, 7, 20, 33, 46, 11, 24]  # 10 cards
    
    result = generator.generate_sequence(test_hand, player_count=4)
    
    print("\n" + "="*80)
    print("UNBEATABLE SEQUENCE MODEL - TEST RESULT")
    print("="*80)
    print(f"Hand: {test_hand}")
    print(f"Should declare Báo Sâm: {result['should_declare_bao_sam']}")
    print(f"Unbeatable probability: {result['unbeatable_probability']:.3f}")
    print(f"User threshold: {result['user_threshold']:.3f}")
    print(f"Model confidence: {result['model_confidence']:.3f}")
    print(f"Reason: {result['reason']}")
    
    if result['unbeatable_sequence']:
        print("\nUnbeatable sequence:")
        for i, combo in enumerate(result['unbeatable_sequence']):
            # Use appropriate strength calculation based on context
            strength = ComboAnalyzer.calculate_unbeatable_strength(combo) if context == "bao_sam" else ComboAnalyzer.calculate_combo_strength(combo)
            print(f"  {i+1}. {combo['combo_type']} rank={combo['rank_value']} cards={combo['cards']} strength={strength:.3f}")
    
    if result['sequence_stats']:
        print(f"\nSequence stats: {result['sequence_stats']}")
    
    print("="*80)


if __name__ == "__main__":
    main()
