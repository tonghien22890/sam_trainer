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

class UnbeatableRuleEngine:
    """Rulebase validation layer - chặn bài quá yếu"""
    
    def __init__(self):
        self.rules = {
            'min_total_cards': 10,           # Đủ bài để tạo sequence
            'max_weak_combos': 1,            # Tối đa 1 combo < 0.5 strength
            'min_strong_combos': 1,          # Ít nhất 1 combo >= 0.7 strength
            'min_avg_strength': 0.55,        # Trung bình strength >= 0.55 (allow borderline)
            'min_unbeatable_combos': 1,      # Ít nhất 1 combo strength >= 0.8
        }
        logger.info(f"UnbeatableRuleEngine initialized with rules: {self.rules}")
    
    def validate_hand(self, possible_combos: List[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate hand against rules"""
        if not possible_combos:
            return False, "no_combos_found", {}
        
        # Calculate total cards
        total_cards = sum(len(combo.get('cards', [])) for combo in possible_combos)
        
        # Calculate strengths
        strengths = [self.calculate_combo_strength(combo) for combo in possible_combos]
        
        # Rule checks
        if total_cards < self.rules['min_total_cards']:
            return False, f"insufficient_cards_{total_cards}", {}
        
        weak_combos = sum(1 for s in strengths if s < 0.5)
        if weak_combos > self.rules['max_weak_combos']:
            return False, f"too_many_weak_combos_{weak_combos}", {}
        
        strong_combos = sum(1 for s in strengths if s >= 0.7)
        if strong_combos < self.rules['min_strong_combos']:
            return False, f"insufficient_strong_combos_{strong_combos}", {}
        
        avg_strength = np.mean(strengths)
        if avg_strength < self.rules['min_avg_strength']:
            return False, f"low_avg_strength_{avg_strength:.2f}", {}
        
        unbeatable_combos = sum(1 for s in strengths if s >= 0.8)
        if unbeatable_combos < self.rules['min_unbeatable_combos']:
            return False, f"no_unbeatable_combos_{unbeatable_combos}", {}
        
        strength_profile = {
            'total_cards': total_cards,
            'avg_strength': avg_strength,
            'strong_combos': strong_combos,
            'unbeatable_combos': unbeatable_combos,
            'strengths': strengths
        }
        
        logger.debug(f"Hand validation passed: {strength_profile}")
        return True, "validation_passed", strength_profile
    
    def calculate_combo_strength(self, combo: Dict[str, Any]) -> float:
        """Calculate Sam-specific combo strength with ultra clear tiers"""
        combo_type = combo['combo_type']
        rank_value = combo.get('rank_value', 0)  # 0..12 where 12 == 2, 11 == A
        cards = combo.get('cards', [])

        is_two = (rank_value == 12)
        is_ace = (rank_value == 11)
        is_face = rank_value in (8, 9, 10)  # J, Q, K

        # Straights
        if combo_type == 'straight':
            length = len(cards)
            if length >= 10:
                return 1.0  # Sảnh rồng
            ranks = [c % 13 for c in cards]
            has_ace = any(r == 11 for r in ranks)
            if has_ace:
                return 1.0  # Ace-high straight
            
            if length >= 7:
                return 0.85 + (length - 7) * 0.02  # 7+ cards
            elif length == 6:
                return 0.6 + (rank_value / 11.0) * 0.05  # 6 cards
            elif length == 5:
                return 0.4 + (rank_value / 11.0) * 0.05  # 5 cards
            else:
                # 3-4 cards
                length_bonus = (length - 3) * 0.05
                rank_bonus = (rank_value / 11.0) * 0.02
                return 0.3 + length_bonus + rank_bonus

        # Singles
        if combo_type == 'single':
            if is_two:
                return 1.0
            if is_ace:
                return 0.3
            return 0.1

        # Pairs
        if combo_type == 'pair':
            if is_two:
                return 1.0
            if is_ace:
                return 0.8
            return 0.2 + (min(rank_value, 7) / 7.0) * 0.1

        # Triples
        if combo_type == 'triple':
            if is_two:
                return 1.0
            if is_ace:
                return 0.9
            if is_face:
                return 0.8
            if rank_value >= 4:  # >= 7
                return 0.5
            return 0.3 + (rank_value / 4.0) * 0.05

        # Quads
        if combo_type == 'quad' or combo_type == 'four_kind':
            if is_two:
                return 1.0
            if is_ace:
                return 0.98
            return 0.95 + (rank_value / 11.0) * 0.03

        return 0.1  # Fallback


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
        features = []
        
        # Basic properties
        combo_types = ['single', 'pair', 'triple', 'straight', 'quad']
        combo_type = combo['combo_type']
        for ct in combo_types:
            features.append(1.0 if ct == combo_type else 0.0)
        
        # Rank normalized
        rank_value = combo.get('rank_value', 0)
        features.append(rank_value / 12.0)
        
        # Absolute strength
        rule_engine = UnbeatableRuleEngine()
        strength = rule_engine.calculate_combo_strength(combo)
        features.append(strength)
        
        # Card count
        card_count = len(combo.get('cards', []))
        features.append(card_count / 10.0)  # Normalize by max possible
        
        return features
    
    def extract_sequence_features(self, combos: List[Dict[str, Any]]) -> List[float]:
        """Extract sequence-level features"""
        if not combos:
            return [0.0] * 10
        
        rule_engine = UnbeatableRuleEngine()
        strengths = [rule_engine.calculate_combo_strength(combo) for combo in combos]
        
        features = []
        
        # Strength distribution
        features.append(np.mean(strengths))  # avg_strength
        features.append(np.var(strengths))   # strength_variance
        features.append(max(strengths) - min(strengths))  # strength_range
        
        # Combo distribution
        combo_types = ['single', 'pair', 'triple', 'straight', 'quad']
        type_counts = [sum(1 for c in combos if c['combo_type'] == ct) for ct in combo_types]
        total_combos = len(combos)
        type_distribution = [count / total_combos for count in type_counts]
        features.extend(type_distribution)
        
        # Power indicators
        power_combo_ratio = sum(1 for s in strengths if s >= 0.8) / len(strengths)
        features.append(power_combo_ratio)
        
        # Coverage efficiency
        total_cards = sum(len(combo.get('cards', [])) for combo in combos)
        features.append(total_cards / 10.0)  # Should be 1.0 for valid hands
        
        return features
    
    def extract_features(self, hand_data: Dict[str, Any]) -> List[float]:
        """Extract all features for validation"""
        combos = hand_data.get('possible_combos', [])
        
        features = []
        
        # Sequence-level features
        seq_features = self.extract_sequence_features(combos)
        features.extend(seq_features)
        
        # First 3 combo features
        for i in range(3):
            if i < len(combos):
                combo_features = self.extract_combo_features(combos[i])
                features.extend(combo_features)
            else:
                # Pad with zeros
                features.extend([0.0] * 8)  # 8 combo features
        
        # Context features
        player_count = hand_data.get('player_count', 4)
        features.append(player_count / 4.0)  # Normalize
        
        return features
    
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
        combos = hand_data.get('possible_combos', [])
        
        if not combos:
            return [0.0] * 15
        
        rule_engine = UnbeatableRuleEngine()
        strengths = [rule_engine.calculate_combo_strength(combo) for combo in combos]
        
        features = []
        
        # Combo diversity
        combo_types = set(combo['combo_type'] for combo in combos)
        combo_diversity = len(combo_types) / 5.0  # Max 5 types
        features.append(combo_diversity)
        
        # Power concentration
        power_combos = sum(1 for s in strengths if s >= 0.8)
        power_concentration = power_combos / len(combos)
        features.append(power_concentration)
        
        # Balance preference
        strength_variance = np.var(strengths)
        features.append(strength_variance)
        
        # Type preferences
        combo_types_list = ['single', 'pair', 'triple', 'straight', 'quad']
        type_counts = [sum(1 for c in combos if c['combo_type'] == ct) for ct in combo_types_list]
        total_combos = len(combos)
        type_prefs = [count / total_combos for count in type_counts]
        features.extend(type_prefs)
        
        # Strength distribution
        features.extend([
            np.mean(strengths),
            np.max(strengths),
            np.min(strengths),
            np.median(strengths)
        ])
        
        # Additional pattern signals to reach 15 features
        # Ratio of singles and pairs (indicates weakness pattern)
        singles_ratio = sum(1 for c in combos if c['combo_type'] == 'single') / total_combos
        pairs_ratio = sum(1 for c in combos if c['combo_type'] == 'pair') / total_combos
        features.extend([singles_ratio, pairs_ratio])
        
        # Context
        player_count = hand_data.get('player_count', 4)
        features.append(player_count / 4.0)
        
        return features
    
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
        features = []
        
        # Hand characteristics
        combos = hand_data.get('possible_combos', [])
        if combos:
            rule_engine = UnbeatableRuleEngine()
            strengths = [rule_engine.calculate_combo_strength(combo) for combo in combos]
            
            features.extend([
                np.mean(strengths),
                np.max(strengths),
                len(combos),
                sum(1 for s in strengths if s >= 0.8)  # unbeatable combos
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # User patterns
        patterns = user_patterns.get('combo_patterns', {})
        features.extend([
            patterns.get('power_concentration', 0.5),
            patterns.get('combo_diversity', 0.5),
            patterns.get('balance_preference', 0.5)
        ])
        
        # Context
        player_count = hand_data.get('player_count', 4)
        features.append(player_count / 4.0)
        
        return features
    
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
        return max(0.5, min(0.95, predicted_threshold))


class UnbeatableSequenceGenerator:
    """Phase 3: Generate optimal unbeatable sequences and decide Báo Sâm"""
    
    def __init__(self):
        self.rule_engine = UnbeatableRuleEngine()
        self.validation_model = SequenceValidationModel()
        self.pattern_model = PatternLearningModel()
        self.threshold_model = ThresholdLearningModel()
        logger.info("UnbeatableSequenceGenerator initialized")
    
    def analyze_hand(self, hand: List[int]) -> List[Dict[str, Any]]:
        """Analyze hand and find possible combos"""
        # Strategy:
        # 1) Prefer detect straights (3-10 cards) greedily from available ranks
        # 2) Then detect quads, triples, pairs
        # 3) Remaining as singles
        # All detections consume cards to avoid duplicates across combos

        # Build rank -> list(cards) map
        rank_to_cards: Dict[int, List[int]] = {}
        for card in sorted(hand):
            rank = card % 13
            rank_to_cards.setdefault(rank, []).append(card)

        combos: List[Dict[str, Any]] = []

        # Helper to get available ranks snapshot
        def available_ranks() -> List[int]:
            return sorted([r for r, cards in rank_to_cards.items() if len(cards) > 0])

        # Detect straights greedily (longest runs first)
        while True:
            # In Sam, rank 12 (the 2) is NOT part of straights -> exclude it
            ranks = [r for r in available_ranks() if r != 12]
            if not ranks:
                break

            # Find longest consecutive run from current availability
            best_start = None
            best_len = 0
            i = 0
            while i < len(ranks):
                j = i
                while j + 1 < len(ranks) and ranks[j + 1] == ranks[j] + 1:
                    j += 1
                run_len = j - i + 1
                if run_len > best_len:
                    best_len = run_len
                    best_start = ranks[i]
                i = j + 1

            # Only create straights of length >= 3 (prefer 5+)
            if best_len < 3:
                break

            # Build the straight from best_start with best_len, but cap at 10 (no wrap, no rank 12)
            start = best_start
            end = best_start + best_len - 1
            length = best_len

            # Prefer taking 10..5 length by trimming from ends if needed to hit max 10
            if length > 10:
                length = 10
                end = start + length - 1

            # Consume one card per rank for the straight
            straight_cards: List[int] = []
            for r in range(start, end + 1):
                if r in rank_to_cards and rank_to_cards[r]:
                    straight_cards.append(rank_to_cards[r].pop(0))

            # Validate minimum usable length (after consumption some ranks might be empty)
            if len(straight_cards) >= 3:
                combos.append({
                    'combo_type': 'straight',
                    'rank_value': (end % 13),  # highest rank in straight (never 12)
                    'cards': straight_cards
                })
            else:
                # If failed to build, put back consumed (rare)
                for c in straight_cards:
                    r = c % 13
                    rank_to_cards.setdefault(r, []).insert(0, c)
                break

            # Continue loop to try detect more straights from remaining cards

        # Detect quads, triples, pairs
        for rank in list(sorted(rank_to_cards.keys())):
            cards = rank_to_cards.get(rank, [])
            if not cards:
                continue
            count = len(cards)
            if count >= 4:
                combos.append({'combo_type': 'quad', 'rank_value': rank, 'cards': cards[:4]})
                rank_to_cards[rank] = cards[4:]
            elif count == 3:
                combos.append({'combo_type': 'triple', 'rank_value': rank, 'cards': cards[:3]})
                rank_to_cards[rank] = cards[3:]
            elif count == 2:
                combos.append({'combo_type': 'pair', 'rank_value': rank, 'cards': cards[:2]})
                rank_to_cards[rank] = cards[2:]

        # Remaining singles
        for rank in list(sorted(rank_to_cards.keys())):
            cards = rank_to_cards.get(rank, [])
            for c in cards:
                combos.append({'combo_type': 'single', 'rank_value': rank, 'cards': [c]})
            rank_to_cards[rank] = []

        return combos
    
    def calculate_unbeatable_probability(self, sequence: List[Dict[str, Any]]) -> float:
        """Calculate probability that sequence is unbeatable"""
        if not sequence:
            return 0.0
        
        strengths = [self.rule_engine.calculate_combo_strength(combo) for combo in sequence]
        
        # Simple heuristic based on strengths
        avg_strength = np.mean(strengths)
        max_strength = max(strengths)
        strong_count = sum(1 for s in strengths if s >= 0.8)
        
        # Probability calculation
        prob = (avg_strength * 0.4 + max_strength * 0.4 + (strong_count / len(strengths)) * 0.2)
        return min(1.0, max(0.0, prob))
    
    def build_sequence_from_patterns(self, combos: List[Dict[str, Any]], user_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build sequence based on user patterns"""
        if not combos:
            return []
        
        # For now, return all combos
        # In real implementation, you'd select optimal subset based on patterns
        return combos.copy()
    
    def calculate_model_confidence(self, validation_result: Dict[str, Any], user_patterns: Dict[str, Any]) -> float:
        """Calculate overall model confidence"""
        validation_conf = validation_result.get('confidence', 0.5)
        pattern_score = user_patterns.get('pattern_score', 0.5)
        
        # Combined confidence
        return (validation_conf + pattern_score) / 2.0
    
    def generate_sequence(self, hand: List[int], player_count: int = 4) -> Dict[str, Any]:
        """Generate unbeatable sequence and decide Báo Sâm"""
        logger.info(f"Generating sequence for hand: {hand}")
        
        # Step 1: Analyze hand
        possible_combos = self.analyze_hand(hand)
        
        hand_data = {
            'hand': hand,
            'player_count': player_count,
            'possible_combos': possible_combos
        }
        
        # Step 2: Rulebase validation
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
        
        # Step 3: ML Validation
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
        
        # Step 4: Extract user patterns
        user_patterns = self.pattern_model.predict(hand_data)
        
        # Step 5: Build sequence from patterns
        sequence = self.build_sequence_from_patterns(possible_combos, user_patterns)
        
        # Step 6: Order by power (strongest first)
        ordered_sequence = sorted(sequence, key=lambda combo: -self.rule_engine.calculate_combo_strength(combo))
        
        # Step 7: Calculate unbeatable probability
        unbeatable_prob = self.calculate_unbeatable_probability(ordered_sequence)
        
        # Step 8: Learn user's threshold preference
        user_threshold = self.threshold_model.predict_user_threshold(hand_data, user_patterns)
        
        # Step 9: Decide Báo Sâm based on learned threshold
        should_declare = unbeatable_prob >= user_threshold
        model_confidence = self.calculate_model_confidence(validation_result, user_patterns)
        
        # Calculate sequence stats
        strengths = [self.rule_engine.calculate_combo_strength(combo) for combo in ordered_sequence]
        sequence_stats = {
            'total_cards': sum(len(combo.get('cards', [])) for combo in ordered_sequence),
            'avg_strength': float(np.mean(strengths)),
            'unbeatable_combos': sum(1 for s in strengths if s >= 0.8),
            'pattern_used': user_patterns['sequence_building_preference']
        }
        
        result = {
            'should_declare_bao_sam': bool(should_declare),
            'unbeatable_probability': float(unbeatable_prob),
            'user_threshold': float(user_threshold),
            'model_confidence': float(model_confidence),
            'reason': f'unbeatable_prob_{unbeatable_prob:.2f}_vs_threshold_{user_threshold:.2f}',
            'unbeatable_sequence': ordered_sequence,
            'sequence_stats': sequence_stats
        }
        
        logger.info(f"Sequence generated - Decision: {should_declare}, Prob: {unbeatable_prob:.3f}, Threshold: {user_threshold:.3f}")
        return result
    
    def save_models(self, model_dir: str = 'models'):
        """Save all trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.validation_model, os.path.join(model_dir, 'validation_model.pkl'))
        joblib.dump(self.pattern_model, os.path.join(model_dir, 'pattern_model.pkl'))
        joblib.dump(self.threshold_model, os.path.join(model_dir, 'threshold_model.pkl'))
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = 'models'):
        """Load pre-trained models"""
        try:
            self.validation_model = joblib.load(os.path.join(model_dir, 'validation_model.pkl'))
            self.pattern_model = joblib.load(os.path.join(model_dir, 'pattern_model.pkl'))
            self.threshold_model = joblib.load(os.path.join(model_dir, 'threshold_model.pkl'))
            logger.info(f"Models loaded from {model_dir}")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")


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
            strength = generator.rule_engine.calculate_combo_strength(combo)
            print(f"  {i+1}. {combo['combo_type']} rank={combo['rank_value']} cards={combo['cards']} strength={strength:.3f}")
    
    if result['sequence_stats']:
        print(f"\nSequence stats: {result['sequence_stats']}")
    
    print("="*80)


if __name__ == "__main__":
    main()
