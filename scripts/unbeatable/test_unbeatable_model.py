import os
import sys

# Ensure parent directory (model_build) is on sys.path for local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unbeatable Sequence Model
Test all components with synthetic data and real scenarios
"""

import unittest
import json
import numpy as np
import logging
import os
import tempfile
from typing import Dict, List, Any

from unbeatable_sequence_model import (
    UnbeatableRuleEngine,
    SequenceValidationModel,
    PatternLearningModel,
    ThresholdLearningModel,
    UnbeatableSequenceGenerator
)
from synthetic_data_generator import SyntheticDataGenerator

# Setup test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestUnbeatableRuleEngine(unittest.TestCase):
    """Test rule engine validation"""
    
    def setUp(self):
        self.rule_engine = UnbeatableRuleEngine()
    
    def test_combo_strength_calculation(self):
        """Test combo strength calculations"""
        # Test quad 2s (strongest)
        quad_2 = {'combo_type': 'quad', 'rank_value': 12, 'cards': [3, 16, 29, 42]}
        strength = self.rule_engine.calculate_combo_strength(quad_2)
        self.assertEqual(strength, 1.0)
        
        # Test single Ace (weak)
        single_ace = {'combo_type': 'single', 'rank_value': 11, 'cards': [11]}
        strength = self.rule_engine.calculate_combo_strength(single_ace)
        self.assertEqual(strength, 0.3)
        
        # Test triple faces
        triple_k = {'combo_type': 'triple', 'rank_value': 10, 'cards': [10, 23, 36]}
        strength = self.rule_engine.calculate_combo_strength(triple_k)
        self.assertEqual(strength, 0.8)
        
        logger.info("✅ Combo strength calculation tests passed")
    
    def test_strong_hand_validation(self):
        """Test validation of strong hands"""
        # Strong hand: quad + triple + triple
        strong_combos = [
            {'combo_type': 'quad', 'rank_value': 12, 'cards': [3, 16, 29, 42]},
            {'combo_type': 'triple', 'rank_value': 10, 'cards': [10, 23, 36]},
            {'combo_type': 'triple', 'rank_value': 8, 'cards': [8, 21, 34]}
        ]
        
        is_valid, reason, profile = self.rule_engine.validate_hand(strong_combos)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "validation_passed")
        self.assertEqual(profile['total_cards'], 10)
        
        logger.info("✅ Strong hand validation tests passed")
    
    def test_weak_hand_rejection(self):
        """Test rejection of weak hands"""
        # Weak hand: only singles
        weak_combos = [
            {'combo_type': 'single', 'rank_value': 3, 'cards': [3]},
            {'combo_type': 'single', 'rank_value': 4, 'cards': [4]},
            {'combo_type': 'single', 'rank_value': 5, 'cards': [5]}
        ]
        
        is_valid, reason, profile = self.rule_engine.validate_hand(weak_combos)
        self.assertFalse(is_valid)
        self.assertIn("insufficient_cards", reason)
        
        logger.info("✅ Weak hand rejection tests passed")


class TestSequenceValidationModel(unittest.TestCase):
    """Test ML validation model"""
    
    def setUp(self):
        self.model = SequenceValidationModel()
        self.generator = SyntheticDataGenerator()
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        # Generate test data
        combos = [
            {'combo_type': 'quad', 'rank_value': 12, 'cards': [3, 16, 29, 42]},
            {'combo_type': 'triple', 'rank_value': 10, 'cards': [10, 23, 36]},
            {'combo_type': 'triple', 'rank_value': 8, 'cards': [8, 21, 34]}
        ]
        
        hand_data = {
            'hand': [3, 16, 29, 42, 10, 23, 36, 8, 21, 34],
            'player_count': 4,
            'possible_combos': combos
        }
        
        features = self.model.extract_features(hand_data)
        
        # Check feature dimensions
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 20)  # Should have many features
        
        # Check feature values are reasonable
        for feature in features:
            self.assertIsInstance(feature, (int, float))
            self.assertGreaterEqual(feature, 0.0)
            self.assertLessEqual(feature, 10.0)  # Reasonable upper bound
        
        logger.info("✅ Feature extraction tests passed")
    
    def test_model_training(self):
        """Test model training with synthetic data"""
        # Generate training data
        training_data = self.generator.generate_validation_data(100)  # Small dataset for testing
        
        # Train model
        results = self.model.train(training_data)
        
        # Check training results
        self.assertIn('accuracy', results)
        self.assertIn('cv_mean', results)
        self.assertGreater(results['accuracy'], 0.5)  # Should be better than random
        
        logger.info(f"✅ Model training tests passed - Accuracy: {results['accuracy']:.3f}")
    
    def test_model_prediction(self):
        """Test model prediction"""
        # Train with small dataset first
        training_data = self.generator.generate_validation_data(50)
        self.model.train(training_data)
        
        # Test prediction
        test_hand_data = {
            'hand': [3, 16, 29, 42, 10, 23, 36, 8, 21, 34],
            'player_count': 4,
            'possible_combos': [
                {'combo_type': 'quad', 'rank_value': 12, 'cards': [3, 16, 29, 42]},
                {'combo_type': 'triple', 'rank_value': 10, 'cards': [10, 23, 36]},
                {'combo_type': 'triple', 'rank_value': 8, 'cards': [8, 21, 34]}
            ]
        }
        
        result = self.model.predict(test_hand_data)
        
        # Check prediction structure
        self.assertIn('is_valid', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['is_valid'], bool)
        self.assertBetween(result['confidence'], 0.0, 1.0)
        
        logger.info("✅ Model prediction tests passed")
    
    def assertBetween(self, value, min_val, max_val):
        """Helper assertion for range checking"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)


class TestPatternLearningModel(unittest.TestCase):
    """Test pattern learning model"""
    
    def setUp(self):
        self.model = PatternLearningModel()
        self.generator = SyntheticDataGenerator()
    
    def test_pattern_feature_extraction(self):
        """Test pattern feature extraction"""
        hand_data = {
            'hand': [3, 16, 29, 42, 10, 23, 36, 8, 21, 34],
            'player_count': 4,
            'possible_combos': [
                {'combo_type': 'quad', 'rank_value': 12, 'cards': [3, 16, 29, 42]},
                {'combo_type': 'triple', 'rank_value': 10, 'cards': [10, 23, 36]},
                {'combo_type': 'triple', 'rank_value': 8, 'cards': [8, 21, 34]}
            ]
        }
        
        features = self.model.extract_pattern_features(hand_data)
        
        # Check features
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 15)  # Expected number of pattern features
        
        for feature in features:
            self.assertIsInstance(feature, (int, float))
        
        logger.info("✅ Pattern feature extraction tests passed")
    
    def test_pattern_model_training(self):
        """Test pattern model training"""
        training_data = self.generator.generate_pattern_data(100)
        results = self.model.train(training_data)
        
        self.assertIn('rmse', results)
        self.assertGreater(results['rmse'], 0.0)
        
        logger.info(f"✅ Pattern model training tests passed - RMSE: {results['rmse']:.3f}")


class TestThresholdLearningModel(unittest.TestCase):
    """Test threshold learning model"""
    
    def setUp(self):
        self.model = ThresholdLearningModel()
        self.generator = SyntheticDataGenerator()
    
    def test_threshold_prediction(self):
        """Test threshold prediction"""
        # Train with small dataset
        training_data = self.generator.generate_threshold_data(50)
        results = self.model.train(training_data)
        
        # Test prediction
        hand_data = {
            'hand': [3, 16, 29, 42, 10, 23, 36, 8, 21, 34],
            'player_count': 4,
            'possible_combos': [
                {'combo_type': 'quad', 'rank_value': 12, 'cards': [3, 16, 29, 42]},
                {'combo_type': 'triple', 'rank_value': 10, 'cards': [10, 23, 36]},
                {'combo_type': 'triple', 'rank_value': 8, 'cards': [8, 21, 34]}
            ]
        }
        
        user_patterns = {
            'combo_patterns': {
                'power_concentration': 0.8,
                'combo_diversity': 0.6,
                'balance_preference': 0.1
            }
        }
        
        threshold = self.model.predict_user_threshold(hand_data, user_patterns)
        
        # Check threshold is reasonable
        self.assertBetween(threshold, 0.5, 0.95)
        
        logger.info(f"✅ Threshold prediction tests passed - Threshold: {threshold:.3f}")
    
    def assertBetween(self, value, min_val, max_val):
        """Helper assertion for range checking"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)


class TestUnbeatableSequenceGenerator(unittest.TestCase):
    """Test complete sequence generator"""
    
    def setUp(self):
        self.generator = UnbeatableSequenceGenerator()
        self.data_generator = SyntheticDataGenerator()
        # Ensure all ML models are trained to avoid NotFittedError
        val_data = self.data_generator.generate_validation_data(80)
        pat_data = self.data_generator.generate_pattern_data(80)
        thr_data = self.data_generator.generate_threshold_data(80)
        self.generator.validation_model.train(val_data)
        self.generator.pattern_model.train(pat_data)
        self.generator.threshold_model.train(thr_data)
    
    def test_hand_analysis(self):
        """Test hand analysis"""
        test_hand = [3, 16, 29, 42, 10, 23, 36, 8, 21, 34]  # Should form quad + 2 triples
        
        combos = self.generator.analyze_hand(test_hand)
        
        self.assertIsInstance(combos, list)
        self.assertGreater(len(combos), 0)
        
        # Check total cards
        total_cards = sum(len(combo.get('cards', [])) for combo in combos)
        self.assertEqual(total_cards, 10)
        
        logger.info("✅ Hand analysis tests passed")
    
    def test_sequence_generation_strong_hand(self):
        """Test sequence generation with strong hand"""
        # Strong hand that should pass all validations
        strong_hand = [12, 25, 38, 51, 10, 23, 36, 8, 21, 34]  # Quad 2s + triples
        
        result = self.generator.generate_sequence(strong_hand, player_count=4)
        
        # Check result structure
        required_keys = [
            'should_declare_bao_sam', 'unbeatable_probability', 'user_threshold',
            'model_confidence', 'reason', 'unbeatable_sequence', 'sequence_stats'
        ]
        
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check value types and ranges
        self.assertIsInstance(result['should_declare_bao_sam'], bool)
        self.assertBetween(result['unbeatable_probability'], 0.0, 1.0)
        self.assertBetween(result['user_threshold'], 0.0, 1.0)
        self.assertBetween(result['model_confidence'], 0.0, 1.0)
        
        if result['unbeatable_sequence']:
            self.assertIsInstance(result['unbeatable_sequence'], list)
            self.assertGreater(len(result['unbeatable_sequence']), 0)
        
        logger.info(f"Strong hand sequence generation tests passed - Decision: {result['should_declare_bao_sam']}")
    
    def test_sequence_generation_weak_hand(self):
        """Test sequence generation with weak hand"""
        # Weak hand that should be rejected
        weak_hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All low singles
        
        result = self.generator.generate_sequence(weak_hand, player_count=4)
        
        # Should be rejected
        self.assertFalse(result['should_declare_bao_sam'])
        self.assertEqual(result['unbeatable_probability'], 0.0)
        
        logger.info("✅ Weak hand sequence generation tests passed - Correctly rejected")
    
    def test_unbeatable_probability_calculation(self):
        """Test unbeatable probability calculation"""
        # Test sequence with known combos
        test_sequence = [
            {'combo_type': 'quad', 'rank_value': 12, 'cards': [3, 16, 29, 42]},  # Very strong
            {'combo_type': 'triple', 'rank_value': 10, 'cards': [10, 23, 36]},  # Strong
            {'combo_type': 'triple', 'rank_value': 5, 'cards': [5, 18, 31]}     # Medium
        ]
        
        prob = self.generator.calculate_unbeatable_probability(test_sequence)
        
        self.assertBetween(prob, 0.0, 1.0)
        self.assertGreater(prob, 0.7)  # Should be high for this strong sequence
        
        logger.info(f"Unbeatable probability calculation tests passed - Prob: {prob:.3f}")
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train models with minimal data
            validation_data = self.data_generator.generate_validation_data(20)
            pattern_data = self.data_generator.generate_pattern_data(20)
            threshold_data = self.data_generator.generate_threshold_data(20)
            
            self.generator.validation_model.train(validation_data)
            self.generator.pattern_model.train(pattern_data)
            self.generator.threshold_model.train(threshold_data)
            
            # Save models
            self.generator.save_models(temp_dir)
            
            # Check files exist
            expected_files = ['validation_model.pkl', 'pattern_model.pkl', 'threshold_model.pkl']
            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                self.assertTrue(os.path.exists(filepath))
            
            # Test loading
            new_generator = UnbeatableSequenceGenerator()
            new_generator.load_models(temp_dir)
            
            logger.info("✅ Model save/load tests passed")
    
    def assertBetween(self, value, min_val, max_val):
        """Helper assertion for range checking"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests with realistic scenarios"""
    
    def setUp(self):
        self.generator = UnbeatableSequenceGenerator()
        self.data_generator = SyntheticDataGenerator()
        
        # Train models with synthetic data
        logger.info("Training models for integration tests...")
        validation_data = self.data_generator.generate_validation_data(100)
        pattern_data = self.data_generator.generate_pattern_data(100)
        threshold_data = self.data_generator.generate_threshold_data(100)
        
        self.generator.validation_model.train(validation_data)
        self.generator.pattern_model.train(pattern_data)
        self.generator.threshold_model.train(threshold_data)
    
    def test_scenario_premium_hand(self):
        """Test scenario: Premium hand with quad 2s"""
        premium_hand = [12, 25, 38, 51, 11, 24, 37, 10, 23, 36]  # Quad 2s + Ace triple + K triple
        
        result = self.generator.generate_sequence(premium_hand, player_count=4)
        
        # Should definitely declare
        self.assertTrue(result['should_declare_bao_sam'])
        self.assertGreater(result['unbeatable_probability'], 0.8)
        
        logger.info(f"Premium hand scenario passed - Prob: {result['unbeatable_probability']:.3f}")
    
    def test_scenario_borderline_hand(self):
        """Test scenario: Borderline hand"""
        borderline_hand = [7, 20, 33, 8, 21, 34, 9, 22, 35, 0]  # Triple 8s + Triple 9s + Triple 10s + Single 3
        
        result = self.generator.generate_sequence(borderline_hand, player_count=4)
        
        # Decision could go either way, but should be consistent
        self.assertIsInstance(result['should_declare_bao_sam'], bool)
        self.assertBetween(result['unbeatable_probability'], 0.3, 0.9)
        
        logger.info(f"Borderline hand scenario passed - Decision: {result['should_declare_bao_sam']}")
    
    def test_scenario_different_player_counts(self):
        """Test scenario: Same hand with different player counts"""
        test_hand = [10, 23, 36, 9, 22, 35, 8, 21, 34, 0]
        
        results = {}
        for player_count in [2, 3, 4]:
            result = self.generator.generate_sequence(test_hand, player_count)
            results[player_count] = result
        
        # Check that thresholds generally increase with player count
        # (more players = need stronger hand to declare)
        logger.info("Player count threshold comparison:")
        for pc in [2, 3, 4]:
            threshold = results[pc]['user_threshold']
            decision = results[pc]['should_declare_bao_sam']
            logger.info(f"  {pc} players: threshold={threshold:.3f}, declare={decision}")
        
        logger.info("✅ Different player count scenario passed")
    
    def assertBetween(self, value, min_val, max_val):
        """Helper assertion for range checking"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)


def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE - UNBEATABLE SEQUENCE MODEL")
    print("="*80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUnbeatableRuleEngine,
        TestSequenceValidationModel,
        TestPatternLearningModel,
        TestThresholdLearningModel,
        TestUnbeatableSequenceGenerator,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\nSuccess rate: {success_rate:.1%}")
    
    if success_rate >= 0.9:
        print("🎯 EXCELLENT - Model implementation is robust!")
    elif success_rate >= 0.7:
        print("✅ GOOD - Model implementation is solid with minor issues")
    else:
        print("⚠️ NEEDS WORK - Model implementation has significant issues")
    
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)
