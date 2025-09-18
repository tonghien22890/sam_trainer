#!/usr/bin/env python3
"""
Training Pipeline for Unbeatable Sequence Model
Complete 3-phase training with comprehensive logging and evaluation
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any

from unbeatable_sequence_model import UnbeatableSequenceGenerator
from synthetic_data_generator import SyntheticDataGenerator

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnbeatableModelTrainer:
    """Complete training pipeline for all model phases"""
    
    def __init__(self):
        self.generator = UnbeatableSequenceGenerator()
        self.data_generator = SyntheticDataGenerator()
        self.training_results = {}
        
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        logger.info("UnbeatableModelTrainer initialized")
    
    def phase1_foundation_training(self) -> Dict[str, Any]:
        """Phase 1: Foundation Training - Learn basic valid/invalid patterns"""
        logger.info("="*80)
        logger.info("PHASE 1: FOUNDATION TRAINING")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Generate training data
        logger.info("Generating Phase 1 training data...")
        validation_data = self.data_generator.generate_validation_data(1000)
        
        # Save data
        data_file = 'data/phase1_validation_data.jsonl'
        self.data_generator.save_data(validation_data, data_file)
        
        # Train validation model
        logger.info("Training SequenceValidationModel...")
        training_results = self.generator.validation_model.train(validation_data)
        
        # Evaluate on test set
        test_data = self.data_generator.generate_validation_data(200)
        correct_predictions = 0
        total_predictions = 0
        
        for record in test_data:
            prediction = self.generator.validation_model.predict(record)
            expected = record.get('is_valid', False)
            actual = prediction['is_valid']
            
            if expected == actual:
                correct_predictions += 1
            total_predictions += 1
        
        test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        duration = time.time() - start_time
        
        phase1_results = {
            'phase': 'Phase 1 - Foundation',
            'duration_seconds': duration,
            'training_samples': len(validation_data),
            'test_samples': len(test_data),
            'training_accuracy': training_results.get('accuracy', 0),
            'cv_accuracy': training_results.get('cv_mean', 0),
            'cv_std': training_results.get('cv_std', 0),
            'test_accuracy': test_accuracy,
            'success_metric_target': 0.85,
            'success_metric_achieved': training_results.get('accuracy', 0),
            'status': 'PASSED' if training_results.get('accuracy', 0) >= 0.85 else 'NEEDS_IMPROVEMENT'
        }
        
        self.training_results['phase1'] = phase1_results
        
        logger.info(f"Phase 1 Results:")
        logger.info(f"  Training Accuracy: {training_results.get('accuracy', 0):.3f}")
        logger.info(f"  CV Accuracy: {training_results.get('cv_mean', 0):.3f} ± {training_results.get('cv_std', 0):.3f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.3f}")
        logger.info(f"  Duration: {duration:.1f} seconds")
        logger.info(f"  Status: {phase1_results['status']}")
        
        return phase1_results
    
    def phase2_pattern_learning(self) -> Dict[str, Any]:
        """Phase 2: Pattern Learning - Learn user combo building patterns"""
        logger.info("="*80)
        logger.info("PHASE 2: PATTERN LEARNING")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Generate training data
        logger.info("Generating Phase 2 training data...")
        pattern_data = self.data_generator.generate_pattern_data(2000)
        
        # Save data
        data_file = 'data/phase2_pattern_data.jsonl'
        self.data_generator.save_data(pattern_data, data_file)
        
        # Train pattern model
        logger.info("Training PatternLearningModel...")
        training_results = self.generator.pattern_model.train(pattern_data)
        
        # Evaluate pattern consistency
        test_data = self.data_generator.generate_pattern_data(300)
        pattern_consistency_scores = []
        
        for record in test_data:
            prediction = self.generator.pattern_model.predict(record)
            expected_score = record.get('pattern_score', 0.5)
            predicted_score = prediction.get('pattern_score', 0.5)
            
            # Calculate consistency (1 - absolute difference)
            consistency = 1.0 - abs(expected_score - predicted_score)
            pattern_consistency_scores.append(consistency)
        
        pattern_consistency = sum(pattern_consistency_scores) / len(pattern_consistency_scores) if pattern_consistency_scores else 0
        
        duration = time.time() - start_time
        
        phase2_results = {
            'phase': 'Phase 2 - Pattern Learning',
            'duration_seconds': duration,
            'training_samples': len(pattern_data),
            'test_samples': len(test_data),
            'training_rmse': training_results.get('rmse', 0),
            'pattern_consistency': pattern_consistency,
            'success_metric_target': 0.8,
            'success_metric_achieved': pattern_consistency,
            'status': 'PASSED' if pattern_consistency >= 0.8 else 'NEEDS_IMPROVEMENT'
        }
        
        self.training_results['phase2'] = phase2_results
        
        logger.info(f"Phase 2 Results:")
        logger.info(f"  Training RMSE: {training_results.get('rmse', 0):.3f}")
        logger.info(f"  Pattern Consistency: {pattern_consistency:.3f}")
        logger.info(f"  Duration: {duration:.1f} seconds")
        logger.info(f"  Status: {phase2_results['status']}")
        
        return phase2_results
    
    def phase3_threshold_optimization(self) -> Dict[str, Any]:
        """Phase 3: Threshold Optimization - Learn user decision thresholds"""
        logger.info("="*80)
        logger.info("PHASE 3: THRESHOLD OPTIMIZATION")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Generate training data
        logger.info("Generating Phase 3 training data...")
        threshold_data = self.data_generator.generate_threshold_data(1500)
        
        # Save data
        data_file = 'data/phase3_threshold_data.jsonl'
        self.data_generator.save_data(threshold_data, data_file)
        
        # Train threshold model
        logger.info("Training ThresholdLearningModel...")
        training_results = self.generator.threshold_model.train(threshold_data)
        
        # Evaluate threshold prediction accuracy
        test_data = self.data_generator.generate_threshold_data(200)
        threshold_accuracy_scores = []
        decision_accuracy_scores = []
        
        for record in test_data:
            user_patterns = record.get('user_patterns', {})
            predicted_threshold = self.generator.threshold_model.predict_user_threshold(record, user_patterns)
            expected_threshold = record.get('user_threshold', 0.75)
            
            # Threshold accuracy (1 - normalized absolute difference)
            threshold_diff = abs(predicted_threshold - expected_threshold)
            threshold_accuracy = max(0, 1.0 - (threshold_diff / 0.5))  # Normalize by max possible diff
            threshold_accuracy_scores.append(threshold_accuracy)
            
            # Decision accuracy (would both make same decision?)
            unbeatable_prob = record.get('unbeatable_probability', 0.5)
            expected_decision = unbeatable_prob >= expected_threshold
            predicted_decision = unbeatable_prob >= predicted_threshold
            
            decision_accuracy_scores.append(1.0 if expected_decision == predicted_decision else 0.0)
        
        threshold_accuracy = sum(threshold_accuracy_scores) / len(threshold_accuracy_scores) if threshold_accuracy_scores else 0
        decision_accuracy = sum(decision_accuracy_scores) / len(decision_accuracy_scores) if decision_accuracy_scores else 0
        
        duration = time.time() - start_time
        
        phase3_results = {
            'phase': 'Phase 3 - Threshold Optimization',
            'duration_seconds': duration,
            'training_samples': len(threshold_data),
            'test_samples': len(test_data),
            'training_rmse': training_results.get('rmse', 0),
            'threshold_accuracy': threshold_accuracy,
            'decision_accuracy': decision_accuracy,
            'success_metric_target': 0.75,
            'success_metric_achieved': decision_accuracy,
            'status': 'PASSED' if decision_accuracy >= 0.75 else 'NEEDS_IMPROVEMENT'
        }
        
        self.training_results['phase3'] = phase3_results
        
        logger.info(f"Phase 3 Results:")
        logger.info(f"  Training RMSE: {training_results.get('rmse', 0):.3f}")
        logger.info(f"  Threshold Accuracy: {threshold_accuracy:.3f}")
        logger.info(f"  Decision Accuracy: {decision_accuracy:.3f}")
        logger.info(f"  Duration: {duration:.1f} seconds")
        logger.info(f"  Status: {phase3_results['status']}")
        
        return phase3_results
    
    def evaluate_end_to_end(self) -> Dict[str, Any]:
        """End-to-end evaluation with realistic scenarios"""
        logger.info("="*80)
        logger.info("END-TO-END EVALUATION")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Premium Hand (Quad 2s)',
                'hand': [12, 25, 38, 51, 10, 23, 36, 8, 21, 34],
                'expected_declare': True,
                'min_probability': 0.8
            },
            {
                'name': 'Strong Hand (Triple A + Triple K)',
                'hand': [11, 24, 37, 10, 23, 36, 9, 22, 35, 0],
                'expected_declare': True,
                'min_probability': 0.6
            },
            {
                'name': 'Medium Hand (Mixed triples)',
                'hand': [7, 20, 33, 6, 19, 32, 5, 18, 31, 0],
                'expected_declare': None,  # Could go either way
                'min_probability': 0.3
            },
            {
                'name': 'Weak Hand (Low singles)',
                'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'expected_declare': False,
                'min_probability': 0.0
            }
        ]
        
        scenario_results = []
        
        for scenario in test_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            
            result = self.generator.generate_sequence(scenario['hand'], player_count=4)
            
            scenario_result = {
                'name': scenario['name'],
                'hand': scenario['hand'],
                'should_declare': result['should_declare_bao_sam'],
                'unbeatable_probability': result['unbeatable_probability'],
                'user_threshold': result['user_threshold'],
                'model_confidence': result['model_confidence'],
                'reason': result['reason'],
                'expected_declare': scenario['expected_declare'],
                'min_probability': scenario['min_probability'],
                'test_passed': True
            }
            
            # Validate results
            if scenario['expected_declare'] is not None:
                if result['should_declare_bao_sam'] != scenario['expected_declare']:
                    scenario_result['test_passed'] = False
                    logger.warning(f"  Declaration mismatch: expected {scenario['expected_declare']}, got {result['should_declare_bao_sam']}")
            
            if result['unbeatable_probability'] < scenario['min_probability']:
                scenario_result['test_passed'] = False
                logger.warning(f"  Probability too low: expected >={scenario['min_probability']}, got {result['unbeatable_probability']:.3f}")
            
            if scenario_result['test_passed']:
                logger.info(f"  PASSED - Declare: {result['should_declare_bao_sam']}, Prob: {result['unbeatable_probability']:.3f}")
            
            scenario_results.append(scenario_result)
        
        # Calculate overall success rate
        passed_scenarios = sum(1 for s in scenario_results if s['test_passed'])
        success_rate = passed_scenarios / len(scenario_results)
        
        duration = time.time() - start_time
        
        e2e_results = {
            'phase': 'End-to-End Evaluation',
            'duration_seconds': duration,
            'total_scenarios': len(scenario_results),
            'passed_scenarios': passed_scenarios,
            'success_rate': success_rate,
            'scenarios': scenario_results,
            'success_metric_target': 0.75,
            'success_metric_achieved': success_rate,
            'status': 'PASSED' if success_rate >= 0.75 else 'NEEDS_IMPROVEMENT'
        }
        
        self.training_results['end_to_end'] = e2e_results
        
        logger.info(f"End-to-End Results:")
        logger.info(f"  Scenarios Passed: {passed_scenarios}/{len(scenario_results)}")
        logger.info(f"  Success Rate: {success_rate:.1%}")
        logger.info(f"  Duration: {duration:.1f} seconds")
        logger.info(f"  Status: {e2e_results['status']}")
        
        return e2e_results
    
    def save_training_results(self):
        """Save comprehensive training results"""
        results_file = f'logs/training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training results saved to {results_file}")
    
    def save_models(self):
        """Save all trained models"""
        self.generator.save_models('models')
        logger.info("All models saved to models/ directory")
    
    def run_complete_training(self) -> Dict[str, Any]:
        """Run complete 3-phase training pipeline"""
        logger.info("STARTING COMPLETE TRAINING PIPELINE")
        logger.info("="*80)
        
        overall_start_time = time.time()
        
        # Phase 1: Foundation
        phase1_results = self.phase1_foundation_training()
        
        # Phase 2: Pattern Learning
        phase2_results = self.phase2_pattern_learning()
        
        # Phase 3: Threshold Optimization
        phase3_results = self.phase3_threshold_optimization()
        
        # End-to-End Evaluation
        e2e_results = self.evaluate_end_to_end()
        
        # Save models
        self.save_models()
        
        # Calculate overall results
        overall_duration = time.time() - overall_start_time
        
        all_phases_passed = all(
            results['status'] == 'PASSED' 
            for results in [phase1_results, phase2_results, phase3_results, e2e_results]
        )
        
        overall_results = {
            'training_completed': datetime.now().isoformat(),
            'total_duration_seconds': overall_duration,
            'total_duration_minutes': overall_duration / 60,
            'all_phases_passed': all_phases_passed,
            'overall_status': 'SUCCESS' if all_phases_passed else 'PARTIAL_SUCCESS',
            'phases': {
                'phase1': phase1_results,
                'phase2': phase2_results,
                'phase3': phase3_results,
                'end_to_end': e2e_results
            }
        }
        
        self.training_results['overall'] = overall_results
        
        # Save results
        self.save_training_results()
        
        # Print final summary
        logger.info("="*80)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("="*80)
        logger.info(f"Total Duration: {overall_duration/60:.1f} minutes")
        logger.info(f"Overall Status: {overall_results['overall_status']}")
        logger.info("")
        logger.info("Phase Results:")
        logger.info(f"  Phase 1 (Foundation): {phase1_results['status']}")
        logger.info(f"  Phase 2 (Pattern Learning): {phase2_results['status']}")
        logger.info(f"  Phase 3 (Threshold Optimization): {phase3_results['status']}")
        logger.info(f"  End-to-End Evaluation: {e2e_results['status']}")
        
        if all_phases_passed:
            logger.info("")
            logger.info("ALL PHASES PASSED - MODEL READY FOR PRODUCTION!")
        else:
            logger.info("")
            logger.info("SOME PHASES NEED IMPROVEMENT - REVIEW RESULTS")
        
        logger.info("="*80)
        
        return overall_results


def main():
    """Main training function"""
    trainer = UnbeatableModelTrainer()
    results = trainer.run_complete_training()
    
    # Exit with appropriate code
    exit(0 if results['all_phases_passed'] else 1)


if __name__ == "__main__":
    main()
