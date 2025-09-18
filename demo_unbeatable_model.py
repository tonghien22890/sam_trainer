#!/usr/bin/env python3
"""
Demo Script for Unbeatable Sequence Model
Interactive demonstration of the complete system
"""

import json
import random
import logging
from typing import Dict, List, Any

from unbeatable_sequence_model import UnbeatableSequenceGenerator
from synthetic_data_generator import SyntheticDataGenerator
from train_unbeatable_model import UnbeatableModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnbeatableModelDemo:
    """Interactive demo of the Unbeatable Sequence Model"""
    
    def __init__(self):
        self.generator = UnbeatableSequenceGenerator()
        self.data_generator = SyntheticDataGenerator()
        self.is_trained = False
        
        print("🎯 Unbeatable Sequence Model - Interactive Demo")
        print("="*60)
    
    def quick_train_models(self):
        """Quick training with minimal data for demo purposes"""
        print("🚀 Quick training models for demo...")
        
        # Generate minimal training data
        validation_data = self.data_generator.generate_validation_data(100)
        pattern_data = self.data_generator.generate_pattern_data(100)
        threshold_data = self.data_generator.generate_threshold_data(100)
        
        # Train models
        self.generator.validation_model.train(validation_data)
        self.generator.pattern_model.train(pattern_data)
        self.generator.threshold_model.train(threshold_data)
        
        self.is_trained = True
        print("✅ Models trained successfully!")
        print()
    
    def demo_hand_scenarios(self):
        """Demonstrate different hand scenarios"""
        print("📋 DEMO: Different Hand Scenarios")
        print("-" * 40)
        
        scenarios = [
            {
                'name': '🏆 PREMIUM: Quad 2s + Strong Triples',
                'hand': [12, 25, 38, 51, 11, 24, 37, 10, 23, 36],
                'description': 'Quad 2s (strongest) + Ace triple + King triple'
            },
            {
                'name': '💪 STRONG: High Triples',
                'hand': [11, 24, 37, 10, 23, 36, 9, 22, 35, 8],
                'description': 'Ace triple + King triple + Queen triple + Jack single'
            },
            {
                'name': '🤔 MEDIUM: Mixed Strength',
                'hand': [7, 20, 33, 6, 19, 32, 5, 18, 4, 3],
                'description': '8 triple + 7 triple + 6 pair + low singles'
            },
            {
                'name': '😬 BORDERLINE: Low Triples',
                'hand': [4, 17, 30, 3, 16, 29, 2, 15, 1, 0],
                'description': '5 triple + 4 triple + 3 pair + low singles'
            },
            {
                'name': '❌ WEAK: All Singles',
                'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'description': 'All low singles - should be rejected'
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            print(f"   Hand: {scenario['hand']}")
            
            result = self.generator.generate_sequence(scenario['hand'], player_count=4)
            
            # Display decision
            decision_emoji = "✅" if result['should_declare_bao_sam'] else "❌"
            print(f"   Decision: {decision_emoji} {'DECLARE BAO SAM' if result['should_declare_bao_sam'] else 'DO NOT DECLARE'}")
            print(f"   Unbeatable Probability: {result['unbeatable_probability']:.3f}")
            print(f"   User Threshold: {result['user_threshold']:.3f}")
            print(f"   Model Confidence: {result['model_confidence']:.3f}")
            print(f"   Reason: {result['reason']}")
            
            # Display sequence if declaring
            if result['should_declare_bao_sam'] and result['unbeatable_sequence']:
                print("   Optimal Sequence:")
                for j, combo in enumerate(result['unbeatable_sequence'], 1):
                    strength = self.generator.rule_engine.calculate_combo_strength(combo)
                    combo_desc = f"{combo['combo_type']} rank={combo['rank_value']}"
                    print(f"     {j}. {combo_desc} (strength={strength:.3f})")
            
            print()
    
    def demo_player_count_effects(self):
        """Demonstrate how player count affects decisions"""
        print("👥 DEMO: Player Count Effects")
        print("-" * 40)
        
        test_hand = [9, 22, 35, 8, 21, 34, 7, 20, 33, 0]  # Medium strength hand
        print(f"Test Hand: {test_hand}")
        print("Description: 10 triple + 9 triple + 8 triple + 3 single")
        print()
        
        for player_count in [2, 3, 4]:
            result = self.generator.generate_sequence(test_hand, player_count)
            
            decision_emoji = "✅" if result['should_declare_bao_sam'] else "❌"
            print(f"{player_count} Players:")
            print(f"  Decision: {decision_emoji} {'DECLARE' if result['should_declare_bao_sam'] else 'DO NOT DECLARE'}")
            print(f"  Threshold: {result['user_threshold']:.3f}")
            print(f"  Unbeatable Prob: {result['unbeatable_probability']:.3f}")
            print()
    
    def demo_interactive_mode(self):
        """Interactive mode - user inputs custom hands"""
        print("🎮 INTERACTIVE MODE")
        print("-" * 40)
        print("Enter your own 10-card hand to test!")
        print("Cards should be numbers 0-51 (0-12=♠, 13-25=♥, 26-38=♦, 39-51=♣)")
        print("Example: 12,25,38,51,11,24,37,10,23,36")
        print("Or type 'random' for a random hand, 'quit' to exit")
        print()
        
        while True:
            try:
                user_input = input("Enter hand (or 'random'/'quit'): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'random':
                    hand = self.data_generator.generate_hand()
                    print(f"Random hand generated: {hand}")
                else:
                    hand = [int(x.strip()) for x in user_input.split(',')]
                    if len(hand) != 10:
                        print("❌ Please enter exactly 10 cards!")
                        continue
                    if any(card < 0 or card > 51 for card in hand):
                        print("❌ Cards must be between 0-51!")
                        continue
                
                # Analyze hand
                result = self.generator.generate_sequence(hand, player_count=4)
                
                print("\n" + "="*50)
                print("ANALYSIS RESULT")
                print("="*50)
                
                decision_emoji = "✅" if result['should_declare_bao_sam'] else "❌"
                print(f"Decision: {decision_emoji} {'DECLARE BAO SAM' if result['should_declare_bao_sam'] else 'DO NOT DECLARE'}")
                print(f"Unbeatable Probability: {result['unbeatable_probability']:.3f}")
                print(f"User Threshold: {result['user_threshold']:.3f}")
                print(f"Model Confidence: {result['model_confidence']:.3f}")
                print(f"Reason: {result['reason']}")
                
                if result['sequence_stats']:
                    stats = result['sequence_stats']
                    print(f"\nSequence Stats:")
                    print(f"  Total Cards: {stats['total_cards']}")
                    print(f"  Average Strength: {stats['avg_strength']:.3f}")
                    print(f"  Unbeatable Combos: {stats['unbeatable_combos']}")
                    print(f"  Pattern Used: {stats['pattern_used']}")
                
                if result['should_declare_bao_sam'] and result['unbeatable_sequence']:
                    print(f"\nOptimal Play Sequence:")
                    for i, combo in enumerate(result['unbeatable_sequence'], 1):
                        strength = self.generator.rule_engine.calculate_combo_strength(combo)
                        cards_str = ','.join(map(str, combo['cards']))
                        print(f"  {i}. {combo['combo_type']} rank={combo['rank_value']} cards=[{cards_str}] strength={strength:.3f}")
                
                print("="*50)
                print()
                
            except ValueError:
                print("❌ Invalid input! Please enter numbers separated by commas.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def demo_model_components(self):
        """Demonstrate individual model components"""
        print("🔧 DEMO: Model Components")
        print("-" * 40)
        
        test_hand = [12, 25, 38, 51, 11, 24, 37, 10, 23, 36]
        
        print("1. Rule Engine Validation:")
        possible_combos = self.generator.analyze_hand(test_hand)
        is_valid, reason, profile = self.generator.rule_engine.validate_hand(possible_combos)
        print(f"   Valid: {is_valid}")
        print(f"   Reason: {reason}")
        if profile:
            print(f"   Total Cards: {profile['total_cards']}")
            print(f"   Average Strength: {profile['avg_strength']:.3f}")
            print(f"   Strong Combos: {profile['strong_combos']}")
        
        print("\n2. ML Validation Model:")
        hand_data = {
            'hand': test_hand,
            'player_count': 4,
            'possible_combos': possible_combos
        }
        ml_result = self.generator.validation_model.predict(hand_data)
        print(f"   ML Valid: {ml_result['is_valid']}")
        print(f"   Confidence: {ml_result['confidence']:.3f}")
        
        print("\n3. Pattern Learning Model:")
        pattern_result = self.generator.pattern_model.predict(hand_data)
        print(f"   Pattern Score: {pattern_result['pattern_score']:.3f}")
        print(f"   Building Preference: {pattern_result['sequence_building_preference']}")
        patterns = pattern_result['combo_patterns']
        print(f"   Power Concentration: {patterns['power_concentration']:.3f}")
        print(f"   Combo Diversity: {patterns['combo_diversity']:.3f}")
        
        print("\n4. Threshold Learning Model:")
        threshold = self.generator.threshold_model.predict_user_threshold(hand_data, pattern_result)
        print(f"   Predicted Threshold: {threshold:.3f}")
        
        print()
    
    def run_demo(self):
        """Run complete demo"""
        if not self.is_trained:
            self.quick_train_models()
        
        while True:
            print("\n" + "="*60)
            print("🎯 UNBEATABLE SEQUENCE MODEL DEMO")
            print("="*60)
            print("Choose a demo option:")
            print("1. 📋 Hand Scenarios (Different strength levels)")
            print("2. 👥 Player Count Effects (2, 3, 4 players)")
            print("3. 🎮 Interactive Mode (Enter your own hands)")
            print("4. 🔧 Model Components (Individual component demo)")
            print("5. 🚀 Full Training Pipeline (Complete training)")
            print("6. 👋 Exit")
            print()
            
            try:
                choice = input("Enter your choice (1-6): ").strip()
                
                if choice == '1':
                    self.demo_hand_scenarios()
                elif choice == '2':
                    self.demo_player_count_effects()
                elif choice == '3':
                    self.demo_interactive_mode()
                elif choice == '4':
                    self.demo_model_components()
                elif choice == '5':
                    print("🚀 Running full training pipeline...")
                    trainer = UnbeatableModelTrainer()
                    results = trainer.run_complete_training()
                    print(f"Training completed with status: {results['overall_status']}")
                elif choice == '6':
                    print("👋 Thank you for using the Unbeatable Sequence Model Demo!")
                    break
                else:
                    print("❌ Invalid choice! Please enter 1-6.")
                
                if choice in ['1', '2', '4']:
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")


def main():
    """Main demo function"""
    demo = UnbeatableModelDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
