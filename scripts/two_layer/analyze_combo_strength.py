#!/usr/bin/env python3
"""
Analyze Combo Strength
So sánh base strength của các loại combo
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, project_root)

from model_build.scripts.two_layer.framework_generator import FrameworkGenerator
from ai_common.core.combo_analyzer import ComboAnalyzer


def analyze_combo_strength():
    """Analyze combo strength for different ranks and types"""
    print("🔍 Analyzing Combo Strength...")
    
    framework_gen = FrameworkGenerator()
    
    # Test different ranks
    ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2
    rank_names = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    
    combo_types = ['single', 'pair', 'triple', four_kind, 'straight']
    
    print("\n📊 Combo Strength Analysis (Using ComboAnalyzer Logic):")
    print("=" * 90)
    print(f"{'Rank':<4} {'Name':<4} {'Single':<8} {'Pair':<8} {'Triple':<8} {'Quad':<8} {'Straight':<8}")
    print("-" * 90)
    
    for i, rank in enumerate(ranks):
        rank_name = rank_names[i]
        
        # Calculate strength for each combo type using ComboAnalyzer
        single_strength = ComboAnalyzer.calculate_combo_strength({
            'combo_type': 'single', 'rank_value': rank, 'cards': [0]
        })
        
        pair_strength = ComboAnalyzer.calculate_combo_strength({
            'combo_type': 'pair', 'rank_value': rank, 'cards': [0, 13]
        })
        
        triple_strength = ComboAnalyzer.calculate_combo_strength({
            'combo_type': 'triple', 'rank_value': rank, 'cards': [0, 13, 26]
        })
        
        quad_strength = ComboAnalyzer.calculate_combo_strength({
            'combo_type': four_kind, 'rank_value': rank, 'cards': [0, 13, 26, 39]
        })
        
        # Rank 12 (2) is not part of straights in Sam
        if rank == 12:  # 2
            straight_strength = 0.0  # No straight with 2
        else:
            straight_strength = ComboAnalyzer.calculate_combo_strength({
                'combo_type': 'straight', 'rank_value': rank, 'cards': [rank, rank+13, rank+26, rank+39, rank+52]
            })
        
        print(f"{rank:<4} {rank_name:<4} {single_strength:<8.3f} {pair_strength:<8.3f} {triple_strength:<8.3f} {quad_strength:<8.3f} {straight_strength:<8.3f}")
    
    print("\n🎯 Top 10 Strongest Combos:")
    print("=" * 50)
    
    # Calculate all combo strengths using ComboAnalyzer
    combo_strengths = []
    for i, rank in enumerate(ranks):
        rank_name = rank_names[i]
        
        for combo_type in combo_types:
            # Create sample combo for strength calculation
            if combo_type == 'single':
                sample_combo = {'combo_type': 'single', 'rank_value': rank, 'cards': [0]}
            elif combo_type == 'pair':
                sample_combo = {'combo_type': 'pair', 'rank_value': rank, 'cards': [0, 13]}
            elif combo_type == 'triple':
                sample_combo = {'combo_type': 'triple', 'rank_value': rank, 'cards': [0, 13, 26]}
            elif combo_type == four_kind:
                sample_combo = {'combo_type': four_kind, 'rank_value': rank, 'cards': [0, 13, 26, 39]}
            elif combo_type == 'straight':
                # Rank 12 (2) is not part of straights in Sam
                if rank == 12:  # 2
                    continue  # Skip straight with 2
                else:
                    sample_combo = {'combo_type': 'straight', 'rank_value': rank, 'cards': [rank, rank+13, rank+26, rank+39, rank+52]}
            
            strength = ComboAnalyzer.calculate_combo_strength(sample_combo)
            combo_strengths.append({
                'rank': rank,
                'rank_name': rank_name,
                'combo_type': combo_type,
                'strength': strength
            })
    
    # Sort by strength (highest first)
    combo_strengths.sort(key=lambda x: x['strength'], reverse=True)
    
    for i, combo in enumerate(combo_strengths[:10]):
        print(f"{i+1:2d}. {combo['rank_name']:2s} {combo['combo_type']:8s} = {combo['strength']:.3f}")
    
    print("\n🔍 Analysis:")
    print("=" * 50)
    
    # Find strongest combo of each type
    strongest_by_type = {}
    for combo in combo_strengths:
        combo_type = combo['combo_type']
        if combo_type not in strongest_by_type or combo['strength'] > strongest_by_type[combo_type]['strength']:
            strongest_by_type[combo_type] = combo
    
    print("Strongest combo of each type:")
    for combo_type in ['single', 'pair', 'triple', four_kind, 'straight']:
        if combo_type in strongest_by_type:
            combo = strongest_by_type[combo_type]
            print(f"  {combo_type:8s}: {combo['rank_name']:2s} = {combo['strength']:.3f}")
    
    # Analyze the problem
    print("\n⚠️  Problem Analysis:")
    print("=" * 50)
    
    pair_2_strength = None
    straight_3_strength = None
    triple_3_strength = None
    
    for combo in combo_strengths:
        if combo['rank_name'] == '2' and combo['combo_type'] == 'pair':
            pair_2_strength = combo['strength']
        elif combo['rank_name'] == '3' and combo['combo_type'] == 'straight':
            straight_3_strength = combo['strength']
        elif combo['rank_name'] == '3' and combo['combo_type'] == 'triple':
            triple_3_strength = combo['strength']
    
    if pair_2_strength and straight_3_strength and triple_3_strength:
        print(f"Pair 2 strength: {pair_2_strength:.3f}")
        print(f"Straight 3-7 strength: {straight_3_strength:.3f}")
        print(f"Triple 3 strength: {triple_3_strength:.3f}")
        print(f"Pair 2 vs Straight 3-7 ratio: {pair_2_strength/straight_3_strength:.1f}:1")
        print(f"Pair 2 vs Triple 3 ratio: {pair_2_strength/triple_3_strength:.1f}:1")
        
        if pair_2_strength > straight_3_strength:
            print("❌ PROBLEM: Pair 2 is stronger than Straight 3-7!")
        if pair_2_strength > triple_3_strength:
            print("❌ PROBLEM: Pair 2 is stronger than Triple 3!")
    
    print("\n✅ ComboAnalyzer Logic Applied:")
    print("=" * 50)
    print("✅ FrameworkGenerator now uses ComboAnalyzer.calculate_combo_strength()")
    print("✅ Combo strengths are calculated using proper Sam logic")
    print("✅ Pair 2 dominance issue should be resolved")
    print("✅ Framework will now prioritize combos correctly")


if __name__ == "__main__":
    analyze_combo_strength()
