#!/usr/bin/env python3
"""Debug training input - Log chi tiết input cho model train"""

import json
import numpy as np
from hybrid_conservative_model import HybridConservativeModel

def debug_training_input():
    print('=' * 80)
    print('DEBUG TRAINING INPUT - CHI TIẾT MODEL ĐANG HỌC GÌ')
    print('=' * 80)
    
    # Load model và data
    model = HybridConservativeModel()
    records = model.load_data('data/sam_training_data.jsonl')
    
    print(f'Total records loaded: {len(records)}')
    print()
    
    # Debug 5 records đầu tiên
    print('=== RAW DATA SAMPLES ===')
    for i in range(min(5, len(records))):
        record = records[i]
        print(f'--- RECORD {i+1} ---')
        print(f'Result (label): {record["result"]}')
        print(f'Hand: {record.get("hand", [])}')
        print(f'Sequence length: {len(record["sammove_sequence"])}')
        
        print('Sammove sequence:')
        for j, combo in enumerate(record["sammove_sequence"]):
            strength = model.calculate_combo_strength(combo)
            print(f'  {j+1}. {combo["combo_type"]} rank={combo["rank_value"]} cards={combo["cards"]} strength={strength:.3f}')
        
        print(f'Meta info: {record.get("meta", {})}')
        print()
    
    # Debug feature extraction
    print('=== FEATURE EXTRACTION DEBUG ===')
    for i in range(min(3, len(records))):
        record = records[i]
        print(f'--- FEATURES FOR RECORD {i+1} ---')
        print(f'Label: {1 if record.get("result") == "success" else 0}')
        
        # Extract features
        features = model.extract_enhanced_features(record)
        print(f'Feature vector length: {len(features)}')
        
        # Decode features theo thứ tự trong code
        idx = 0
        print(f'[{idx}] Sequence length: {features[idx]}')
        idx += 1
        
        # First combo type (5 dims)
        combo_types = ['single', 'pair', 'triple', 'straight', 'quad']
        first_combo_onehot = features[idx:idx+5]
        first_combo = combo_types[np.argmax(first_combo_onehot)] if max(first_combo_onehot) > 0 else 'none'
        print(f'[{idx}-{idx+4}] First combo type: {first_combo} {first_combo_onehot}')
        idx += 5
        
        # Second combo type (5 dims)
        second_combo_onehot = features[idx:idx+5]
        second_combo = combo_types[np.argmax(second_combo_onehot)] if max(second_combo_onehot) > 0 else 'none'
        print(f'[{idx}-{idx+4}] Second combo type: {second_combo} {second_combo_onehot}')
        idx += 5
        
        # Third combo type (5 dims)
        third_combo_onehot = features[idx:idx+5]
        third_combo = combo_types[np.argmax(third_combo_onehot)] if max(third_combo_onehot) > 0 else 'none'
        print(f'[{idx}-{idx+4}] Third combo type: {third_combo} {third_combo_onehot}')
        idx += 5
        
        # Rank values (3 dims)
        rank_values = features[idx:idx+3]
        print(f'[{idx}-{idx+2}] Rank values: {rank_values}')
        idx += 3
        
        # Combo strengths (3 dims)
        combo_strengths = features[idx:idx+3]
        print(f'[{idx}-{idx+2}] Combo strengths: {combo_strengths}')
        idx += 3
        
        # Statistics (6 dims)
        stats = features[idx:idx+6]
        print(f'[{idx}-{idx+5}] Statistics [max_str, min_str, avg_str, max_rank, min_rank, avg_rank]: {stats}')
        idx += 6
        
        # Pattern indicators (6 dims)
        patterns = features[idx:idx+6]
        print(f'[{idx}-{idx+5}] Patterns [strong_start, strong_finish, ascending, descending, strong_count, high_rank_count]: {patterns}')
        
        print()
    
    # Prepare dataset và debug
    print('=== DATASET PREPARATION ===')
    X, y = model.prepare_dataset(records)
    
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print(f'Feature dimensions: {X.shape[1]}')
    print(f'Label distribution: {np.bincount(y)}')
    print(f'Success rate: {np.mean(y):.3f}')
    print()
    
    # Feature statistics
    print('=== FEATURE STATISTICS ===')
    print('Feature ranges:')
    for i in range(X.shape[1]):
        print(f'  Feature {i:2d}: min={X[:, i].min():.3f}, max={X[:, i].max():.3f}, mean={X[:, i].mean():.3f}')
    
    print()
    
    # Label correlation analysis
    print('=== LABEL CORRELATION ANALYSIS ===')
    success_indices = np.where(y == 1)[0]
    fail_indices = np.where(y == 0)[0]
    
    print(f'Success samples: {len(success_indices)}')
    print(f'Fail samples: {len(fail_indices)}')
    
    # So sánh feature means giữa success và fail
    print('\nFeature differences (Success vs Fail):')
    for i in range(min(10, X.shape[1])):  # Chỉ show 10 features đầu
        success_mean = X[success_indices, i].mean()
        fail_mean = X[fail_indices, i].mean()
        diff = success_mean - fail_mean
        print(f'  Feature {i:2d}: Success={success_mean:.3f}, Fail={fail_mean:.3f}, Diff={diff:.3f}')
    
    print()
    print('=== SUMMARY ===')
    print('Model đang học:')
    print('1. Input: 34 features từ sammove_sequence patterns')
    print('2. Output: Binary classification (success=1, fail=0)')
    print('3. Features chủ yếu: combo types, ranks, strengths, patterns')
    print('4. Data: Synthetic với success_prob rules')
    print('5. Learning objective: Predict synthetic success/fail labels')

if __name__ == "__main__":
    debug_training_input()

