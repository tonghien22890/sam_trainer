#!/usr/bin/env python3
"""Test pure ML without threshold - xem model học được gì thực sự"""

import json
import numpy as np
from hybrid_conservative_model import HybridConservativeModel

def test_pure_ml():
    print('=== TEST PURE ML (NO THRESHOLD) ===')
    
    # Load và train model
    model = HybridConservativeModel()
    records = model.load_data('data/sam_training_data.jsonl')
    X, y = model.prepare_dataset(records)
    model.train_model(X, y)
    print('Model trained successfully')
    
    print()
    print('=== TESTING SCENARIOS WITH PURE ML ===')
    
    def test_pure_ml_scenario(name, sequence):
        record = {'sammove_sequence': sequence}
        
        # Extract features và predict trực tiếp
        features = model.extract_enhanced_features(record)
        X_test = np.array(features).reshape(1, -1)
        
        # Pure ML prediction (no threshold)
        prediction = model.model.predict(X_test)[0]
        probabilities = model.model.predict_proba(X_test)[0]
        confidence = max(probabilities)
        
        print(f"{name}:")
        print(f"  Pure ML Prediction: {prediction} ({'BAO SAM' if prediction == 1 else 'KHONG BAO'})")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Probabilities: [fail={probabilities[0]:.3f}, success={probabilities[1]:.3f}]")
        print()
        
        return prediction, confidence

    # Test cases
    results = []
    
    results.append(test_pure_ml_scenario("Sảnh rồng (10 lá)", [
        {'cards': list(range(10)), 'combo_type': 'straight', 'rank_value': 9}
    ]))
    
    results.append(test_pure_ml_scenario("Tứ quý 2 (tuyệt đối)", [
        {'cards': [12, 25, 38, 51], 'combo_type': 'quad', 'rank_value': 12},
        {'cards': [0, 1, 2, 3, 4, 5], 'combo_type': 'straight', 'rank_value': 5}
    ]))
    
    results.append(test_pure_ml_scenario("Triple K + Pair A", [
        {'cards': [10, 23, 36], 'combo_type': 'triple', 'rank_value': 10},
        {'cards': [11, 24], 'combo_type': 'pair', 'rank_value': 11}
    ]))
    
    results.append(test_pure_ml_scenario("Nhiều single yếu", [
        {'cards': [0], 'combo_type': 'single', 'rank_value': 0},
        {'cards': [2], 'combo_type': 'single', 'rank_value': 2},
        {'cards': [3], 'combo_type': 'single', 'rank_value': 3}
    ]))
    
    results.append(test_pure_ml_scenario("Single 2 (mạnh nhất)", [
        {'cards': [12], 'combo_type': 'single', 'rank_value': 12}
    ]))
    
    results.append(test_pure_ml_scenario("Sảnh 5 lá thấp", [
        {'cards': [0, 1, 2, 3, 4], 'combo_type': 'straight', 'rank_value': 4}
    ]))
    
    results.append(test_pure_ml_scenario("Pair 2 + Single A", [
        {'cards': [12, 25], 'combo_type': 'pair', 'rank_value': 12},
        {'cards': [11], 'combo_type': 'single', 'rank_value': 11}
    ]))
    
    # Phân tích kết quả
    print('=== PHÂN TÍCH PURE ML ===')
    positive_predictions = sum(1 for pred, conf in results if pred == 1)
    avg_confidence = np.mean([conf for pred, conf in results])
    
    print(f'Tổng số scenario: {len(results)}')
    print(f'So scenario model du doan BAO SAM: {positive_predictions}')
    print(f'Ty le bao sam: {positive_predictions/len(results)*100:.1f}%')
    print(f'Average confidence: {avg_confidence:.3f}')
    print()
    
    print('=== SO SÁNH VỚI SYNTHETIC RULE ===')
    print('Kiểm tra xem model có học đúng synthetic rule không:')
    
    for i, (name, sequence) in enumerate([
        ("Sảnh rồng (10 lá)", [{'cards': list(range(10)), 'combo_type': 'straight', 'rank_value': 9}]),
        ("Tứ quý 2 (tuyệt đối)", [{'cards': [12, 25, 38, 51], 'combo_type': 'quad', 'rank_value': 12}, {'cards': [0, 1, 2, 3, 4, 5], 'combo_type': 'straight', 'rank_value': 5}]),
        ("Nhiều single yếu", [{'cards': [0], 'combo_type': 'single', 'rank_value': 0}, {'cards': [2], 'combo_type': 'single', 'rank_value': 2}, {'cards': [3], 'combo_type': 'single', 'rank_value': 3}])
    ]):
        # Tính synthetic success_prob
        temp_model = HybridConservativeModel()
        strengths = [temp_model.calculate_combo_strength(combo) for combo in sequence]
        avg_strength = np.mean(strengths)
        
        # Simplified synthetic rule approximation
        synthetic_prob = avg_strength
        if any(combo['combo_type'] in ['straight', 'quad'] for combo in sequence):
            synthetic_prob += 0.1
        synthetic_prob = min(0.95, max(0.1, synthetic_prob))
        
        synthetic_prediction = "BAO SAM" if synthetic_prob > 0.7 else "KHONG BAO"
        ml_prediction = "BAO SAM" if results[i][0] == 1 else "KHONG BAO"
        
        match = "OK" if synthetic_prediction == ml_prediction else "DIFF"
        print(f'{name}: Synthetic={synthetic_prediction}, ML={ml_prediction} {match}')
    
    print()
    print('=== KẾT LUẬN ===')
    print('1. Pure ML có khả năng ra quyết định không cần threshold')
    print('2. Nhưng nó chỉ học lại synthetic pattern, không phải strategy thật')
    print('3. Confidence levels có thể thay thế threshold nhưng vẫn là fake ML')
    print('4. Model có thể hữu ích như "consistent rule engine" nhưng không phải AI thông minh')

if __name__ == "__main__":
    test_pure_ml()
