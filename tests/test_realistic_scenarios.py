"""
Test Hybrid Conservative Model với các tình huống thực tế
Mô phỏng các trường hợp: bài mạnh không báo, bài yếu mà báo, etc.
"""
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any
import joblib
import sys
import os

# Add parent directory to path để import HybridConservativeModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_conservative_model import HybridConservativeModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_scenarios():
    """Tạo các tình huống test thực tế"""
    
    scenarios = [
        # SCENARIO 1: Bài rất mạnh - nên báo
        {
            "name": "Bài rất mạnh - Straight + Tứ Quý + Triple",
            "description": "3 combo mạnh, rank cao, strength cao",
            "sammove_sequence": [
                {"cards": [15, 16, 17, 18, 19], "combo_type": "straight", "rank_value": 10},
                {"cards": [25, 26, 27, 28], "combo_type": "quad", "rank_value": 11},
                {"cards": [35, 36, 37], "combo_type": "triple", "rank_value": 12}
            ],
            "expected": "should_declare",
            "expected_reason": "very_strong_hand"
        },
        
        # SCENARIO 2: Bài mạnh nhưng không quá mạnh - có thể báo
        {
            "name": "Bài mạnh vừa - Straight + Triple + Pair",
            "description": "2 combo mạnh, 1 combo trung bình",
            "sammove_sequence": [
                {"cards": [8, 9, 10, 11, 12], "combo_type": "straight", "rank_value": 8},
                {"cards": [25, 26, 27], "combo_type": "triple", "rank_value": 9},
                {"cards": [35, 36], "combo_type": "pair", "rank_value": 10}
            ],
            "expected": "might_declare",
            "expected_reason": "moderate_strong_hand"
        },
        
        # SCENARIO 3: Bài yếu - không nên báo
        {
            "name": "Bài yếu - Toàn single và pair",
            "description": "Không có combo mạnh, rank thấp",
            "sammove_sequence": [
                {"cards": [5], "combo_type": "single", "rank_value": 2},
                {"cards": [15, 16], "combo_type": "pair", "rank_value": 3},
                {"cards": [25], "combo_type": "single", "rank_value": 4},
                {"cards": [35, 36], "combo_type": "pair", "rank_value": 5},
                {"cards": [45], "combo_type": "single", "rank_value": 6},
                {"cards": [55, 56], "combo_type": "pair", "rank_value": 7},
                {"cards": [65], "combo_type": "single", "rank_value": 8}
            ],
            "expected": "should_not_declare",
            "expected_reason": "weak_hand"
        },
        
        # SCENARIO 4: Bài rất yếu - chắc chắn không báo
        {
            "name": "Bài cực yếu - Toàn single rank thấp",
            "description": "Chỉ có single, rank rất thấp",
            "sammove_sequence": [
                {"cards": [5], "combo_type": "single", "rank_value": 1},
                {"cards": [15], "combo_type": "single", "rank_value": 2},
                {"cards": [25], "combo_type": "single", "rank_value": 3},
                {"cards": [35], "combo_type": "single", "rank_value": 4},
                {"cards": [45], "combo_type": "single", "rank_value": 5},
                {"cards": [55], "combo_type": "single", "rank_value": 6},
                {"cards": [65], "combo_type": "single", "rank_value": 7},
                {"cards": [75], "combo_type": "single", "rank_value": 8},
                {"cards": [85], "combo_type": "single", "rank_value": 9},
                {"cards": [95], "combo_type": "single", "rank_value": 10}
            ],
            "expected": "should_not_declare",
            "expected_reason": "very_weak_hand"
        },
        
        # SCENARIO 5: Bài mixed - có mạnh có yếu
        {
            "name": "Bài mixed - Tứ Quý + Single + Pair yếu",
            "description": "1 combo rất mạnh, 2 combo yếu",
            "sammove_sequence": [
                {"cards": [25, 26, 27, 28], "combo_type": "quad", "rank_value": 11},
                {"cards": [5], "combo_type": "single", "rank_value": 2},
                {"cards": [15, 16], "combo_type": "pair", "rank_value": 3},
                {"cards": [35], "combo_type": "single", "rank_value": 4},
                {"cards": [45, 46], "combo_type": "pair", "rank_value": 5}
            ],
            "expected": "might_declare",
            "expected_reason": "mixed_hand"
        },
        
        # SCENARIO 6: Bài 2 combo mạnh
        {
            "name": "Bài 2 combo mạnh - Tứ Quý + Straight",
            "description": "2 combo rất mạnh",
            "sammove_sequence": [
                {"cards": [35, 36, 37, 38], "combo_type": "quad", "rank_value": 12},
                {"cards": [8, 9, 10, 11, 12], "combo_type": "straight", "rank_value": 11},
                {"cards": [45], "combo_type": "single", "rank_value": 5}
            ],
            "expected": "should_declare",
            "expected_reason": "very_strong_two_combos"
        },
        
        # SCENARIO 7: Bài 10 single yếu - không báo
        {
            "name": "Bài 10 single yếu - không báo",
            "description": "10 combo single rank thấp",
            "sammove_sequence": [
                {"cards": [5], "combo_type": "single", "rank_value": 1},
                {"cards": [15], "combo_type": "single", "rank_value": 2},
                {"cards": [25], "combo_type": "single", "rank_value": 3},
                {"cards": [35], "combo_type": "single", "rank_value": 4},
                {"cards": [45], "combo_type": "single", "rank_value": 5},
                {"cards": [55], "combo_type": "single", "rank_value": 6},
                {"cards": [65], "combo_type": "single", "rank_value": 7},
                {"cards": [75], "combo_type": "single", "rank_value": 8},
                {"cards": [85], "combo_type": "single", "rank_value": 9},
                {"cards": [95], "combo_type": "single", "rank_value": 10}
            ],
            "expected": "should_not_declare",
            "expected_reason": "all_weak_singles"
        },
        
        # SCENARIO 8: Bài edge case - 2 triple + 4 single
        {
            "name": "Bài edge case - 2 triple + 4 single",
            "description": "2 combo triple rank trung bình + 4 single",
            "sammove_sequence": [
                {"cards": [25, 26, 27], "combo_type": "triple", "rank_value": 7},
                {"cards": [35, 36, 37], "combo_type": "triple", "rank_value": 8},
                {"cards": [5], "combo_type": "single", "rank_value": 1},
                {"cards": [15], "combo_type": "single", "rank_value": 2},
                {"cards": [45], "combo_type": "single", "rank_value": 3},
                {"cards": [55], "combo_type": "single", "rank_value": 4}
            ],
            "expected": "might_declare",
            "expected_reason": "edge_case_moderate"
        },
        
        # SCENARIO 9: Bài có straight nhưng rank thấp
        {
            "name": "Bài có straight rank thấp",
            "description": "Có straight nhưng rank chỉ 5",
            "sammove_sequence": [
                {"cards": [5, 6, 7, 8, 9], "combo_type": "straight", "rank_value": 5},
                {"cards": [25, 26], "combo_type": "pair", "rank_value": 6},
                {"cards": [35], "combo_type": "single", "rank_value": 7},
                {"cards": [45], "combo_type": "single", "rank_value": 8},
                {"cards": [55], "combo_type": "single", "rank_value": 9}
            ],
            "expected": "might_declare",
            "expected_reason": "straight_low_rank"
        },
        
        # SCENARIO 10: Bài có tứ quý rank cao + combo yếu
        {
            "name": "Bài tứ quý rank cao + combo yếu",
            "description": "1 tứ quý rank 12, 1 straight rank thấp",
            "sammove_sequence": [
                {"cards": [35, 36, 37, 38], "combo_type": "quad", "rank_value": 12},
                {"cards": [5, 6, 7, 8, 9], "combo_type": "straight", "rank_value": 5},
                {"cards": [45], "combo_type": "single", "rank_value": 2}
            ],
            "expected": "should_declare",
            "expected_reason": "quad_high_rank_plus_weak"
        }
    ]
    
    return scenarios

def test_scenario(model, scenario):
    """Test một scenario với model"""
    
    # Tạo record theo format training data
    record = {
        "game_id": f"test_{scenario['name'].replace(' ', '_')}",
        "player_id": 1,
        "hand": [],  # Không cần thiết cho test
        "sammove_sequence": scenario["sammove_sequence"],
        "result": "success",  # Giả sử success để test
        "meta": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Predict với model
    result = model.predict_hybrid(record)
    
    return {
        "scenario_name": scenario["name"],
        "description": scenario["description"],
        "expected": scenario["expected"],
        "expected_reason": scenario["expected_reason"],
        "actual_declare": result["should_declare"],
        "confidence": result["confidence"],
        "reason": result["reason"],
        "rulebase_blocked": result.get("rulebase_blocked", False),
        "sequence_length": result.get("sequence_length", len(scenario["sammove_sequence"])),
        "correct": _evaluate_correctness(scenario["expected"], result["should_declare"])
    }

def _evaluate_correctness(expected, actual):
    """Đánh giá xem prediction có đúng không"""
    if expected == "should_declare":
        return actual == True
    elif expected == "should_not_declare":
        return actual == False
    elif expected == "might_declare":
        return True  # Cả hai đều OK
    else:
        return True

def analyze_results(results):
    """Phân tích kết quả test"""
    
    total_tests = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    
    # Phân loại theo expected
    should_declare_tests = [r for r in results if r["expected"] == "should_declare"]
    should_not_declare_tests = [r for r in results if r["expected"] == "should_not_declare"]
    might_declare_tests = [r for r in results if r["expected"] == "might_declare"]
    
    # Thống kê
    stats = {
        "total_tests": total_tests,
        "correct_predictions": correct_predictions,
        "accuracy": correct_predictions / total_tests,
        "should_declare": {
            "total": len(should_declare_tests),
            "correct": sum(1 for r in should_declare_tests if r["actual_declare"]),
            "accuracy": sum(1 for r in should_declare_tests if r["actual_declare"]) / len(should_declare_tests) if should_declare_tests else 0
        },
        "should_not_declare": {
            "total": len(should_not_declare_tests),
            "correct": sum(1 for r in should_not_declare_tests if not r["actual_declare"]),
            "accuracy": sum(1 for r in should_not_declare_tests if not r["actual_declare"]) / len(should_not_declare_tests) if should_not_declare_tests else 0
        },
        "rulebase_blocked": sum(1 for r in results if r["rulebase_blocked"]),
        "avg_confidence": np.mean([r["confidence"] for r in results])
    }
    
    return stats

def main():
    """Main test function"""
    
    print("🔄 Testing Hybrid Conservative Model với Realistic Scenarios...")
    
    # Load model
    try:
        model = joblib.load('models/hybrid_conservative_bao_sam_model.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    print(f"📋 Created {len(scenarios)} test scenarios")
    
    # Test each scenario
    results = []
    for scenario in scenarios:
        result = test_scenario(model, scenario)
        results.append(result)
        
        # Print individual result
        status = "✅" if result["correct"] else "❌"
        declare_status = "BÁO" if result["actual_declare"] else "KHÔNG BÁO"
        rulebase_status = " (BLOCKED)" if result["rulebase_blocked"] else ""
        
        print(f"\n{status} {result['scenario_name']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Actual: {declare_status}{rulebase_status}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Reason: {result['reason']}")
    
    # Analyze results
    stats = analyze_results(results)
    
    print(f"\n📊 OVERALL TEST RESULTS:")
    print(f"   Total Tests: {stats['total_tests']}")
    print(f"   Correct Predictions: {stats['correct_predictions']}")
    print(f"   Overall Accuracy: {stats['accuracy']:.3f}")
    print(f"   Average Confidence: {stats['avg_confidence']:.3f}")
    print(f"   Rulebase Blocked: {stats['rulebase_blocked']}")
    
    print(f"\n📈 BREAKDOWN BY EXPECTED BEHAVIOR:")
    print(f"   Should Declare: {stats['should_declare']['correct']}/{stats['should_declare']['total']} ({stats['should_declare']['accuracy']:.3f})")
    print(f"   Should Not Declare: {stats['should_not_declare']['correct']}/{stats['should_not_declare']['total']} ({stats['should_not_declare']['accuracy']:.3f})")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/realistic_scenario_test_results_{timestamp}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert results for JSON serialization
    converted_results = convert_numpy_types(results)
    converted_stats = convert_numpy_types(stats)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "stats": converted_stats,
            "detailed_results": converted_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Highlight interesting cases
    print(f"\n🔍 INTERESTING CASES:")
    
    # Cases where strong hand didn't declare
    strong_not_declared = [r for r in results if r["expected"] == "should_declare" and not r["actual_declare"]]
    if strong_not_declared:
        print(f"   ⚠️ Strong hands that didn't declare: {len(strong_not_declared)}")
        for case in strong_not_declared:
            print(f"      - {case['scenario_name']}: {case['reason']}")
    
    # Cases where weak hand declared
    weak_declared = [r for r in results if r["expected"] == "should_not_declare" and r["actual_declare"]]
    if weak_declared:
        print(f"   ⚠️ Weak hands that declared: {len(weak_declared)}")
        for case in weak_declared:
            print(f"      - {case['scenario_name']}: {case['reason']}")
    
    if not strong_not_declared and not weak_declared:
        print(f"   ✅ No problematic cases found!")

if __name__ == "__main__":
    main()
