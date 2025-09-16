"""
Debug script để log chi tiết thông tin features và labels cho Stage 2 training
và Stage 1 (label + features tối giản theo yêu cầu).
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Any, Tuple
from optimized_general_model_v3 import OptimizedGeneralModelV3

def load_training_data(filepath: str):
    """Load training data from JSONL file"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    return records

def debug_stage1_candidates(records: List[Dict[str, Any]], max_turns: int = 5):
    """
    Train & evaluate per-candidate Stage 1 model and print debug for a few turns.
    - Shows per-turn candidates with predicted probabilities and marks the chosen one.
    """
    print("=" * 80)
    print("STAGE 1 PER-CANDIDATE - TRAIN/EVAL/DEBUG")
    print("=" * 80)

    model = OptimizedGeneralModelV3()

    # Build dataset once for debugging view
    X, y, groups = model.build_stage1_candidate_dataset(records)
    print(f"Dataset: samples={X.shape[0]} features={X.shape[1] if X.shape[0] else 0} turns={len(groups)}")

    # Train
    train_acc = model.train_stage1_candidates(records)
    print(f"Sample-level training accuracy: {train_acc:.4f}")

    # Evaluate per-turn
    eval_res = model.evaluate_stage1_candidates(records)
    print(f"Per-turn top-1 accuracy: {eval_res['turn_accuracy']:.4f}  (turns={eval_res['num_turns']}, samples={eval_res['num_samples']})")
    print()

    if X.shape[0] == 0 or not groups or model.stage1_candidate_model is None:
        print("No data for candidate debugging.")
        print("=" * 80)
        return

    # Predict probabilities for debug
    probs = model.stage1_candidate_model.predict_proba(X)[:, 1]

    # Print first N turns' candidate lists with probabilities
    print("DEBUG (first turns):")
    turn_shown = 0
    for turn_idx, turn_indices in enumerate(groups):
        if turn_shown >= max_turns:
            break

        # Reconstruct basic move info for this turn from original records
        # Find the record by approximating: walk through records and accumulate candidates
        # until we pass the starting index; for readability we instead rebuild candidates per record.
        record = None
        cum = 0
        target_start = turn_indices[0]
        for rec in records:
            lm = rec.get("meta", {}).get("legal_moves", []) or []
            cand_count = sum(1 for mv in lm if mv.get("type") in ("play_cards", "pass"))
            if target_start < cum + cand_count:
                record = rec
                break
            cum += cand_count

        legal_moves = record.get("meta", {}).get("legal_moves", []) if record else []
        # Build the candidate list for printing
        candidates = [mv for mv in legal_moves if mv.get("type") in ("play_cards", "pass")]

        # Pair indices with probs and label
        turn_pairs = [(idx, probs[idx], y[idx]) for idx in turn_indices]
        # Sort by prob desc
        turn_pairs.sort(key=lambda t: t[1], reverse=True)

        print(f"Turn {turn_idx+1} - candidates={len(turn_indices)}")
        for rank, (glob_idx, p, label) in enumerate(turn_pairs[:10]):
            # Map back to local candidate index
            local_idx = turn_indices.index(glob_idx) if glob_idx in turn_indices else -1
            mv = candidates[local_idx] if 0 <= local_idx < len(candidates) else {}
            marker = "<-- CHOSEN" if label == 1 else ""
            print(f"  [{rank}] p={p:.3f} label={label} type={mv.get('type')} combo={mv.get('combo_type')} rank_value={mv.get('rank_value')} cards={mv.get('cards')} {marker}")

        print()
        turn_shown += 1

    print("=" * 80)

def debug_stage1_training(records: List[Dict[str, Any]], max_samples: int = 10):
    """
    Debug Stage 1: chỉ log LABEL (combo_type) và FEATURES (12 dims)
    - Chỉ áp dụng cho các record khi không có combo trước đó (pass situations)
    """
    print("=" * 80)
    print("DEBUG STAGE 1 TRAINING - LABELS & FEATURES (12 dims)")
    print("=" * 80)

    model = OptimizedGeneralModelV3()

    samples = []
    for i, record in enumerate(records):
        last_move = record.get("last_move")
        if last_move and last_move.get("combo_type"):
            continue  # Chỉ lấy tình huống pass cho Stage 1

        # Extract label from action.stage1.value (fallback 'pass')
        action = record.get("action", {})
        stage1 = action.get("stage1", {})
        combo_type = stage1.get("value", "pass")

        # Extract features (12 dims)
        features = model.extract_stage1_features(record)

        samples.append({
            'record_index': i,
            'label': combo_type,
            'features': features
        })

    print(f"Total Stage 1 samples: {len(samples)}")
    print(f"Showing first {min(max_samples, len(samples))} samples:\n")

    for idx, sample in enumerate(samples[:max_samples]):
        print(f"--- SAMPLE {idx + 1} ---")
        print(f"Record Index: {sample['record_index']}")
        print(f"Label (combo_type): {sample['label']}")
        # Features order reference: [6 combo_counts, 4 cards_left, 1 hand_count, 1 combo_strength_relative]
        feats = sample['features']
        print(f"Features (len={len(feats)}): {feats.tolist() if hasattr(feats, 'tolist') else feats}")
        print()

    # Summary stats
    if samples:
        X = np.stack([s['features'] for s in samples], axis=0)
        print("SUMMARY STATISTICS (Stage 1 Features):")
        print(f"  Shape: {X.shape}")
        print(f"  Means: {X.mean(axis=0)}")
        print(f"  Stds:  {X.std(axis=0)}")
        print(f"  Mins:  {X.min(axis=0)}")
        print(f"  Maxs:  {X.max(axis=0)}")
    print("=" * 80)

def debug_stage2_training(records: List[Dict[str, Any]], max_samples: int = 10):
    """
    Debug Stage 2 training với logging chi tiết
    """
    print("=" * 80)
    print("DEBUG STAGE 2 TRAINING - FEATURES & LABELS")
    print("=" * 80)
    
    model = OptimizedGeneralModelV3()
    
    # Collect Stage 2 training data
    stage2_samples = []
    
    for i, record in enumerate(records):
        # Only train on non-pass moves
        action = record.get("action", {})
        stage1 = action.get("stage1", {})
        stage2 = action.get("stage2", {})
        
        combo_type = stage1.get("value", "pass")
        if combo_type == "pass":
            continue
        
        # Extract features
        features = model.extract_stage2_features(record, combo_type)
        encoded_features = model.encode_stage2_features_for_model(features)
        
        # Extract label (index in legal_moves)
        legal_moves = record.get("meta", {}).get("legal_moves", [])
        chosen_cards = stage2.get("cards", [])
        
        # Find index of chosen move
        label = -1
        for j, move in enumerate(legal_moves):
            if move.get("cards") == chosen_cards:
                label = j
                break
        
        if label == -1:
            continue  # Skip if not found
        
        stage2_samples.append({
            'record_index': i,
            'combo_type': combo_type,
            'features': features,
            'encoded_features': encoded_features,
            'label': label,
            'legal_moves': legal_moves,
            'chosen_cards': chosen_cards,
            'record': record
        })
    
    print(f"Total Stage 2 samples: {len(stage2_samples)}")
    print(f"Showing first {min(max_samples, len(stage2_samples))} samples:")
    print()
    
    # Log detailed information for first few samples
    for idx, sample in enumerate(stage2_samples[:max_samples]):
        print(f"--- SAMPLE {idx + 1} ---")
        print(f"Record Index: {sample['record_index']}")
        print(f"Combo Type: {sample['combo_type']}")
        print(f"Label (Move Index): {sample['label']}")
        print()
        
        # Log features breakdown
        features = sample['features']
        print("FEATURES BREAKDOWN:")
        print(f"  Combo Type ID: {features['combo_type']}")
        print(f"  Cards Left: {features['cards_left']}")
        print(f"  Hand Count: {features['hand_count']}")
        print()
        
        # Log combo strength ranking
        ranking = features['combo_strength_ranking']
        print("COMBO STRENGTH RANKING:")
        for rank_idx, rank_item in enumerate(ranking):
            print(f"  {rank_idx}: {rank_item['combo_type']} (rank={rank_item['rank_value']}, strength={rank_item['strength']}, cards={rank_item['cards']})")
        print()
        
        # Log encoded features
        encoded = sample['encoded_features']
        print("ENCODED FEATURES (9 dims):")
        print(f"  [0] Combo Type ID: {encoded[0]}")
        print(f"  [1-3] Top 3 Strengths: {encoded[1:4]}")
        print(f"  [4-7] Cards Left: {encoded[4:8]}")
        print(f"  [8] Hand Count: {encoded[8]}")
        print()
        
        # Log legal moves
        legal_moves = sample['legal_moves']
        print("LEGAL MOVES:")
        for move_idx, move in enumerate(legal_moves):
            marker = " <-- CHOSEN" if move_idx == sample['label'] else ""
            print(f"  {move_idx}: {move.get('type', 'unknown')} - {move.get('combo_type', 'N/A')} (rank={move.get('rank_value', 'N/A')}) - cards={move.get('cards', [])}{marker}")
        print()
        
        # Log chosen move details
        chosen_move = legal_moves[sample['label']] if sample['label'] < len(legal_moves) else None
        print("CHOSEN MOVE:")
        if chosen_move:
            print(f"  Type: {chosen_move.get('type', 'unknown')}")
            print(f"  Combo Type: {chosen_move.get('combo_type', 'N/A')}")
            print(f"  Rank Value: {chosen_move.get('rank_value', 'N/A')}")
            print(f"  Cards: {chosen_move.get('cards', [])}")
        else:
            print("  ERROR: Chosen move not found in legal moves!")
        print()
        
        # Log raw record for context
        record = sample['record']
        print("RAW RECORD CONTEXT:")
        print(f"  Hand: {record.get('hand', [])}")
        print(f"  Cards Left: {record.get('cards_left', [])}")
        print(f"  Last Move: {record.get('last_move', {})}")
        print(f"  Action Stage1: {action.get('stage1', {})}")
        print(f"  Action Stage2: {action.get('stage2', {})}")
        print()
        
        print("-" * 80)
        print()
    
    # Summary statistics
    print("SUMMARY STATISTICS:")
    print(f"Total Stage 2 samples: {len(stage2_samples)}")
    
    # Combo type distribution
    combo_counts = {}
    for sample in stage2_samples:
        combo_type = sample['combo_type']
        combo_counts[combo_type] = combo_counts.get(combo_type, 0) + 1
    
    print("Combo Type Distribution:")
    for combo_type, count in sorted(combo_counts.items()):
        print(f"  {combo_type}: {count}")
    
    # Label distribution
    labels = [sample['label'] for sample in stage2_samples]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Label (Move Index) Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Index {label}: {count}")
    
    # Feature statistics
    if stage2_samples:
        encoded_features_list = [sample['encoded_features'] for sample in stage2_samples]
        X = np.stack(encoded_features_list, axis=0)
        
        print("Feature Statistics:")
        print(f"  Feature shape: {X.shape}")
        print(f"  Feature means: {X.mean(axis=0)}")
        print(f"  Feature stds: {X.std(axis=0)}")
        print(f"  Feature mins: {X.min(axis=0)}")
        print(f"  Feature maxs: {X.max(axis=0)}")
    
    print("=" * 80)

def main():
    # Toggle giữa Stage 1 và Stage 2 debug nếu cần
    mode = os.environ.get("DEBUG_MODE", "stage1")  # stage1 | stage2

    if mode == "stage1":
        print("DEBUG STAGE 1 TRAINING")
        print("=" * 60)
        # Dữ liệu tổng quát có last_move (null/pass) phù hợp cho Stage 1
        data_file = os.environ.get("DATA_FILE", "data/sam_improved_training_data.jsonl")
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found")
            return
        records = load_training_data(data_file)
        print(f"Loaded {len(records)} training records")
        debug_stage1_training(records, max_samples=10)
    elif mode == "stage1c":
        print("DEBUG STAGE 1 PER-CANDIDATE")
        print("=" * 60)
        data_file = os.environ.get("DATA_FILE", "data/sam_improved_training_data.jsonl")
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found")
            return
        records = load_training_data(data_file)
        print(f"Loaded {len(records)} training records")
        debug_stage1_candidates(records, max_turns=5)
    elif mode == "stage1cmp":
        print("COMPARE STAGE 1 PER-CANDIDATE MODELS (DT/RF/XGB)")
        print("=" * 60)
        data_file = os.environ.get("DATA_FILE", "data/sam_improved_training_data.jsonl")
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found")
            return
        records = load_training_data(data_file)
        print(f"Loaded {len(records)} training records")
        model = OptimizedGeneralModelV3()
        results = model.compare_stage1_candidate_models(records)
        print("Results:")
        for mt, res in results.items():
            print(f"  {mt}: sample_acc={res['sample_acc']:.4f} turn@1={res['turn_accuracy']:.4f} turn@3={res['turn_top3']:.4f} turns={res['num_turns']} samples={res['num_samples']}")
    else:
        print("DEBUG STAGE 2 TRAINING")
        print("=" * 60)
        data_file = "data/synthetic_training_data.jsonl"
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found")
            print("Please run generate_test_data.py first")
            return
        records = load_training_data(data_file)
        print(f"Loaded {len(records)} training records")
        debug_stage2_training(records, max_samples=5)

if __name__ == "__main__":
    main()
