#!/usr/bin/env python3
"""
Core trainer for StyleLearner with --game_type flag.
Loads JSONL training data that already includes framework + sequence_context.
Outputs model to models/style_learner_{game_type}.pkl by default.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Ensure imports work when running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)            # .../model_build/scripts
model_build_dir = os.path.dirname(scripts_dir)        # .../model_build
project_root = os.path.dirname(model_build_dir)       # .../AI-Sam
for p in (project_root, model_build_dir, scripts_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from model_build.scripts.two_layer.style_learner import StyleLearner


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data


def main(args=None):
    parser = argparse.ArgumentParser(description='Train StyleLearner (Two-Layer)')
    parser.add_argument('--game_type', type=str, default='sam', choices=['sam', 'tlmn'], help='Game type to train for')
    parser.add_argument('--data_path', type=str, default='simple_sam.jsonl', help='Path to JSONL training data')
    parser.add_argument('--model_path', type=str, default=None, help='Output model path (.pkl)')
    
    # Ensemble mode arguments
    parser.add_argument('--ensemble', action='store_true', help='Enable ensemble mode (combine base + new data)')
    parser.add_argument('--base_data', type=str, help='Base data file path (for ensemble mode)')
    parser.add_argument('--new_data', type=str, help='New data file path (for ensemble mode)')
    parser.add_argument('--base_weight', type=int, default=1, help='Weight for base data samples')
    parser.add_argument('--new_weight', type=int, default=5, help='Weight for new data samples')
    
    args = parser.parse_args(args)

    game_type = args.game_type.lower()
    model_path = args.model_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', f'style_learner_{game_type}.pkl')

    if args.ensemble:
        # Ensemble mode: combine base + new data with weights
        print(f"[Trainer] Ensemble mode enabled")
        print(f"[Trainer] Base data: {args.base_data} (weight={args.base_weight})")
        print(f"[Trainer] New data: {args.new_data} (weight={args.new_weight})")
        
        # Load base data
        base_data = read_jsonl(args.base_data)
        print(f"[Trainer] Base data loaded: {len(base_data)} records")
        
        # Load new data
        new_data = read_jsonl(args.new_data)
        print(f"[Trainer] New data loaded: {len(new_data)} records")
        
        # Combine with weights (duplicate samples)
        combined_data = []
        
        # Add base data with weight
        for _ in range(args.base_weight):
            for record in base_data:
                combined_data.append(record)
        
        # Add new data with weight
        for _ in range(args.new_weight):
            for record in new_data:
                combined_data.append(record)
        
        print(f"[Trainer] Combined data: {len(combined_data)} records ({len(base_data)} base × {args.base_weight} + {len(new_data)} new × {args.new_weight})")
        training_data = combined_data
        
    else:
        # Single data source mode
        data_path = args.data_path
        print(f"[Trainer] Loading data: {data_path}")
        training_data = read_jsonl(data_path)

    # Optional filter by game_type if records include it
    filtered: List[Dict[str, Any]] = []
    for r in training_data:
        gt = (r.get('game_type') or r.get('game', '') or '').strip().lower()
        if not gt:
            # If not present, assume matches requested game_type (e.g., synthetic Sam)
            filtered.append(r)
        elif gt in ['sam', 'bao_sam', 'bao-sam'] and game_type == 'sam':
            filtered.append(r)
        elif gt in ['tlmn', 'tien len', 'tien_len', 'tien-len'] and game_type == 'tlmn':
            filtered.append(r)

    print(f"[Trainer] Records: {len(filtered)} (game_type={game_type})")
    if len(filtered) == 0:
        print("[Trainer] WARNING: No records matched the requested game_type. Check --data_path and record 'game_type' fields.")

    learner = StyleLearner()
    metrics = learner.train(filtered)
    print(f"[Trainer] Train metrics: {metrics}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    learner.save(model_path)
    print(f"[Trainer] Saved: {model_path}")


if __name__ == '__main__':
    main()


