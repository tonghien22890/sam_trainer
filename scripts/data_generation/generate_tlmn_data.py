#!/usr/bin/env python3
"""
Generate TLMN training data
"""

import logging
import argparse
import os
import sys

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_build_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(model_build_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if model_build_dir not in sys.path:
    sys.path.insert(0, model_build_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tlmn.tlmn_generator import TLMNGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate TLMN training data')
    parser.add_argument('--sessions', type=int, default=100,
                        help='Number of game sessions (default: 100)')
    parser.add_argument('--output-file', type=str, default='data/tlmn_training_data.jsonl',
                        help='Output file path (default: data/tlmn_training_data.jsonl)')
    parser.add_argument('--players', type=int, default=4,
                        help='Number of players per game (default: 4)')
    parser.add_argument('--pattern-samples', type=int, default=1500,
                        help='Number of pattern analysis samples (default: 1500)')
    parser.add_argument('--generate-patterns', action='store_true',
                        help='Generate pattern analysis data')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = TLMNGenerator()
    
    try:
        logger.info(f"Generating {args.sessions} TLMN game sessions...")
        logger.info(f"Each session will have {args.players} players")
        
        # Generate training data
        training_data = generator.generate_training_data(args.sessions)
        
        # Save data
        generator.save_data(training_data, args.output_file)
        
        logger.info(f"Generated {len(training_data)} TLMN training records")
        logger.info(f"Data saved to: {args.output_file}")
        
        # Calculate statistics
        total_moves = len(training_data)
        players_count = len(set(record.get('player_id', 0) for record in training_data))
        avg_moves_per_player = total_moves / players_count if players_count > 0 else 0
        
        logger.info(f"Statistics:")
        logger.info(f"  Total moves: {total_moves}")
        logger.info(f"  Players: {players_count}")
        logger.info(f"  Avg moves per player: {avg_moves_per_player:.2f}")
        
        # Generate pattern data if requested
        if args.generate_patterns:
            logger.info(f"Generating {args.pattern_samples} TLMN pattern samples...")
            pattern_data = generator.generate_tlmn_pattern_data(args.pattern_samples)
            
            pattern_output_file = args.output_file.replace('.jsonl', '_patterns.jsonl')
            generator.save_data(pattern_data, pattern_output_file)
            
            logger.info(f"Generated {len(pattern_data)} TLMN pattern records")
            logger.info(f"Pattern data saved to: {pattern_output_file}")
        
    except Exception as e:
        logger.error(f"Error generating TLMN data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
