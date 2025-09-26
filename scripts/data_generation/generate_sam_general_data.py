#!/usr/bin/env python3
"""
Generate Sam General training data
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

from sam_general.sam_general_generator import SamGeneralGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate Sam General training data')
    parser.add_argument('--sessions', type=int, default=100,
                        help='Number of game sessions (default: 100)')
    parser.add_argument('--output-file', type=str, default='data/sam_general_training_data.jsonl',
                        help='Output file path (default: data/sam_general_training_data.jsonl)')
    parser.add_argument('--players', type=int, default=4,
                        help='Number of players per game (default: 4)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = SamGeneralGenerator()
    
    try:
        logger.info(f"Generating {args.sessions} Sam General game sessions...")
        logger.info(f"Each session will have {args.players} players")
        
        # Generate training data
        training_data = generator.generate_training_data(args.sessions)
        
        # Save data
        generator.save_data(training_data, args.output_file)
        
        logger.info(f"Generated {len(training_data)} Sam General training records")
        logger.info(f"Data saved to: {args.output_file}")
        
        # Calculate statistics
        total_moves = len(training_data)
        players_count = len(set(record.get('player_id', 0) for record in training_data))
        avg_moves_per_player = total_moves / players_count if players_count > 0 else 0
        
        logger.info(f"Statistics:")
        logger.info(f"  Total moves: {total_moves}")
        logger.info(f"  Players: {players_count}")
        logger.info(f"  Avg moves per player: {avg_moves_per_player:.2f}")
        
    except Exception as e:
        logger.error(f"Error generating Sam General data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
