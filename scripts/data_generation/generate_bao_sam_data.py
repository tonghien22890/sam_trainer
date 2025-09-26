#!/usr/bin/env python3
"""
Generate Báo Sâm training data
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

from bao_sam.bao_sam_generator import BaoSamGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate Báo Sâm training data')
    parser.add_argument('--validation-samples', type=int, default=1000,
                        help='Number of validation samples (default: 1000)')
    parser.add_argument('--pattern-samples', type=int, default=2000,
                        help='Number of pattern samples (default: 2000)')
    parser.add_argument('--threshold-samples', type=int, default=1500,
                        help='Number of threshold samples (default: 1500)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--all-phases', action='store_true',
                        help='Generate all phases of training data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = BaoSamGenerator()
    
    try:
        if args.all_phases:
            logger.info("Generating all Báo Sâm training data phases...")
            results = generator.generate_all_phases(
                validation_samples=args.validation_samples,
                pattern_samples=args.pattern_samples,
                threshold_samples=args.threshold_samples
            )
            
            logger.info(f"Generated {len(results['validation'])} validation records")
            logger.info(f"Generated {len(results['pattern'])} pattern records")
            logger.info(f"Generated {len(results['threshold'])} threshold records")
            
        else:
            # Generate individual phases
            logger.info(f"Generating {args.validation_samples} validation samples...")
            validation_data = generator.generate_validation_data(args.validation_samples)
            generator.save_data(validation_data, 
                              os.path.join(args.output_dir, 'bao_sam_validation_training_data.jsonl'))
            
            logger.info(f"Generating {args.pattern_samples} pattern samples...")
            pattern_data = generator.generate_pattern_data(args.pattern_samples)
            generator.save_data(pattern_data, 
                              os.path.join(args.output_dir, 'bao_sam_pattern_training_data.jsonl'))
            
            logger.info(f"Generating {args.threshold_samples} threshold samples...")
            threshold_data = generator.generate_threshold_data(args.threshold_samples)
            generator.save_data(threshold_data, 
                              os.path.join(args.output_dir, 'bao_sam_threshold_training_data.jsonl'))
        
        logger.info("Báo Sâm data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating Báo Sâm data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
