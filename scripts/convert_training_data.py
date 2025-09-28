#!/usr/bin/env python3
"""
Convert training data from data_logger format to model training format
This script bridges the gap between web_backend logging and model training
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the data converter
sys.path.insert(0, os.path.join(project_root, "ai_bots", "adapters"))
from data_converter import DataConverter


def convert_web_backend_data():
    """
    Convert training data from web_backend to model_build format
    """
    print("🔄 [ConvertTrainingData] Starting conversion...")
    
    # Paths
    web_backend_dir = os.path.join(project_root, "web_backend")
    model_build_dir = os.path.join(project_root, "model_build")
    scripts_dir = os.path.join(model_build_dir, "scripts")
    
    # Input files (from web_backend)
    training_data_file = os.path.join(web_backend_dir, "training_data.jsonl")
    bao_sam_data_file = os.path.join(web_backend_dir, "bao_sam_data.jsonl")
    
    # Output files (to model_build)
    two_layer_output = os.path.join(scripts_dir, "two_layer", "converted_two_layer_data.jsonl")
    unbeatable_output = os.path.join(scripts_dir, "unbeatable", "converted_unbeatable_data.jsonl")
    
    converter = DataConverter()
    results = {}
    
    # General format conversion removed - general model not used
    
    # Convert training_data.jsonl to two-layer format
    if os.path.exists(training_data_file):
        print(f"\n🎯 Converting {training_data_file} to two-layer format...")
        results["two_layer"] = converter.convert_file(training_data_file, two_layer_output, "two_layer")
    else:
        print(f"⚠️ Training data file not found: {training_data_file}")
        results["two_layer"] = {"converted": 0, "skipped": 0, "total": 0}
    
    # Convert training_data.jsonl to unbeatable format
    if os.path.exists(training_data_file):
        print(f"\n🏆 Converting {training_data_file} to unbeatable format...")
        results["unbeatable"] = converter.convert_file(training_data_file, unbeatable_output, "unbeatable")
    else:
        print(f"⚠️ Training data file not found: {training_data_file}")
        results["unbeatable"] = {"converted": 0, "skipped": 0, "total": 0}
    
    # Summary
    print(f"\n📈 CONVERSION SUMMARY:")
    print(f"=" * 50)
    total_converted = 0
    for format_name, result in results.items():
        converted = result["converted"]
        total = result["total"]
        print(f"{format_name:12}: {converted:4d}/{total:4d} records")
        total_converted += converted
    
    print(f"=" * 50)
    print(f"Total converted: {total_converted} records")
    
    if total_converted > 0:
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 Output files:")
        print(f"   Two-Layer: {two_layer_output}")
        print(f"   Unbeatable: {unbeatable_output}")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Train two-layer model: python scripts/two_layer/train_style_learner_core.py")
        print(f"   2. Train unbeatable model: python scripts/unbeatable/train_unbeatable_model.py")
    else:
        print(f"\n❌ No data was converted. Check if training_data.jsonl exists and has valid records.")
    
    return results


def convert_specific_file(input_file: str, output_file: str, target_format: str):
    """Convert a specific file to target format"""
    converter = DataConverter()
    result = converter.convert_file(input_file, output_file, target_format)
    
    print(f"\n📊 Conversion result:")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Format: {target_format}")
    print(f"   Converted: {result['converted']} records")
    print(f"   Skipped: {result['skipped']} records")
    
    return result


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Convert training data from data_logger format')
    parser.add_argument('--auto', action='store_true', 
                       help='Auto-convert web_backend data to all formats')
    parser.add_argument('--input', type=str, help='Input JSONL file')
    parser.add_argument('--output', type=str, help='Output JSONL file')
    parser.add_argument('--format', choices=['two_layer', 'unbeatable'], 
                       help='Target format')
    
    args = parser.parse_args()
    
    if args.auto:
        # Auto-convert web_backend data
        convert_web_backend_data()
    elif args.input and args.output and args.format:
        # Convert specific file
        convert_specific_file(args.input, args.output, args.format)
    else:
        print("❌ Please specify either --auto or --input/--output/--format")
        parser.print_help()


if __name__ == '__main__':
    main()
