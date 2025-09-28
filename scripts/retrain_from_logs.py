#!/usr/bin/env python3
"""
Retrain models from web_backend logs
Automatically converts data_logger format and retrains all models
"""

import os
import sys
import subprocess
import argparse
from typing import Dict, List, Any

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_command(cmd: List[str], cwd: str = None) -> bool:
    """Run a command and return success status"""
    try:
        print(f"🚀 Running: {' '.join(cmd)}")
        if cwd:
            print(f"   Working directory: {cwd}")
        
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def convert_training_data() -> bool:
    """Convert web_backend training data to model formats"""
    print("\n🔄 STEP 1: Converting training data...")
    
    convert_script = os.path.join(current_dir, "convert_training_data.py")
    cmd = [sys.executable, convert_script, "--auto"]
    
    return run_command(cmd, cwd=current_dir)


# train_general_model removed - general model not used


def train_two_layer_model() -> bool:
    """Train two-layer model"""
    print("\n🎯 STEP 2: Training two-layer model...")
    
    # Check if converted data exists
    converted_data = os.path.join(current_dir, "two_layer", "converted_two_layer_data.jsonl")
    if not os.path.exists(converted_data):
        print(f"❌ Converted two-layer data not found: {converted_data}")
        return False
    
    # Copy converted data to expected location
    training_data = os.path.join(current_dir, "two_layer", "simple_synthetic_training_data_with_sequence.jsonl")
    try:
        import shutil
        shutil.copy2(converted_data, training_data)
        print(f"📋 Copied {converted_data} to {training_data}")
    except Exception as e:
        print(f"❌ Failed to copy data: {e}")
        return False
    
    # Train model for both Sam and TLMN
    train_script = os.path.join(current_dir, "two_layer", "train_style_learner_core.py")
    
    # Train for Sam
    print("   Training for Sam...")
    cmd_sam = [sys.executable, train_script, "--game_type", "sam", "--data_path", training_data]
    sam_success = run_command(cmd_sam, cwd=os.path.join(current_dir, "two_layer"))
    
    # Train for TLMN
    print("   Training for TLMN...")
    cmd_tlmn = [sys.executable, train_script, "--game_type", "tlmn", "--data_path", training_data]
    tlmn_success = run_command(cmd_tlmn, cwd=os.path.join(current_dir, "two_layer"))
    
    return sam_success and tlmn_success


def train_unbeatable_model() -> bool:
    """Train unbeatable model"""
    print("\n🏆 STEP 3: Training unbeatable model...")
    
    # Check if converted data exists
    converted_data = os.path.join(current_dir, "unbeatable", "converted_unbeatable_data.jsonl")
    if not os.path.exists(converted_data):
        print(f"❌ Converted unbeatable data not found: {converted_data}")
        return False
    
    # Train model
    train_script = os.path.join(current_dir, "unbeatable", "train_unbeatable_model.py")
    cmd = [sys.executable, train_script, "--data_path", converted_data]
    
    return run_command(cmd, cwd=os.path.join(current_dir, "unbeatable"))


def check_prerequisites() -> bool:
    """Check if prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check web_backend training data
    web_backend_dir = os.path.join(project_root, "web_backend")
    training_data_file = os.path.join(web_backend_dir, "training_data.jsonl")
    
    if not os.path.exists(training_data_file):
        print(f"❌ Training data not found: {training_data_file}")
        print("   Please run some games to generate training data first.")
        return False
    
    # Check file size
    file_size = os.path.getsize(training_data_file)
    if file_size == 0:
        print(f"❌ Training data file is empty: {training_data_file}")
        print("   Please run some games to generate training data first.")
        return False
    
    print(f"✅ Training data found: {training_data_file} ({file_size} bytes)")
    
    # Check if we have any training records
    try:
        with open(training_data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            non_empty_lines = [line for line in lines if line.strip()]
            
        if len(non_empty_lines) == 0:
            print(f"❌ No training records found in {training_data_file}")
            return False
        
        print(f"✅ Found {len(non_empty_lines)} training records")
        
    except Exception as e:
        print(f"❌ Error reading training data: {e}")
        return False
    
    return True


def main():
    """Main retraining pipeline"""
    parser = argparse.ArgumentParser(description='Retrain models from web_backend logs')
    parser.add_argument('--skip-conversion', action='store_true', 
                       help='Skip data conversion step')
    # --skip-general removed - general model not used
    parser.add_argument('--skip-two-layer', action='store_true',
                       help='Skip two-layer model training')
    parser.add_argument('--skip-unbeatable', action='store_true',
                       help='Skip unbeatable model training')
    
    args = parser.parse_args()
    
    print("🤖 AI-SAM Model Retraining Pipeline")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return
    
    # Track success
    steps_completed = []
    steps_failed = []
    
    # Step 1: Convert training data
    if not args.skip_conversion:
        if convert_training_data():
            steps_completed.append("Data Conversion")
        else:
            steps_failed.append("Data Conversion")
            print("\n❌ Data conversion failed. Stopping pipeline.")
            return
    else:
        print("\n⏭️ Skipping data conversion step")
    
    # Step 2: Train two-layer model
    if not args.skip_two_layer:
        if train_two_layer_model():
            steps_completed.append("Two-Layer Model Training")
        else:
            steps_failed.append("Two-Layer Model Training")
    else:
        print("\n⏭️ Skipping two-layer model training")
    
    # Step 3: Train unbeatable model
    if not args.skip_unbeatable:
        if train_unbeatable_model():
            steps_completed.append("Unbeatable Model Training")
        else:
            steps_failed.append("Unbeatable Model Training")
    else:
        print("\n⏭️ Skipping unbeatable model training")
    
    # Final summary
    print("\n📊 PIPELINE SUMMARY")
    print("=" * 50)
    
    if steps_completed:
        print("✅ Completed steps:")
        for step in steps_completed:
            print(f"   ✓ {step}")
    
    if steps_failed:
        print("\n❌ Failed steps:")
        for step in steps_failed:
            print(f"   ✗ {step}")
    
    if not steps_failed:
        print("\n🎉 All steps completed successfully!")
        print("🚀 Models are ready for deployment!")
    else:
        print(f"\n⚠️ Pipeline completed with {len(steps_failed)} failures.")
        print("   Check the error messages above for details.")


if __name__ == '__main__':
    main()
