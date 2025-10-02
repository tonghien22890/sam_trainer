#!/usr/bin/env python3
import subprocess
import sys
import os


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    core = os.path.join(current_dir, 'train_style_learner_core.py')
    data_default = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'simple_sam.jsonl')
    cmd = [sys.executable, core, '--game_type', 'sam', '--data_path', data_default]
    sys.exit(subprocess.call(cmd))


if __name__ == '__main__':
    main()


