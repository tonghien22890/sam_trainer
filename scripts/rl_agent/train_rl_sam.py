#!/usr/bin/env python3
import os
import subprocess
import sys


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    core_script = os.path.join(current_dir, "train_rl_core.py")
    default_save = os.path.join(models_dir, "rl_policy_sam.pt")

    cmd = [
        sys.executable,
        core_script,
        "--game_type",
        "sam",
        "--save_path",
        default_save,
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

