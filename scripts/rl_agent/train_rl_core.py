#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from trainer import RLTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL policy for Sam/TLMN.")
    parser.add_argument("--game_type", type=str, default="sam", choices=["sam", "tlmn"])
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--seats", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--self_play", action="store_true", default=True, help="Enable self-play (agent vs itself)")
    parser.add_argument("--no_self_play", dest="self_play", action="store_false", help="Disable self-play (use scripted opponents)")
    parser.add_argument("--load_path", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = os.path.dirname(os.path.abspath(args.save_path))
    os.makedirs(save_dir, exist_ok=True)

    config = TrainerConfig(
        game_type=args.game_type,
        seats=args.seats,
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        save_path=args.save_path,
        use_self_play=args.self_play,
        load_path=args.load_path,
    )

    trainer = RLTrainer(config)
    metrics = trainer.train()
    print(f"[train_rl_core] Finished training. Metrics: {metrics}")


if __name__ == "__main__":
    main()

