#!/usr/bin/env python3
import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm
from datetime import datetime

# Add current directory to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from env import CardGameEnv
from policy import PolicyNetwork, RLLearner
from features import FrameworkAwareFeatureBuilder

class RLEvaluator:
    def __init__(self, model_path: str, game_type: str = "sam", seats: int = 4):
        self.model_path = model_path
        self.game_type = game_type
        self.seats = seats
        
        # Load model metadata
        checkpoint = torch.load(model_path, map_location="cpu")
        self.hidden_dim = checkpoint.get("metadata", {}).get("hidden_dim", 128)
        
        # Initialize feature builder
        self.feature_builder = FrameworkAwareFeatureBuilder()
        
        # Load policy
        self.policy = PolicyNetwork(self.feature_builder.feature_dim, self.hidden_dim)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        
        # Initialize environment (no self-play for evaluation against scripted)
        self.env = CardGameEnv(
            game_type=game_type,
            seats=seats,
            feature_builder=self.feature_builder,
            use_self_play=False  # Compare against scripted bot
        )

    def evaluate(self, episodes: int = 100) -> Dict[str, Any]:
        wins = 0
        total_rewards = []
        cards_left_when_lost = []
        blocking_attempts = 0
        blocking_successes = 0
        total_moves = 0
        
        print(f"[*] Evaluating model {os.path.basename(self.model_path)} over {episodes} episodes...")
        
        for _ in tqdm(range(episodes)):
            obs = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Check for blocking situation before moving
                # Situation: Agent is about to play, and an opponent has < 3 cards
                is_blocking_situation = self._is_blocking_situation()
                
                # Model predicts move
                move_features = torch.tensor(obs["move_features"], dtype=torch.float32)
                with torch.no_grad():
                    logits = self.policy(move_features)
                    action = torch.argmax(logits).item()
                
                selected_move = self.env._latest_legal_moves[action]
                
                # Track blocking success
                if is_blocking_situation:
                    blocking_attempts += 1
                    # If action is not PASS and can beat last move, it's a blocking attempt
                    if selected_move["type"] != "pass":
                        blocking_successes += 1
                
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                total_moves += 1
                
            total_rewards.append(total_reward)
            
            # Record winner
            if self.env.game.state.winner_id == self.env.agent_id:
                wins += 1
            else:
                agent_player = self.env.game.state.get_player(self.env.agent_id)
                cards_left_when_lost.append(len(agent_player.hand))
        
        win_rate = (wins / episodes) * 100
        avg_reward = sum(total_rewards) / episodes
        avg_cards_lost = sum(cards_left_when_lost) / len(cards_left_when_lost) if cards_left_when_lost else 0
        block_rate = (blocking_successes / blocking_attempts * 100) if blocking_attempts > 0 else 0
        
        results = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_cards_left_when_lost": avg_cards_lost,
            "blocking_success_rate": block_rate,
            "total_episodes": episodes,
            "avg_moves_per_game": total_moves / episodes
        }
        
        return results

    def _is_blocking_situation(self) -> bool:
        """Helper to detect if agent SHOULD block"""
        for player in self.env.game.state.players:
            if player.player_id != self.env.agent_id and len(player.hand) <= 2:
                return True
        return False

def generate_report(results: Dict[str, Any], output_path: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# RL Agent Evaluation Report
    
Generated on: {timestamp}

## Summary Metrics
- **Win Rate**: {results['win_rate']:.2f}%
- **Avg Reward**: {results['avg_reward']:.4f}
- **Avg Cards Left (Loss)**: {results['avg_cards_left_when_lost']:.2f}
- **Blocking Success Rate**: {results['blocking_success_rate']:.2f}%
- **Avg Moves per Game**: {results['avg_moves_per_game']:.1f}

## Detailed Analysis
- AI wins **{results['win_rate']:.2f}%** of games against ScriptedBot.
- When losing, the AI has **{results['avg_cards_left_when_lost']:.2f}** cards remaining.
- AI attempts to block threatening opponents **{results['blocking_success_rate']:.2f}%** of the time.

---
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[+] Evaluation report saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--game_type", type=str, default="sam")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    
    evaluator = RLEvaluator(args.model, args.game_type)
    results = evaluator.evaluate(args.episodes)
    
    report_path = os.path.join(os.path.dirname(args.model), "TRAINING_REPORT.md")
    generate_report(results, report_path)
