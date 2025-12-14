from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

try:
    from .env import CardGameEnv
    from .policy import PolicyNetwork, ValueNetwork
    from .features import FrameworkAwareFeatureBuilder
except ImportError:
    from env import CardGameEnv
    from policy import PolicyNetwork, ValueNetwork
    from features import FrameworkAwareFeatureBuilder


@dataclass
class TrainerConfig:
    game_type: str = "sam"
    seats: int = 2
    episodes: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    hidden_dim: int = 128
    max_steps: int = 400
    log_interval: int = 25
    save_path: Optional[str] = None
    use_self_play: bool = True  # Enable self-play by default
    load_path: Optional[str] = None  # Path to checkpoint to resume training from
    # PPO hyperparameters
    ppo_clip_epsilon: float = 0.2  # PPO clipping parameter
    ppo_epochs: int = 6  # Number of update epochs per batch
    ppo_batch_size: int = 256  # Batch size for PPO updates
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.02  # Entropy bonus coefficient


class RLTrainer:
    """
    PPO (Proximal Policy Optimization) trainer with self-play.
    More stable and sample-efficient than REINFORCE.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_builder = FrameworkAwareFeatureBuilder()
        
        # Initialize policy and value networks
        self.policy = PolicyNetwork(
            input_dim=self.feature_builder.feature_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            input_dim=self.feature_builder.feature_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        
        # Load checkpoint if provided
        self.start_episode = 0
        if config.load_path and os.path.exists(config.load_path):
            print(f"[RLTrainer] Loading checkpoint from {config.load_path}")
            checkpoint = torch.load(config.load_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint["model_state_dict"])
            
            # Load value network if available
            if "value_state_dict" in checkpoint:
                self.value_net.load_state_dict(checkpoint["value_state_dict"])
                print(f"[RLTrainer] Loaded value network from checkpoint")
            
            # Try to load optimizer state if available
            if "optimizer_state_dict" in checkpoint:
                self.optimizer = torch.optim.Adam(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    lr=config.lr
                )
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print(f"[RLTrainer] Loaded optimizer state from checkpoint")
            else:
                self.optimizer = torch.optim.Adam(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    lr=config.lr
                )
            
            # Get starting episode from metadata if available
            metadata = checkpoint.get("metadata", {})
            if "episodes" in metadata:
                self.start_episode = int(metadata.get("episodes", 0))
                print(f"[RLTrainer] Resuming from episode {self.start_episode}")
        else:
            self.optimizer = torch.optim.Adam(
                list(self.policy.parameters()) + list(self.value_net.parameters()),
                lr=config.lr
            )
            if config.load_path:
                print(f"[RLTrainer] Warning: Load path {config.load_path} not found, starting from scratch")
        
        # Create environment with policy for self-play
        self.env = CardGameEnv(
            game_type=config.game_type,
            seats=config.seats,
            max_steps=config.max_steps,
            feature_builder=self.feature_builder,
            policy_network=self.policy,  # Pass policy for self-play
            use_self_play=config.use_self_play,
        )

    def train(self) -> Dict[str, float]:
        episode_rewards: List[float] = []
        # Start from resume episode if loading checkpoint
        start_episode = self.start_episode + 1 if self.start_episode > 0 else 1
        total_episodes = self.start_episode + self.config.episodes
        
        # PPO: Collect batch of episodes before updating
        batch_obs = []
        batch_actions = []
        batch_old_log_probs = []
        batch_rewards = []
        batch_dones = []
        
        episode = start_episode
        while episode <= total_episodes:
            # Collect one episode
            obs = self.env.reset()
            episode_obs = []
            episode_actions = []
            episode_old_log_probs = []
            episode_rewards_list = []
            episode_dones = []
            done = False

            while not done:
                move_features = torch.tensor(
                    obs["move_features"], dtype=torch.float32, device=self.device
                )
                action, log_prob = self.policy.sample_action(move_features)
                next_obs, reward, done, info = self.env.step(action)
                
                # Store for PPO update
                episode_obs.append(obs)
                episode_actions.append(action)
                episode_old_log_probs.append(log_prob.detach())
                episode_rewards_list.append(reward)
                episode_dones.append(done)
                
                obs = next_obs

            episode_return = sum(episode_rewards_list)
            episode_rewards.append(episode_return)
            
            # Add to batch
            batch_obs.extend(episode_obs)
            batch_actions.extend(episode_actions)
            batch_old_log_probs.extend(episode_old_log_probs)
            batch_rewards.extend(episode_rewards_list)
            batch_dones.extend(episode_dones)
            
            # Update when batch is large enough or at end
            if len(batch_obs) >= self.config.ppo_batch_size or episode == total_episodes:
                if len(batch_obs) > 0:
                    self._ppo_update(
                        batch_obs, batch_actions, batch_old_log_probs,
                        batch_rewards, batch_dones
                    )
                    # Clear batch
                    batch_obs = []
                    batch_actions = []
                    batch_old_log_probs = []
                    batch_rewards = []
                    batch_dones = []

            if episode % self.config.log_interval == 0:
                avg_reward = sum(episode_rewards[-self.config.log_interval :]) / max(
                    1, self.config.log_interval
                )
                print(
                    f"[RLTrainer] Episode {episode}/{total_episodes} "
                    f"| avg_reward={avg_reward:.3f} | last_return={episode_return:.3f}"
                )
            
            episode += 1

        metrics = {
            "episodes": float(self.config.episodes),
            "avg_reward": sum(episode_rewards) / max(1, len(episode_rewards)),
        }

        if self.config.save_path:
            self._save_checkpoint(self.config.save_path, metrics)

        return metrics

    def _ppo_update(
        self,
        batch_obs: List[Dict[str, Any]],
        batch_actions: List[int],
        batch_old_log_probs: List[torch.Tensor],
        batch_rewards: List[float],
        batch_dones: List[bool],
    ) -> None:
        """PPO update with clipped objective."""
        if len(batch_obs) == 0:
            return
        
        # Calculate returns (discounted cumulative rewards)
        returns = self._calculate_returns(batch_rewards, batch_dones)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize returns
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )
        
        # Calculate baseline values for advantage estimation (before update)
        with torch.no_grad():
            values = []
            for obs in batch_obs:
                # Use average of move features as state representation
                move_features = torch.tensor(
                    obs["move_features"], dtype=torch.float32, device=self.device
                )
                state_features = move_features.mean(dim=0, keepdim=True)  # [1, feature_dim]
                value = self.value_net(state_features)
                values.append(value.item())
        
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        advantages = returns_tensor - values_tensor
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_log_probs_tensor = torch.stack(batch_old_log_probs).to(self.device)
        actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        
        # Store observations as tensors for efficiency
        obs_move_features = []
        for obs in batch_obs:
            move_features = torch.tensor(
                obs["move_features"], dtype=torch.float32, device=self.device
            )
            obs_move_features.append(move_features)
        
        # PPO update: multiple epochs
        for epoch in range(self.config.ppo_epochs):
            # Shuffle for better training
            indices = torch.randperm(len(batch_obs), device=self.device)
            
            # Mini-batch updates
            for start_idx in range(0, len(batch_obs), self.config.ppo_batch_size):
                end_idx = min(start_idx + self.config.ppo_batch_size, len(batch_obs))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_move_features_subset = [obs_move_features[i] for i in batch_indices.cpu().numpy()]
                batch_actions_subset = actions_tensor[batch_indices]
                batch_old_log_probs_subset = old_log_probs_tensor[batch_indices]
                batch_advantages_subset = advantages[batch_indices]
                batch_returns_subset = returns_tensor[batch_indices]
                
                # Calculate new log probs, values, and entropies
                new_log_probs = []
                new_values = []
                entropies = []
                for move_features, action in zip(batch_move_features_subset, batch_actions_subset):
                    logits = self.policy(move_features)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs.append(dist.log_prob(action))
                    entropies.append(dist.entropy())
                    
                    # Value estimate: use average of move features as state representation
                    state_features = move_features.mean(dim=0, keepdim=True)
                    new_values.append(self.value_net(state_features))
                
                new_log_probs_tensor = torch.stack(new_log_probs)
                new_values_tensor = torch.stack(new_values).squeeze()
                entropy_tensor = torch.stack(entropies)
                
                # Calculate ratio for PPO clipping
                ratio = torch.exp(new_log_probs_tensor - batch_old_log_probs_subset)
                
                # PPO clipped objective
                policy_loss_1 = ratio * batch_advantages_subset
                policy_loss_2 = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo_clip_epsilon,
                    1.0 + self.config.ppo_clip_epsilon,
                ) * batch_advantages_subset
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                value_loss = torch.nn.functional.mse_loss(
                    new_values_tensor, batch_returns_subset
                )
                
                # Entropy bonus (encourage exploration)
                entropy = entropy_tensor.mean()
                
                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()
    
    def _calculate_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """Calculate discounted returns."""
        returns = []
        running_return = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_return = 0.0
            running_return = reward + self.config.gamma * running_return
            returns.insert(0, running_return)
        return returns

    def _save_checkpoint(self, path: str, metrics: Dict[str, float]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Calculate total episodes (including resumed episodes)
        total_episodes = self.start_episode + self.config.episodes
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value_net.state_dict(),  # Save value network
                "optimizer_state_dict": self.optimizer.state_dict(),  # Save optimizer state for resume
                "metadata": {
                    "game_type": self.config.game_type,
                    "hidden_dim": self.config.hidden_dim,
                    "episodes": total_episodes,  # Total episodes trained so far
                    "start_episode": self.start_episode,
                    "metrics": metrics,
                },
                "feature_dim": self.feature_builder.feature_dim,
            },
            path,
        )
        print(f"[RLTrainer] Saved PPO checkpoint to {path} (total episodes: {total_episodes})")

