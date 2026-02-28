from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    seats: int = 4
    episodes: int = 1000
    gamma: float = 0.99
    lr: float = 3e-4
    hidden_dim: int = 128
    max_steps: int = 400
    log_interval: int = 25
    save_path: Optional[str] = None
    use_self_play: bool = True  # Enable self-play by default
    load_path: Optional[str] = None  # Path to checkpoint to resume training from
    # PPO hyperparameters
    ppo_clip_epsilon: float = 0.2  # PPO clipping parameter
    ppo_epochs: int = 3  # Number of update epochs per batch (reduced from 6 to prevent overfitting)
    ppo_batch_size: int = 128  # Batch size for PPO updates
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.03  # Reduced from 0.1 to stabilize strategy
    normalize_returns: bool = False  # Whether to normalize returns (False preserves magnitude information)
    max_grad_norm: float = 1.0  # Gradient clipping norm
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation (0=TD, 1=MC)
    # Opponent pool for diverse self-play
    opponent_pool_size: int = 20  # Reduced from 50 to focus on recent high-quality versions
    opponent_pool_checkpoint_interval: int = 1000  # Save checkpoint to pool every N episodes
    opponent_temperature_min: float = 0.5  # Min temperature for opponent variation
    opponent_temperature_max: float = 1.5  # Max temperature (reduced from 2.0)
    opponent_weight_noise_std: float = 0.0  # Set to 0.0 to remove unnecessary noise
    opponent_sampling_temperature: float = 1.0  # Default/fallback temperature
    scripted_opponent_ratio: float = 0.2  # Reduced from 0.3 as self-play models become more stable


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
        
        # Value network uses state_dim (stateless state features), not feature_dim
        # Value network uses enriched state (state + pooled moves)
        self.value_net = ValueNetwork(
            state_dim=self.feature_builder.state_dim,
            move_feature_dim=self.feature_builder.feature_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        
        # Load checkpoint if provided
        self.start_episode = 0
        if config.load_path and os.path.exists(config.load_path):
            print(f"[RLTrainer] Loading checkpoint from {config.load_path}")
            checkpoint = torch.load(config.load_path, map_location=self.device)
            metadata = checkpoint.get("metadata", {})
            state_dim = metadata.get("state_dim", self.feature_builder.state_dim)
            if state_dim != self.feature_builder.state_dim:
                self.value_net = ValueNetwork(
                    state_dim=self.feature_builder.state_dim,
                    move_feature_dim=self.feature_builder.feature_dim,
                    hidden_dim=config.hidden_dim,
                ).to(self.device)
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
        
        # Opponent pool management for diverse self-play
        self.opponent_pool: List[str] = []  # List of checkpoint paths
        self.opponent_pool_dir: Optional[str] = None
        if config.use_self_play and config.opponent_pool_size > 0:
            # Create opponent pool directory
            if config.save_path:
                pool_dir = os.path.join(os.path.dirname(config.save_path), "opponent_pool")
                os.makedirs(pool_dir, exist_ok=True)
                self.opponent_pool_dir = pool_dir
                print(f"[RLTrainer] Opponent pool enabled: size={config.opponent_pool_size}, interval={config.opponent_pool_checkpoint_interval}")
        
        # Create environment with policy for self-play
        # Pass opponent pool info to environment
        self.env = CardGameEnv(
            game_type=config.game_type,
            seats=config.seats,
            max_steps=config.max_steps,
            feature_builder=self.feature_builder,
            policy_network=self.policy,  # Current policy (fallback)
            use_self_play=config.use_self_play,
            opponent_pool_dir=self.opponent_pool_dir,
            opponent_pool=self.opponent_pool,
            opponent_sampling_temperature=config.opponent_sampling_temperature,
            feature_dim=self.feature_builder.feature_dim,
            hidden_dim=config.hidden_dim,
            scripted_opponent_ratio=config.scripted_opponent_ratio,
        )
        
        # Initialize pool with initial checkpoint (episode 0) after env is created
        if config.use_self_play and config.opponent_pool_size > 0 and self.opponent_pool_dir:
            self._save_opponent_checkpoint(0)
            print(f"[RLTrainer] Initialized opponent pool with starting checkpoint (pool size: {len(self.opponent_pool)})")

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
            
            # Save checkpoint to opponent pool periodically (skip episode 0, already saved in __init__)
            if (self.opponent_pool_dir is not None and 
                episode % self.config.opponent_pool_checkpoint_interval == 0 and 
                episode > 0):
                self._save_opponent_checkpoint(episode)
            
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
        """PPO update with clipped objective and GAE."""
        if len(batch_obs) == 0:
            return
        
        # Calculate baseline values for GAE (before update)
        with torch.no_grad():
            values = []
            for obs in batch_obs:
                value = self._get_value(obs)
                values.append(value.item())
        
        # Calculate GAE advantages and returns
        advantages_list, returns_list = self._calculate_gae(
            batch_rewards, values, batch_dones,
            gamma=self.config.gamma, lam=self.config.gae_lambda
        )
        returns_tensor = torch.tensor(returns_list, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_log_probs_tensor = torch.stack(batch_old_log_probs).to(self.device)
        actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        
        # Store observations as tensors for efficiency
        obs_move_features = []
        obs_state_features = []
        for obs in batch_obs:
            move_features = torch.tensor(
                obs["move_features"], dtype=torch.float32, device=self.device
            )
            obs_move_features.append(move_features)
            state_features = torch.tensor(
                obs["state_features"], dtype=torch.float32, device=self.device
            )
            obs_state_features.append(state_features)
        
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
                batch_state_features_subset = [obs_state_features[i] for i in batch_indices.cpu().numpy()]
                batch_actions_subset = actions_tensor[batch_indices]
                batch_old_log_probs_subset = old_log_probs_tensor[batch_indices]
                batch_advantages_subset = advantages[batch_indices]
                batch_returns_subset = returns_tensor[batch_indices]
                
                # Calculate new log probs, values, and entropies
                new_log_probs = []
                new_values = []
                entropies = []
                for move_features, state_features, action in zip(
                    batch_move_features_subset,
                    batch_state_features_subset,
                    batch_actions_subset,
                ):
                    logits = self.policy(move_features)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs.append(dist.log_prob(action))
                    entropies.append(dist.entropy())
                    
                    # Compute enriched features for Critic
                    mean_move = move_features.mean(dim=0).unsqueeze(0)
                    max_move = move_features.max(dim=0)[0].unsqueeze(0)
                    combined = torch.cat([state_features.unsqueeze(0), mean_move, max_move], dim=-1)
                    value = self.value_net(combined)
                    new_values.append(value)
                
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
                    max_norm=self.config.max_grad_norm
                )
                self.optimizer.step()
    
    def _calculate_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate Generalized Advantage Estimation (GAE).
        
        GAE reduces variance compared to Monte Carlo returns while maintaining
        acceptable bias. This is critical for long episodes where final reward
        dominates step rewards.
        
        Args:
            rewards: List of rewards at each timestep
            values: List of value estimates at each timestep
            dones: List of done flags at each timestep
            gamma: Discount factor
            lam: GAE lambda (0=TD(0), 1=Monte Carlo)
            
        Returns:
            Tuple of (advantages, returns) lists
        """
        advantages = [0.0] * len(rewards)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                gae = 0.0  # Reset GAE at episode boundary
            else:
                next_value = values[t + 1] if t + 1 < len(values) else 0.0
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]
            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        
        # Returns = advantages + values
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def _get_value(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Helper to compute value using enriched features (state + pooled moves)."""
        state_features = torch.tensor(
            obs["state_features"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        move_features = torch.tensor(
            obs["move_features"], dtype=torch.float32, device=self.device
        )
        # Pooling: mean and max across moves
        mean_move = move_features.mean(dim=0).unsqueeze(0)
        max_move = move_features.max(dim=0)[0].unsqueeze(0)
        # Concatenate state and pooled moves
        combined = torch.cat([state_features, mean_move, max_move], dim=-1)
        return self.value_net(combined)

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
                    "state_dim": self.feature_builder.state_dim,
                },
                "feature_dim": self.feature_builder.feature_dim,
            },
            path,
        )
        print(f"[RLTrainer] Saved PPO checkpoint to {path} (total episodes: {total_episodes})")
    
    def _save_opponent_checkpoint(self, episode: int) -> None:
        """Save current policy to opponent pool with variation parameters for diverse self-play."""
        if self.opponent_pool_dir is None:
            return
        
        # Sample variation parameters for this opponent
        # Each opponent gets different temperature and potentially weight noise
        opponent_temperature = random.uniform(
            self.config.opponent_temperature_min,
            self.config.opponent_temperature_max
        )
        
        checkpoint_path = os.path.join(self.opponent_pool_dir, f"opponent_ep{episode}.pt")
        
        # Optionally add weight noise for diversity
        state_dict = self.policy.state_dict()
        if self.config.opponent_weight_noise_std > 0:
            # Add small random noise to weights
            noisy_state_dict = {}
            for key, value in state_dict.items():
                noise = torch.randn_like(value) * self.config.opponent_weight_noise_std
                noisy_state_dict[key] = value + noise
            state_dict = noisy_state_dict
        
        torch.save(
            {
                "model_state_dict": state_dict,
                "metadata": {
                    "episode": episode,
                    "game_type": self.config.game_type,
                    "hidden_dim": self.config.hidden_dim,
                    "opponent_temperature": opponent_temperature,  # Variation parameter
                },
                "feature_dim": self.feature_builder.feature_dim,
            },
            checkpoint_path,
        )
        
        # Add to pool
        self.opponent_pool.append(checkpoint_path)
        
        # Maintain pool size: remove oldest if exceeded
        if len(self.opponent_pool) > self.config.opponent_pool_size:
            oldest_checkpoint = self.opponent_pool.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"[RLTrainer] Removed oldest opponent checkpoint: {os.path.basename(oldest_checkpoint)}")
        
        # Update environment's opponent pool
        self.env.update_opponent_pool(self.opponent_pool)
        
        print(f"[RLTrainer] Saved opponent checkpoint (pool size: {len(self.opponent_pool)}, temp: {opponent_temperature:.2f})")

