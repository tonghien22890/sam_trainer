from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from .features import FrameworkAwareFeatureBuilder
except ImportError:
    from features import FrameworkAwareFeatureBuilder


class PolicyNetwork(nn.Module):
    """Simple feed-forward policy that scores each move feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, move_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            move_features: Tensor of shape [num_moves, feature_dim]

        Returns:
            Logits tensor of shape [num_moves]
        """
        logits = self.net(move_features).squeeze(-1)
        return logits

    def sample_action(
        self, move_features: torch.Tensor
    ) -> (int, torch.Tensor):
        logits = self.forward(move_features)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def get_action_log_probs(
        self, move_features: torch.Tensor, action_idx: int
    ) -> torch.Tensor:
        """
        Get log probability for a specific action (for PPO).
        
        Args:
            move_features: Tensor of shape [num_moves, feature_dim]
            action_idx: Integer index of the selected action
            
        Returns:
            Log probability tensor (scalar)
        """
        logits = self.forward(move_features)  # [num_moves]
        dist = Categorical(logits=logits)
        return dist.log_prob(torch.tensor(action_idx, device=move_features.device))


class ValueNetwork(nn.Module):
    """Value network (critic) for PPO to estimate state values."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_features: Tensor of shape [batch_size, feature_dim]
                           (aggregated features representing game state)

        Returns:
            Value estimates tensor of shape [batch_size]
        """
        return self.net(state_features).squeeze(-1)


class RLLearner:
    """
    Thin inference wrapper so the RL policy can be used similarly to StyleLearner.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 128,
    ):
        self.feature_builder = FrameworkAwareFeatureBuilder()
        self.policy = PolicyNetwork(
            input_dim=self.feature_builder.feature_dim, hidden_dim=hidden_dim
        )
        self.metadata: Dict[str, Any] = {"hidden_dim": hidden_dim}
        if model_path:
            self.load(model_path)

    def predict_move(
        self,
        game_record: Dict[str, Any],
        legal_moves: List[Dict[str, Any]],
        framework: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Pick the highest scoring move deterministicly.
        """
        feature_matrix = self.feature_builder.build_feature_matrix(
            game_record, legal_moves, framework
        )
        
        # Check feature_dim match
        expected_dim = self.policy.net[0].in_features
        actual_dim = len(feature_matrix[0]) if feature_matrix else 0
        if actual_dim != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_dim} (from model), "
                f"got {actual_dim} (from feature_builder). "
                f"Model was trained with different feature configuration. "
                f"Please retrain model or adjust feature flags."
            )
        
        tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        logits = self.policy(tensor)
        action_idx = torch.argmax(logits).item()
        action_idx = max(0, min(action_idx, len(legal_moves) - 1))
        return legal_moves[action_idx]

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "metadata": self.metadata,
                "feature_dim": self.feature_builder.feature_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        
        # Read hidden_dim and feature_dim from checkpoint
        checkpoint_metadata = checkpoint.get("metadata", {})
        checkpoint_hidden_dim = checkpoint_metadata.get("hidden_dim")
        checkpoint_feature_dim = checkpoint.get("feature_dim")  # Saved in root, not metadata
        
        current_feature_dim = self.feature_builder.feature_dim
        
        # Check if we need to rebuild network (different hidden_dim or feature_dim)
        needs_rebuild = False
        if checkpoint_hidden_dim and checkpoint_hidden_dim != self.policy.net[0].out_features:
            needs_rebuild = True
            print(f"[RLLearner] Checkpoint hidden_dim={checkpoint_hidden_dim} differs from current {self.policy.net[0].out_features}. Rebuilding network...")
        
        if checkpoint_feature_dim and checkpoint_feature_dim != current_feature_dim:
            needs_rebuild = True
            print(f"[RLLearner] Checkpoint feature_dim={checkpoint_feature_dim} differs from current {current_feature_dim}. Rebuilding network...")
        
        if needs_rebuild:
            # Use checkpoint's dimensions for rebuilding
            rebuild_hidden_dim = checkpoint_hidden_dim or self.policy.net[0].out_features
            rebuild_input_dim = checkpoint_feature_dim or current_feature_dim
            self.policy = PolicyNetwork(
                input_dim=rebuild_input_dim,
                hidden_dim=rebuild_hidden_dim
            )
            self.metadata["hidden_dim"] = rebuild_hidden_dim
        
        # Load state dict
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.metadata = checkpoint_metadata

