from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from .features import FrameworkAwareFeatureBuilder
except ImportError:
    from features import FrameworkAwareFeatureBuilder


class CrossMoveAttention(nn.Module):
    """
    Relational module that allows moves to attend to each other.
    Essential for card games where a move's value depends on other possible moves.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_moves, embed_dim]
        Returns:
            [num_moves, embed_dim]
        """
        # MultiheadAttention expects [batch, seq, feature]
        x_batch = x.unsqueeze(0)
        attn_out, _ = self.attn(x_batch, x_batch, x_batch)
        x = x_batch + self.dropout(attn_out)
        x = self.norm(x)
        return x.squeeze(0)


class PolicyNetwork(nn.Module):
    """Simple feed-forward policy that scores each move feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = CrossMoveAttention(hidden_dim, num_heads=4, dropout=dropout)
        self.scorer = nn.Sequential(
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
        x = self.embedding(move_features)  # [num_moves, hidden_dim]
        x = self.attention(x)             # [num_moves, hidden_dim]
        logits = self.scorer(x).squeeze(-1) # [num_moves]
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

    def __init__(self, state_dim: int, move_feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # Input: state_features + pooled_move_features (mean + max)
        # Combined dim: state_dim + 2 * move_feature_dim
        combined_dim = state_dim + (2 * move_feature_dim)
        
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            combined_features: Tensor of shape [batch_size, combined_dim]
        """
        return self.net(combined_features).squeeze(-1)


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
        # Always use embedding[0] for input dim check as per Phase 2 architecture
        expected_dim = self.policy.embedding[0].in_features
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
        # Use self.policy.embedding[0].out_features for hidden_dim check
        if checkpoint_hidden_dim and checkpoint_hidden_dim != self.policy.embedding[0].out_features:
            needs_rebuild = True
            print(f"[RLLearner] Checkpoint hidden_dim={checkpoint_hidden_dim} differs from current {self.policy.embedding[0].out_features}. Rebuilding network...")
        
        if checkpoint_feature_dim and checkpoint_feature_dim != current_feature_dim:
            needs_rebuild = True
            print(f"[RLLearner] Checkpoint feature_dim={checkpoint_feature_dim} differs from current {current_feature_dim}. Rebuilding network...")
        
        if needs_rebuild:
            # Use checkpoint's dimensions for rebuilding
            rebuild_hidden_dim = checkpoint_hidden_dim or self.policy.embedding[0].out_features
            rebuild_input_dim = checkpoint_feature_dim or current_feature_dim
            self.policy = PolicyNetwork(
                input_dim=rebuild_input_dim,
                hidden_dim=rebuild_hidden_dim
            )
            self.metadata["hidden_dim"] = rebuild_hidden_dim
        
        if "model_state_dict" in checkpoint:
            # After Phase 2, we use self.policy.embedding[0] instead of net[0]
            # Check first layer weight shape if possible, or just load
            self.policy.load_state_dict(checkpoint["model_state_dict"])
            print(f"[RLLearner] Successfully loaded weights from {path}") # Changed weight_path to path
        
        self.metadata = checkpoint_metadata
