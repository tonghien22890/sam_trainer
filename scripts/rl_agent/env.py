from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_engine.core.card_encoding import CardEncoder
from game_engine.core.game_types import PlayerAction
from game_engine.games.sam_game import SamGame
from game_engine.games.tlmn_game import TLMNGame

from model_build.scripts.two_layer.framework_generator import FrameworkGenerator

try:
    from .features import FrameworkAwareFeatureBuilder
except ImportError:
    from features import FrameworkAwareFeatureBuilder


class CardGameEnv:
    """
    Lightweight RL environment that wraps the existing Sam/TLMN game engine.

    The environment controls seat 0 (agent) while other seats are driven by
    a lightweight scripted opponent. Observations expose one feature vector per legal move,
    matching the StyleLearner feature space to allow policy sharing.
    """

    def __init__(
        self,
        game_type: str = "sam",
        seats: int = 4,
        reward_config: Optional[Dict[str, float]] = None,
        max_steps: int = 400,
        feature_builder: Optional[FrameworkAwareFeatureBuilder] = None,
        seed: Optional[int] = None,
        policy_network: Optional[Any] = None,
        use_self_play: bool = False,
    ) -> None:
        self.game_type = game_type.lower()
        if self.game_type not in {"sam", "tlmn"}:
            raise ValueError("game_type must be 'sam' or 'tlmn'")

        self.seats = max(2, seats)
        self.agent_id = 0
        self.max_steps = max_steps
        self.reward_cfg = reward_config or {
            "win": 1.0,
            "loss": -1.0,
            "step": 0,
            "invalid": -1.0,
        }

        self.feature_builder = feature_builder or FrameworkAwareFeatureBuilder()
        self.framework_generator = FrameworkGenerator()
        self.policy_network = policy_network  # Policy network for self-play
        self.use_self_play = use_self_play

        self._rng = random.Random(seed)

        self.game = None
        self.steps = 0
        self._latest_legal_moves: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng.seed(seed)

        self.game = self._create_game()
        self.steps = 0

        self._advance_until_agent_turn()
        return self._build_observation()

    def step(
        self, action_index: int
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.game is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        legal_moves = self._latest_legal_moves or self._build_legal_moves(self.agent_id)
        if not legal_moves:
            # No moves available -> force pass to avoid crashes
            legal_moves = [
                {"type": "pass", "cards": [], "combo_type": "pass", "rank_value": -1}
            ]

        if action_index < 0 or action_index >= len(legal_moves):
            obs = self._build_observation()
            return obs, self.reward_cfg["invalid"], True, {
                "error": "illegal_action_index",
                "legal_moves": len(legal_moves),
            }

        selected_move = legal_moves[action_index]
        success = self._apply_move(self.agent_id, selected_move)
        self.steps += 1

        if not success:
            obs = self._build_observation()
            return obs, self.reward_cfg["invalid"], True, {"error": "apply_failed"}

        if self.game.state.is_finished:
            reward = self._final_reward()
            return self._terminal_observation(), reward, True, {
                "winner_id": self.game.state.winner_id
            }

        self._advance_until_agent_turn()
        done = self.game.state.is_finished or self.steps >= self.max_steps
        if done:
            reward = self._final_reward() if self.game.state.is_finished else 0.0
            return self._terminal_observation(), reward, True, {
                "winner_id": self.game.state.winner_id
            }

        obs = self._build_observation()
        return obs, self.reward_cfg["step"], False, {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _create_game(self):
        game_id = f"rl_{uuid.uuid4().hex[:8]}"
        names = [f"RL-Agent"] + [f"Bot-{i}" for i in range(1, self.seats)]

        if self.game_type == "sam":
            return SamGame(game_id, names)
        return TLMNGame(game_id, names)

    def _advance_until_agent_turn(self) -> None:
        guard = 0
        while (
            self.game is not None
            and not self.game.state.is_finished
            and self.game.state.current_player_id != self.agent_id
        ):
            self._play_opponent_turn()
            guard += 1
            if guard > self.seats * 20:
                break

    def _play_opponent_turn(self) -> None:
        player_id = self.game.state.current_player_id
        if player_id == self.agent_id:
            return

        legal_moves = self._build_legal_moves(player_id)
        selected_move = self._select_opponent_move(legal_moves, player_id)
        success = self._apply_move(player_id, selected_move)
        if not success:
            # Force pass to keep environment stable
            self.game.play_move(player_id, PlayerAction.PASS, [])

    def _apply_move(self, player_id: int, move: Dict[str, Any]) -> bool:
        if move.get("type") == "pass":
            return self.game.play_move(player_id, PlayerAction.PASS, [])

        cards_ids = move.get("cards", []) or []
        cards = CardEncoder.decode_hand(cards_ids)
        return self.game.play_move(player_id, PlayerAction.PLAY_CARDS, cards)

    def _build_game_record_for_player(self, player_id: int, hand: List[int]) -> Dict[str, Any]:
        """Build game record for a specific player (for self-play)."""
        if self.game is None:
            return {}
        
        # Calculate cards left for each player
        cards_left = []
        for p in self.game.state.players:
            cards_left.append(len(p.hand))
        
        # Build last_move info if exists
        last_move_info = None
        if self.game.state.last_move:
            lm = self.game.state.last_move
            if lm.cards:
                last_move_info = {
                    "player_id": lm.player_id,
                    "cards": [c.card_id for c in lm.cards],
                    "combo_type": lm.combo.combo_type.value if lm.combo else "pass",
                    "rank_value": lm.combo.rank_value if lm.combo else -1,
                }
        
        record = {
            "hand": hand,
            "cards_left": cards_left,
            "last_move": last_move_info,
            "current_player_id": self.game.state.current_player_id,
            "meta": {
                "legal_moves": []  # Will be filled by feature builder if needed
            }
        }
        return record

    def _build_observation(self) -> Dict[str, Any]:
        record = self.game.get_game_record()
        legal_moves = self._build_legal_moves(self.agent_id)
        framework = self.framework_generator.generate_framework(
            record.get("hand", []), game_type=self.game_type
        )
        feature_matrix = self.feature_builder.build_feature_matrix(
            record, legal_moves, framework
        )
        obs = {
            "move_features": feature_matrix,
            "legal_moves": legal_moves,
            "game_record": record,
            "framework": framework,
        }
        return obs

    def _terminal_observation(self) -> Dict[str, Any]:
        return {
            "move_features": [[0.0] * self.feature_builder.feature_dim],
            "legal_moves": [{"type": "pass", "cards": [], "combo_type": "pass", "rank_value": -1}],
            "game_record": {
                "is_finished": True,
                "winner_id": self.game.state.winner_id if self.game else None,
            },
            "framework": {},
        }

    def _build_legal_moves(self, player_id: int) -> List[Dict[str, Any]]:
        combos = self.game.get_legal_moves(player_id)
        moves: List[Dict[str, Any]] = []
        for combo in combos:
            if hasattr(combo, "cards"):
                cards = CardEncoder.encode_hand(list(combo.cards))
                combo_type = combo.combo_type.value
                rank_value = combo.rank_value
                secondary_rank = getattr(combo, "secondary_rank", None)
            else:
                cards = combo.get("cards", [])
                combo_type = combo.get("combo_type")
                rank_value = combo.get("rank_value")
                secondary_rank = combo.get("secondary_rank")

            moves.append(
                {
                    "type": "play_cards",
                    "cards": cards,
                    "combo_type": combo_type,
                    "rank_value": rank_value,
                    "secondary_rank": secondary_rank,
                }
            )
        if self.game.can_pass(player_id):
            moves.append({"type": "pass", "cards": [], "combo_type": "pass", "rank_value": -1})
        if player_id == self.agent_id:
            self._latest_legal_moves = moves
        return moves

    def _select_opponent_move(self, legal_moves: List[Dict[str, Any]], player_id: int) -> Dict[str, Any]:
        """
        Select move for opponent player.
        If self-play is enabled and policy_network is provided, use policy.
        Otherwise, use simple scripted strategy.
        """
        # Use policy network for self-play
        if self.use_self_play and self.policy_network is not None:
            try:
                # Get hand for this specific player
                player = self.game.state.get_player(player_id)
                if not player:
                    raise ValueError(f"Player {player_id} not found")
                
                hand = [card.card_id for card in player.hand]
                
                # Build game record for this player
                record = self._build_game_record_for_player(player_id, hand)
                
                # Generate framework for this player's hand
                framework = self.framework_generator.generate_framework(
                    hand, game_type=self.game_type
                )
                
                # Build feature matrix for all legal moves
                feature_matrix = self.feature_builder.build_feature_matrix(
                    record, legal_moves, framework
                )
                
                # Use policy to select move (deterministic: argmax)
                import torch
                tensor = torch.tensor(feature_matrix, dtype=torch.float32)
                with torch.no_grad():
                    logits = self.policy_network(tensor)
                    action_idx = torch.argmax(logits).item()
                    action_idx = max(0, min(action_idx, len(legal_moves) - 1))
                    return legal_moves[action_idx]
            except Exception as e:
                # Fallback to scripted if policy fails
                print(f"[CardGameEnv] Policy selection failed for player {player_id}: {e}")
        
        # Scripted opponent: choose smallest move
        playable = [m for m in legal_moves if m.get("type") == "play_cards"]
        if not playable:
            return {"type": "pass", "cards": [], "combo_type": "pass", "rank_value": -1}
        playable.sort(key=lambda m: (len(m.get("cards", [])), m.get("rank_value", -1)))
        return playable[0]

    def _final_reward(self) -> float:
        """
        Calculate final reward based on remaining cards:
        - If agent wins (0 cards left): reward = sum of remaining cards of other players
        - If agent loses (>0 cards left): 
          - Base penalty = -agent's remaining cards
          - Special penalties:
            * If still has all 10 cards: -15 points
            * If has card 2 (rank 12) or four_kind: -3 points per card (instead of -1)
        """
        if self.game is None or not self.game.state.is_finished:
            return 0.0
        
        agent_player = self.game.state.get_player(self.agent_id)
        if not agent_player:
            return 0.0
        
        agent_cards_left = len(agent_player.hand)
        
        # Agent wins (0 cards left)
        if agent_cards_left == 0:
            # Reward = sum of remaining cards of all other players
            total_opponent_cards = 0
            for player in self.game.state.players:
                if player.player_id != self.agent_id:
                    total_opponent_cards += len(player.hand)
            return float(total_opponent_cards)
        
        # Agent loses (>0 cards left)
        # Special penalty: if still has all 10 cards (initial hand size)
        initial_hand_size = 10 if self.game_type == "sam" else 13  # Sam: 10, TLMN: 13
        if agent_cards_left == initial_hand_size:
            return -15.0
        
        # Check for special cards: card 2 (rank 12) or four_kind
        has_card_2 = any(card.rank.value == 12 for card in agent_player.hand)
        has_four_kind = self._has_four_of_a_kind(agent_player.hand)
        
        # Calculate penalty
        if has_card_2 or has_four_kind:
            # Special penalty: -3 points per card (instead of -1)
            return -3.0 * float(agent_cards_left)
        
        # Normal penalty: -1 point per card
        return -float(agent_cards_left)
    
    def _has_four_of_a_kind(self, hand: List) -> bool:
        """Check if hand contains four of a kind (4 cards of same rank)"""
        if not hand or len(hand) < 4:
            return False
        
        rank_counts = {}
        for card in hand:
            # hand is list of Card objects
            rank = card.rank.value
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        return any(count >= 4 for count in rank_counts.values())

