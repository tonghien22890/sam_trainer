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
    from .policy import PolicyNetwork
except ImportError:
    from features import FrameworkAwareFeatureBuilder
    from policy import PolicyNetwork


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
        opponent_pool_dir: Optional[str] = None,
        opponent_pool: Optional[List[str]] = None,
        opponent_sampling_temperature: float = 1.0,
        feature_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        scripted_opponent_ratio: float = 0.0,
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
        self.policy_network = policy_network  # Policy network for self-play (fallback)
        self.use_self_play = use_self_play
        
        # Opponent pool for diverse self-play
        self.opponent_pool_dir = opponent_pool_dir
        self.opponent_pool: List[str] = opponent_pool or []
        self.opponent_sampling_temperature = opponent_sampling_temperature  # Default/fallback temperature
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.scripted_opponent_ratio = scripted_opponent_ratio  # Phase 3: Mixed Training support
        self._opponent_networks: Dict[str, Any] = {}  # Cache loaded opponent networks
        self._opponent_temperatures: Dict[str, float] = {}  # Cache opponent-specific temperatures
        self._seat_to_opponent: Dict[int, Optional[Tuple[str, float]]] = {}  # Map seat_id -> (checkpoint_path, temp) or None for scripted

        self._rng = random.Random(seed)

        self.game = None
        self.steps = 0
        self._latest_legal_moves: List[Dict[str, Any]] = []
        # Track cards for reward shaping
        self._cards_before: Optional[Tuple[int, List[int]]] = None  # (agent_cards, [opponent_cards...])
        # Track if last move broke combo (for winning bonus)
        self._last_move_broke_combo: bool = False
        # Snapshot before agent move (for chặt reward: only +5 when actually beating previous move)
        self._game_record_before_agent_move: Optional[Dict[str, Any]] = None

        # Card counting: track seen ranks (cards that have been played)
        # 13 ranks: 0=3, 1=4, ..., 11=A, 12=2
        # Each element = number of cards of that rank seen (0-4)
        self.seen_ranks: List[int] = [0] * 13

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng.seed(seed)

        self.game = self._create_game()
        self.steps = 0
        self._cards_before = None
        self._last_move_broke_combo = False
        self._game_record_before_agent_move = None

        # Reset card counting for new game
        self.seen_ranks = [0] * 13

        # Assign fixed opponents to seats for this game
        self._assign_opponents_to_seats()

        self._advance_until_agent_turn()
        # Track initial cards state for step reward
        self._cards_before = self._get_cards_state()
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
        self._last_selected_move = selected_move  # Store for potential chặt reward calculation
        # Snapshot game state before agent move so we can check if this move actually beats previous (chặt)
        self._game_record_before_agent_move = self.game.get_game_record()
        success = self._apply_move(self.agent_id, selected_move)
        self.steps += 1

        if not success:
            obs = self._build_observation()
            return obs, self.reward_cfg["invalid"], True, {"error": "apply_failed"}

        if self.game.state.is_finished:
            # Final reward with step reward shaping for last move
            cards_after = self._get_cards_state()
            step_r = self._step_reward(self._cards_before, cards_after) if self._cards_before else 0.0
            # Add chặt reward for last move (if game finished immediately, this is the move that finished it)
            chat_reward = self._calculate_chat_reward(selected_move)
            step_r += chat_reward
            # Add situation bonus for blocking opponent about to win
            if self._cards_before:
                step_r += self._calculate_situation_bonus(self._cards_before, selected_move)
            final_r = self._final_reward()
            # Add strategic failure penalty (hoarding strong cards while losing)
            strat_penalty = self._calculate_strategic_failure_penalty()
            reward = final_r + step_r + strat_penalty
            self._cards_before = None
            self._last_move_broke_combo = False
            return self._terminal_observation(), reward, True, {
                "winner_id": self.game.state.winner_id,
                "strat_penalty": strat_penalty
            }

        self._advance_until_agent_turn()
        done = self.game.state.is_finished or self.steps >= self.max_steps
        
        if done:
            # Final reward with step reward shaping if game finished
            if self.game.state.is_finished:
                cards_after = self._get_cards_state()
                step_r = self._step_reward(self._cards_before, cards_after) if self._cards_before else 0.0
                # Add chặt reward for TLMN (need to get the last move played by agent)
                if hasattr(self, '_last_selected_move'):
                    chat_reward = self._calculate_chat_reward(self._last_selected_move)
                    step_r += chat_reward
                # Add situation bonus for blocking opponent about to win
                if self._cards_before and hasattr(self, '_last_selected_move'):
                    step_r += self._calculate_situation_bonus(self._cards_before, self._last_selected_move)
                final_r = self._final_reward()
                # Add strategic failure penalty
                strat_penalty = self._calculate_strategic_failure_penalty()
                reward = final_r + step_r + strat_penalty
            else:
                reward = 0.0
            self._cards_before = None
            self._last_move_broke_combo = False
            return self._terminal_observation(), reward, True, {
                "winner_id": self.game.state.winner_id,
                "strat_penalty": strat_penalty if self.game.state.is_finished else 0.0
            }

        # Step reward for intermediate moves (includes chặt reward)
        cards_after = self._get_cards_state()
        step_r = self._step_reward(self._cards_before, cards_after) if self._cards_before else 0.0
        
        # Add chặt reward (3 đôi thông, 4 đôi thông, tứ quý) - immediate feedback
        chat_reward = self._calculate_chat_reward(selected_move)
        step_r += chat_reward
        
        # Add situation bonus for blocking opponent about to win
        if self._cards_before:
            step_r += self._calculate_situation_bonus(self._cards_before, selected_move)
        
        # Update cards_before for next step
        self._cards_before = cards_after
        
        obs = self._build_observation()
        return obs, step_r, False, {}

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
        success = self.game.play_move(player_id, PlayerAction.PLAY_CARDS, cards)
        
        # Update seen cards for card counting (tracks both agent and opponent moves)
        if success:
            self._update_seen_cards(move)
        
        return success
    
    def _update_seen_cards(self, move: Dict[str, Any]) -> None:
        """
        Update seen_ranks from a played move.
        Tracks cards that have been played (by any player).
        Does NOT track cards still in agent's hand.
        """
        if move.get("type") != "play_cards":
            return
        
        cards = move.get("cards", []) or []
        for card_id in cards:
            rank = card_id % 13
            self.seen_ranks[rank] += 1
            self.seen_ranks[rank] = min(4, self.seen_ranks[rank])  # Cap at 4 cards per rank

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

        # Tính min_opponent_cards để model học được khi nào nên giật cái
        cards_left = record.get("cards_left", []) or []
        current_player_id = record.get("current_player_id", self.agent_id)
        min_opp = None
        if cards_left and len(cards_left) > 1:
            opp_counts = [
                c for idx, c in enumerate(cards_left) 
                if idx != current_player_id and c > 0
            ]
            if opp_counts:
                min_opp = min(opp_counts)
        
        # Add seen_ranks to record for feature builder (card counting)
        record["seen_ranks"] = self.seen_ranks.copy()  # Copy to avoid mutation
        
        framework = self.framework_generator.generate_framework(
            record.get("hand", []), 
            game_type=self.game_type,
            min_opponent_cards=min_opp
        )
        feature_matrix = self.feature_builder.build_feature_matrix(
            record, legal_moves, framework
        )
        state_features = self.feature_builder.build_state_features(record, framework)
        obs = {
            "move_features": feature_matrix,
            "legal_moves": legal_moves,
            "game_record": record,
            "framework": framework,
            "state_features": state_features,
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
            "state_features": [0.0] * self.feature_builder.state_dim,
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
        If self-play is enabled, use policy from opponent pool or current policy.
        Each seat gets assigned a fixed opponent checkpoint at game start for consistency.
        Otherwise, use simple scripted strategy.
        """
        # Use policy network for self-play
        if self.use_self_play:
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
                
                # Get opponent network assigned to this seat (or random if not assigned)
                opponent_net, opponent_temp = self._get_opponent_network_for_seat(player_id)
                
                # Use policy to select move with sampling (stochastic for diversity)
                import torch
                from torch.distributions import Categorical
                
                tensor = torch.tensor(feature_matrix, dtype=torch.float32)
                with torch.no_grad():
                    logits = opponent_net(tensor)
                    
                    # Apply opponent-specific temperature for diversity
                    if opponent_temp != 1.0:
                        logits = logits / opponent_temp
                    
                    # Sample from distribution instead of argmax
                    dist = Categorical(logits=logits)
                    action_idx = dist.sample().item()
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
        initial_hand_size = 10 if self.game_type == "sam" else 13  # Sam: 10, TLMN: 13
        
        # Agent wins (0 cards left)
        if agent_cards_left == 0:
            # Reward = sum of remaining cards of all other players
            total_opponent_cards = 0
            for player in self.game.state.players:
                if player.player_id != self.agent_id:
                    total_opponent_cards += len(player.hand)
            # Normalize by initial_hand_size to bring into ~[0, 3.0] range
            return float(total_opponent_cards) / float(initial_hand_size)
        
        # Agent loses (>0 cards left)
        # Special penalty: if still has all 10 cards (initial hand size)
        if agent_cards_left == initial_hand_size:
            # Normalized: -15 / 10 = -1.5 (Sam), -15 / 13 ≈ -1.15 (TLMN)
            return -15.0 / float(initial_hand_size)
        
        # Check for special cards: card 2 (rank 12) or four_kind
        has_card_2 = any(card.rank.value == 12 for card in agent_player.hand)
        has_four_kind = self._has_four_of_a_kind(agent_player.hand)
        
        # Calculate penalty (normalized by initial_hand_size)
        if has_card_2 or has_four_kind:
            # Special penalty: -3 points per card, normalized
            return -3.0 * float(agent_cards_left) / float(initial_hand_size)
        
        # Normal penalty: -1 point per card, normalized
        return -float(agent_cards_left) / float(initial_hand_size)
    
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
    
    # ------------------------------------------------------------------ #
    # Reward shaping helpers
    # ------------------------------------------------------------------ #
    def _get_cards_state(self) -> Tuple[int, List[int]]:
        """Get current cards state: (agent_cards, [opponent_cards...])"""
        if self.game is None:
            return (0, [])
        
        agent_player = self.game.state.get_player(self.agent_id)
        agent_cards = len(agent_player.hand) if agent_player else 0
        
        opponent_cards = []
        for player in self.game.state.players:
            if player.player_id != self.agent_id:
                opponent_cards.append(len(player.hand))
        
        return (agent_cards, opponent_cards)
    
    def _phi(self, c_opp_list: List[int], c_me: int, k: float = 0.5) -> float:
        """
        Potential function: khuyến khích mình ít bài hơn đối thủ
        c_opp_list: list số bài của từng đối thủ
        c_me: số bài của agent
        k: scaling factor
        """
        if not c_opp_list:
            return 0.0
        # Dùng min để focus vào đối thủ nguy hiểm nhất (ít bài nhất)
        min_opp = min(c_opp_list)
        return k * (min_opp - c_me)
    
    def _step_reward(
        self,
        cards_before: Tuple[int, List[int]],
        cards_after: Tuple[int, List[int]],
        w1: float = 0.5,
        w2: float = 0.1,
        k: float = 0.1,
        gamma: float = 0.99,
    ) -> float:
        """
        Calculate step reward with potential-based shaping.
        
        Args:
            cards_before: (agent_cards, [opponent_cards...]) before moves
            cards_after: (agent_cards, [opponent_cards...]) after moves
            w1: weight for agent reducing cards
            w2: weight for opponent reducing cards (penalty when opponents play cards)
            k: potential function scaling
            gamma: discount factor
        """
        c_me_before, c_opp_list_before = cards_before
        c_me_after, c_opp_list_after = cards_after
        
        d_me = c_me_before - c_me_after  # >=0 nếu agent đánh
        # Tính tổng số bài đối thủ giảm (normalized by number of opponents)
        d_opp_total = sum(c_opp_list_before) - sum(c_opp_list_after)
        d_opp_avg = d_opp_total / max(1, len(c_opp_list_before))
        
        # Potential-based shaping
        shaping = gamma * self._phi(c_opp_list_after, c_me_after, k) - self._phi(c_opp_list_before, c_me_before, k)
        
        # Step reward (no time penalty)
        r_step = w1 * d_me - w2 * d_opp_avg + shaping
        
        # Clip để ổn định (mở rộng range để situation bonus/chặt reward có trọng lượng)
        return max(-3.0, min(3.0, r_step))
    
    def _calculate_situation_bonus(
        self, cards_before: Tuple[int, List[int]], selected_move: Dict[str, Any]
    ) -> float:
        """
        Calculate bonus for situation-based actions.
        
        Returns:
            +0.3 if blocking opponent who is about to win (< 3 cards)
        """
        if self.game is None:
            return 0.0
        
        _, c_opp_list_before = cards_before
        
        # Check if any opponent is about to win (< 3 cards)
        opponent_winning_threat = any(cards < 3 for cards in c_opp_list_before if cards > 0)
        
        if not opponent_winning_threat:
            return 0.0
        
        # Check if agent is blocking (can beat last_move)
        game_record = self.game.get_game_record()
        can_beat = self.feature_builder._can_beat_last_move(selected_move, game_record)
        
        if can_beat == 1.0:
            return 0.5  # Bonus for blocking opponent about to win
        
        return 0.0
    
    def _calculate_chat_reward(self, move: Dict[str, Any]) -> float:
        """
        Calculate reward for chặt (cutting) combos.
        Chặt reward chỉ khi chặt được 2 (Heo) = +15/initial_hand_size (chuẩn hóa);
        chặn lá thường không thưởng.
        
        Returns:
            Reward value (0.0 if not chặt or incorrect target)
        """
        if self.game is None:
            return 0.0
        
        # Check if this is a chặt move (combo type + beats last move)
        if not self._is_chat_move(move):
            return 0.0
            
        # Get last move info from snapshot taken before agent move
        record = getattr(self, "_game_record_before_agent_move", None)
        if not record:
            return 0.0
            
        last_move = record.get("last_move")
        if not last_move:
            return 0.0
            
        # ONLY Reward if the move beats a '2' (rank 12)
        # Chặt Heo = ăn 1 ván móm (+15 normalized)
        if last_move.get("rank_value") == 12:
            initial_hand_size = 10 if self.game_type == "sam" else 13
            return 15.0 / float(initial_hand_size)
        
        # Other chat moves (beating a smaller tứ quý with a larger one) get 0 reward
        # as they are just tactical blocks, not immediate "jackpot" rewards.
        return 0.0
    
    def _is_chat_move(self, move: Dict[str, Any]) -> bool:
        """
        Check if move is actually a chặt (cutting) move.
        Chặt = đánh đè nước trước: combo is a chặt type (tứ quý / 3–4 đôi thông)
        AND there was a previous move on table AND this move beats that move.
        """
        if self.game is None:
            return False

        combo_type = move.get("combo_type", "")

        # (a) Must be a chặt combo type
        is_chat_combo = False
        if combo_type == "four_kind":
            is_chat_combo = True
        elif self.game_type == "tlmn" and combo_type in (
            "three_consecutive_pairs",
            "four_consecutive_pairs",
        ):
            is_chat_combo = True
        if not is_chat_combo:
            return False

        # (b) Must have a previous move on table (snapshot taken before agent moved)
        record = getattr(self, "_game_record_before_agent_move", None)
        if not record or record.get("last_move") is None:
            return False

        # (c) This move must actually beat the previous move
        if self.feature_builder._can_beat_last_move(move, record) != 1.0:
            return False

        return True
    
    def _calculate_winning_bonus(
        self, cards_before: Tuple[int, List[int]], selected_move: Dict[str, Any]
    ) -> float:
        """
        Calculate bonus for breaking combo to win.
        
        Returns:
            +1.0 if agent broke combo and won the game
        """
        if self.game is None or not self.game.state.is_finished:
            return 0.0
        
        # Check if agent won
        if self.game.state.winner_id != self.agent_id:
            return 0.0
        
        # Check if agent broke combo (stored before applying move)
        if self._last_move_broke_combo:
            return 1.0  # Bonus for breaking combo to win
        
        return 0.0
    
    def _final_reward_with_shaping(self, cards_before: Optional[Tuple[int, List[int]]]) -> float:
        """
        Calculate final reward, optionally adding step reward if cards_before is provided.
        """
        final_r = self._final_reward()
        
        # If we have cards_before, add step reward for the final move
        if cards_before is not None:
            cards_after = self._get_cards_state()
            step_r = self._step_reward(cards_before, cards_after)
            return final_r + step_r
        
        return final_r
    
    def update_opponent_pool(self, opponent_pool: List[str]) -> None:
        """Update opponent pool list (called by trainer when pool changes)."""
        self.opponent_pool = opponent_pool
        # Clear cache when pool updates (new opponents may have different variations)
        self._opponent_networks.clear()
        self._opponent_temperatures.clear()
    
    def _assign_opponents_to_seats(self) -> None:
        """Assign opponent checkpoints to each opponent seat at game start for consistent diversity."""
        self._seat_to_opponent = {}
        
        if not self.use_self_play or not self.opponent_pool:
            return
        
        # Assign a checkpoint to each opponent seat
        for seat_id in range(1, self.seats):
            # Check if this seat should be scripted (Mixed Mode)
            if self._rng.random() < self.scripted_opponent_ratio:
                self._seat_to_opponent[seat_id] = None  # None indicates scripted
                continue

            if self.opponent_pool:
                # Weighted sampling: favor newer models but include old ones
                # idx_in_pool = 0 (oldest), idx_in_pool = len-1 (newest)
                # Using a simple power distribution biased towards the end
                idx = int(self._rng.triangular(0, len(self.opponent_pool) - 1, len(self.opponent_pool) - 1))
                checkpoint_path = self.opponent_pool[idx]
                
                # Get temperature for this checkpoint
                temp = self._opponent_temperatures.get(
                    checkpoint_path,
                    self.opponent_sampling_temperature
                )
                self._seat_to_opponent[seat_id] = (checkpoint_path, temp)
    
    def _get_opponent_network_for_seat(self, seat_id: int) -> Tuple[Any, float]:
        """
        Get opponent network for a specific seat.
        Each seat has a fixed checkpoint assigned at game start for consistency.
        Falls back to random selection if not assigned.
        
        Returns:
            Tuple of (network, temperature)
        """
        import torch
        
        # If pool is empty or disabled, fallback to current policy
        if not self.opponent_pool:
            if self.policy_network:
                print(f"[CardGameEnv] Warning: Opponent pool is empty, using current policy as fallback")
                return self.policy_network, self.opponent_sampling_temperature
            else:
                raise RuntimeError("No opponent pool and no policy network available")
        
        # Check if this seat has an assigned opponent
        if seat_id in self._seat_to_opponent:
            checkpoint_path, temp = self._seat_to_opponent[seat_id]
        else:
            # Fallback: randomly select (shouldn't happen if _assign_opponents_to_seats was called)
            checkpoint_path = self._rng.choice(self.opponent_pool)
            temp = self.opponent_sampling_temperature
        
        # Check cache first
        if checkpoint_path in self._opponent_networks:
            # Use temp from seat assignment if available, otherwise lookup
            if checkpoint_path not in self._opponent_temperatures:
                # Load temp from checkpoint metadata if not cached
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    temp = checkpoint.get("metadata", {}).get(
                        "opponent_temperature",
                        self.opponent_sampling_temperature
                    )
                    self._opponent_temperatures[checkpoint_path] = temp
                except:
                    temp = self.opponent_sampling_temperature
            return self._opponent_networks[checkpoint_path], temp
        
        # Load opponent network from checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Get network dimensions from checkpoint or use defaults
            hidden_dim = (
                checkpoint.get("metadata", {}).get("hidden_dim") 
                or self.hidden_dim 
                or 128
            )
            feature_dim = checkpoint.get("feature_dim") or self.feature_dim or self.feature_builder.feature_dim
            
            # Get opponent-specific temperature from metadata (if not already from seat assignment)
            if checkpoint_path not in self._opponent_temperatures:
                opponent_temp = checkpoint.get("metadata", {}).get(
                    "opponent_temperature",
                    self.opponent_sampling_temperature  # Fallback to default
                )
                self._opponent_temperatures[checkpoint_path] = opponent_temp
            else:
                opponent_temp = self._opponent_temperatures[checkpoint_path]
            
            # Create and load opponent network
            opponent_net = PolicyNetwork(
                input_dim=feature_dim,
                hidden_dim=hidden_dim
            )
            opponent_net.load_state_dict(checkpoint["model_state_dict"])
            opponent_net.eval()  # Set to eval mode
            
            # Cache network
            self._opponent_networks[checkpoint_path] = opponent_net
            
            return opponent_net, opponent_temp
        except Exception as e:
            print(f"[CardGameEnv] Failed to load opponent from {checkpoint_path}: {e}")
            # Fallback to current policy
            return self.policy_network, self.opponent_sampling_temperature

    def _calculate_strategic_failure_penalty(self) -> float:
        """
        Phạt nếu thua mà vẫn găm hàng (Heo hoặc bộ dây dài).
        Chỉ tính khi agent là người thua.
        """
        if self.game is None or not self.game.state.is_finished:
            return 0.0
        
        if self.game.state.winner_id == self.agent_id:
            return 0.0
        
        agent_player = self.game.state.get_player(self.agent_id)
        if not agent_player or not agent_player.hand:
            return 0.0
        
        penalty = 0.0
        hand_cards = agent_player.hand
        
        # 1. Găm Heo (2)
        has_two = any(card.rank.value == 12 for card in hand_cards)
        if has_two:
            penalty -= 1.0  # Phạt nặng vì không đánh Heo để giật cái
            
        # 2. Găm bộ dây dài (>5 quân)
        # (Sử dụng FrameworkGenerator để check cho nhanh)
        fw = self.framework_generator.generate_framework(
            [c.card_id for c in hand_cards], 
            game_type=self.game_type
        )
        for combo in fw.get("core_combos", []):
            if combo.get("combo_type") == "straight" and len(combo.get("cards", [])) >= 5:
                penalty -= 0.5
                break
                
        return penalty
