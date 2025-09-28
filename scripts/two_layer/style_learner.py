#!/usr/bin/env python3
"""
Style Learner - Layer 2
Học style đánh dựa trên framework từ Layer 1
Thay thế OPTIMIZED_GENERAL_MODEL_SOLUTION.md
"""

import numpy as np
import joblib
from typing import Dict, List, Any
import xgboost as xgb


class StyleLearner:
    """
    Layer 2: Style Learner
    Học cách đánh theo framework từ Layer 1
    
    Features: 51 dims total (27 original + 9 framework + 15 multi-sequence with HEAVY SCALING)
    - Original: legal_moves_combo_counts, cards_left, hand_count, combo features
    - Framework: framework_alignment, framework_priority, framework_breaking_severity, etc.
    - Multi-sequence: top 3 sequences x 5 features each
    """
    
    def __init__(self):
        self.model = None
        self.combo_types = ["single", "pair", "triple", "four_kind", "straight", "double_seq", "pass"]
        self.combo_type_to_id = {ct: i for i, ct in enumerate(self.combo_types)}
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> List[str]:
        """Get feature names for debugging"""
        names = []
        
        # Original features (23 dims)
        names.extend(['single_count', 'pair_count', 'triple_count', 'four_kind_count', 'straight_count', 'double_seq_count'])
        names.extend(['cards_left_0', 'cards_left_1', 'cards_left_2', 'cards_left_3'])
        names.append('hand_count')
        names.extend(['single', 'pair', 'triple', 'four_kind', 'straight', 'double_seq', 'pass'])
        names.extend(['hybrid_rank', 'combo_length', 'breaks_combo_flag', 'individual_move_strength', 
                     'enhanced_breaks_penalty'])
        
        # Framework features (9 dims) - HEAVY SCALED
        names.extend(['framework_alignment_x15', 'framework_priority_x15', 'framework_breaking_severity_x30', 
                     'framework_strength_x8', 'framework_position_x10', 'combo_type_preference_x5', 
                     'rank_preference_x5', 'timing_preference_x3', 'sequence_compliance_x12'])
        
        # Multi-sequence features (15 dims) - 3 sequences x 5 features each
        for i in range(3):
            names.extend([f'seq{i+1}_alignment_x15', f'seq{i+1}_priority_x15', f'seq{i+1}_breaking_x30', 
                         f'seq{i+1}_position_x10', f'seq{i+1}_compliance_x12'])
        
        return names
    
    def extract_original_features(self, move: Dict[str, Any], game_record: Dict[str, Any]) -> List[float]:
        """Extract original 25-dim features (từ OPTIMIZED_GENERAL_MODEL_SOLUTION.md)"""
        features = []
        
        # 1. Legal moves combo counts (6 dims)
        legal_moves = game_record.get('meta', {}).get('legal_moves', [])
        combo_counts = [0] * 6
        for m in legal_moves:
            combo_type = m.get('combo_type', 'pass')
            if combo_type in self.combo_type_to_id and combo_type != 'pass':
                idx = self.combo_type_to_id[combo_type]
                if idx < 6:
                    combo_counts[idx] += 1
        features.extend(combo_counts)
        
        # 2. Cards left (4 dims)
        cards_left = game_record.get('cards_left', [0, 0, 0, 0])
        features.extend(cards_left[:4])  # Ensure exactly 4 values
        
        # 3. Hand count (1 dim)
        hand = game_record.get('hand', [])
        features.append(len(hand))
        
        # 4. Combo type onehot (7 dims)
        combo_type = move.get('combo_type', 'pass')
        onehot = [0.0] * 7
        if combo_type in self.combo_type_to_id:
            onehot[self.combo_type_to_id[combo_type]] = 1.0
        features.extend(onehot)
        
        # 5. Hybrid rank feature (1 dim) - Sam rank system: 0=3, 1=4, ..., 11=A, 12=2
        rank_value = move.get('rank_value', 0)
        if combo_type == 'single':
            if rank_value == 12:  # 2 - strongest
                features.append(3.0)
            elif rank_value == 11:  # A - strong
                features.append(2.0)
            elif rank_value >= 8:  # J, Q, K - medium
                features.append(1.5)
            elif rank_value >= 4:  # 7, 8, 9, 10 - decent
                features.append(1.0)
            else:  # 3, 4, 5, 6 - weak
                features.append(0.5)
        elif combo_type == 'pair':
            if rank_value == 12:  # 2 - strongest
                features.append(4.0)
            elif rank_value == 11:  # A - strong
                features.append(3.0)
            elif rank_value >= 8:  # J, Q, K - medium
                features.append(2.0)
            elif rank_value >= 4:  # 7, 8, 9, 10 - decent
                features.append(1.5)
            else:  # 3, 4, 5, 6 - weak
                features.append(1.0)
        elif combo_type == 'triple':
            if rank_value == 12:  # 2 - strongest
                features.append(5.0)
            elif rank_value == 11:  # A - strong
                features.append(4.0)
            elif rank_value >= 8:  # J, Q, K - medium
                features.append(3.0)
            elif rank_value >= 4:  # 7, 8, 9, 10 - decent
                features.append(2.0)
            else:  # 3, 4, 5, 6 - weak
                features.append(1.5)
        elif combo_type == 'four_kind':
            if rank_value == 12:  # 2 - strongest
                features.append(6.0)
            elif rank_value == 11:  # A - strong
                features.append(5.0)
            elif rank_value >= 8:  # J, Q, K - medium
                features.append(4.0)
            elif rank_value >= 4:  # 7, 8, 9, 10 - decent
                features.append(3.0)
            else:  # 3, 4, 5, 6 - weak
                features.append(2.0)
        elif combo_type == 'straight':
            cards = move.get('cards', [])
            # For straights, use length as base strength, then adjust by starting rank
            base_strength = len(cards)
            
            # Special case: A-2-3-4-5 straight (absolute power)
            if rank_value == 11:  # A start - absolute power
                features.append(10.0)  # Maximum strength for A straight
            elif rank_value == 12:  # 2 start - very strong
                features.append(8.0)
            elif rank_value >= 8:  # J, Q, K start - strong
                features.append(base_strength + 2.0)
            elif rank_value >= 4:  # 7, 8, 9, 10 start - decent
                features.append(base_strength + 1.0)
            else:  # 3, 4, 5, 6 start - normal
                features.append(base_strength)
        elif combo_type == 'double_seq':
            # Double_seq (đôi thông) is always absolute power in Sam
            features.append(12.0)  # Maximum strength for any double_seq
        else:
            features.append(0.0)
        
        # 6. Combo length (1 dim)
        cards = move.get('cards', [])
        features.append(len(cards))
        
        # 7. Breaks combo flag (1 dim)
        breaks_severity = self._calculate_breaks_combo_flag(game_record.get('hand', []), cards)
        features.append(breaks_severity)
        
        # 8. Individual move strength (1 dim)
        individual_strength = self._calculate_individual_move_strength(move)
        features.append(individual_strength)
        
        # 9. Combo type strength multiplier (1 dim)
        type_multipliers = {
            'single': 1.0, 'pair': 2.0, 'triple': 3.0, 'four_kind': 4.0,
            'straight': 2.5, 'double_seq': 3.5, 'pass': 0.0
        }
        features.append(type_multipliers.get(combo_type, 0.0))
        
        # 10. Enhanced breaks penalty (1 dim)
        enhanced_penalty = 0.0
        if breaks_severity == 2:
            enhanced_penalty = 0.7
        elif breaks_severity == 1:
            enhanced_penalty = 0.3
        features.append(enhanced_penalty)
        
        # 11. Combo efficiency score (1 dim)
        efficiency_scores = {
            'single': 0.2, 'pair': 0.4, 'triple': 0.6, 'four_kind': 0.8,
            'straight': 0.5, 'double_seq': 0.7, 'pass': 0.0
        }
        features.append(efficiency_scores.get(combo_type, 0.0))
        
        # 12-13. Removed placeholder bonuses to avoid noise and importance misreporting
        
        return features
    
    def extract_framework_features(self, move: Dict[str, Any], framework: Dict[str, Any]) -> List[float]:
        """Extract framework-aware features (9 dims) with HEAVY SCALING to override data bias"""
        features = []
        
        # Allow runtime control over scales via environment variables
        import os as _os
        _S_ALIGN = float(_os.environ.get('STYLE_SCALE_ALIGN', '15'))
        _S_PRIORITY = float(_os.environ.get('STYLE_SCALE_PRIORITY', '15'))
        _S_BREAK = float(_os.environ.get('STYLE_SCALE_BREAK', '26'))
        _S_STRENGTH = float(_os.environ.get('STYLE_SCALE_STRENGTH', '8'))
        _S_POSITION = float(_os.environ.get('STYLE_SCALE_POSITION', '12'))
        _S_TYPE = float(_os.environ.get('STYLE_SCALE_TYPE', '3'))
        _S_RANK = float(_os.environ.get('STYLE_SCALE_RANK', '4'))
        _S_TIMING = float(_os.environ.get('STYLE_SCALE_TIMING', '3'))
        _S_COMPLIANCE = float(_os.environ.get('STYLE_SCALE_COMPLIANCE', '16'))

        # 1. Framework alignment (1 dim)
        features.append(self._is_in_framework(move, framework) * _S_ALIGN)
        
        # 2. Framework priority (1 dim)
        features.append(self._framework_priority_score(move, framework) * _S_PRIORITY)
        
        # 3. Framework breaking severity (1 dim)
        features.append(-self._framework_breaking_severity(move, framework) * _S_BREAK)
        
        # 4. Framework strength (1 dim)
        features.append(framework.get('framework_strength', 0.0) * _S_STRENGTH)
        
        # 5. Framework position (1 dim)
        features.append(self._framework_position(move, framework) * _S_POSITION)
        
        # 6. Combo type preference (1 dim)
        features.append(self._combo_type_preference(move, framework) * _S_TYPE)
        
        # 7. Rank preference (1 dim)
        features.append(self._rank_preference(move, framework) * _S_RANK)
        
        # 8. Timing preference (1 dim)
        features.append(self._timing_preference(move, framework) * _S_TIMING)
        
        # 9. Sequence compliance (1 dim)
        features.append(self._sequence_compliance(move, framework) * _S_COMPLIANCE)
        
        return features
    
    def extract_multi_sequence_features(self, move: Dict[str, Any], framework: Dict[str, Any]) -> List[float]:
        """Extract features considering top 3 sequences (15 dims total) - FIXED LENGTH"""
        features = []
        
        # Get all sequences (best + alternatives)
        all_sequences = [framework.get('core_combos', [])]
        alt_sequences = framework.get('alternative_sequences', [])
        for alt_seq in alt_sequences:
            all_sequences.append(alt_seq.get('sequence', []))
        
        # Always extract exactly 3 sequences x 5 features = 15 dims
        for i in range(3):  # Always 3 sequences
            if i < len(all_sequences):
                sequence = all_sequences[i]
                seq_framework = {
                    'core_combos': sequence,
                    'framework_strength': framework.get('framework_strength', 0.0) if i == 0 else alt_sequences[i-1].get('total_strength', 0.0),
                    'recommended_moves': [combo.get('cards', []) for combo in sequence if combo.get('cards')]
                }
            else:
                # Empty sequence for padding
                seq_framework = {
                    'core_combos': [],
                    'framework_strength': 0.0,
                    'recommended_moves': []
                }
            
            # 5 features per sequence (always 5)
            features.append(self._is_in_framework(move, seq_framework) * 15.0)  # alignment
            features.append(self._framework_priority_score(move, seq_framework) * 15.0)  # priority
            features.append(-self._framework_breaking_severity(move, seq_framework) * 26.0)  # breaking penalty
            features.append(self._framework_position(move, seq_framework) * 10.0)  # position
            features.append(self._sequence_compliance(move, seq_framework) * 12.0)  # compliance
        
        return features
    
    def _is_in_framework(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """Check if move aligns với framework (0/1)"""
        move_cards = set(move.get('cards', []))
        core_combos = framework.get('core_combos', [])
        
        for combo in core_combos:
            combo_cards = set(combo.get('cards', []))
            if move_cards.issubset(combo_cards):
                return 1.0
        return 0.0
    
    def _framework_priority_score(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """Score dựa trên priority trong framework (0-1)"""
        move_cards = set(move.get('cards', []))
        core_combos = framework.get('core_combos', [])
        
        for combo in core_combos:
            combo_cards = set(combo.get('cards', []))
            if move_cards.issubset(combo_cards):
                return combo.get('strength', 0.0)
        return 0.0
    
    def _framework_breaking_severity(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """Severity of breaking framework (0-2)"""
        move_cards = set(move.get('cards', []))
        core_combos = framework.get('core_combos', [])
        
        max_severity = 0.0
        for combo in core_combos:
            combo_cards = set(combo.get('cards', []))
            combo_type = combo.get('type', '')
            
            # Check if move breaks this combo
            if move_cards.intersection(combo_cards) and not move_cards.issubset(combo_cards):
                if combo_type in ['four_kind', 'double_seq']:
                    max_severity = max(max_severity, 2.0)  # Heavy break
                elif combo_type in ['triple', 'straight']:
                    max_severity = max(max_severity, 1.0)  # Normal break
                else:
                    max_severity = max(max_severity, 0.5)  # Light break
        
        return max_severity
    
    def _sequence_compliance(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """Sequence compliance - move này có theo đúng sequence order không (0-1)"""
        move_cards = set(move.get('cards', []))
        recommended_moves = framework.get('recommended_moves', [])
        
        if not recommended_moves:
            return 0.0
        
        # Check if move matches any recommended move
        for i, rec_move in enumerate(recommended_moves):
            if set(rec_move) == move_cards:
                # Return score based on position: earlier moves get higher scores
                return 1.0 - (i / max(1, len(recommended_moves) - 1))
        
        # If move is not in recommended sequence, check if it's a subset of any combo
        for i, rec_move in enumerate(recommended_moves):
            if move_cards.issubset(set(rec_move)):
                # Partial match - give partial score
                return (1.0 - (i / max(1, len(recommended_moves) - 1))) * 0.5
        
        return 0.0  # No match with recommended sequence
    
    def _framework_position(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """Position trong framework sequence (0-1) - càng sớm càng tốt"""
        move_cards = set(move.get('cards', []))
        core_combos = framework.get('core_combos', [])
        
        # Sort core_combos by position để có thứ tự đúng
        sorted_combos = sorted(core_combos, key=lambda x: x.get('position', 0))
        
        for i, combo in enumerate(sorted_combos):
            combo_cards = set(combo.get('cards', []))
            if move_cards.issubset(combo_cards):
                # Return inverse position: 1.0 for first combo, 0.0 for last combo
                return 1.0 - (i / max(1, len(sorted_combos) - 1))
        return 0.0  # Nếu không trong framework
    
    def _combo_type_preference(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """Combo type preference trong framework (0-1)"""
        move_type = move.get('combo_type', 'pass')
        core_combos = framework.get('core_combos', [])
        
        type_counts = {}
        for combo in core_combos:
            combo_type = combo.get('type', '')
            type_counts[combo_type] = type_counts.get(combo_type, 0) + 1
        
        if not type_counts:
            return 0.0
        
        # Calculate preference based on frequency in framework
        total_combos = len(core_combos)
        move_type_count = type_counts.get(move_type, 0)
        return move_type_count / total_combos
    
    def _rank_preference(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """Rank preference trong framework (0-1)"""
        move_rank = move.get('rank_value', 0)
        core_combos = framework.get('core_combos', [])
        
        rank_counts = {}
        for combo in core_combos:
            rank_value = combo.get('rank_value', 0)
            rank_counts[rank_value] = rank_counts.get(rank_value, 0) + 1
        
        if not rank_counts:
            return 0.0
        
        # Calculate preference based on rank frequency in framework
        total_combos = len(core_combos)
        move_rank_count = rank_counts.get(move_rank, 0)
        return move_rank_count / total_combos
    
    def _timing_preference(self, move: Dict[str, Any], framework: Dict[str, Any]) -> float:
        """Timing preference dựa trên game state (0-1)"""
        # This can be enhanced based on game state analysis
        # For now, return a simple heuristic
        return 0.5  # Placeholder
    
    def _calculate_breaks_combo_flag(self, hand: List[int], move_cards: List[int]) -> float:
        """Calculate breaks combo flag severity (0/1/2)"""
        # Simplified implementation - can be enhanced
        if not move_cards:
            return 0.0
        
        # Check if move breaks any potential combos in hand
        remaining_cards = [c for c in hand if c not in move_cards]
        
        # Simple heuristic: if removing cards reduces potential combo strength
        if len(move_cards) >= 3:  # Potentially breaking a strong combo
            return 2.0
        elif len(move_cards) >= 2:  # Potentially breaking a medium combo
            return 1.0
        else:
            return 0.0
    
    def _calculate_individual_move_strength(self, move: Dict[str, Any]) -> float:
        """Calculate individual move strength (0-1)"""
        combo_type = move.get('combo_type', 'pass')
        rank_value = move.get('rank_value', 0)
        
        # Simplified strength calculation
        base_strengths = {
            'single': 0.1, 'pair': 0.3, 'triple': 0.5, 'four_kind': 0.8,
            'straight': 0.4, 'double_seq': 0.7, 'pass': 0.0
        }
        
        base_strength = base_strengths.get(combo_type, 0.0)
        
        # Adjust by rank value
        if combo_type in ['single', 'pair', 'triple', 'four_kind']:
            if rank_value == 1:  # 2 - strongest
                rank_multiplier = 1.0
            elif rank_value == 0:  # A - strong
                rank_multiplier = 0.8
            elif rank_value >= 10:  # J, Q, K - medium
                rank_multiplier = 0.6
            else:
                rank_multiplier = 0.4
        else:
            rank_multiplier = 1.0
        
        return base_strength * rank_multiplier
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the style learner model"""
        print("🎯 [StyleLearner] Training model...")
        
        X = []
        y = []
        sample_weights = []
        # Optional debug logging controls (non-intrusive)
        import os as _os
        _log_train = _os.environ.get('STYLE_LOG_TRAIN', '0') == '1'
        _train_logs = []
        
        for record in training_data:
            hand = record.get('hand', [])
            
            # Use framework từ training data (đã được generate bởi FrameworkGenerator)
            framework = record.get('framework', {
                'unbeatable_sequence': [],
                'framework_strength': 0.0,
                'core_combos': [],
                'protected_ranks': [],
                'protected_windows': [],
                'recommended_moves': []
            })
            
            legal_moves = record.get('meta', {}).get('legal_moves', [])
            chosen_move = record.get('action', {}).get('stage2', {})
            # Skip if no legal moves or malformed record
            if not legal_moves or not isinstance(legal_moves, list):
                continue
            
            for move in legal_moves:
                # Extract features
                original_features = self.extract_original_features(move, record)
                framework_features = self.extract_framework_features(move, framework)
                multi_sequence_features = self.extract_multi_sequence_features(move, framework)
                combined_features = original_features + framework_features + multi_sequence_features
                
                X.append(combined_features)
                
                # Label: 1 if this move was chosen, 0 otherwise
                is_chosen = self._moves_equal(move, chosen_move)
                y.append(1 if is_chosen else 0)

                # Sample weighting: boost planned moves to bias learning toward sequence plan
                # Exact-plan match: move cards equal to any recommended move
                move_cards = set(move.get('cards', []))
                # Compliance-based weighting (data-driven):
                # - For positives: boost proportional to sequence compliance (exact match highest)
                # - For negatives: downweight proportional to compliance (so we don't punish planned steps)
                compliance = self._sequence_compliance(move, framework)
                breaking = self._framework_breaking_severity(move, framework)
                # Penalize negatives that break hard; boost positives that follow plan
                if is_chosen:
                    weight = 1.0 + 12.0 * compliance
                else:
                    weight = max(0.05, 1.0 - 0.9 * compliance - 0.5 * breaking)
                sample_weights.append(weight)
                if _log_train:
                    _phase = self._infer_game_phase(len(hand))
                    _train_logs.append({
                        'phase': _phase,
                        'hand_count': len(hand),
                        'chosen': bool(is_chosen),
                        'combo_type': move.get('combo_type'),
                        'len': len(move.get('cards', [])),
                        'rank': move.get('rank_value'),
                        'compliance': float(compliance),
                        'position': float(self._framework_position(move, framework)),
                        'breaking': float(breaking),
                        'weight': float(weight),
                    })
        
        # Guard: no samples
        if not X:
            print("⚠️ [StyleLearner] No training samples found. Check dataset format and legal_moves.")
            return {'accuracy': 0.0}

        X = np.array(X)
        y = np.array(y)
        
        print(f"🎯 [StyleLearner] Training data: {X.shape[0]} samples, {X.shape[1]} features (25 original + 9 framework + 15 multi-sequence with HEAVY SCALING)")
        print(f"🎯 [StyleLearner] Positive rate: {np.mean(y):.3f}")
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss'
        )
        
        try:
            self.model.fit(X, y, sample_weight=np.array(sample_weights))
        except TypeError:
            # Fallback if the model signature doesn't support sample_weight (shouldn't happen with xgboost)
            self.model.fit(X, y)
        
        # Calculate accuracy
        y_pred = self.model.predict(X)
        accuracy = np.mean(y == y_pred)
        
        try:
            import os
            if os.environ.get('STYLE_DEBUG', '0') == '1':
                print(f"🎯 [StyleLearner] Training accuracy: {accuracy:.3f}")
                if _log_train and _train_logs:
                    # Summarize by phase to diagnose early aggression
                    _by_phase = {'early': [], 'mid': [], 'late': []}
                    for row in _train_logs:
                        _by_phase[row['phase']].append(row)
                    def _summ(rows):
                        if not rows:
                            return {'n': 0}
                        import numpy as _np
                        return {
                            'n': len(rows),
                            'chosen_rate': float(_np.mean([1 if r['chosen'] else 0 for r in rows])),
                            'avg_position': float(_np.mean([r['position'] for r in rows])),
                            'avg_compliance': float(_np.mean([r['compliance'] for r in rows])),
                            'avg_breaking': float(_np.mean([r['breaking'] for r in rows])),
                        }
                    print({'train_phase_summary': {k: _summ(v) for k, v in _by_phase.items()}})
        except Exception:
            pass
        
        return {'accuracy': accuracy}
    
    def predict_with_framework(self, game_record: Dict[str, Any], legal_moves: List[Dict[str, Any]], 
                              framework: Dict[str, Any]) -> Dict[str, Any]:
        """Predict best move với framework guidance và context-aware penalties"""
        if self.model is None:
            print("⚠️ [StyleLearner] Model not trained, using fallback")
            return legal_moves[0] if legal_moves else {"type": "pass", "cards": []}
        
        # Defensive: ensure moves are legal for this hand
        hand_set = set(game_record.get('hand', []))
        legal_moves = [m for m in legal_moves if set(m.get('cards', [])) <= hand_set] or [{"type": "pass", "cards": [], "combo_type": "pass", "rank_value": -1}]

        # Extract features cho từng legal move
        features_list = []
        for move in legal_moves:
            original_features = self.extract_original_features(move, game_record)
            framework_features = self.extract_framework_features(move, framework)  # Already heavily scaled
            multi_sequence_features = self.extract_multi_sequence_features(move, framework)
            combined_features = original_features + framework_features + multi_sequence_features
            features_list.append(combined_features)
        
        if not features_list:
            return {"type": "pass", "cards": []}
        
        # Predict với heavily scaled features (thuần data-driven)
        X = np.array(features_list)
        base_scores = self.model.predict_proba(X)[:, 1]

        # Tie-break: nudge toward exact plan compliance and earlier framework position; punish breaking
        adjusted_scores = []
        _debug_rows = []
        import os as _os
        _NO_TIEBREAK = _os.environ.get('STYLE_DISABLE_TIEBREAK', '0') == '1'
        for i, move in enumerate(legal_moves):
            compliance = self._sequence_compliance(move, framework)
            position = self._framework_position(move, framework)
            breaking = self._framework_breaking_severity(move, framework)
            bonus = 0.0 if _NO_TIEBREAK else (0.02 * compliance + 0.01 * position - 0.02 * breaking)
            adjusted_scores.append(float(base_scores[i]) + bonus)
            _debug_rows.append({
                'i': i,
                'type': move.get('combo_type'),
                'len': len(move.get('cards', [])),
                'rank': move.get('rank_value'),
                'base': float(base_scores[i]),
                'position': float(position),
                'compliance': float(compliance),
                'breaking': float(breaking),
                'bonus': float(bonus),
            })
        final_scores = np.array(adjusted_scores)

        # Optional per-candidate debug
        try:
            import os
            if os.environ.get('STYLE_DEBUG', '0') == '1':
                rows = []
                for i, move in enumerate(legal_moves):
                    rows.append({
                        'i': i,
                        'type': move.get('combo_type'),
                        'len': len(move.get('cards', [])),
                        'rank': move.get('rank_value'),
                        'base': round(float(base_scores[i]), 4),
                        'final': round(float(final_scores[i]), 4),
                    })
                print(f"[STYLE_DEBUG] moves={rows}")
                if os.environ.get('STYLE_LOG_PREDICT', '0') == '1':
                    _phase = self._infer_game_phase(len(game_record.get('hand', [])))
                    print({'phase': _phase, 'hand_count': len(game_record.get('hand', [])), 'rows': _debug_rows})
        except Exception:
            pass
        
        # Choose best move
        best_idx = np.argmax(final_scores)
        best_move = legal_moves[best_idx]
        best_score = float(final_scores[best_idx])
        
        try:
            import os
            if os.environ.get('STYLE_DEBUG', '0') == '1':
                print(f"🎯 [StyleLearner] Best move: {best_move}")
                print(f"🎯 [StyleLearner] Score: {best_score:.4f}")
        except Exception:
            pass
        
        return best_move

    def _infer_game_phase(self, hand_count: int) -> str:
        """Heuristic phase inference based on remaining hand size."""
        if hand_count >= 8:
            return 'early'
        if hand_count >= 4:
            return 'mid'
        return 'late'
    
    def _moves_equal(self, move1: Dict[str, Any], move2: Dict[str, Any]) -> bool:
        """Check if two moves are equal"""
        if not move1 or not move2:
            return False
        
        return (move1.get('type') == move2.get('type') and
                set(move1.get('cards', [])) == set(move2.get('cards', [])) and
                move1.get('combo_type') == move2.get('combo_type') and
                move1.get('rank_value') == move2.get('rank_value'))
    
    def save(self, model_path: str):
        """Save trained model"""
        if self.model is not None:
            joblib.dump(self.model, model_path)
            print(f"🎯 [StyleLearner] Model saved to {model_path}")
    
    def load(self, model_path: str):
        """Load trained model"""
        self.model = joblib.load(model_path)
        print(f"🎯 [StyleLearner] Model loaded from {model_path}")
