#!/usr/bin/env python3
"""
Framework Generator - Layer 1
Sử dụng SequenceEvaluator để tạo framework cho Style Learner
"""

import os
import sys
from typing import Dict, List, Any

# Setup paths properly
current_dir = os.path.dirname(os.path.abspath(__file__))
model_build_dir = os.path.dirname(os.path.dirname(current_dir))  # model_build/
project_root = os.path.dirname(model_build_dir)  # AI-Sam/

# Add to path
if model_build_dir not in sys.path:
    sys.path.insert(0, model_build_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import SequenceEvaluator (preferred method)
try:
    from ai_common.core.sequence_evaluator import SequenceEvaluator
except ImportError:
    print("⚠️ [FrameworkGenerator] SequenceEvaluator not available")
    SequenceEvaluator = None


class FrameworkGenerator:
    """
    Layer 1: Framework Generator
    Sử dụng SequenceEvaluator để tạo "khung bài" cho Style Learner
    """
    
    def __init__(self, enforce_full_coverage: bool = True):
        # If True, always include all leftover cards (including rank 12) so the
        # produced sequence covers the entire hand.
        self.enforce_full_coverage = enforce_full_coverage
        if SequenceEvaluator is not None:
            # Instantiate evaluator with the same policy
            self.sequence_evaluator = SequenceEvaluator(enforce_full_coverage=self.enforce_full_coverage)
        else:
            self.sequence_evaluator = None
            print("⚠️ [FrameworkGenerator] SequenceEvaluator not available - using fallback")
        
    def generate_framework(self, hand: List[int]) -> Dict[str, Any]:
        """
        Generate framework using SequenceEvaluator (preferred) or simple analysis
        
        Args:
            hand: List of card IDs (0-51)
            
        Returns:
            framework: Dict containing framework structure
        """
        try:
            # Try SequenceEvaluator first (preferred)
            if self.sequence_evaluator is not None:
                return self._generate_sequence_framework(hand)
            else:
                # Fallback to simple framework generation
                return self._generate_simple_framework(hand)
            
        except Exception as e:
            print(f"⚠️ [FrameworkGenerator] Error generating framework for hand {hand}: {e}")
            return self._get_empty_framework()
    
    def _get_empty_framework(self) -> Dict[str, Any]:
        """Return empty framework as fallback"""
        return {
            'unbeatable_sequence': [],
            'framework_strength': 0.0,
            'core_combos': [],
            'protected_ranks': [],
            'protected_windows': [],
            'recommended_moves': []
        }
    
    def _extract_core_combos(self, sequence_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract core combos từ unbeatable_sequence"""
        core_combos = []
        unbeatable_sequence = sequence_result.get('unbeatable_sequence', [])
        
        # Handle None case
        if unbeatable_sequence is None:
            return core_combos
        
        for combo_info in unbeatable_sequence:
            combo = combo_info.get('combo', {})
            if combo:
                core_combos.append({
                    'type': combo.get('type'),
                    'rank': combo.get('rank'),
                    'cards': combo.get('cards', []),
                    'strength': combo_info.get('strength', 0.0),
                    'position': combo_info.get('position', 0),
                    'rank_value': combo.get('rank_value', 0)
                })
        
        return core_combos
    
    def _extract_protected_ranks(self, sequence_result: Dict[str, Any]) -> List[int]:
        """Extract protected ranks từ core combos"""
        protected_ranks = []
        core_combos = self._extract_core_combos(sequence_result)
        
        for combo in core_combos:
            protected_ranks.extend(combo.get('cards', []))
        
        return list(set(protected_ranks))  # Remove duplicates
    
    def _extract_protected_windows(self, sequence_result: Dict[str, Any]) -> List[Dict[str, int]]:
        """Extract protected windows cho straights"""
        protected_windows = []
        core_combos = self._extract_core_combos(sequence_result)
        
        for combo in core_combos:
            if combo.get('type') in ['straight', 'double_seq']:
                cards = combo.get('cards', [])
                if len(cards) >= 3:  # Minimum straight length
                    ranks = [c % 13 for c in cards]  # Convert to ranks
                    ranks.sort()
                    
                    # Find consecutive windows
                    for i in range(len(ranks) - 2):
                        window_length = 1
                        for j in range(i + 1, len(ranks)):
                            if ranks[j] == ranks[j-1] + 1:
                                window_length += 1
                            else:
                                break
                        
                        if window_length >= 3:
                            protected_windows.append({
                                'start_rank': ranks[i],
                                'length': window_length,
                                'cards': cards[i:i+window_length]
                            })
        
        return protected_windows
    
    def _extract_recommended_moves(self, sequence_result: Dict[str, Any]) -> List[List[int]]:
        """Extract recommended moves từ core combos"""
        recommended_moves = []
        core_combos = self._extract_core_combos(sequence_result)
        
        # Sort by position để có thứ tự đánh
        core_combos.sort(key=lambda x: x.get('position', 0))
        
        for combo in core_combos:
            cards = combo.get('cards', [])
            if cards:
                recommended_moves.append(cards)
        
        return recommended_moves
    
    def _generate_simple_framework(self, hand: List[int]) -> Dict[str, Any]:
        """Generate simple framework using basic combo analysis"""
        if not hand:
            return self._get_empty_framework()
        
        # Analyze hand and find combos
        combos = self._analyze_hand_for_combos(hand)
        
        # Recalculate strengths using ComboAnalyzer logic
        for combo in combos:
            combo['strength'] = self._calculate_combo_strength(combo)
        
        # Sort combos by strength (strongest first) - keep all combos, don't filter
        sorted_combos = sorted(combos, key=lambda x: x['strength'], reverse=True)
        
        # Create framework structure
        framework = {
            'unbeatable_sequence': sorted_combos,
            'framework_strength': self._calculate_framework_strength(sorted_combos),
            'core_combos': sorted_combos,
            'protected_ranks': self._extract_protected_ranks_from_combos(sorted_combos),
            'protected_windows': self._extract_protected_windows_from_combos(sorted_combos),
            'recommended_moves': [combo.get('cards', []) for combo in sorted_combos if combo.get('cards')]
        }
        
        return framework
    
    def _generate_sequence_framework(self, hand: List[int]) -> Dict[str, Any]:
        """Generate framework using SequenceEvaluator"""
        if not hand:
            return self._get_empty_framework()
        
        # Get top sequences from SequenceEvaluator
        top_sequences = self.sequence_evaluator.evaluate_top_sequences(hand, k=3, enforce_full_coverage=self.enforce_full_coverage)
        
        if not top_sequences:
            return self._get_empty_framework()
        
        # Use the best sequence (first one)
        best_sequence = top_sequences[0]
        sequence_combos = best_sequence['sequence']

        # Ensure each combo has an explicit position index for downstream features
        for idx, combo in enumerate(sequence_combos):
            combo['position'] = idx
        
        # Phase-aware adjustment: cap strong strengths and delay positions in early phase
        hand_count = len(hand)
        phase = 'early' if hand_count >= 8 else ('mid' if hand_count >= 4 else 'late')
        adjusted_combos = []
        if phase == 'early':
            for combo in sequence_combos:
                c = dict(combo)
                raw_strength = float(c.get('strength', 0.0))
                combo_type = c.get('type', '')
                is_strong = combo_type in ['four_kind', 'triple', 'double_seq', 'straight'] or (combo_type == 'pair' and c.get('rank_value', -1) == 12)
                if is_strong:
                    c['strength'] = min(raw_strength, 0.8)
                    c['position'] = int(c.get('position', 0)) + 1
                adjusted_combos.append(c)
            sequence_combos = adjusted_combos

        # Create framework structure
        framework = {
            'unbeatable_sequence': sequence_combos,
            'framework_strength': (min(best_sequence['total_strength'], 0.9) if phase == 'early' else best_sequence['total_strength']),
            'core_combos': sequence_combos,
            'protected_ranks': self._extract_protected_ranks_from_combos(sequence_combos),
            'protected_windows': self._extract_protected_windows_from_combos(sequence_combos),
            'recommended_moves': [combo.get('cards', []) for combo in sequence_combos if combo.get('cards')]
        }
        
        # Add metadata from SequenceEvaluator
        framework['coverage_score'] = best_sequence['coverage_score']
        framework['end_rule_compliance'] = best_sequence['end_rule_compliance']
        framework['combo_count'] = best_sequence['combo_count']
        framework['avg_combo_strength'] = best_sequence['avg_combo_strength']
        
        # Add alternative sequences for reference
        framework['alternative_sequences'] = top_sequences[1:] if len(top_sequences) > 1 else []
        
        # Optional debug logging
        try:
            import os
            if os.environ.get('FRAMEWORK_DEBUG', '0') == '1':
                preview = [
                    {
                        'i': i,
                        't': c.get('type'),
                        'r': c.get('rank_value'),
                        'len': len(c.get('cards', [])),
                        's': round(c.get('strength', 0.0), 3),
                        'pos': c.get('position', None)
                    } for i, c in enumerate(sequence_combos[:10])
                ]
                print(f"[FRAMEWORK_DEBUG] combos={len(sequence_combos)} alt={len(top_sequences)-1} strength={framework.get('framework_strength', 0):.3f}")
                print(f"[FRAMEWORK_DEBUG] order={preview}")
        except Exception:
            pass

        return framework
    
    def _analyze_hand_for_combos(self, hand: List[int]) -> List[Dict[str, Any]]:
        """Analyze hand and find all possible combos"""
        combos = []
        
        # Group cards by rank
        rank_groups = {}
        for card in hand:
            rank = card % 13
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)
        
        # Find singles, pairs, triples, four_kinds
        for rank, cards in rank_groups.items():
            if len(cards) >= 1:
                combos.append({
                    'type': 'single',
                    'rank_value': rank,
                    'cards': [cards[0]],
                    'strength': self._get_rank_strength(rank),
                    'position': len(combos)
                })
            if len(cards) >= 2:
                combos.append({
                    'type': 'pair',
                    'rank_value': rank,
                    'cards': cards[:2],
                    'strength': self._get_rank_strength(rank) * 1.5,
                    'position': len(combos)
                })
            if len(cards) >= 3:
                combos.append({
                    'type': 'triple',
                    'rank_value': rank,
                    'cards': cards[:3],
                    'strength': self._get_rank_strength(rank) * 2.0,
                    'position': len(combos)
                })
            if len(cards) >= 4:
                combos.append({
                    'type': 'quad',  # Match ComboAnalyzer naming
                    'rank_value': rank,
                    'cards': cards[:4],
                    'strength': self._get_rank_strength(rank) * 3.0,  # Will be recalculated
                    'position': len(combos)
                })
        
        # Find straights (simplified)
        sorted_ranks = sorted(rank_groups.keys())
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i+4] - sorted_ranks[i] == 4:  # 5 consecutive ranks
                straight_cards = []
                for rank in sorted_ranks[i:i+5]:
                    straight_cards.append(rank_groups[rank][0])
                combos.append({
                    'type': 'straight',
                    'rank_value': sorted_ranks[i],
                    'cards': straight_cards,
                    'strength': self._get_rank_strength(sorted_ranks[i]) * 2.5,
                    'position': len(combos)
                })
        
        return combos
    
    def _get_rank_strength(self, rank: int) -> float:
        """Get strength value for rank in Sam system"""
        # Sam rank system: 0=3, 1=4, 2=5, ..., 10=K, 11=A, 12=2
        # Strength follows combo_analyzer.py logic
        if rank == 12:  # 2 - STRONGEST
            return 1.0
        elif rank == 11:  # A - special case, weak
            return 0.3
        else:
            # For ranks 0-10, strength scales with rank (higher rank = stronger)
            # Following combo_analyzer.py: 0.2 + (min(rank_value, 7) / 7.0) * 0.1
            return 0.1 + (min(rank, 7) / 7.0) * 0.1
    
    def _calculate_combo_strength(self, combo: Dict[str, Any]) -> float:
        """Calculate overall strength of a combo using ComboAnalyzer logic"""
        # Convert combo format to ComboAnalyzer format
        combo_analyzer_combo = {
            'combo_type': combo.get('type', 'single'),
            'rank_value': combo.get('rank_value', 0),
            'cards': combo.get('cards', [])
        }
        
        # Use ComboAnalyzer logic for accurate strength calculation
        from ai_common.core.combo_analyzer import ComboAnalyzer
        return ComboAnalyzer.calculate_combo_strength(combo_analyzer_combo)
    
    def _calculate_framework_strength(self, combos: List[Dict[str, Any]]) -> float:
        """Calculate overall framework strength"""
        if not combos:
            return 0.0
        
        total_strength = sum(self._calculate_combo_strength(combo) for combo in combos)
        max_possible_strength = len(combos) * 3.0  # Maximum strength per combo
        
        return min(total_strength / max_possible_strength, 1.0)
    
    def _extract_protected_ranks_from_combos(self, combos: List[Dict[str, Any]]) -> List[int]:
        """Extract protected ranks from combos"""
        protected_ranks = []
        for combo in combos:
            protected_ranks.extend(combo.get('cards', []))
        return list(set(protected_ranks))  # Remove duplicates
    
    def _extract_protected_windows_from_combos(self, combos: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        """Extract protected windows for straights"""
        protected_windows = []
        for combo in combos:
            if combo.get('type') in ['straight', 'double_seq']:
                cards = combo.get('cards', [])
                if len(cards) >= 3:
                    ranks = [c % 13 for c in cards]
                    ranks.sort()
                    
                    protected_windows.append({
                        'start_rank': ranks[0],
                        'length': len(ranks),
                        'cards': cards
                    })
        return protected_windows
    
    def get_framework_summary(self, framework: Dict[str, Any]) -> str:
        """Get human-readable framework summary"""
        core_combos = framework.get('core_combos', [])
        framework_strength = framework.get('framework_strength', 0.0)
        coverage_score = framework.get('coverage_score', 0.0)
        end_rule_compliance = framework.get('end_rule_compliance', True)
        combo_count = framework.get('combo_count', len(core_combos))
        avg_combo_strength = framework.get('avg_combo_strength', 0.0)
        
        summary = f"Framework Strength: {framework_strength:.3f}\n"
        summary += f"Coverage Score: {coverage_score:.3f}\n"
        summary += f"End Rule Compliance: {end_rule_compliance}\n"
        summary += f"Combo Count: {combo_count}\n"
        summary += f"Avg Combo Strength: {avg_combo_strength:.3f}\n"
        summary += f"Core Combos: {len(core_combos)}\n"
        
        for i, combo in enumerate(core_combos):
            combo_type = combo.get('type', 'unknown')
            rank_value = combo.get('rank_value', 0)
            strength = combo.get('strength', 0.0)
            cards = combo.get('cards', [])
            summary += f"  {i+1}. {combo_type} rank={rank_value} strength={strength:.3f} cards={cards}\n"
        
        # Show alternative sequences if available
        alt_sequences = framework.get('alternative_sequences', [])
        if alt_sequences:
            summary += f"\nAlternative Sequences: {len(alt_sequences)}\n"
            for i, alt_seq in enumerate(alt_sequences):
                alt_strength = alt_seq.get('total_strength', 0.0)
                alt_coverage = alt_seq.get('coverage_score', 0.0)
                summary += f"  Alt {i+1}: strength={alt_strength:.3f}, coverage={alt_coverage:.3f}\n"
        
        return summary
