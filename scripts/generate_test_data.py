"""
Generate test data phù hợp với optimized general model
"""

import json
import random
from typing import List, Dict, Any


def generate_hand(num_cards: int = 13) -> List[int]:
    """Generate random hand"""
    all_cards = list(range(52))  # 0-51
    return random.sample(all_cards, num_cards)


def generate_legal_moves(hand: List[int]) -> List[Dict[str, Any]]:
    """Generate legal moves from hand (simplified)"""
    legal_moves = []
    
    # Count ranks
    rank_counts = {}
    for card_id in hand:
        rank = card_id % 13
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    # Generate single moves
    for rank, count in rank_counts.items():
        if count >= 1:
            cards = [card_id for card_id in hand if card_id % 13 == rank][:1]
            legal_moves.append({
                "type": "play_cards",
                "cards": cards,
                "combo_type": "single",
                "rank_value": rank
            })
    
    # Generate pair moves
    for rank, count in rank_counts.items():
        if count >= 2:
            cards = [card_id for card_id in hand if card_id % 13 == rank][:2]
            legal_moves.append({
                "type": "play_cards",
                "cards": cards,
                "combo_type": "pair",
                "rank_value": rank
            })
    
    # Generate triple moves
    for rank, count in rank_counts.items():
        if count >= 3:
            cards = [card_id for card_id in hand if card_id % 13 == rank][:3]
            legal_moves.append({
                "type": "play_cards",
                "cards": cards,
                "combo_type": "triple",
                "rank_value": rank
            })
    
    # Always add pass move
    legal_moves.append({
        "type": "pass",
        "cards": [],
        "combo_type": "pass",
        "rank_value": -1
    })
    
    return legal_moves


def generate_cards_left() -> List[int]:
    """Generate cards left for each player"""
    total_remaining = random.randint(20, 40)  # 20-40 cards left
    players = 4
    
    cards_left = []
    remaining = total_remaining
    
    for i in range(players - 1):
        # Ensure minimum 1 card per player and enough remaining
        min_cards = 1
        max_cards = min(remaining - (players - i - 1), 13)  # Max 13 cards per player
        
        if max_cards < min_cards:
            max_cards = min_cards
            
        cards = random.randint(min_cards, max_cards)
        cards_left.append(cards)
        remaining -= cards
    
    cards_left.append(min(max(1, remaining), 13))  # Last player gets remaining (at least 1, max 13)
    return cards_left


def generate_record() -> Dict[str, Any]:
    """Generate a single test record"""
    
    # Generate hand
    hand = generate_hand(random.randint(5, 15))
    
    # Generate legal moves
    legal_moves = generate_legal_moves(hand)
    
    # Generate cards left
    cards_left = generate_cards_left()
    
    # Randomly choose last move (50% pass, 50% combo)
    if random.random() < 0.5:
        last_move = None  # Pass situation
    else:
        # Generate a random last move
        last_rank = random.randint(0, 12)
        last_combo = random.choice(["single", "pair", "triple"])
        last_move = {
            "type": "play_cards",
            "cards": [last_rank],  # Simplified
            "combo_type": last_combo,
            "rank_value": last_rank
        }
    
    # Choose action based on last move
    if last_move:
        # Must play combo to beat last move
        available_combos = [move for move in legal_moves if move.get("combo_type") == last_move.get("combo_type")]
        if available_combos:
            chosen_move = random.choice(available_combos)
            action = {
                "stage1": {
                    "value": chosen_move.get("combo_type")
                },
                "stage2": {
                    "cards": chosen_move.get("cards")
                }
            }
        else:
            # No matching combo, must pass
            action = {
                "stage1": {
                    "value": "pass"
                },
                "stage2": {
                    "cards": []
                }
            }
    else:
        # Pass situation, can choose any combo or pass
        chosen_move = random.choice(legal_moves)
        action = {
            "stage1": {
                "value": chosen_move.get("combo_type")
            },
            "stage2": {
                "cards": chosen_move.get("cards")
            }
        }
    
    # Create record
    record = {
        "hand": hand,
        "cards_left": cards_left,
        "last_move": last_move,
        "action": action,
        "meta": {
            "legal_moves": legal_moves
        }
    }
    
    return record


def main():
    """Generate test data"""
    
    print("Generating test data...")
    
    records = []
    for i in range(1000):  # Generate 1000 records
        record = generate_record()
        records.append(record)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} records...")
    
    # Save to file
    output_file = "data/test_general_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Generated {len(records)} test records")
    print(f"Saved to: {output_file}")
    
    # Print sample record
    print("\nSample record:")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()
