"""
Generate training data với combo types đúng cho Sâm
"""
import json
import random
from datetime import datetime
from typing import Dict, List, Any

def generate_sam_combo_sequence():
    """Generate combo sequence với combo types đúng cho Sâm"""
    
    # Sâm combo types: single, pair, triple, straight, quad
    combo_types = ['single', 'pair', 'triple', 'straight', 'quad']
    
    # Generate random sequence length (1-4 combos)
    sequence_length = random.randint(1, 4)
    
    sequence = []
    used_cards = set()
    
    for i in range(sequence_length):
        combo_type = random.choice(combo_types)
        rank_value = random.randint(0, 11)  # 0-11: 3 to A, không được có 2
        
        # Generate cards based on combo type
        if combo_type == 'single':
            cards = [random.randint(1, 52)]
        elif combo_type == 'pair':
            base_card = random.randint(1, 52)
            cards = [base_card, base_card + 13]  # Same rank, different suit
        elif combo_type == 'triple':
            base_card = random.randint(1, 52)
            cards = [base_card, base_card + 13, base_card + 26]
        elif combo_type == 'straight':
            start_card = random.randint(1, 40)  # Ensure we have 5 consecutive cards
            cards = [start_card + i for i in range(5)]
        elif combo_type == 'quad':
            base_card = random.randint(1, 52)
            cards = [base_card, base_card + 13, base_card + 26, base_card + 39]
        
        # Check if cards are already used
        if any(card in used_cards for card in cards):
            continue
            
        used_cards.update(cards)
        sequence.append({
            "cards": cards,
            "combo_type": combo_type,
            "rank_value": rank_value
        })
    
    return sequence

def calculate_sequence_strength(sequence):
    """Calculate strength of sequence"""
    if not sequence:
        return 0.0
    
    # Base strength by combo type
    base_strength = {
        'single': 0.1, 'pair': 0.3, 'triple': 0.5,
        'straight': 0.7, 'quad': 0.9
    }
    
    strengths = []
    for combo in sequence:
        combo_type = combo['combo_type']
        rank_value = combo['rank_value']
        
        strength = base_strength.get(combo_type, 0.1)
        # Rank bonus: dây tới Át (11) = yếu nhất, dây tới J (8) = mạnh nhất
        if combo_type == 'straight':
            # Dây: rank cao = yếu hơn (A-2-3-4-5 yếu hơn J-Q-K-A)
            rank_bonus = ((11 - rank_value) / 11.0) * 0.3  # Invert for straight
        else:
            # Các combo khác: rank cao = mạnh hơn
            rank_bonus = (rank_value / 11.0) * 0.3
        strengths.append(strength + rank_bonus)
    
    return sum(strengths) / len(strengths)

def determine_success_probability(sequence, sequence_strength):
    """Determine success probability based on sequence strength"""
    
    # Count strong combos
    strong_combos = sum(1 for combo in sequence if combo['combo_type'] in ['straight', 'quad'])
    # High ranks: cho straight là rank thấp (J-Q-K-A), cho combo khác là rank cao
    high_ranks = sum(1 for combo in sequence if 
                    (combo['combo_type'] == 'straight' and combo['rank_value'] <= 3) or
                    (combo['combo_type'] != 'straight' and combo['rank_value'] >= 8))
    
    # Base probability from strength
    base_prob = sequence_strength
    
    # Bonus for strong combos
    if strong_combos >= 2:
        base_prob += 0.2
    elif strong_combos >= 1:
        base_prob += 0.1
    
    # Bonus for high ranks
    if high_ranks >= 2:
        base_prob += 0.15
    elif high_ranks >= 1:
        base_prob += 0.05
    
    # Penalty for too many weak combos
    weak_combos = sum(1 for combo in sequence if combo['combo_type'] in ['single', 'pair'])
    if weak_combos > 2:
        base_prob -= 0.2
    
    # Ensure probability is between 0.1 and 0.95
    base_prob = max(0.1, min(0.95, base_prob))
    
    return base_prob

def generate_training_records(num_records=1500):
    """Generate training records"""
    records = []
    
    for i in range(num_records):
        sequence = generate_sam_combo_sequence()
        sequence_strength = calculate_sequence_strength(sequence)
        success_prob = determine_success_probability(sequence, sequence_strength)
        
        # Determine result based on probability
        result = "success" if random.random() < success_prob else "fail"
        
        record = {
            "game_id": f"sam_game_{i}",
            "player_id": random.randint(0, 3),
            "hand": [random.randint(1, 52) for _ in range(random.randint(8, 15))],
            "sammove_sequence": sequence,
            "result": result,
            "meta": {
                "winner_id": random.randint(0, 3),
                "total_players": 4,
                "num_combos": len(sequence),
                "num_cards": sum(len(combo['cards']) for combo in sequence),
                "sequence_strength": sequence_strength,
                "success_probability": success_prob
            },
            "timestamp": datetime.now().isoformat()
        }
        
        records.append(record)
    
    return records

def main():
    """Generate and save training data"""
    print("🔄 Generating Sâm training data with correct combo types...")
    
    # Generate records
    records = generate_training_records(1500)
    
    # Save to file
    output_file = "data/sam_training_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Generated {len(records)} training records")
    print(f"💾 Saved to {output_file}")
    
    # Show statistics
    success_count = sum(1 for r in records if r['result'] == 'success')
    print(f"📊 Success rate: {success_count/len(records):.3f}")
    
    # Show combo type distribution
    combo_counts = {}
    for record in records:
        for combo in record['sammove_sequence']:
            combo_type = combo['combo_type']
            combo_counts[combo_type] = combo_counts.get(combo_type, 0) + 1
    
    print(f"📈 Combo type distribution:")
    for combo_type, count in combo_counts.items():
        print(f"   {combo_type}: {count}")

if __name__ == "__main__":
    main()
