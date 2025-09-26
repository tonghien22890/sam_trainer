import json

# Check first 10 records
with open('simple_synthetic_training_data.jsonl', 'r') as f:
    lines = f.readlines()[:10]
    
print("=== TRAINING DATA ANALYSIS ===")
combo_counts = {}
rank_counts = {}

for i, line in enumerate(lines):
    data = json.loads(line)
    action = data.get('action', {}).get('stage2', {})
    combo_type = action.get('combo_type', 'unknown')
    rank_value = action.get('rank_value', -1)
    
    combo_counts[combo_type] = combo_counts.get(combo_type, 0) + 1
    rank_counts[rank_value] = rank_counts.get(rank_value, 0) + 1
    
    print(f"Record {i+1}: {combo_type} rank={rank_value}")

print(f"\n=== COMBO TYPE DISTRIBUTION ===")
for combo, count in combo_counts.items():
    print(f"{combo}: {count}")

print(f"\n=== RANK VALUE DISTRIBUTION ===")
for rank, count in sorted(rank_counts.items()):
    print(f"rank {rank}: {count}")

