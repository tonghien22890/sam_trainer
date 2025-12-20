# Migration từ XGBClassifier sang XGBRanker

## Tổng quan

Style Learner đã được refactor từ **XGBClassifier** (binary classification) sang **XGBRanker** (learning-to-rank) để phù hợp hơn với bản chất bài toán: **ranking các legal moves**.

## Lý do chuyển đổi

### Vấn đề với Classifier
1. **Overload rule-based logic**: Classifier phải gánh quá nhiều heuristic thông qua sample weighting và feature scaling
2. **Binary classification không tự nhiên**: Bài toán thực tế là "move nào tốt nhất?" chứ không phải "move này tốt hay không?"
3. **Sample imbalance nghiêm trọng**: Mỗi game state có 1 chosen move (positive) vs 10-20 non-chosen moves (negatives)
4. **Hard để tune**: Phải điều chỉnh nhiều hyperparameter về weighting, scaling để model học đúng

### Ưu điểm của Ranker
1. **Phù hợp bản chất bài toán**: Ranking các moves trong cùng game state
2. **Relevance scores linh hoạt**: Thay vì 0/1, có thể encode multiple levels (chosen=3, good alternative=2, mediocre=1, bad=0)
3. **Tự nhiên hơn**: Model học so sánh tương đối giữa các moves trong cùng context
4. **Ít phụ thuộc rule heuristic**: Sample weighting đơn giản hơn nhiều

## Thay đổi chính

### 1. Model Type
```python
# Before
self.model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    eval_metric='logloss'
)

# After
self.model = xgb.XGBRanker(
    objective='rank:ndcg',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    eval_metric='ndcg@5'
)
```

### 2. Data Structure

#### Before (Classifier)
```python
X = []  # Features
y = []  # Binary labels (0 or 1)
sample_weights = []  # Heavy weighting logic

for record in training_data:
    for move in legal_moves:
        X.append(features)
        y.append(1 if is_chosen else 0)
        weight = complex_weighting_logic(...)
        sample_weights.append(weight)

model.fit(X, y, sample_weight=sample_weights)
```

#### After (Ranker)
```python
X = []  # Features
y = []  # Relevance scores (0-5)
groups = []  # Group sizes
sample_weights = []  # Simplified weighting

for record in training_data:
    group_size = 0
    for move in legal_moves:
        X.append(features)
        relevance = calculate_relevance(move, framework, is_chosen)
        y.append(relevance)
        weight = 1.0 if is_chosen else 0.5  # Much simpler
        sample_weights.append(weight)
        group_size += 1
    groups.append(group_size)

model.fit(X, y, group=groups, sample_weight=sample_weights)
```

### 3. Relevance Calculation

```python
def _calculate_relevance(move, framework, is_chosen):
    """
    Relevance scoring (0-5):
    - Chosen move: 3.0 base + compliance bonus - breaking penalty
    - Non-chosen: 1.0 base + compliance bonus - breaking penalty
    - Pass: 0.5 (neutral)
    """
    compliance = _sequence_compliance(move, framework)
    breaking = _framework_breaking_severity(move, framework)
    
    if is_chosen:
        relevance = 3.0 + 2.0 * compliance - 1.0 * breaking
    else:
        relevance = 1.0 + 1.0 * compliance - 0.5 * breaking
        if move['combo_type'] == 'pass':
            relevance = 0.5
    
    return clip(relevance, 0.0, 5.0)
```

**Logic:**
- Chosen move luôn có base relevance cao (3.0)
- Framework compliance boost điểm
- Breaking important combos giảm điểm
- Non-chosen moves vẫn có điểm nếu chúng align với framework (good alternatives)
- Pass có điểm neutral

### 4. Inference

#### Before (Classifier)
```python
base_scores = model.predict_proba(X)[:, 1]  # Probability of class 1
```

#### After (Ranker)
```python
base_scores = model.predict(X)  # Direct scores (higher = better)
```

### 5. Evaluation Metric

#### Before (Classifier)
```python
# Binary accuracy: % correctly classified
y_pred = model.predict(X)
accuracy = np.mean(y == y_pred)
```

#### After (Ranker)
```python
# Top-1 accuracy: % times chosen move is ranked first
y_pred = model.predict(X)

top1_correct = 0
offset = 0
for group_size in groups:
    group_relevances = y[offset:offset + group_size]
    group_predictions = y_pred[offset:offset + group_size]
    
    if np.argmax(group_relevances) == np.argmax(group_predictions):
        top1_correct += 1
    offset += group_size

accuracy = top1_correct / len(groups)
```

## Ví dụ cụ thể

### Scenario: Hand có [3♥, 4♦, 5♣, 6♠, 7♥, A♠, 2♠]

#### Legal moves:
1. Single 3♥ - không breaking
2. Single A♠ - không breaking  
3. Single 2♠ - breaking (nên giữ 2 cho sau)
4. Straight 3♥-4♦-5♣-6♠-7♥ - exact framework combo
5. Pass

#### Framework:
- Core combo: Straight [3♥, 4♦, 5♣, 6♠, 7♥]
- Recommended moves: [straight first, then singles]

#### Chosen move: Straight (move 4)

### Relevance scores (Ranker)

| Move | Compliance | Breaking | Is Chosen | Relevance |
|------|-----------|----------|-----------|-----------|
| Single 3♥ | 0.0 | 2.0 (breaks straight) | No | 1.0 + 0 - 1.0 = **0.0** |
| Single A♠ | 0.0 | 0.0 | No | 1.0 + 0 - 0 = **1.0** |
| Single 2♠ | 0.0 | 2.0 (breaks protection) | No | 1.0 + 0 - 1.0 = **0.0** |
| Straight | 1.0 | 0.0 | **Yes** | 3.0 + 2.0 - 0 = **5.0** |
| Pass | N/A | N/A | No | **0.5** |

**Model học**: Straight có relevance 5.0 >> others, nên model sẽ học rank nó cao nhất.

### Binary labels (Classifier - old)

| Move | Is Chosen | Weight | Label |
|------|-----------|--------|-------|
| Single 3♥ | No | 0.05 (heavy penalty for breaking) | 0 |
| Single A♠ | No | 0.5 | 0 |
| Single 2♠ | No | 0.05 (heavy penalty) | 0 |
| Straight | **Yes** | 13.0 (1 + 12*compliance) | 1 |
| Pass | No | 0.5 | 0 |

**Problem**: Model phải dựa vào weight để phân biệt, không tự nhiên và khó học.

## Sample Weighting Simplification

### Before (Classifier)
```python
if is_chosen:
    weight = 1.0 + 12.0 * compliance
    if breaking < 1.0:
        weight *= 1.5
else:
    weight = max(0.05, 1.0 - 0.9 * compliance - 0.5 * breaking)
    if breaking >= 2.0 and move_type == 'single':
        weight *= 0.3
```

### After (Ranker)
```python
if is_chosen:
    weight = 1.0  # Simple base weight
else:
    weight = 0.5  # Light downweight
```

**Reason**: Relevance scores đã encode preferences rồi, không cần phức tạp hoá weighting.

## Class Balancing

Vẫn giữ class balancing logic (reweight theo combo type frequency) vì:
- Dataset có thể bias heavy về singles
- Muốn model học cả rare combos (four_kind, double_seq)
- Ranker vẫn benefit từ balanced distribution

## Backward Compatibility

- Features vẫn giữ nguyên 51 dims
- Framework features vẫn heavily scaled (có thể giảm sau khi test)
- Tie-break logic trong inference vẫn giữ
- Environment variables (STYLE_DEBUG, STYLE_SCALE_*) vẫn hoạt động

## Testing Checklist

- [ ] Train model với data hiện có → check top-1 accuracy
- [ ] So sánh accuracy: Ranker vs Classifier (expect tương đương hoặc cao hơn)
- [ ] Test inference với game states thực tế
- [ ] Kiểm tra edge cases: all pass, single legal move, etc.
- [ ] Monitor performance: latency, memory usage
- [ ] A/B test với human players (optional)

## Tuning Guidelines

### Nếu model quá conservative (không chơi combos)
1. Tăng relevance gap giữa chosen vs non-chosen
2. Boost compliance weight trong relevance calculation
3. Increase n_estimators để model học sâu hơn

### Nếu model breaking combos quá nhiều
1. Tăng breaking penalty trong relevance
2. Thêm explicit negative samples cho breaking moves
3. Adjust feature scaling cho framework_breaking_severity

### Nếu model không follow framework
1. Check framework generation quality
2. Increase framework feature scales (STYLE_SCALE_*)
3. Add more training data với good framework examples

## Migration Steps

1. ✅ Refactor `train()` để dùng groups và relevance
2. ✅ Add `_calculate_relevance()` helper
3. ✅ Update `predict_with_framework()` để dùng Ranker predict
4. ✅ Update evaluation metric (top-1 accuracy)
5. ⏳ Test với existing pipeline
6. ⏳ Compare performance với Classifier baseline
7. ⏳ Deploy và monitor

## Rollback Plan

Nếu Ranker không hoạt động tốt:
1. Giữ Classifier code trong branch riêng
2. Add flag `USE_RANKER=False` để switch back
3. Document lessons learned
4. Re-evaluate approach

---

**Status**: Refactoring completed, pending testing  
**Next**: Run training pipeline và compare metrics với Classifier baseline

