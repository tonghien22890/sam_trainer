# Unbeatable Model - Refactor TODO

## 🔧 Cần Refactor: Model Save/Load Pattern

### Vấn Đề Hiện Tại

**Current approach:**
```python
# Save WHOLE class instances
def save_models(self, model_dir: str = 'models'):
    joblib.dump(self.validation_model, 'validation_model.pkl')  # ❌ Whole instance
    joblib.dump(self.pattern_model, 'pattern_model.pkl')
    joblib.dump(self.threshold_model, 'threshold_model.pkl')
```

**Issues:**
- Pickle cả class instance → Joblib cần module definition khi unpickle
- Nuitka build cần `--include-module=model_build.scripts.unbeatable.unbeatable_sequence_model`
- Breaking change khi refactor class
- Không consistent với Two-Layer pattern

### Giải Pháp Đề Xuất

**Refactor theo Two-Layer pattern:**
```python
# Save ONLY pure sklearn/ML objects
def save_models(self, model_dir: str = 'models'):
    joblib.dump(self.validation_model.model, 'validation_model.pkl')  # ✅ Pure RandomForest
    joblib.dump(self.pattern_model.model, 'pattern_model.pkl')        # ✅ Pure GradientBoosting
    joblib.dump(self.threshold_model.model, 'threshold_model.pkl')    # ✅ Pure RandomForest

def load_models(self, model_dir: str = 'models'):
    self.validation_model.model = joblib.load('validation_model.pkl')
    self.pattern_model.model = joblib.load('pattern_model.pkl')
    self.threshold_model.model = joblib.load('threshold_model.pkl')
```

**Benefits:**
- ✅ Chỉ pickle pure sklearn objects → không cần module definition
- ✅ Nuitka không cần `--include-module`
- ✅ Consistent với StyleLearner pattern
- ✅ Dễ refactor class logic
- ✅ Smaller pickle files

### Impact

**Cần làm khi refactor:**
1. Sửa `save_models()` và `load_models()` trong `unbeatable_sequence_model.py`
2. **RETRAIN tất cả models** (existing .pkl files không compatible)
3. Remove `--include-module` từ `build_process.py`
4. Test lại toàn bộ pipeline

### Tạm Thời Hotfix (CURRENT)

**Files đã sửa:**

1. **`ai_bots/nuitka/build_process.py`** line 56:
   - Added `--include-module=model_build.scripts.unbeatable.unbeatable_sequence_model`
   - Attempt to include module cho joblib unpickling

2. **`ai_bots/adapters/unbeatable_adapter.py`** lines 52-93:
   - Handle frozen mode path detection
   - Track `_models_loaded` flag
   - **Fallback to `context="general"`** if models load fail

3. **`ai_bots/adapters/unbeatable_adapter.py`** lines 137-141:
   - Auto-detect context based on `_models_loaded`
   - If models not loaded → use "general" context (no ML calls)
   - If models loaded → use "bao_sam" context (full ML pipeline)

**Fallback Behavior:**
```python
# If joblib.load fails in Nuitka:
context = "general"  # No ML models called
# → Uses rule-based combo detection + simple probability
# → Still works, just less precise
# → Combo priority still correct (Four-kind first)
```

**Vẫn hoạt động nhưng:**
- General context: probability calculation đơn giản hơn
- Không dùng được ML validation/pattern/threshold learning
- Executable size lớn hơn (nếu module được include)
- Technical debt

### Fixed Threshold vs ML Threshold

**Current Decision (KEPT):**
- Adapter OVERRIDES ML threshold with fixed values based on player count
- Fixed thresholds: 2p=0.5, 3p=0.55, 4p=0.65, 5p=0.7

**Why Keep Fixed?**
- Stable behavior across different user profiles
- SDK clients depend on predictable thresholds
- Easier to tune for game balance

**Trade-off:**
- ThresholdLearningModel learns user preferences but gets overridden
- Could use ML threshold với safety bounds instead:
  ```python
  ml_threshold = result.get('user_threshold', 0.65)
  threshold = max(0.5, min(0.8, ml_threshold))  # Clamp to safe range
  ```

**For Future:** Consider using ML threshold with bounds for personalization.

---

### Priority

- **Priority**: Medium
- **Effort**: ~2 hours (code + retrain)
- **Risk**: Low (chỉ change save/load, không đổi logic)

---

**Created**: 2025-10-11  
**Status**: NOTED - Để refactor sau release  
**Model Save/Load**: ✅ DONE (pure sklearn objects)  
**Fixed Threshold**: ✅ KEPT (stable for SDK)

