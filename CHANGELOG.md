# Changelog

## Fix: CellTypist Annotation Order (2025-10-01)

### Problem
CellTypist was failing with error:
```
🛑 Invalid expression matrix in `.X`, expect log1p normalized expression to 10000 counts per cell
```

### Root Cause
CellTypist annotation was running **before** data normalization, but CellTypist requires log1p normalized data to 10k counts per cell.

### Solution
**Reordered pipeline steps:**

1. ✅ Load data
2. ✅ Check normalization status  
3. ✅ QC plots (on raw data)
4. ✅ **Normalize for CellTypist** (if `--annotate` flag used)
   - Creates temporary copy of data
   - Normalizes to 10k counts + log1p
   - Runs CellTypist annotation
   - Transfers annotations back to original data
5. ✅ Preprocessing for integration (normalize, HVG, scale, PCA)
6. ✅ Run integration methods
7. ✅ Save trained models
8. ✅ Benchmark with scib
9. ✅ Visualize results
10. ✅ Save all results

### Key Changes in `main.py`

**Before:**
```python
# 1. Load data
# 2. Annotate with CellTypist ❌ (fails - data not normalized)
# 3. Check normalization
# 4. Preprocessing
```

**After:**
```python
# 1. Load data
# 2. Check normalization status
# 3. QC plots
# 4. Normalize for CellTypist if --annotate ✅
#    - Normalizes temporary copy
#    - Runs annotation
#    - Transfers results back
# 5. Preprocessing for integration
```

### Usage
Now works correctly:
```bash
python main.py --input merged_adata.h5ad --batch_key batch --annotate
```

The pipeline will:
1. Detect if data needs normalization
2. Normalize specifically for CellTypist (10k + log1p)
3. Run annotation successfully
4. Continue with integration preprocessing
