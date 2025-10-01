# Pipeline Summary

## What This Pipeline Does

When you pass a **merged dataset** (multiple batches/datasets concatenated) with a **batch_key** and optionally **cell type labels**, the pipeline will:

### 1. **Data Loading & QC** ✅
- Loads your h5ad file
- **Auto-detects** if data needs normalization (checks for raw counts vs normalized)
- Generates QC violin plots (total counts, genes detected, MT%)
- Plots batch and cell type distributions

### 2. **Optional Cell Type Annotation** ✅
- If you use `--annotate`, **normalizes data first** (CellTypist requires log1p normalized to 10k)
- Runs **CellTypist** to automatically annotate cell types
- Transfers annotations back to original data
- Uses the predictions as labels for benchmarking if no label_key provided
- Supports multiple pre-trained models (Immune, Lung, Fetal, etc.)

### 3. **Preprocessing for Integration** ✅
- Normalizes if needed (10k counts per cell + log transform)
- Selects highly variable genes (default 2000)
- Scales data
- Computes PCA

### 4. **Integration** ✅
Runs multiple integration methods that **remove batch effects**:

#### Methods Returning Corrected Expression Matrices:
- **scGen**: Conditional VAE - outputs corrected gene expression
- **Scanorama**: Mutual nearest neighbors - outputs corrected gene expression  
- **ComBat**: Empirical Bayes - outputs corrected gene expression

#### Methods Returning Embeddings:
- **Harmony**: Iterative clustering - outputs corrected PCA
- **scVI**: Variational inference - outputs latent space
- **pyLemur**: Latent embedding - outputs latent space

All results stored in `adata.obsm['X_<method>']`

### 5. **Model Saving** ✅
- Trained models (scGen, scVI) are **automatically saved** to `results/models/`
- Can be reloaded later for inference on new data

### 6. **Comprehensive Benchmarking** ✅
Uses **scib metrics** to evaluate integration quality:

#### Batch Correction Metrics (lower = better):
- **ASW batch**: Silhouette width for batch mixing
- **PCR batch**: Principal component regression
- **Graph connectivity**: How well batches mix

#### Bio-Conservation Metrics (higher = better):
- **ASW label**: Silhouette width for cell type separation
- **NMI**: Normalized mutual information
- **ARI**: Adjusted Rand index

### 7. **Visualization** ✅
- **Benchmark heatmap**: Compare all methods across all metrics
- **UMAP plots**: Before/after integration, colored by batch and cell type
- Shows which methods best remove batch effects while preserving biology

### 8. **Results Saved** ✅
```
results/
├── qc/                          # QC plots
├── models/                      # Trained models
├── benchmark_metrics.csv        # All scores
├── benchmark_heatmap.png        # Visual comparison
├── umap_by_batch.png           # Integration quality
├── umap_by_celltype.png        # Biology preservation
└── integrated_data.h5ad        # All integrated matrices
```

## Example Use Cases

### Case 1: You have batch labels and cell type labels
```bash
python main.py --input merged_data.h5ad --batch_key batch --label_key cell_type
```
→ Integrates and benchmarks using your labels

### Case 2: You have batch labels but NO cell type labels
```bash
python main.py --input merged_data.h5ad --batch_key batch --annotate
```
→ Annotates with CellTypist, then integrates and benchmarks

### Case 3: Test specific methods only
```bash
python main.py --input merged_data.h5ad --batch_key batch --label_key cell_type \
  --methods scanorama harmony combat
```
→ Only runs methods that output corrected expression

## Key Points

✅ **Yes**, it handles merged datasets with batch_key  
✅ **Yes**, it integrates them using multiple methods  
✅ **Yes**, it gives you comprehensive scores (scib metrics)  
✅ **Yes**, it saves trained models  
✅ **Yes**, it includes methods that output corrected count matrices (scGen, Scanorama, ComBat)  
✅ **Yes**, it can auto-annotate cell types with CellTypist  

The pipeline is **fully automated** - just point it at your data and it handles everything!
