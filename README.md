# Single-Cell RNA Integration Benchmark Pipeline

A clean, modular pipeline for benchmarking single-cell RNA-seq integration methods.

## Features

- **Data Loading & QC**: Automatic detection of normalization status, quality control metrics, and visualization
- **Cell Type Annotation**: Optional CellTypist-based automatic annotation
- **Integration Methods**: 
  - **scGen** - Conditional VAE (outputs corrected expression)
  - **Scanorama** - Mutual nearest neighbors (outputs corrected expression)
  - **Harmony** - Iterative clustering (outputs corrected PCA)
  - **ComBat** - Empirical Bayes (outputs corrected expression)
  - **scVI** - Variational inference (outputs latent representation)
  - **pyLemur** - Latent embedding (outputs latent representation)
- **Benchmarking**: Comprehensive evaluation using scib metrics
- **Model Saving**: Trained models are automatically saved for reuse

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py --input data.h5ad --batch_key batch --label_key cell_type --output results/
```

### With CellTypist Annotation
```bash
# Annotate cells and use predictions for benchmarking
# When --annotate is used WITHOUT --label_key, CellTypist predictions 
# automatically become the label_key for benchmarking
python main.py --input data.h5ad --batch_key batch --annotate --output results/

# Use specific CellTypist model
python main.py --input data.h5ad --batch_key batch --annotate \
  --celltypist_model Immune_All_High.pkl --output results/

# Annotate AND use your own labels for benchmarking
# CellTypist predictions will be saved in adata.obs['celltypist_predicted']
python main.py --input data.h5ad --batch_key batch --label_key my_cell_types \
  --annotate --output results/
```

### List Available CellTypist Models
```bash
python main.py --list_models
```

### Select Specific Integration Methods
```bash
python main.py --input data.h5ad --batch_key batch --label_key cell_type \
  --methods scanorama harmony combat --output results/
```

## Input Requirements

- **H5AD file** with multiple batches
- **Batch key**: Column in `.obs` indicating batch/dataset origin
- **Label key** (optional): Column in `.obs` with cell type labels
  - If not provided and `--annotate` is used, CellTypist predictions (`celltypist_predicted`) will **automatically become the label_key**
  - This ensures benchmarking runs successfully with auto-annotations

## Output Structure

```
results/
├── qc/
│   ├── qc_violins.png
│   └── batch_distribution.png
├── models/
│   ├── scgen_model/
│   └── scvi_model/
├── benchmark_metrics.csv
├── benchmark_heatmap.png
├── umap_by_batch.png
├── umap_by_celltype.png
└── integrated_data.h5ad
```

## Integration Methods Details

### Methods Returning Corrected Expression
- **scGen**: Uses conditional VAE to remove batch effects
- **Scanorama**: Panoramic stitching of datasets
- **ComBat**: Linear model-based batch correction

### Methods Returning Embeddings
- **Harmony**: Corrected PCA space
- **scVI**: Latent representation from variational inference
- **pyLemur**: Latent embedding

## Project Structure

- `main.py` - Main pipeline orchestrator
- `preprocessing.py` - Data loading, QC, and preprocessing
- `integration.py` - Integration methods
- `benchmark.py` - scib-based evaluation
- `annotation.py` - CellTypist annotation
