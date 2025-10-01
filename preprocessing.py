"""
Data loading, quality control, and preprocessing for scRNA-seq data.
"""
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try to import GPU-accelerated scanpy
try:
    import rapids_singlecell as rsc
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False


def load_data(file_path: str):
    """Load h5ad file."""
    print(f"Loading data from {file_path}...")
    adata = sc.read_h5ad(file_path)
    print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")
    return adata


def check_normalization_status(adata):
    """
    Check if data is already normalized by examining count distributions.
    Returns: dict with normalization status information
    """
    print("\n=== Checking normalization status ===")
    
    # Check if counts are integers (raw) or floats (normalized)
    is_integer = np.allclose(adata.X.data if hasattr(adata.X, 'data') else adata.X, 
                             np.round(adata.X.data if hasattr(adata.X, 'data') else adata.X))
    
    # Check count distribution
    if hasattr(adata.X, 'toarray'):
        sample_data = adata.X[:100].toarray()
    else:
        sample_data = adata.X[:100]
    
    max_val = np.max(sample_data)
    mean_val = np.mean(sample_data)
    
    status = {
        'is_integer': is_integer,
        'max_value': max_val,
        'mean_value': mean_val,
        'needs_normalization': is_integer and max_val > 100,
        'likely_log_normalized': not is_integer and max_val < 20,
        'likely_raw': is_integer and max_val > 100
    }
    
    print(f"Data type: {'Integer counts' if is_integer else 'Float values'}")
    print(f"Max value: {max_val:.2f}, Mean value: {mean_val:.2f}")
    
    if status['likely_raw']:
        print("✓ Data appears to be raw counts - normalization needed")
    elif status['likely_log_normalized']:
        print("✓ Data appears to be log-normalized")
    else:
        print("⚠ Data normalization status unclear - will normalize to be safe")
    
    return status


def quality_control(adata, output_dir: str = "qc_plots"):
    """
    Perform quality control and generate QC plots.
    Following best practices from Scanpy tutorials.
    """
    print("\n=== Quality Control ===")
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Create QC violin plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total counts per cell
    sns.violinplot(data=adata.obs, y='total_counts', ax=axes[0])
    axes[0].set_ylabel('Total counts')
    axes[0].set_title('Total counts per cell')
    
    # Number of genes per cell
    sns.violinplot(data=adata.obs, y='n_genes_by_counts', ax=axes[1])
    axes[1].set_ylabel('Number of genes')
    axes[1].set_title('Genes detected per cell')
    
    # Percentage of mitochondrial genes (if available)
    if 'pct_counts_mt' in adata.obs.columns:
        sns.violinplot(data=adata.obs, y='pct_counts_mt', ax=axes[2])
        axes[2].set_ylabel('% MT genes')
        axes[2].set_title('Mitochondrial gene percentage')
    else:
        axes[2].text(0.5, 0.5, 'MT genes not calculated', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Mitochondrial genes')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qc_violins.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ QC plots saved to {output_dir}/qc_violins.png")
    
    # Print summary statistics
    print(f"\nQC Summary:")
    print(f"  Mean counts per cell: {adata.obs['total_counts'].mean():.0f}")
    print(f"  Median genes per cell: {adata.obs['n_genes_by_counts'].median():.0f}")
    
    return adata


def preprocess_data(adata, normalize: bool = True, hvg: bool = True, 
                   n_top_genes: int = 2000, scale: bool = True, use_gpu: bool = True):
    """
    Preprocess scRNA-seq data following best practices.
    GPU-accelerated if rapids-singlecell is available.
    
    Args:
        adata: AnnData object
        normalize: Whether to normalize (if needed)
        hvg: Whether to select highly variable genes
        n_top_genes: Number of HVGs to select
        scale: Whether to scale data
        use_gpu: Whether to use GPU acceleration (if available)
    """
    print("\n=== Preprocessing ===")
    
    # Check if we can use GPU
    use_rapids = use_gpu and RAPIDS_AVAILABLE
    if use_rapids:
        print("✓ Using GPU-accelerated preprocessing (rapids-singlecell)")
    else:
        if use_gpu and not RAPIDS_AVAILABLE:
            print("⚠ rapids-singlecell not available, using CPU")
        else:
            print("Using CPU preprocessing")
    
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()
    
    # Normalize if needed
    if normalize:
        print("Normalizing to 10,000 counts per cell...")
        if use_rapids:
            rsc.pp.normalize_total(adata, target_sum=1e4)
            print("Log-transforming...")
            rsc.pp.log1p(adata)
        else:
            sc.pp.normalize_total(adata, target_sum=1e4)
            print("Log-transforming...")
            sc.pp.log1p(adata)
    
    # Highly variable genes
    if hvg:
        print(f"Selecting {n_top_genes} highly variable genes...")
        if use_rapids:
            rsc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, 
                                        subset=False, flavor='seurat_v3', 
                                        layer='counts')
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, 
                                        subset=False, flavor='seurat_v3', 
                                        layer='counts')
        print(f"✓ Found {adata.var['highly_variable'].sum()} HVGs")
    
    # Scale data
    if scale:
        print("Scaling data...")
        if use_rapids:
            rsc.pp.scale(adata, max_value=10)
        else:
            sc.pp.scale(adata, max_value=10)
    
    # PCA
    print("Computing PCA...")
    if use_rapids:
        rsc.tl.pca(adata, svd_solver='arpack')
    else:
        sc.tl.pca(adata, svd_solver='arpack')
    
    print("✓ Preprocessing complete")
    return adata


def plot_batch_distribution(adata, batch_key: str, label_key: str = None, 
                           output_dir: str = "qc_plots"):
    """Plot batch and cell type distributions."""
    print("\n=== Plotting batch distribution ===")
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2 if label_key else 1, figsize=(12, 4))
    if not label_key:
        axes = [axes]
    
    # Batch distribution
    batch_counts = adata.obs[batch_key].value_counts()
    axes[0].bar(range(len(batch_counts)), batch_counts.values)
    axes[0].set_xticks(range(len(batch_counts)))
    axes[0].set_xticklabels(batch_counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Number of cells')
    axes[0].set_title(f'Cells per {batch_key}')
    
    # Cell type distribution (if available)
    if label_key and label_key in adata.obs.columns:
        label_counts = adata.obs[label_key].value_counts()
        axes[1].bar(range(len(label_counts)), label_counts.values)
        axes[1].set_xticks(range(len(label_counts)))
        axes[1].set_xticklabels(label_counts.index, rotation=45, ha='right')
        axes[1].set_ylabel('Number of cells')
        axes[1].set_title(f'Cells per {label_key}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/batch_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Batch distribution plot saved to {output_dir}/batch_distribution.png")
