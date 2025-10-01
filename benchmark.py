"""
Benchmarking integration methods using scib metrics.
"""
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def compute_scib_metrics(adata, batch_key: str, label_key: str, 
                        embed_key: str, method_name: str):
    """
    Compute scib metrics for a given integration method.
    
    Args:
        adata: AnnData object with integration results
        batch_key: Column name for batch information
        label_key: Column name for cell type labels
        embed_key: Key in adata.obsm for the integration embedding
        method_name: Name of the integration method
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n=== Computing scib metrics for {method_name} ===")
    
    try:
        import scib
        
        # Prepare data for scib
        adata_int = adata.copy()
        adata_int.obsm['X_emb'] = adata_int.obsm[embed_key]
        
        # Compute neighbor graph on integrated embedding
        print("Computing neighbor graph...")
        sc.pp.neighbors(adata_int, use_rep='X_emb')
        
        # Compute UMAP for visualization
        print("Computing UMAP...")
        sc.tl.umap(adata_int)
        
        metrics = {}
        
        # Batch correction metrics (lower batch effect is better)
        print("Computing batch correction metrics...")
        
        try:
            # ASW (Average Silhouette Width) - batch
            # Lower is better (less batch effect)
            asw_batch = scib.metrics.silhouette_batch(
                adata_int, batch_key=batch_key, label_key=label_key,
                embed='X_emb', verbose=False
            )
            metrics['ASW_batch'] = asw_batch
            print(f"  ASW batch: {asw_batch:.3f}")
        except Exception as e:
            print(f"  ⚠ ASW batch failed: {e}")
            metrics['ASW_batch'] = np.nan
        
        try:
            # PCR (Principal Component Regression) - batch
            pcr_batch = scib.metrics.pcr_comparison(
                adata_int, adata, covariate=batch_key, embed='X_emb', verbose=False
            )
            metrics['PCR_batch'] = pcr_batch
            print(f"  PCR batch: {pcr_batch:.3f}")
        except Exception as e:
            print(f"  ⚠ PCR batch failed: {e}")
            metrics['PCR_batch'] = np.nan
        
        try:
            # Graph connectivity
            graph_conn = scib.metrics.graph_connectivity(adata_int, label_key=label_key)
            metrics['Graph_connectivity'] = graph_conn
            print(f"  Graph connectivity: {graph_conn:.3f}")
        except Exception as e:
            print(f"  ⚠ Graph connectivity failed: {e}")
            metrics['Graph_connectivity'] = np.nan
        
        # Bio-conservation metrics (higher is better)
        print("Computing bio-conservation metrics...")
        
        try:
            # ASW (Average Silhouette Width) - cell type
            # Higher is better (better cell type separation)
            asw_label = scib.metrics.silhouette(
                adata_int, label_key=label_key, embed='X_emb'
            )
            metrics['ASW_label'] = asw_label
            print(f"  ASW label: {asw_label:.3f}")
        except Exception as e:
            print(f"  ⚠ ASW label failed: {e}")
            metrics['ASW_label'] = np.nan
        
        try:
            # NMI (Normalized Mutual Information)
            nmi = scib.metrics.nmi(adata_int, group1=label_key, group2='leiden', 
                                  method='arithmetic')
            metrics['NMI'] = nmi
            print(f"  NMI: {nmi:.3f}")
        except Exception as e:
            # Compute leiden clustering if not present
            try:
                sc.tl.leiden(adata_int, resolution=0.5)
                nmi = scib.metrics.nmi(adata_int, group1=label_key, group2='leiden',
                                      method='arithmetic')
                metrics['NMI'] = nmi
                print(f"  NMI: {nmi:.3f}")
            except Exception as e2:
                print(f"  ⚠ NMI failed: {e2}")
                metrics['NMI'] = np.nan
        
        try:
            # ARI (Adjusted Rand Index)
            if 'leiden' not in adata_int.obs.columns:
                sc.tl.leiden(adata_int, resolution=0.5)
            ari = scib.metrics.ari(adata_int, group1=label_key, group2='leiden')
            metrics['ARI'] = ari
            print(f"  ARI: {ari:.3f}")
        except Exception as e:
            print(f"  ⚠ ARI failed: {e}")
            metrics['ARI'] = np.nan
        
        # Store UMAP coordinates for visualization
        adata.obsm[f'X_umap_{method_name}'] = adata_int.obsm['X_umap']
        
        print(f"✓ Metrics computed for {method_name}")
        return metrics, adata_int
        
    except ImportError:
        print("⚠ scib not available. Install with: pip install scib")
        return {}, adata
    except Exception as e:
        print(f"⚠ Metric computation failed: {str(e)}")
        return {}, adata


def benchmark_all_methods(adata, batch_key: str, label_key: str, 
                         integration_results: dict):
    """
    Benchmark all integration methods.
    
    Args:
        adata: AnnData object with integration results
        batch_key: Column name for batch information
        label_key: Column name for cell type labels
        integration_results: Dict mapping method names to obsm keys
    
    Returns:
        DataFrame with all metrics
    """
    print(f"\n{'='*60}")
    print("BENCHMARKING INTEGRATION METHODS")
    print(f"{'='*60}")
    
    all_metrics = {}
    integrated_adatas = {}
    
    for method_name, embed_key in integration_results.items():
        if embed_key in adata.obsm:
            metrics, adata_int = compute_scib_metrics(
                adata, batch_key, label_key, embed_key, method_name
            )
            all_metrics[method_name] = metrics
            integrated_adatas[method_name] = adata_int
    
    # Create results dataframe
    results_df = pd.DataFrame(all_metrics).T
    
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(results_df.to_string())
    
    return results_df, integrated_adatas


def plot_benchmark_results(results_df, output_dir: str = "results"):
    """Plot benchmark results as a heatmap."""
    print("\n=== Plotting benchmark results ===")
    Path(output_dir).mkdir(exist_ok=True)
    
    if results_df.empty:
        print("⚠ No results to plot")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize metrics to 0-1 scale for visualization
    results_normalized = results_df.copy()
    for col in results_normalized.columns:
        if not results_normalized[col].isna().all():
            min_val = results_normalized[col].min()
            max_val = results_normalized[col].max()
            if max_val > min_val:
                results_normalized[col] = (results_normalized[col] - min_val) / (max_val - min_val)
    
    sns.heatmap(results_normalized.T, annot=results_df.T, fmt='.3f', 
                cmap='RdYlGn', center=0.5, ax=ax, cbar_kws={'label': 'Normalized Score'})
    ax.set_xlabel('Integration Method')
    ax.set_ylabel('Metric')
    ax.set_title('Integration Benchmark Results')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Benchmark heatmap saved to {output_dir}/benchmark_heatmap.png")


def plot_integration_umaps(adata, integrated_adatas: dict, batch_key: str, 
                          label_key: str, output_dir: str = "results"):
    """Plot UMAPs for all integration methods."""
    print("\n=== Plotting integration UMAPs ===")
    Path(output_dir).mkdir(exist_ok=True)
    
    n_methods = len(integrated_adatas)
    if n_methods == 0:
        print("⚠ No integrated data to plot")
        return
    
    # Plot by batch
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 4))
    if n_methods == 0:
        axes = [axes]
    
    # Original data
    if 'X_pca' in adata.obsm:
        sc.pp.neighbors(adata, use_rep='X_pca')
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=batch_key, ax=axes[0], show=False, title='Original (by batch)')
    
    # Integrated data
    for idx, (method_name, adata_int) in enumerate(integrated_adatas.items()):
        sc.pl.umap(adata_int, color=batch_key, ax=axes[idx + 1], show=False, 
                  title=f'{method_name} (by batch)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_by_batch.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot by cell type
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 4))
    if n_methods == 0:
        axes = [axes]
    
    # Original data
    if 'X_umap' in adata.obsm:
        sc.pl.umap(adata, color=label_key, ax=axes[0], show=False, title='Original (by cell type)')
    
    # Integrated data
    for idx, (method_name, adata_int) in enumerate(integrated_adatas.items()):
        sc.pl.umap(adata_int, color=label_key, ax=axes[idx + 1], show=False,
                  title=f'{method_name} (by cell type)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_by_celltype.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ UMAP plots saved to {output_dir}/")


def save_results(results_df, adata, output_dir: str = "results"):
    """Save benchmark results and integrated data."""
    print("\n=== Saving results ===")
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save metrics
    results_df.to_csv(f"{output_dir}/benchmark_metrics.csv")
    print(f"✓ Metrics saved to {output_dir}/benchmark_metrics.csv")
    
    # Save integrated data
    adata.write_h5ad(f"{output_dir}/integrated_data.h5ad")
    print(f"✓ Integrated data saved to {output_dir}/integrated_data.h5ad")
