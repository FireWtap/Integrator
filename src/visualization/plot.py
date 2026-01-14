import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union

class IntegrationVisualizer:
    """
    Visualizer for comparing integration methods.
    """
    
    def __init__(self, adata: ad.AnnData):
        self.adata = adata
        self.computed_embeddings = []

    def compute_umaps(self, embedding_keys: List[str] = None, n_neighbors: int = 15, min_dist: float = 0.5):
        """
        Compute UMAPs for the specified embeddings.
        
        Parameters:
        -----------
        embedding_keys : List[str]
            List of keys in .obsm to compute UMAPs for (e.g., ['X_pca', 'X_pca_harmony', 'X_scVI']).
            If None, tries to find common integration keys.
        """
        if embedding_keys is None:
            # Auto-detect
            potential_keys = ['X_pca', 'X_pca_harmony', 'X_scVI', 'X_pca_scgen', 'X_scanorama']
            embedding_keys = [k for k in potential_keys if k in self.adata.obsm]
            
        print(f"Computing UMAPs for: {embedding_keys}")
        
        for key in embedding_keys:
            if key not in self.adata.obsm:
                print(f"Warning: Embedding '{key}' not found in .obsm. Skipping.")
                continue
                
            print(f"  Processing '{key}'...")
            
            # We need to be careful not to overwrite the main 'X_umap' if we want to keep them separate
            # Scanpy usually uses 'X_pca' by default for neighbors.
            # We can use use_rep in neighbors.
            
            # Compute neighbors
            # We store neighbors in a specific key to avoid conflicts? 
            # Scanpy's neighbors function writes to .uns['neighbors'] and .obsp['distances'] etc.
            # If we run it sequentially, it overwrites.
            # For plotting, we usually need the UMAP coordinates.
            
            # Strategy: Run neighbors -> Run UMAP -> Store UMAP in specific obsm key -> Repeat
            
            try:
                sc.pp.neighbors(self.adata, use_rep=key, n_neighbors=n_neighbors, key_added=f'neighbors_{key}')
                
                # Run UMAP using the specific neighbors key
                # sc.tl.umap(adata, neighbors_key=...)
                # Note: sc.tl.umap writes to .obsm['X_umap'] by default.
                
                # We can use the 'neighbors_key' argument in sc.tl.umap (available in recent scanpy)
                # And we can copy the result to a custom key.
                
                sc.tl.umap(self.adata, neighbors_key=f'neighbors_{key}', min_dist=min_dist)
                
                # Move result to custom key
                self.adata.obsm[f'X_umap_{key}'] = self.adata.obsm['X_umap'].copy()
                self.computed_embeddings.append(key)
                
            except Exception as e:
                print(f"  Error computing UMAP for {key}: {e}")
                
        print(f"✓ Computed UMAPs for {len(self.computed_embeddings)} embeddings")

    def plot_umaps(self, color_by: List[str] = ['batch'], save_path: str = 'integration_comparison.pdf', show: bool = True):
        """
        Plot UMAPs for all computed embeddings.
        """
        if not self.computed_embeddings:
            print("No UMAPs computed. Run compute_umaps() first.")
            return

        n_embeddings = len(self.computed_embeddings)
        n_colors = len(color_by)
        
        print(f"Plotting {n_embeddings} embeddings colored by {color_by}...")
        
        # Create a grid of plots
        # Rows = embeddings, Cols = colors
        
        fig, axes = plt.subplots(n_embeddings, n_colors, figsize=(5 * n_colors, 4 * n_embeddings), squeeze=False)
        
        for i, emb_key in enumerate(self.computed_embeddings):
            umap_key = f'X_umap_{emb_key}'
            
            # Temporarily set X_umap to this key for scanpy plotting
            # (sc.pl.umap expects X_umap or basis argument)
            # We can use basis=umap_key if we registered it? 
            # Actually sc.pl.embedding(..., basis=...) is more generic.
            
            for j, color in enumerate(color_by):
                ax = axes[i, j]
                
                # Title
                title = f"{emb_key} - {color}"
                
                sc.pl.embedding(
                    self.adata, 
                    basis=umap_key, 
                    color=color, 
                    ax=ax, 
                    show=False, 
                    title=title,
                    frameon=False
                )
        
        plt.tight_layout()
        
        if save_path:
            print(f"Saving plot to {save_path}...")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
        print("✓ Plots generated")
