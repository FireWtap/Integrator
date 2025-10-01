"""
Cell type annotation using CellTypist.
"""
import scanpy as sc
import numpy as np
from pathlib import Path


def annotate_celltypist(adata, model: str = 'Immune_All_Low.pkl', 
                        majority_voting: bool = True):
    """
    Annotate cell types using CellTypist.
    
    Args:
        adata: AnnData object (MUST be log1p normalized to 10k counts)
        model: CellTypist model name or path
               Popular models: 'Immune_All_Low.pkl', 'Immune_All_High.pkl',
                              'Human_Lung_Atlas.pkl', 'Pan_Fetal_Human.pkl'
        majority_voting: Whether to use majority voting for refinement
    
    Returns:
        adata: AnnData with cell type predictions in .obs
    """
    print("\n=== Running CellTypist annotation ===")
    
    try:
        import celltypist
        from celltypist import models
        
        # Download model if needed
        print(f"Loading CellTypist model: {model}")
        try:
            ct_model = models.Model.load(model=model)
        except:
            print(f"Downloading model {model}...")
            models.download_models(model=model)
            ct_model = models.Model.load(model=model)
        
        # CellTypist expects log1p normalized data in .X
        # Data should already be normalized by caller
        print("Running cell type prediction...")
        predictions = celltypist.annotate(
            adata, 
            model=ct_model,
            majority_voting=majority_voting
        )
        
        # Store predictions
        adata.obs['celltypist_predicted'] = predictions.predicted_labels.predicted_labels
        
        if majority_voting:
            adata.obs['celltypist_majority_voting'] = predictions.predicted_labels.majority_voting
            adata.obs['celltypist_conf_score'] = predictions.predicted_labels.conf_score
        
        print(f"✓ CellTypist annotation complete")
        print(f"  Found {adata.obs['celltypist_predicted'].nunique()} cell types")
        print(f"  Top 5 cell types:")
        for ct, count in adata.obs['celltypist_predicted'].value_counts().head(5).items():
            print(f"    - {ct}: {count} cells")
        
        return adata
        
    except ImportError:
        print("⚠ CellTypist not available. Install with: pip install celltypist")
        return adata
    except Exception as e:
        print(f"⚠ CellTypist annotation failed: {str(e)}")
        return adata


def list_celltypist_models():
    """List available CellTypist models."""
    try:
        import celltypist
        from celltypist import models
        
        print("\n=== Available CellTypist Models ===")
        available_models = models.models_description()
        print(available_models)
        
    except ImportError:
        print("⚠ CellTypist not available. Install with: pip install celltypist")
    except Exception as e:
        print(f"⚠ Failed to list models: {str(e)}")


def annotate_with_markers(adata, marker_genes: dict, layer: str = None):
    """
    Simple marker-based annotation.
    
    Args:
        adata: AnnData object
        marker_genes: Dict mapping cell types to marker gene lists
                     e.g., {'T cells': ['CD3D', 'CD3E'], 'B cells': ['CD19', 'MS4A1']}
        layer: Layer to use for expression (None = .X)
    
    Returns:
        adata: AnnData with marker scores in .obs
    """
    print("\n=== Running marker-based annotation ===")
    
    for cell_type, genes in marker_genes.items():
        # Filter genes present in dataset
        genes_present = [g for g in genes if g in adata.var_names]
        
        if len(genes_present) == 0:
            print(f"⚠ No marker genes found for {cell_type}")
            continue
        
        # Compute mean expression
        if layer:
            expr = adata[:, genes_present].layers[layer]
        else:
            expr = adata[:, genes_present].X
        
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        
        score = np.mean(expr, axis=1)
        adata.obs[f'marker_score_{cell_type}'] = score
        
        print(f"✓ Computed marker score for {cell_type} ({len(genes_present)} genes)")
    
    return adata
