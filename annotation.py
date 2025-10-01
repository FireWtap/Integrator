"""
Cell type annotation using CellTypist.
"""
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path


def annotate_celltypist(adata, model: str = 'Immune_All_Low.pkl', 
                        majority_voting: bool = True, batch_key: str = None,
                        use_gpu: bool = True):
    """
    Annotate cell types using CellTypist.
    Processes batches separately if batch_key is provided.
    
    Args:
        adata: AnnData object (MUST be log1p normalized to 10k counts)
        model: CellTypist model name or path
               Popular models: 'Immune_All_Low.pkl', 'Immune_All_High.pkl',
                              'Human_Lung_Atlas.pkl', 'Pan_Fetal_Human.pkl'
        majority_voting: Whether to use majority voting for refinement
        batch_key: If provided, process each batch separately then combine
        use_gpu: Whether to use GPU acceleration (if available)
    
    Returns:
        adata: AnnData with cell type predictions in .obs
    """
    print("\n=== Running CellTypist annotation ===")
    
    try:
        import celltypist
        from celltypist import models
        import torch
        
        # Check GPU availability
        if use_gpu and torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            device = 'cuda'
        else:
            print("Using CPU for CellTypist")
            device = 'cpu'
        
        # Download model if needed
        print(f"Loading CellTypist model: {model}")
        try:
            ct_model = models.Model.load(model=model)
        except:
            print(f"Downloading model {model}...")
            models.download_models(model=model)
            ct_model = models.Model.load(model=model)
        
        # Process by batch if batch_key provided
        if batch_key and batch_key in adata.obs.columns:
            print(f"Processing batches separately using '{batch_key}'...")
            batches = adata.obs[batch_key].unique()
            print(f"Found {len(batches)} batches: {list(batches)}")
            
            all_predictions = []
            all_majority_voting = []
            all_conf_scores = []
            
            for batch in batches:
                print(f"\n  Annotating batch: {batch}")
                batch_mask = adata.obs[batch_key] == batch
                adata_batch = adata[batch_mask].copy()
                
                # Run prediction on this batch
                predictions = celltypist.annotate(
                    adata_batch, 
                    model=ct_model,
                    majority_voting=majority_voting,
                    use_gpu=use_gpu
                )
                
                # Store predictions in order
                all_predictions.append(predictions.predicted_labels.predicted_labels)
                
                if majority_voting:
                    all_majority_voting.append(predictions.predicted_labels.majority_voting)
                    all_conf_scores.append(predictions.predicted_labels.conf_score)
                
                print(f"    ✓ Batch {batch}: {len(adata_batch)} cells, {predictions.predicted_labels.predicted_labels.nunique()} cell types")
            
            # Combine predictions in original order
            adata.obs['celltypist_predicted'] = pd.concat(all_predictions)
            
            if majority_voting:
                adata.obs['celltypist_majority_voting'] = pd.concat(all_majority_voting)
                adata.obs['celltypist_conf_score'] = pd.concat(all_conf_scores)
            
            print(f"\n✓ CellTypist annotation complete (batch-wise)")
        
        else:
            # Process entire dataset at once
            print("Running cell type prediction on entire dataset...")
            predictions = celltypist.annotate(
                adata, 
                model=ct_model,
                majority_voting=majority_voting,
                use_gpu=use_gpu
            )
            
            # Store predictions
            adata.obs['celltypist_predicted'] = predictions.predicted_labels.predicted_labels
            
            if majority_voting:
                adata.obs['celltypist_majority_voting'] = predictions.predicted_labels.majority_voting
                adata.obs['celltypist_conf_score'] = predictions.predicted_labels.conf_score
            
            print(f"✓ CellTypist annotation complete")
        
        # Print summary
        print(f"  Total: {len(adata)} cells")
        print(f"  Found {adata.obs['celltypist_predicted'].nunique()} unique cell types")
        print(f"  Top 5 cell types:")
        for ct, count in adata.obs['celltypist_predicted'].value_counts().head(5).items():
            print(f"    - {ct}: {count} cells")
        
        return adata
        
    except ImportError:
        print("⚠ CellTypist not available. Install with: pip install celltypist")
        return adata
    except Exception as e:
        print(f"⚠ CellTypist annotation failed: {str(e)}")
        import traceback
        traceback.print_exc()
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
