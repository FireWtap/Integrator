"""
Cell type annotation module.
Provides a base class for annotators and implementations for specific tools (e.g., CellTypist).
"""

from abc import ABC, abstractmethod
import scanpy as sc
import anndata as ad
import pandas as pd
from typing import Optional, Union, List

class BaseAnnotator(ABC):
    """
    Abstract base class for cell type annotators.
    """
    
    @abstractmethod
    def annotate(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Annotate the AnnData object with cell types.
        
        Parameters:
        -----------
        adata : AnnData
            The dataset to annotate.
            
        Returns:
        --------
        AnnData : The annotated dataset (potentially modified in-place or a copy).
        """
        pass

class CellTypistAnnotator(BaseAnnotator):
    """
    Annotator using CellTypist (https://github.com/Teichlab/celltypist).
    """
    
    def __init__(self, model: str = 'Immune_All_Low.pkl', majority_voting: bool = True, mode: str = 'best match', p_thres: float = 0.5, use_gpu: bool = False):
        """
        Initialize CellTypistAnnotator.
        
        Parameters:
        -----------
        model : str
            Model name to use (e.g., 'Immune_All_Low.pkl', 'Cells_Fetal_Lung.pkl').
            See celltypist.models.models_path for available models.
        majority_voting : bool
            Whether to refine predictions using majority voting within clusters.
        mode : str
            Prediction mode ('best match' or 'prob match').
        p_thres : float
            Probability threshold for 'prob_match' mode.
        use_gpu : bool
            Whether to use GPU acceleration (if supported by underlying model/tools).
        """
        self.model = model
        self.majority_voting = majority_voting
        self.mode = mode
        self.p_thres = p_thres
        self.use_gpu = use_gpu
        
    def annotate(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Run CellTypist annotation.
        """
        print(f"\n=== Running CellTypist Annotation ===")
        print(f"Model: {self.model}")
        print(f"GPU Acceleration: {self.use_gpu}")
        
        try:
            import celltypist
        except ImportError:
            raise ImportError("CellTypist is not installed. Please install it with `pip install celltypist`.")
        
        # CellTypist requires normalized data (log1p) usually.
        # We assume adata is already normalized/log1p-ed by Preprocessor.
        
        # Helper to check if we can actually use GPU
        use_gpu_arg = self.use_gpu
        if use_gpu_arg:
            try:
                import rapids_singlecell
            except ImportError:
                print("Warning: GPU requested but 'rapids_singlecell' not found. Falling back to CPU for over-clustering.")
                use_gpu_arg = False

        print("Predicting cell types...")
        # Note: celltypist.annotate signature varies, but typically:
        # annotate(filename, model=..., majority_voting=..., use_GPU=...)
        # We'll pass use_GPU if it allows, or rely on it handling kwargs.
        # Recent versions support use_GPU for over-clustering.
        
        try:
            predictions = celltypist.annotate(
                adata, 
                model=self.model, 
                majority_voting=self.majority_voting, 
                mode=self.mode,
                p_thres=self.p_thres,
                use_GPU=use_gpu_arg # Attempt to pass GPU flag (note: param might be use_gpu or use_GPU depending on version)
            )
        except TypeError:
             # Fallback for older versions that might not accept use_GPU
            print("Warning: `use_GPU` argument not accepted by installed celltypist version. Running without explicit GPU flag.")
            predictions = celltypist.annotate(
                adata, 
                model=self.model, 
                majority_voting=self.majority_voting, 
                mode=self.mode,
                p_thres=self.p_thres
            )
            
        # Transfer results to adata
        # predictions.predicted_labels is a DataFrame
        adata = predictions.to_adata(prefix='')
        
        print(f"✓ Annotation complete")
        print(f"  Added columns: {', '.join([c for c in adata.obs.columns if 'predicted_labels' in c or 'conf_score' in c])}")
        
        return adata

class AnnotationManager:
    """
    Manager to handle annotation workflows.
    """
    
    def __init__(self, annotator: Optional[BaseAnnotator] = None):
        self.annotator = annotator
        
    def check_annotation(self, adata: ad.AnnData, annotation_key: str) -> bool:
        """
        Check if annotation column exists.
        """
        if annotation_key in adata.obs.columns:
            print(f"✓ Annotation column '{annotation_key}' found.")
            return True
        print(f"⚠ Annotation column '{annotation_key}' NOT found.")
        return False
        
    def run_annotation(self, adata: ad.AnnData, annotation_key: str = 'cell_type') -> ad.AnnData:
        """
        Run annotation if needed.
        """
        if self.check_annotation(adata, annotation_key):
            print("Skipping annotation (already present).")
            return adata
            
        if self.annotator is None:
            raise ValueError("No annotator provided and annotation column missing.")
            
        print("Running annotator...")
        adata = self.annotator.annotate(adata)
        
        # Standardize column name if possible, or just let the user know
        # CellTypist usually outputs 'predicted_labels' or 'majority_voting'
        
        # If we want to map the result to 'annotation_key', we can do it here
        source_key = 'majority_voting' if isinstance(self.annotator, CellTypistAnnotator) and self.annotator.majority_voting else 'predicted_labels'
        
        if source_key in adata.obs.columns and source_key != annotation_key:
            print(f"Mapping '{source_key}' to '{annotation_key}'")
            adata.obs[annotation_key] = adata.obs[source_key]
            
        return adata
