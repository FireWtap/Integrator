import utils.env_setup
import scanpy as sc
import anndata as ad
from typing import Union, Dict
from .base import IntegrationMethod
from utils.gpu import get_device, is_gpu_available, is_rapids_available

class HarmonyIntegration(IntegrationMethod):
    """
    Harmony integration wrapper.
    Supports harmony-pytorch for GPU acceleration.
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__("Harmony", use_gpu=use_gpu)

    def check_dependencies(self) -> bool:
        # Check for harmony-pytorch (works on CPU/GPU) or harmony (CPU)
        try:
            from harmony import harmonize
            return True
        except ImportError:
            pass

        # Fallback check for scanpy's wrapper requirements
        try:
            import harmonypy
            return True
        except ImportError:
            pass

        print("Warning: Neither 'harmony-pytorch' nor 'harmonypy' package found.")
        return False

    def run(self, adata: Union[ad.AnnData, Dict[str, ad.AnnData]], batch_key: str, **kwargs) -> ad.AnnData:
        """
        Run Harmony integration.
        Prioritizes harmony-pytorch if installed (faster, supports GPU/CPU).
        Falls back to scanpy (harmonypy) otherwise.
        """
        # Ensure single AnnData
        adata = self._prepare_input(adata, batch_key)
        
        device = get_device(self.use_gpu)
        print(f"Running Harmony integration on '{batch_key}' (Target Device: {device})...")
        
        # Ensure PCA is computed
        if 'X_pca' not in adata.obsm:
            print("PCA not found. Computing PCA...")
            sc.tl.pca(adata)
            
        # Try harmony-pytorch first (works for both GPU and CPU if installed)
        try:
            from harmony import harmonize
            use_gpu_flag = (device != 'cpu')
            print(f"Using harmony-pytorch (use_gpu={use_gpu_flag})...")

            # Prepare data
            X = adata.obsm['X_pca']
            batch_mat = adata.obs[[batch_key]]
            
            # Ensure X is numpy/torch array
            if hasattr(X, 'get'): # Cupy
                X = X.get()
            elif hasattr(X, 'to_numpy'): # DataFrame
                X = X.to_numpy()
            
            # Run harmonize
            import traceback
            try:
                Z_corr = harmonize(X, batch_mat, batch_key=batch_key, use_gpu=use_gpu_flag, **kwargs)
                adata.obsm['X_pca_harmony'] = Z_corr
                print(f"✓ Harmony integration complete (Provider: harmony-pytorch, GPU: {use_gpu_flag}).")
                return adata
            except Exception as inner_e:
                print(f"  harmonize() call failed: {inner_e}")
                traceback.print_exc()
                raise inner_e

        except ImportError:
            # Fallback to Scanpy (needs harmonypy)
            print("harmony-pytorch not found. Falling back to scanpy.external.pp.harmony_integrate...")
            try:
                sc.external.pp.harmony_integrate(
                    adata, 
                    key=batch_key, 
                    basis='X_pca', 
                    adjusted_basis='X_pca_harmony',
                    **kwargs
                )
                print("✓ Harmony integration complete (Provider: Scanpy/harmonypy).")
            except Exception as e:
                raise RuntimeError(f"Harmony integration failed: {e}")
            
        return adata
