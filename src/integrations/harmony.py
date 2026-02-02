# CRITICAL: Set thread limits BEFORE any imports
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import scanpy as sc
import anndata as ad
from typing import Union, Dict
from .base import IntegrationMethod
from utils.gpu import get_device, is_gpu_available, is_rapids_available, print_device_info


class HarmonyIntegration(IntegrationMethod):
    """
    Harmony integration wrapper with fallback chain:
    1. rapids-singlecell (GPU, fastest, no OpenBLAS issues)
    2. harmony-pytorch (GPU/CPU, with thread safety)
    3. scanpy/harmonypy (CPU fallback)
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__("Harmony", use_gpu=use_gpu)

    def check_dependencies(self) -> bool:
        # Check in priority order
        try:
            import rapids_singlecell as rsc
            return True
        except ImportError:
            pass
        
        try:
            from harmony import harmonize
            return True
        except ImportError:
            pass
        
        try:
            import harmonypy
            return True
        except ImportError:
            pass

        print("Warning: No Harmony implementation found (rapids_singlecell, harmony-pytorch, or harmonypy).")
        return False

    def run(self, adata: Union[ad.AnnData, Dict[str, ad.AnnData]], batch_key: str, **kwargs) -> ad.AnnData:
        """
        Run Harmony integration with intelligent fallback.
        Priority: rapids-singlecell > harmony-pytorch > scanpy/harmonypy
        """
        # Ensure single AnnData
        adata = self._prepare_input(adata, batch_key)
        
        device = get_device(self.use_gpu)
        print("\n--- Harmony Integration Setup ---")
        print_device_info()
        print(f"Target Device for Harmony: {device}")
        
        # Ensure PCA is computed
        if 'X_pca' not in adata.obsm:
            print("PCA not found. Computing PCA...")
            sc.tl.pca(adata)
        
        # PRIORITY 1: rapids-singlecell (GPU, no OpenBLAS issues, fastest)
        if self.use_gpu and is_rapids_available():
            try:
                import rapids_singlecell as rsc
                print("Using rapids-singlecell.pp.harmony_integrate (GPU-accelerated)...")
                
                rsc.pp.harmony_integrate(
                    adata,
                    key=batch_key,
                    basis='X_pca',
                    adjusted_basis='X_pca_harmony',
                    correction_method='fast',  # Faster improved method
                    **kwargs
                )
                print("✓ Harmony integration complete (Provider: rapids-singlecell, GPU: True).")
                return adata
                
            except Exception as e:
                print(f"rapids-singlecell failed: {e}")
                print("Falling back to harmony-pytorch...")
        
        # PRIORITY 2: harmony-pytorch (with thread safety)
        try:
            from harmony import harmonize
            use_gpu_flag = (device != 'cpu')
            print(f"Using harmony-pytorch (use_gpu={use_gpu_flag})...")

            # Prepare data
            X = adata.obsm['X_pca']
            batch_mat = adata.obs[[batch_key]]
            
            # Convert to numpy if needed
            if hasattr(X, 'get'):  # CuPy array
                X = X.get()
            elif hasattr(X, 'to_numpy'):  # DataFrame
                X = X.to_numpy()
            
            # Enforce strict thread limits to prevent OpenBLAS crashes
            try:
                from threadpoolctl import threadpool_limits
                thread_limit_ctx = threadpool_limits(limits=1, user_api='blas')
            except ImportError:
                print("Warning: threadpoolctl not found. Install with: pip install threadpoolctl")
                from contextlib import nullcontext
                thread_limit_ctx = nullcontext()
            
            with thread_limit_ctx:
                Z_corr = harmonize(X, batch_mat, batch_key=batch_key, use_gpu=use_gpu_flag, **kwargs)
            
            adata.obsm['X_pca_harmony'] = Z_corr
            print(f"✓ Harmony integration complete (Provider: harmony-pytorch, GPU: {use_gpu_flag}).")
            return adata
            
        except ImportError:
            print("harmony-pytorch not found. Falling back to scanpy/harmonypy...")
        except Exception as e:
            print(f"harmony-pytorch failed: {e}")
            print("Falling back to scanpy/harmonypy...")
        
        # PRIORITY 3: Scanpy fallback (CPU only, requires harmonypy)
        try:
            print("Using scanpy.external.pp.harmony_integrate (CPU)...")
            sc.external.pp.harmony_integrate(
                adata, 
                key=batch_key, 
                basis='X_pca', 
                adjusted_basis='X_pca_harmony',
                **kwargs
            )
            print("✓ Harmony integration complete (Provider: scanpy/harmonypy, CPU).")
            return adata
            
        except Exception as e:
            raise RuntimeError(f"All Harmony implementations failed. Last error: {e}")
