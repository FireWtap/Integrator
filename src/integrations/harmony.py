import src.utils.env_setup
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
        # Check for either harmony-pytorch (GPU) or harmony (CPU)
        try:
            import harmony
            return True
        except ImportError:
            pass
            
        try:
            from harmony import harmonize
            return True
        except ImportError:
            print("Warning: Neither 'harmony-pytorch' nor 'harmony' package found.")
            return False

    def run(self, adata: Union[ad.AnnData, Dict[str, ad.AnnData]], batch_key: str, **kwargs) -> ad.AnnData:
        """
        Run Harmony integration.
        """
        # Ensure single AnnData
        adata = self._prepare_input(adata, batch_key)
        
        device = get_device(self.use_gpu)
        print(f"Running Harmony integration on '{batch_key}' (Target Device: {device})...")
        
        # Ensure PCA is computed
        if 'X_pca' not in adata.obsm:
            print("PCA not found. Computing PCA...")
            sc.tl.pca(adata)
            
        # Try GPU implementation first if requested
        if device in ['cuda', 'mps']:
            # RAPIDS check (CUDA only)
            # User requested to avoid RAPIDS
            # if device == 'cuda' and is_rapids_available():
            #     try:
            #         import rapids_singlecell as rsc
            #         ...
            #     except Exception as e:
            #         print(f"RAPIDS Harmony failed: {e}. Trying harmony-pytorch...")

            try:
                from harmony import harmonize
                print(f"Using harmony-pytorch on {device}...")
                
                # harmony-pytorch expects the matrix and batch info
                # It returns the corrected PCA matrix
                
                # Prepare data
                X = adata.obsm['X_pca']
                batch_mat = adata.obs[[batch_key]]
                
                # Debug info
                print(f"  X type: {type(X)}")
                print(f"  batch_mat type: {type(batch_mat)}")
                
                # Ensure X is numpy array (if it's cupy or tensor)
                if hasattr(X, 'get'): # Cupy
                    print("  Converting Cupy to Numpy...")
                    X = X.get()
                elif hasattr(X, 'numpy'): # Tensor
                    print("  Converting Tensor to Numpy...")
                    X = X.numpy()
                elif hasattr(X, 'to_numpy'): # DataFrame/Series
                    X = X.to_numpy()
                    
                # Run harmonize
                # harmony-pytorch might not support MPS explicitly in older versions, 
                # but usually falls back or works if torch tensors are on MPS.
                # We'll let it try.
                
                import traceback
                try:
                    Z_corr = harmonize(X, batch_mat, batch_key=batch_key, use_gpu=(device != 'cpu'), **kwargs)
                    adata.obsm['X_pca_harmony'] = Z_corr
                    print("✓ Harmony (GPU) integration complete.")
                    return adata
                except Exception as inner_e:
                    print(f"  harmonize() call failed: {inner_e}")
                    traceback.print_exc()
                    raise inner_e
                
            except ImportError:
                print("harmony-pytorch not installed. Falling back to CPU implementation.")
            except Exception as e:
                print(f"GPU integration failed: {e}. Falling back to CPU.")

        # CPU Fallback (scanpy external)
        try:
            print("Using scanpy.external.pp.harmony_integrate (CPU)...")
            sc.external.pp.harmony_integrate(
                adata, 
                key=batch_key, 
                basis='X_pca', 
                adjusted_basis='X_pca_harmony',
                **kwargs
            )
            print("✓ Harmony (CPU) integration complete.")
        except Exception as e:
            raise RuntimeError(f"Harmony integration failed: {e}")
            
        return adata
