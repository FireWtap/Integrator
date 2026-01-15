
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import harmony
    print(f"Harmony module found: {harmony}")
    print(f"Harmony file: {harmony.__file__}")
    print(f"Harmony dir: {dir(harmony)}")
except ImportError:
    print("Harmony module NOT found.")

# Create dummy AnnData
n_obs = 100
n_vars = 50
n_pca = 20

X = np.random.rand(n_obs, n_vars)
obs = pd.DataFrame({
    'batch': np.random.choice(['A', 'B'], n_obs),
    'celltype': np.random.choice(['T', 'B'], n_obs)
}, index=[f"cell_{i}" for i in range(n_obs)])

adata = ad.AnnData(X=X, obs=obs)
adata.obsm['X_pca'] = np.random.rand(n_obs, n_pca)

print("\n--- Testing scanpy.external.pp.harmony_integrate ---")
try:
    sc.external.pp.harmony_integrate(adata, key='batch', basis='X_pca', adjusted_basis='X_pca_harmony')
    print("Scanpy harmony_integrate SUCCESS")
    print(f"Result shape: {adata.obsm['X_pca_harmony'].shape}")
except Exception as e:
    print(f"Scanpy harmony_integrate FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Testing harmony.harmonize (if available) ---")
if 'harmony' in locals() and hasattr(harmony, 'harmonize'):
    try:
        Z = harmony.harmonize(adata.obsm['X_pca'], adata.obs[['batch']], batch_key='batch')
        print("harmony.harmonize SUCCESS")
        print(f"Result shape: {Z.shape}")
    except Exception as e:
        print(f"harmony.harmonize FAILED: {e}")
else:
    print("harmony.harmonize NOT available.")

print("\n--- Testing harmony.run_harmony (if available) ---")
if 'harmony' in locals() and hasattr(harmony, 'run_harmony'):
    try:
        # harmonypy specific signature might be needed
        # run_harmony(data_mat, meta_data, vars_use, ...)
        # Just checking existence primarily
        print("harmony.run_harmony exists.")
    except Exception as e:
        print(f"harmony.run_harmony FAILED: {e}")
else:
    print("harmony.run_harmony NOT available.")
