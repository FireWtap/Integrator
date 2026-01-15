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

print("\n--- Using harmony.harmonize directly (FIXED) ---")
try:
    # Use harmony.harmonize directly instead of broken scanpy wrapper
    Z_corrected = harmony.harmonize(
        adata.obsm['X_pca'], 
        adata.obs, 
        batch_key='batch'
    )
    adata.obsm['X_pca_harmony'] = Z_corrected
    print("✓ Harmony integration SUCCESS")
    print(f"✓ Result shape: {adata.obsm['X_pca_harmony'].shape}")
    print(f"✓ Expected shape: ({n_obs}, {n_pca})")
except Exception as e:
    print(f"✗ Harmony integration FAILED: {e}")
    import traceback
    traceback.print_exc()

# Verify the result
print("\n--- Verification ---")
print(f"Original PCA shape: {adata.obsm['X_pca'].shape}")
print(f"Harmony corrected shape: {adata.obsm['X_pca_harmony'].shape}")
print(f"Batch distribution: {adata.obs['batch'].value_counts().to_dict()}")
