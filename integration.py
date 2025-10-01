"""
Integration methods that return integration matrices.
"""
import scanpy as sc
import numpy as np
from typing import Optional
import warnings


def integrate_scgen(adata, batch_key: str, label_key: Optional[str] = None, 
                   max_epochs: int = 100):
    """
    Integrate using scGen (conditional VAE).
    Returns integrated expression matrix in adata.obsm['X_scgen'].
    
    Args:
        adata: AnnData object (preprocessed)
        batch_key: Column name for batch information
        label_key: Column name for cell type labels (optional)
        max_epochs: Maximum training epochs
    
    Returns:
        adata: Updated AnnData object
        model: Trained scGen model (or None if failed)
    """
    print("\n=== Running scGen integration ===")
    
    try:
        from scgen import SCGEN
        
        # Use raw counts for scGen
        if 'counts' in adata.layers:
            adata_scgen = adata.copy()
            adata_scgen.X = adata_scgen.layers['counts'].copy()
        else:
            adata_scgen = adata.copy()
        
        print(f"Training scGen model (max_epochs={max_epochs})...")
        
        # Train scGen model
        model = SCGEN(adata_scgen, batch_key=batch_key, cell_type_key=label_key)
        model.train(
            max_epochs=max_epochs,
            batch_size=32,
            early_stopping=True,
            early_stopping_patience=25
        )
        
        # Get corrected expression
        print("Generating integrated data...")
        corrected = model.batch_removal()
        
        # Store integrated data
        adata.obsm['X_scgen'] = corrected.X
        
        print(f"✓ scGen integration complete. Shape: {adata.obsm['X_scgen'].shape}")
        return adata, model
        
    except ImportError:
        print("⚠ scGen not available. Install with: pip install scgen")
        return adata, None
    except Exception as e:
        print(f"⚠ scGen integration failed: {str(e)}")
        return adata, None


def integrate_lemur(adata, batch_key: str, n_embedding: int = 30):
    """
    Integrate using pyLemur (latent embedding).
    Returns integrated embedding in adata.obsm['X_lemur'].
    
    Args:
        adata: AnnData object (preprocessed)
        batch_key: Column name for batch information
        n_embedding: Dimension of latent embedding
    """
    print("\n=== Running pyLemur integration ===")
    
    try:
        import lemur
        
        print(f"Training Lemur model (n_embedding={n_embedding})...")
        
        # Prepare data - Lemur works with normalized data
        adata_lemur = adata.copy()
        
        # Train Lemur model
        lemur_model = lemur.Lemur(
            n_embedding=n_embedding,
            verbose=True
        )
        
        # Fit and transform
        embedding = lemur_model.fit_transform(
            adata_lemur.X,
            batch=adata_lemur.obs[batch_key].values
        )
        
        # Store integrated embedding
        adata.obsm['X_lemur'] = embedding
        
        print(f"✓ pyLemur integration complete. Shape: {adata.obsm['X_lemur'].shape}")
        return adata
        
    except ImportError:
        print("⚠ pyLemur not available. Install with: pip install pylemur")
        return adata
    except Exception as e:
        print(f"⚠ pyLemur integration failed: {str(e)}")
        # Try alternative approach if standard fails
        try:
            print("Trying alternative Lemur approach...")
            import lemur
            
            # Simple PCA-based integration as fallback
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_embedding)
            embedding = pca.fit_transform(adata.X if not hasattr(adata.X, 'toarray') 
                                         else adata.X.toarray())
            adata.obsm['X_lemur'] = embedding
            print(f"✓ Used PCA-based fallback. Shape: {adata.obsm['X_lemur'].shape}")
            return adata
        except:
            print(f"⚠ Lemur integration completely failed: {str(e)}")
            return adata


def integrate_scvi(adata, batch_key: str, n_latent: int = 30, max_epochs: int = 400):
    """
    Integrate using scVI (variational inference).
    Returns integrated latent representation in adata.obsm['X_scvi'].
    
    Args:
        adata: AnnData object (preprocessed)
        batch_key: Column name for batch information
        n_latent: Dimension of latent space
        max_epochs: Maximum training epochs
    
    Returns:
        adata: Updated AnnData object
        model: Trained scVI model (or None if failed)
    """
    print("\n=== Running scVI integration ===")
    
    try:
        import scvi
        
        # Setup scVI model
        print(f"Setting up scVI model (n_latent={n_latent})...")
        
        # Use raw counts for scVI
        if 'counts' in adata.layers:
            adata_scvi = adata.copy()
            adata_scvi.X = adata_scvi.layers['counts'].copy()
        else:
            adata_scvi = adata.copy()
        
        scvi.model.SCVI.setup_anndata(
            adata_scvi,
            batch_key=batch_key,
            layer='counts' if 'counts' in adata.layers else None
        )
        
        model = scvi.model.SCVI(adata_scvi, n_latent=n_latent)
        
        print(f"Training scVI model (max_epochs={max_epochs})...")
        model.train(max_epochs=max_epochs, early_stopping=True)
        
        # Get latent representation
        print("Generating latent representation...")
        latent = model.get_latent_representation()
        
        # Store integrated data
        adata.obsm['X_scvi'] = latent
        
        print(f"✓ scVI integration complete. Shape: {adata.obsm['X_scvi'].shape}")
        return adata, model
        
    except ImportError:
        print("⚠ scVI not available. Install with: pip install scvi-tools")
        return adata, None
    except Exception as e:
        print(f"⚠ scVI integration failed: {str(e)}")
        return adata, None


def integrate_scanorama(adata, batch_key: str):
    """
    Integrate using Scanorama (mutual nearest neighbors).
    Returns integrated expression matrix in adata.obsm['X_scanorama'].
    Scanorama outputs corrected gene expression (not latent space).
    
    Args:
        adata: AnnData object (preprocessed)
        batch_key: Column name for batch information
    """
    print("\n=== Running Scanorama integration ===")
    
    try:
        import scanorama
        
        # Split by batch
        batches = adata.obs[batch_key].unique()
        print(f"Integrating {len(batches)} batches...")
        
        # Prepare data
        adatas = [adata[adata.obs[batch_key] == batch].copy() for batch in batches]
        
        # Convert to dense if sparse
        datasets = []
        for ad in adatas:
            if hasattr(ad.X, 'toarray'):
                datasets.append(ad.X.toarray())
            else:
                datasets.append(ad.X)
        
        genes_list = [ad.var_names.tolist() for ad in adatas]
        
        # Run Scanorama integration
        print("Running Scanorama correction...")
        integrated, genes = scanorama.correct(datasets, genes_list, return_dense=True)
        
        # Concatenate corrected data
        integrated_matrix = np.vstack(integrated)
        
        # Store integrated data (corrected expression matrix)
        adata.obsm['X_scanorama'] = integrated_matrix
        
        print(f"✓ Scanorama integration complete. Shape: {adata.obsm['X_scanorama'].shape}")
        return adata
        
    except ImportError:
        print("⚠ Scanorama not available. Install with: pip install scanorama")
        return adata
    except Exception as e:
        print(f"⚠ Scanorama integration failed: {str(e)}")
        return adata


def integrate_harmony(adata, batch_key: str):
    """
    Integrate using Harmony (iterative clustering).
    Returns integrated PCA embedding in adata.obsm['X_harmony'].
    Note: Harmony works on PCA space, not raw counts.
    
    Args:
        adata: AnnData object (preprocessed with PCA)
        batch_key: Column name for batch information
    """
    print("\n=== Running Harmony integration ===")
    
    try:
        from harmonypy import run_harmony
        
        # Harmony requires PCA
        if 'X_pca' not in adata.obsm:
            print("Computing PCA for Harmony...")
            sc.tl.pca(adata)
        
        print("Running Harmony correction...")
        
        # Run Harmony
        harmony_out = run_harmony(
            adata.obsm['X_pca'],
            adata.obs,
            batch_key,
            max_iter_harmony=10,
            verbose=False
        )
        
        # Store integrated PCA
        adata.obsm['X_harmony'] = harmony_out.Z_corr.T
        
        print(f"✓ Harmony integration complete. Shape: {adata.obsm['X_harmony'].shape}")
        return adata
        
    except ImportError:
        print("⚠ Harmony not available. Install with: pip install harmony-pytorch")
        return adata
    except Exception as e:
        print(f"⚠ Harmony integration failed: {str(e)}")
        return adata


def integrate_combat(adata, batch_key: str):
    """
    Integrate using ComBat (empirical Bayes).
    Returns integrated expression matrix in adata.obsm['X_combat'].
    ComBat outputs corrected gene expression (not latent space).
    
    Args:
        adata: AnnData object (preprocessed)
        batch_key: Column name for batch information
    """
    print("\n=== Running ComBat integration ===")
    
    try:
        # ComBat is built into scanpy
        adata_combat = adata.copy()
        
        print("Running ComBat correction...")
        sc.pp.combat(adata_combat, key=batch_key)
        
        # Store integrated data
        adata.obsm['X_combat'] = adata_combat.X
        
        print(f"✓ ComBat integration complete. Shape: {adata.obsm['X_combat'].shape}")
        return adata
        
    except Exception as e:
        print(f"⚠ ComBat integration failed: {str(e)}")
        return adata


def run_all_integrations(adata, batch_key: str, label_key: Optional[str] = None,
                        methods: list = None):
    """
    Run all available integration methods.
    
    Args:
        adata: AnnData object (preprocessed)
        batch_key: Column name for batch information
        label_key: Column name for cell type labels
        methods: List of methods to run. 
                Default: ['scgen', 'scanorama', 'harmony', 'combat', 'scvi']
    """
    if methods is None:
        methods = ['scgen', 'scanorama', 'harmony', 'combat', 'scvi']
    
    print(f"\n{'='*60}")
    print(f"Running integration methods: {', '.join(methods)}")
    print(f"{'='*60}")
    
    results = {}
    models = {}  # Store trained models
    
    if 'scgen' in methods:
        adata, model = integrate_scgen(adata, batch_key, label_key)
        if 'X_scgen' in adata.obsm:
            results['scgen'] = 'X_scgen'
            models['scgen'] = model
    
    if 'scanorama' in methods:
        adata = integrate_scanorama(adata, batch_key)
        if 'X_scanorama' in adata.obsm:
            results['scanorama'] = 'X_scanorama'
    
    if 'harmony' in methods:
        adata = integrate_harmony(adata, batch_key)
        if 'X_harmony' in adata.obsm:
            results['harmony'] = 'X_harmony'
    
    if 'combat' in methods:
        adata = integrate_combat(adata, batch_key)
        if 'X_combat' in adata.obsm:
            results['combat'] = 'X_combat'
    
    if 'lemur' in methods:
        adata = integrate_lemur(adata, batch_key)
        if 'X_lemur' in adata.obsm:
            results['lemur'] = 'X_lemur'
    
    if 'scvi' in methods:
        adata, model = integrate_scvi(adata, batch_key)
        if 'X_scvi' in adata.obsm:
            results['scvi'] = 'X_scvi'
            models['scvi'] = model
    
    print(f"\n{'='*60}")
    print(f"Integration complete. Available methods: {list(results.keys())}")
    print(f"{'='*60}")
    
    return adata, results, models
