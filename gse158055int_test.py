import src.utils.env_setup
import os
import scanpy as sc
import anndata as ad
import pandas as pd
import warnings

from src.data.loader import DatasetLoader
from src.data.preprocessing import Preprocessor
from src.data.annotation import AnnotationManager, CellTypistAnnotator
from src.integrations.harmony import HarmonyIntegration
from src.integrations.scvi import SCVIIntegration
from src.integrations.scgen import SCGENIntegration
from src.evaluation.evaluator import IntegrationEvaluator
from src.visualization.plot import IntegrationVisualizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("=======================================================")
    print("   GSE158055 Full Pipeline Test (Integration & Eval)   ")
    print("=======================================================")

    # 1. Load Data
    dataset_path = os.path.join(os.path.dirname(__file__), 'GSE158055_subsampled.h5ad')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"\n[1] Loading data from {dataset_path}...")
    loader = DatasetLoader(dataset_path=dataset_path)
    adata = loader.get_adata()
    print(f"    Loaded {adata.n_obs} cells and {adata.n_vars} genes.")
    
    # Define keys
    batch_key = 'sampleID'
    label_key = 'celltype' # Ground truth for evaluation
    
    if batch_key not in adata.obs.columns:
        print(f"Error: Batch key '{batch_key}' not found in adata.obs")
        return
    if label_key not in adata.obs.columns:
        print(f"Error: Label key '{label_key}' not found in adata.obs")
        return

    # 2. Preprocessing
    print(f"\n[2] Preprocessing (Batch Key: {batch_key})...")
    # Initialize preprocessor (handles splitting internally if needed, but we keep it global for simple pipeline)
    # GPU is NOT enabled for preprocessing
    pp = Preprocessor(adata, batch_key=batch_key)
    
    # Check status
    pp.check_normalization_status()
    
    # Normalize and Log1p
    adata = pp.normalize_and_log1p()
    
    # HVG
    adata = pp.select_hvg(n_top_genes=2000, batch_key=batch_key)
    
    # PCA (needed for some integrations and initial visualization)
    print("    Computing PCA...")
    sc.tl.pca(adata)
    
    # 3. Annotation
    print("\n[3] Annotation...")
    # We use CellTypist
    annotator = CellTypistAnnotator(use_gpu=True) # CellTypist supports GPU over-clustering
    manager = AnnotationManager(annotator)
    
    # Annotate and store in 'predicted_celltype'
    # Note: CellTypistAnnotator.annotate usually returns a new adata or modifies in place
    # Our wrapper returns adata.
    # We want to store it in a specific column.
    # The AnnotationManager.run_annotation checks if key exists.
    
    # Let's run it manually to ensure we control the key
    # Actually AnnotationManager.run_annotation takes annotation_key
    adata = manager.run_annotation(adata, annotation_key='predicted_celltype')
    
    if 'predicted_celltype' in adata.obs.columns:
        print(f"    Annotation successful. Labels in 'predicted_celltype'.")
        # Sync splits to ensure they have the new annotation
        pp.sync_splits()
    else:
        print("    Annotation might have failed or been skipped.")

    # 4. Integration
    print("\n[4] Integration...")
    
    # A. Harmony
    print("    --- Harmony ---")
    harmony = HarmonyIntegration(use_gpu=True)
    adata = harmony.run(adata, batch_key=batch_key)
    
    # B. scVI
    print("    --- scVI ---")
    # scVI needs raw counts usually. 
    # If we normalized in place, X is normalized.
    # Hopefully 'counts' layer exists or we use X (suboptimal but works for test).
    # DatasetLoader doesn't strictly enforce 'counts' layer creation yet, 
    # but let's check.
    if 'counts' not in adata.layers:
        # If we don't have counts, scVI might warn. 
        # For this test, we proceed.
        pass
        
    scvi_int = SCVIIntegration(use_gpu=True)
    if scvi_int.check_dependencies():
        adata = scvi_int.run(adata, batch_key=batch_key, max_epochs=10) # Low epochs for speed
    
    # C. SCGEN
    print("    --- SCGEN ---")
    scgen_int = SCGENIntegration(use_gpu=True)
    if scgen_int.check_dependencies():
        # SCGEN needs labels. We use ground truth 'celltype' for best integration training
        adata = scgen_int.run(adata, batch_key=batch_key, cell_type_key=label_key, max_epochs=10)


    # Save the integrated dataset
    adata.write_h5ad("gse158055_integrated.h5ad")   
    
    
    # 5. Evaluation
    print("\n[5] Evaluation...")
    evaluator = IntegrationEvaluator(adata)
    if evaluator.check_dependencies():
        # Compare Uncorrected (PCA), Harmony, scVI, SCGEN
        keys_to_eval = ['X_pca', 'X_pca_harmony', 'X_scVI', 'X_pca_scgen']
        
        results_df = evaluator.evaluate_all(
            keys_to_eval,
            batch_key=batch_key,
            label_key=label_key
        )
        
        print("\n    Evaluation Results:")
        print(results_df)
        
        # Save results
        results_path = os.path.join(os.path.dirname(__file__), 'integration_results.csv')
        results_df.to_csv(results_path)
        print(f"    Saved results to {results_path}")

    # 6. Visualization
    print("\n[6] Visualization...")
    viz = IntegrationVisualizer(adata)
    
    # Compute UMAPs for all available embeddings
    keys_to_plot = ['X_pca', 'X_pca_harmony', 'X_scVI', 'X_pca_scgen']
    # Filter keys that exist
    keys_to_plot = [k for k in keys_to_plot if k in adata.obsm]
    
    viz.compute_umaps(keys_to_plot)
    
    # Plot
    plot_path = os.path.join(os.path.dirname(__file__), 'integration_comparison.pdf')
    viz.plot_umaps(color_by=[batch_key, label_key, 'predicted_celltype'], save_path=plot_path, show=False)
    print(f"    Saved plots to {plot_path}")

    print("\n=======================================================")
    print("               Pipeline Test Completed                 ")
    print("=======================================================")

if __name__ == "__main__":
    main()
