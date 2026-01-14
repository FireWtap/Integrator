import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union

class IntegrationEvaluator:
    """
    Evaluator for comparing integration methods using scib-metrics (accelerated).
    Follows OpenProblems metrics guidelines where possible.
    """
    
    def __init__(self, adata: ad.AnnData):
        self.adata = adata

    def check_dependencies(self) -> bool:
        try:
            import scib_metrics
            return True
        except ImportError:
            print("Warning: 'scib-metrics' package not found. Please install it.")
            return False

    def evaluate_embedding(self, embedding_key: str, batch_key: str, label_key: str) -> Dict[str, float]:
        """
        Compute metrics for a specific embedding using scib-metrics.
        """
        print(f"Evaluating embedding '{embedding_key}'...")
        
        if embedding_key not in self.adata.obsm:
            print(f"Warning: Embedding '{embedding_key}' not found. Skipping.")
            return {}

        results = {}
        
        try:
            from scib_metrics import (
                # Biological conservation
                silhouette_label, 
                nmi_ari_cluster_labels_leiden,
                graph_connectivity, 
                isolated_labels,
                # Batch correction
                silhouette_batch, 
                principal_component_regression,
                # kbet # kBET might not be in the init if older version, but let's try
            )
            # Try specific imports for optional ones
            try:
                from scib_metrics import kbet
            except ImportError:
                kbet = None

        except ImportError as e:
            print(f"Error importing metrics from scib_metrics: {e}")
            return {}

        X_emb = self.adata.obsm[embedding_key]
        
        # Ensure obs columns are strings/categories
        labels = self.adata.obs[label_key].astype(str)
        batch = self.adata.obs[batch_key].astype(str)
        
        # 1. Biological Conservation
        print("  [Bio] Computing Silhouette Label (ASW)...")
        try:
            results['Bio_ASW_Label'] = silhouette_label(X_emb, labels, rescale=True)
        except Exception as e: 
            print(f"    Failed: {e}")
            results['Bio_ASW_Label'] = np.nan

        print("  [Bio] Computing NMI & ARI...")
        try:
            # Performs Leiden clustering on X_emb and compares to labels
            nmi_ari = nmi_ari_cluster_labels_leiden(X_emb, labels, optimize_resolution=True)
            results['Bio_NMI'] = nmi_ari['nmi']
            results['Bio_ARI'] = nmi_ari['ari']
        except Exception as e: 
            print(f"    Failed: {e}")
            results['Bio_NMI'] = np.nan
            results['Bio_ARI'] = np.nan

        print("  [Bio] Computing Graph Connectivity...")
        try:
            results['Bio_Graph_Conn'] = graph_connectivity(X_emb, labels)
        except Exception as e: 
            print(f"    Failed: {e}")
            results['Bio_Graph_Conn'] = np.nan
        
        print("  [Bio] Computing Isolated Labels (F1)...")
        try:
            results['Bio_Iso_F1'] = isolated_labels(X_emb, labels, batch)
        except Exception as e: 
            print(f"    Failed: {e}")
            results['Bio_Iso_F1'] = np.nan

        # 2. Batch Correction
        print("  [Batch] Computing Silhouette Batch (ASW)...")
        try:
            results['Batch_ASW'] = silhouette_batch(X_emb, labels, batch, rescale=True)
        except Exception as e: 
            print(f"    Failed: {e}")
            results['Batch_ASW'] = np.nan
        
        print("  [Batch] Computing PCR...")
        try:
            # PCR checks how much batch variance is in the embedding
            # principal_component_regression(X, covariate, categorical=True)
            results['Batch_PCR'] = principal_component_regression(X_emb, batch, categorical=True)
        except Exception as e: 
            print(f"    Failed: {e}")
            results['Batch_PCR'] = np.nan

        if kbet is not None:
            print("  [Batch] Computing kBET...")
            try:
                # kBET measures mixing
                results['Batch_kBET'] = kbet(X_emb, batch)
            except Exception as e:
                print(f"    Failed: {e}")
                pass

        # Calculate Overall Score (0.6 Bio + 0.4 Batch weighting)
        bio_vals = [v for k, v in results.items() if k.startswith('Bio_') and not np.isnan(v)]
        batch_vals = [v for k, v in results.items() if k.startswith('Batch_') and not np.isnan(v)]
        
        bio_score = np.mean(bio_vals) if bio_vals else 0
        batch_score = np.mean(batch_vals) if batch_vals else 0
        
        results['Score_Bio'] = bio_score
        results['Score_Batch'] = batch_score
        results['Score_Overall'] = 0.6 * bio_score + 0.4 * batch_score
        
        return results

    def evaluate_all(self, embedding_keys: List[str], batch_key: str, label_key: str) -> pd.DataFrame:
        """
        Evaluate multiple embeddings and return a comparison DataFrame.
        """
        all_results = {}
        
        for key in embedding_keys:
            if key in self.adata.obsm:
                metrics = self.evaluate_embedding(key, batch_key, label_key)
                if metrics:
                    all_results[key] = metrics
            else:
                print(f"Skipping '{key}' (not found)")
                
        if not all_results:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_results).T
        # Sort by Overall Score if available
        if 'Score_Overall' in df.columns:
            df = df.sort_values('Score_Overall', ascending=False)
            
        return df
