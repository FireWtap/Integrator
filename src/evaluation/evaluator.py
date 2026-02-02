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
            import pynndescent
            return True
        except ImportError:
            print("Warning: 'scib-metrics' or 'pynndescent' package not found.")
            print("Install with: pip install scib-metrics pynndescent")
            return False



    def evaluate_embedding(self, embedding_key: str, batch_key: str, label_key: str, n_neighbors: int = 90) -> Dict[str, float]:
        """
        Compute metrics for a specific embedding using scib-metrics.
        
        Args:
            embedding_key: Key in adata.obsm
            batch_key: Column in adata.obs for batch labels
            label_key: Column in adata.obs for cell type labels
            n_neighbors: Number of neighbors for LISI/kBET metrics
        """
        print(f"\nEvaluating embedding '{embedding_key}'...")
        
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
                clisi_knn,
                # Batch correction
                silhouette_batch, 
                pcr_comparison,
                ilisi_knn,
                kbet_per_label
            )
            from scib_metrics.nearest_neighbors import NeighborsResults
            from pynndescent import NNDescent
        except ImportError as e:
            print(f"Error importing metrics from scib_metrics: {e}")
            return {}


        X_emb = self.adata.obsm[embedding_key]
        
        # Ensure obs columns are strings/categories
        labels = self.adata.obs[label_key].astype(str).values
        batch = self.adata.obs[batch_key].astype(str).values
        
        # Precompute neighbors (required for multiple metrics)
        neighbors = None
        print("  Computing k-NN graph for neighbor-based metrics...")
        try:
            index = NNDescent(
                X_emb,
                n_neighbors=n_neighbors,
                metric='euclidean',
                random_state=42,
                n_jobs=-1
            )
            indices, distances = index.query(X_emb, k=n_neighbors)
            neighbors = NeighborsResults(indices=indices, distances=distances)
            print(f"    ✓ Computed {n_neighbors}-NN graph")
        except Exception as e:
            print(f"    ✗ k-NN computation failed: {e}")
            print(f"    Many metrics will be skipped")
            return {}
        
        # ==================== BIOLOGICAL CONSERVATION ====================
        print("  [Bio] Computing Silhouette Label (ASW)...")
        try:
            results['Bio_ASW_Label'] = silhouette_label(X_emb, labels, rescale=True)
            print(f"    ✓ {results['Bio_ASW_Label']:.3f}")
        except Exception as e: 
            print(f"    ✗ Failed: {e}")
            results['Bio_ASW_Label'] = np.nan


        print("  [Bio] Computing NMI & ARI...")
        try:
            # FIXED: Pass neighbors object, not embedding
            nmi_ari = nmi_ari_cluster_labels_leiden(neighbors, labels, optimize_resolution=True)
            results['Bio_NMI'] = nmi_ari['nmi']
            results['Bio_ARI'] = nmi_ari['ari']
            print(f"    ✓ NMI={results['Bio_NMI']:.3f}, ARI={results['Bio_ARI']:.3f}")
        except Exception as e: 
            print(f"    ✗ Failed: {e}")
            results['Bio_NMI'] = np.nan
            results['Bio_ARI'] = np.nan


        print("  [Bio] Computing Graph Connectivity...")
        try:
            # FIXED: Pass neighbors object, not embedding
            results['Bio_Graph_Conn'] = graph_connectivity(neighbors, labels)
            print(f"    ✓ {results['Bio_Graph_Conn']:.3f}")
        except Exception as e: 
            print(f"    ✗ Failed: {e}")
            results['Bio_Graph_Conn'] = np.nan
        
        print("  [Bio] Computing Isolated Labels...")
        try:
            # Ensure proper types for isolated_labels
            # scib-metrics might prefer categorical codes if strings fail
            iso_dict = isolated_labels(X_emb, labels, batch, rescale=True, iso_threshold=None)
            results['Bio_Iso_F1'] = iso_dict['isolated_labels_f1']
            results['Bio_Iso_ASW'] = iso_dict['isolated_labels_asw']
            print(f"    ✓ F1={results['Bio_Iso_F1']:.3f}, ASW={results['Bio_Iso_ASW']:.3f}")
        except Exception as e: 
            print(f"    ✗ Failed: {e}")
            results['Bio_Iso_F1'] = np.nan
            results['Bio_Iso_ASW'] = np.nan


        print("  [Bio] Computing cLISI...")
        try:
            # FIXED: Pass neighbors object, not just indices
            results['Bio_cLISI'] = clisi_knn(neighbors, labels)
            print(f"    ✓ {results['Bio_cLISI']:.3f}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results['Bio_cLISI'] = np.nan


        # ==================== BATCH CORRECTION ====================
        print("  [Batch] Computing Silhouette Batch (ASW)...")
        try:
            results['Batch_ASW'] = silhouette_batch(X_emb, labels, batch, rescale=True)
            print(f"    ✓ {results['Batch_ASW']:.3f} (lower=better)")
        except Exception as e: 
            print(f"    ✗ Failed: {e}")
            results['Batch_ASW'] = np.nan
        
        print("  [Batch] Computing iLISI...")
        try:
            # FIXED: Pass neighbors object, not just indices
            results['Batch_iLISI'] = ilisi_knn(neighbors, batch)
            print(f"    ✓ {results['Batch_iLISI']:.3f}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results['Batch_iLISI'] = np.nan


        print("  [Batch] Computing Graph Connectivity (batch)...")
        try:
            # FIXED: Pass neighbors object, not embedding
            results['Batch_Graph_Conn'] = graph_connectivity(neighbors, batch)
            print(f"    ✓ {results['Batch_Graph_Conn']:.3f}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results['Batch_Graph_Conn'] = np.nan
        
        print("  [Batch] Computing PCR...")
        try:
            # Get pre-integration embedding (PCA of original data)
            if 'X_pca' in self.adata.obsm and embedding_key != 'X_pca':
                X_pre = self.adata.obsm['X_pca']
            else:
                # Compute PCA if not available
                adata_temp = self.adata.copy()
                sc.tl.pca(adata_temp, n_comps=50)
                X_pre = adata_temp.obsm['X_pca']
            
            results['Batch_PCR'] = pcr_comparison(
                X_pre, 
                X_emb, 
                batch,
                categorical=True,
                scale=True
            )
            print(f"    ✓ {results['Batch_PCR']:.3f} (lower=better)")
        except Exception as e: 
            print(f"    ✗ Failed: {e}")
            results['Batch_PCR'] = np.nan


        print("  [Batch] Computing kBET...")
        try:
            results['Batch_kBET'] = kbet_per_label(neighbors, batch, labels)
            print(f"    ✓ {results['Batch_kBET']:.3f}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results['Batch_kBET'] = np.nan


        # ==================== CALCULATE SCORES ====================
        bio_metrics = ['Bio_ASW_Label', 'Bio_NMI', 'Bio_ARI', 'Bio_Graph_Conn', 
                      'Bio_Iso_F1', 'Bio_Iso_ASW', 'Bio_cLISI']
        batch_metrics_high = ['Batch_iLISI', 'Batch_Graph_Conn', 'Batch_kBET']
        batch_metrics_low = ['Batch_ASW', 'Batch_PCR']
        
        bio_vals = [results[k] for k in bio_metrics if k in results and not np.isnan(results[k])]
        batch_vals = []
        for k in batch_metrics_high:
            if k in results and not np.isnan(results[k]):
                batch_vals.append(results[k])
        for k in batch_metrics_low:
            if k in results and not np.isnan(results[k]):
                batch_vals.append(results[k])
        
        results['Score_Bio'] = np.mean(bio_vals) if bio_vals else np.nan
        results['Score_Batch'] = np.mean(batch_vals) if batch_vals else np.nan
        
        if not np.isnan(results['Score_Bio']) and not np.isnan(results['Score_Batch']):
            results['Score_Overall'] = 0.6 * results['Score_Bio'] + 0.4 * results['Score_Batch']
        else:
            results['Score_Overall'] = np.nan
        
        return results



    def evaluate_all(self, embedding_keys: List[str], batch_key: str, label_key: str, n_neighbors: int = 90) -> pd.DataFrame:
        """
        Evaluate multiple embeddings and return a comparison DataFrame.
        
        Args:
            embedding_keys: List of embedding keys to evaluate
            batch_key: Batch column in adata.obs
            label_key: Label column in adata.obs
            n_neighbors: Number of neighbors for LISI/kBET
        """
        print("\n" + "="*80)
        print("INTEGRATION EVALUATION - BENCHMARKING ALL METHODS")
        print("="*80)
        
        all_results = {}
        
        for key in embedding_keys:
            if key in self.adata.obsm:
                metrics = self.evaluate_embedding(key, batch_key, label_key, n_neighbors)
                if metrics:
                    all_results[key] = metrics
            else:
                print(f"\nSkipping '{key}' (not found)")
                
        if not all_results:
            print("\n❌ No results computed.")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_results).T
        
        # Recalculate scores with proper normalization across methods
        df = self._compute_normalized_scores(df)
        
        # Sort by Overall Score if available
        if 'Score_Overall_Norm' in df.columns:
            df = df.sort_values('Score_Overall_Norm', ascending=False)
        elif 'Score_Overall' in df.columns:
            df = df.sort_values('Score_Overall', ascending=False)
        
        # Print summary
        self._print_summary(df)
            
        return df
    
    
    def _compute_normalized_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute normalized scores across all methods.
        Uses min-max normalization so scores are 0-1 and comparable.
        """
        bio_metrics = ['Bio_ASW_Label', 'Bio_NMI', 'Bio_ARI', 'Bio_Graph_Conn', 
                      'Bio_Iso_F1', 'Bio_Iso_ASW', 'Bio_cLISI']
        batch_metrics_high = ['Batch_iLISI', 'Batch_Graph_Conn', 'Batch_kBET']
        batch_metrics_low = ['Batch_ASW', 'Batch_PCR']
        
        def normalize_col(col):
            """Min-max normalize to 0-1"""
            col_min, col_max = col.min(), col.max()
            if col_max > col_min and not col.isna().all():
                return (col - col_min) / (col_max - col_min)
            else:
                return pd.Series(1.0, index=col.index)
        
        # Normalize bio metrics
        bio_scores = []
        for metric in bio_metrics:
            if metric in df.columns and not df[metric].isna().all():
                bio_scores.append(normalize_col(df[metric]))
        
        # Normalize batch metrics (higher is better)
        batch_scores = []
        for metric in batch_metrics_high:
            if metric in df.columns and not df[metric].isna().all():
                batch_scores.append(normalize_col(df[metric]))
        
        # Normalize batch metrics (lower is better - invert)
        for metric in batch_metrics_low:
            if metric in df.columns and not df[metric].isna().all():
                batch_scores.append(1 - normalize_col(df[metric]))
        
        # Calculate normalized aggregate scores
        if bio_scores:
            df['Score_Bio_Norm'] = pd.concat(bio_scores, axis=1).mean(axis=1)
        
        if batch_scores:
            df['Score_Batch_Norm'] = pd.concat(batch_scores, axis=1).mean(axis=1)
        
        # Overall score (60% bio, 40% batch - scIB standard)
        if 'Score_Bio_Norm' in df.columns and 'Score_Batch_Norm' in df.columns:
            df['Score_Overall_Norm'] = 0.6 * df['Score_Bio_Norm'] + 0.4 * df['Score_Batch_Norm']
        
        return df
    
    
    def _print_summary(self, df: pd.DataFrame):
        """Print formatted summary of results."""
        print("\n" + "="*80)
        print("📈 EVALUATION SUMMARY")
        print("="*80)
        
        # Overall rankings
        score_col = 'Score_Overall_Norm' if 'Score_Overall_Norm' in df.columns else 'Score_Overall'
        if score_col in df.columns and not df[score_col].isna().all():
            print(f"\n🏆 OVERALL RANKINGS (60% Bio + 40% Batch):")
            for rank, (method, row) in enumerate(df.iterrows(), 1):
                overall = row[score_col]
                bio = row.get('Score_Bio_Norm', row.get('Score_Bio', np.nan))
                batch = row.get('Score_Batch_Norm', row.get('Score_Batch', np.nan))
                
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                print(f"  {emoji} {method:20s} | Overall: {overall:.3f} | Bio: {bio:.3f} | Batch: {batch:.3f}")
        
        # Best per metric
        print("\n📊 BEST METHODS PER METRIC:")
        print("\n  Biological Conservation (higher = better):")
        bio_metrics = ['Bio_NMI', 'Bio_ARI', 'Bio_ASW_Label', 'Bio_Graph_Conn', 
                      'Bio_Iso_F1', 'Bio_cLISI']
        for metric in bio_metrics:
            if metric in df.columns and not df[metric].isna().all():
                best = df[metric].idxmax()
                print(f"    • {metric:20s}: {best:20s} ({df.loc[best, metric]:.3f})")
        
        print("\n  Batch Correction:")
        batch_high = ['Batch_iLISI', 'Batch_Graph_Conn', 'Batch_kBET']
        for metric in batch_high:
            if metric in df.columns and not df[metric].isna().all():
                best = df[metric].idxmax()
                print(f"    • {metric:20s}: {best:20s} ({df.loc[best, metric]:.3f})")
        
        batch_low = ['Batch_ASW', 'Batch_PCR']
        for metric in batch_low:
            if metric in df.columns and not df[metric].isna().all():
                best = df[metric].idxmin()
                print(f"    • {metric:20s}: {best:20s} ({df.loc[best, metric]:.3f}) *lower=better")
        
        print("\n" + "="*80)
        print("\n📋 COMPLETE RESULTS TABLE:")
        print(df.to_string(float_format=lambda x: f'{x:.3f}'))
        print("\n" + "="*80 + "\n")
