"""
Preprocessing module for single-cell RNA-seq data
Handles QC, filtering, normalization, doublet detection, and HVG selection.
"""

import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import numpy as np
from dataclasses import dataclass
import copy
from typing import Optional, Union, Dict, Any, List

@dataclass
class QCConfig:
    """Configuration for quality control filtering thresholds"""
    # Cell filtering
    min_genes: int = 200          # Minimum genes per cell
    max_genes: int = 6000         # Maximum genes per cell (upper outliers)
    min_counts: int = 500         # Minimum UMI counts per cell
    max_counts: int = 50000       # Maximum UMI counts per cell
    max_pct_mt: float = 15.0      # Maximum mitochondrial percentage
    
    # Gene filtering
    min_cells: int = 3            # Minimum cells expressing a gene
    
    # Doublet detection
    expected_doublet_rate: float = 0.06    # Expected doublet rate (default 6%)
    
    def __repr__(self):
        return (f"QCConfig(min_genes={self.min_genes}, max_genes={self.max_genes}, "
                f"max_pct_mt={self.max_pct_mt})")
    
    @staticmethod
    def create_per_batch_config(base_config: 'QCConfig', batch_overrides: Dict[str, Dict[str, Any]]) -> Dict[str, 'QCConfig']:
        """
        Create per-batch configurations
        
        Parameters:
        -----------
        base_config : QCConfig
            Base configuration
        batch_overrides : dict
            Dictionary of {batch_name: {param: value}} for overrides
        
        Returns:
        --------
        dict : {batch_name: QCConfig}
        """
        configs = {}
        for batch_name, overrides in batch_overrides.items():
            batch_config = copy.deepcopy(base_config)
            for param, value in overrides.items():
                if hasattr(batch_config, param):
                    setattr(batch_config, param, value)
            configs[batch_name] = batch_config
        return configs


class Preprocessor:
    """
    Handles preprocessing of single-cell RNA-seq data:
    QC, filtering, doublet detection, normalization, and HVG selection.
    
    NOTE: This module is strictly CPU-only to ensure stability and correctness 
    during the critical preprocessing phase. GPU acceleration should only be 
    used in the integration step if supported by the specific method.
    """
    
    def __init__(self, adata: Union[ad.AnnData, Dict[str, ad.AnnData]], batch_key: Optional[str] = None):
        """
        Initialize Preprocessor
        
        Parameters:
        -----------
        adata : AnnData or Dict[str, AnnData]
            The data to process. Can be a single AnnData object or a dictionary of split objects.
        batch_key : str, optional
            Column name in .obs for batch information. Used if data is a single AnnData object.
        """
        self.adata = adata
        self.batch_key = batch_key
        self.adata_splits: Optional[Dict[str, ad.AnnData]] = None
        self.adata: Optional[ad.AnnData] = None

        if isinstance(adata, dict):
            print("✓ Initialized with split dataset (dictionary)")
            self.adata_splits = adata
            self._update_global_from_splits()
                
        elif isinstance(adata, ad.AnnData):
            self.adata = adata
            if self.batch_key and self.batch_key in self.adata.obs.columns:
                print(f"✓ Initialized with single AnnData, splitting by '{self.batch_key}'")
                self._split_by_obs(self.batch_key)
            else:
                print("✓ Initialized with single AnnData (no batch splitting)")
        else:
            raise TypeError(f"data must be AnnData or dict, got {type(adata)}")

    def _split_by_obs(self, column_key: str):
        """Helper to split adata by batch for batch-aware processing"""
        unique_values = self.adata.obs[column_key].unique()
        split_dict = {}
        for value in unique_values:
            if pd.isna(value): continue
            mask = self.adata.obs[column_key] == value
            split_dict[str(value)] = self.adata[mask, :].copy()
        self.adata_splits = split_dict

    def _update_global_from_splits(self):
        """Re-concatenate splits to update the global self.adata view"""
        if self.adata_splits:
            try:
                # print("  Updating global view via concatenation...")
                self.adata = ad.concat(
                    list(self.adata_splits.values()),
                    label=self.batch_key if self.batch_key else 'batch',
                    keys=list(self.adata_splits.keys()),
                    join='outer',
                    index_unique='_'
                )
            except Exception as e:
                print(f"  Warning: Could not create global view: {e}")

    def sync_splits(self):
        """
        Synchronize splits from the global adata.
        Useful if global adata has been modified (e.g. annotated) and we want splits to reflect that.
        """
        if self.adata is not None and self.batch_key:
            print(f"Synchronizing splits from global adata (batch_key='{self.batch_key}')...")
            self._split_by_obs(self.batch_key)
            print(f"✓ Splits updated ({len(self.adata_splits)} batches)")
        else:
            print("Warning: Cannot sync splits (no global adata or batch_key missing)")

    def validate_and_compute_qc(self):
        """Validate loaded data and compute QC metrics"""
        print("\n=== Computing QC Metrics ===")
        
        def _compute_qc_for_adata(adata_obj):
            # Ensure CPU
            adata_obj = self._ensure_cpu(adata_obj)

            # Store raw counts if not already present
            if 'counts' not in adata_obj.layers:
                adata_obj.layers['counts'] = adata_obj.X.copy()
            
            # Identify gene groups
            adata_obj.var['mt'] = adata_obj.var_names.str.startswith(('MT-', 'mt-'))
            adata_obj.var['ribo'] = adata_obj.var_names.str.startswith(('RPS', 'RPL', 'Rps', 'Rpl'))
            adata_obj.var['hb'] = adata_obj.var_names.str.contains('^HB[^(P)]', case=False)
            
            # Calculate QC metrics
            sc.pp.calculate_qc_metrics(
                adata_obj,
                qc_vars=['mt', 'ribo', 'hb'],
                percent_top=None,
                log1p=False,
                inplace=True
            )
            return adata_obj

        # Process splits if available
        if self.adata_splits is not None:
            print(f"Computing QC metrics for {len(self.adata_splits)} batches...")
            for batch_name, batch_adata in self.adata_splits.items():
                self.adata_splits[batch_name] = _compute_qc_for_adata(batch_adata)
            
            self._update_global_from_splits()
        
        # If no splits, process global
        elif self.adata is not None:
            self.adata = _compute_qc_for_adata(self.adata)
        
        # Print summary
        if self.adata is not None:
            print(f"✓ QC metrics computed")
            print(f"  Cells: {self.adata.n_obs}")
            print(f"  Genes: {self.adata.n_vars}")
            print(f"  Median genes/cell: {self.adata.obs['n_genes_by_counts'].median():.0f}")
            print(f"  Median UMIs/cell: {self.adata.obs['total_counts'].median():.0f}")
            print(f"  Median % MT: {self.adata.obs['pct_counts_mt'].median():.2f}%")
        
        return self.adata
    
    def check_normalization_status(self) -> Dict[str, Any]:
        """
        Check if data is already normalized.
        
        Checks:
        1. Are counts integers? (If yes -> Raw)
        2. Is library size constant across cells? (If yes -> Normalized)
        3. Log check (max value < 50 usually implies log-transformed)
        """
        print("\n=== Checking normalization status ===")
        
        results = {}

        def _analyze_single_adata(adata_obj, label):
            # Ensure CPU for checks (avoiding potential GPU pointer dereferences)
            adata_obj = self._ensure_cpu(adata_obj)

            status = {
                'is_integer': False,
                'constant_library_size': False,
                'is_log_like': False,
                'conclusion': 'unknown'
            }
            
            # 1. Check for integers (Raw counts)
            # Sample data for speed
            if hasattr(adata_obj.X, "data"): # Sparse
                data_sample = adata_obj.X.data[:10000] if len(adata_obj.X.data) > 10000 else adata_obj.X.data
            else: # Dense
                flat = adata_obj.X.flatten()
                data_sample = flat[:10000] if len(flat) > 10000 else flat
            
            if len(data_sample) == 0:
                return {'conclusion': 'empty'}

            is_integer = np.allclose(data_sample, np.round(data_sample))
            status['is_integer'] = bool(is_integer)
            
            # 2. Check library sizes (Sum of counts per cell)
            # We need to sum across genes (axis 1)
            lib_sizes = np.array(adata_obj.X.sum(axis=1)).flatten()
            
            # Check variance of library sizes
            # If normalized, they should be very close (e.g. all 10000)
            mean_lib = np.mean(lib_sizes)
            cv = np.std(lib_sizes) / mean_lib if mean_lib > 0 else 0
            is_constant_size = cv < 1e-4 # Very low coefficient of variation
            
            status['constant_library_size'] = bool(is_constant_size)
            status['mean_library_size'] = float(mean_lib)
            
            # 3. Check value range (Log check)
            max_val = np.max(data_sample)
            status['max_value'] = float(max_val)
            is_log_like = max_val < 50 and not is_integer # Heuristic
            status['is_log_like'] = is_log_like

            # Conclusion
            if is_constant_size:
                status['conclusion'] = 'normalized'
                print(f"  → {label}: Normalized (Constant library size ~{status['mean_library_size']:.0f})")
            elif is_integer:
                status['conclusion'] = 'raw'
                print(f"  → {label}: Raw counts (Integers detected)")
            elif is_log_like:
                status['conclusion'] = 'log_transformed'
                print(f"  → {label}: Log-transformed (Low max value, non-integers)")
            else:
                status['conclusion'] = 'ambiguous'
                print(f"  → {label}: Ambiguous (Non-integers, variable library size)")

            return status

        if self.adata_splits:
            for name, adata_obj in self.adata_splits.items():
                results[name] = _analyze_single_adata(adata_obj, f"Batch {name}")
        elif self.adata is not None:
            results['global'] = _analyze_single_adata(self.adata, "Global")
            
        return results

    def plot_qc_metrics(self, save=None, show=True):
        """
        Generate comprehensive QC plots, batch-aware if batch_key is set
        """
        if self.adata is None:
            print("No data to plot.")
            return

        # Check if QC metrics have been computed
        if 'n_genes_by_counts' not in self.adata.obs:
            print("QC metrics not computed. Running validate_and_compute_qc()...")
            self.validate_and_compute_qc()
        
        print("\n=== Generating QC Plots ===")
        
        # 1. Highest expressed genes
        print("Plotting highest expressed genes...")
        with plt.rc_context({'figure.figsize': (8, 6)}):
            sc.pl.highest_expr_genes(self.adata, n_top=20, show=show, 
                                    save='_highest_expr' if save else None)
        
        # 2. Violin plots
        print("Plotting QC metric violins...")
        
        # Use batch_key if available in obs
        batch_col = self.batch_key if (self.batch_key and self.batch_key in self.adata.obs.columns) else None
        
        if batch_col:
            n_batches = self.adata.obs[batch_col].nunique()
            print(f"  → Grouping by batch: '{batch_col}' ({n_batches} batches)")
            
            # Dynamic figure sizing
            width_per_batch = 0.5
            fig_width = max(10, min(n_batches * width_per_batch, 30))
            
            with plt.rc_context({'figure.figsize': (fig_width, 10)}):
                sc.pl.violin(
                    self.adata,
                    keys=['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo'],
                    groupby=batch_col,
                    jitter=0.3,
                    multi_panel=True,
                    rotation=90,
                    stripplot=False if n_batches > 10 else True,
                    show=show,
                    save='_qc_violin_by_batch' if save else None
                )
        
        # Overall distribution
        print("  → Overall distribution")
        with plt.rc_context({'figure.figsize': (12, 8)}):
            sc.pl.violin(
                self.adata,
                keys=['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo'],
                jitter=0.4,
                multi_panel=True,
                show=show,
                save='_qc_violin_overall' if save else None
            )
        
        # 3. Scatter plots
        print("Plotting scatter relationships...")
        point_size = 10 if self.adata.n_obs < 50000 else 5
        
        with plt.rc_context({'figure.figsize': (8, 6)}):
            sc.pl.scatter(
                self.adata, 
                x='total_counts', 
                y='n_genes_by_counts',
                color=batch_col if batch_col else 'pct_counts_mt',
                size=point_size,
                alpha=0.6,
                show=show, 
                save='_counts_vs_genes' if save else None
            )
        
        print("✓ QC plots generated")

    def filter_cells(self, config: Optional[QCConfig] = None, plot_before_after=True, per_batch=True):
        """
        Filter cells based on QC thresholds
        """
        if config is None:
            config = QCConfig()
        
        print(f"\n=== Filtering Cells ===")
        print(f"Configuration: {config}")

        # Ensure QC metrics are present
        if self.adata is not None and 'n_genes_by_counts' not in self.adata.obs:
            self.validate_and_compute_qc()

        # Plot before
        if plot_before_after and self.adata is not None:
            print("\nBefore filtering:")
            batch_col = self.batch_key if (self.batch_key and self.batch_key in self.adata.obs.columns) else None
            sc.pl.violin(
                self.adata,
                keys=['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                groupby=batch_col,
                jitter=0.4,
                multi_panel=True,
                save='_before_filter'
            )

        # Helper to filter a single AnnData
        def _filter_single(adata_obj):
            # Ensure CPU
            adata_obj = self._ensure_cpu(adata_obj)
            
            n_before = adata_obj.n_obs
            sc.pp.filter_cells(adata_obj, min_genes=config.min_genes)
            sc.pp.filter_cells(adata_obj, min_counts=config.min_counts)
            adata_obj = adata_obj[adata_obj.obs['n_genes_by_counts'] < config.max_genes, :]
            adata_obj = adata_obj[adata_obj.obs['total_counts'] < config.max_counts, :]
            adata_obj = adata_obj[adata_obj.obs['pct_counts_mt'] < config.max_pct_mt, :]
            sc.pp.filter_genes(adata_obj, min_cells=config.min_cells)
            n_after = adata_obj.n_obs
            return adata_obj, n_before, n_after

        # Apply filtering
        if self.adata_splits is not None and per_batch:
            print("Filtering each batch separately...")
            filtered_splits = {}
            for batch_name, batch_adata in self.adata_splits.items():
                filtered_adata, n_before, n_after = _filter_single(batch_adata)
                print(f"  Batch {batch_name}: {n_before} → {n_after} ({100 * n_after / n_before:.1f}% retained)")
                filtered_splits[batch_name] = filtered_adata
            self.adata_splits = filtered_splits
            self._update_global_from_splits()
            
        elif self.adata is not None:
            print("Filtering global dataset...")
            self.adata, n_before, n_after = _filter_single(self.adata)
            print(f"  Cells: {n_before} → {n_after} ({100 * n_after / n_before:.1f}% retained)")

        # Plot after
        if plot_before_after and self.adata is not None:
            print("\nAfter filtering:")
            batch_col = self.batch_key if (self.batch_key and self.batch_key in self.adata.obs.columns) else None
            sc.pl.violin(
                self.adata,
                keys=['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                groupby=batch_col,
                jitter=0.4,
                multi_panel=True,
                save='_after_filter'
            )
        
        return self.adata

    def detect_doublets(self, expected_doublet_rate=0.06):
        """
        Detect doublets using Scrublet algorithm
        """
        print(f"\n=== Detecting Doublets ===")
        print(f"Expected doublet rate: {expected_doublet_rate:.1%}")
        
        def _run_scrublet(adata_obj, batch_name=None):
            # Ensure CPU
            adata_obj = self._ensure_cpu(adata_obj)
            
            try:
                sc.pp.scrublet(
                    adata_obj,
                    batch_key=None, # Already split or single batch
                    expected_doublet_rate=expected_doublet_rate,
                    random_state=42
                )
                n_doublets = adata_obj.obs['predicted_doublet'].sum()
                pct_doublets = 100 * n_doublets / adata_obj.n_obs
                label = f"Batch {batch_name}" if batch_name else "Dataset"
                print(f"  {label}: {n_doublets} doublets ({pct_doublets:.2f}%)")
                return adata_obj
            except Exception as e:
                print(f"  Warning: Doublet detection failed for {batch_name}: {e}")
                return adata_obj

        if self.adata_splits is not None:
            print("Running doublet detection per batch...")
            for batch_name, batch_adata in self.adata_splits.items():
                self.adata_splits[batch_name] = _run_scrublet(batch_adata, batch_name)
            self._update_global_from_splits()
        
        elif self.adata is not None:
            if self.batch_key and self.batch_key in self.adata.obs.columns:
                print(f"Running doublet detection with batch_key='{self.batch_key}'")
                # Ensure CPU
                self.adata = self._ensure_cpu(self.adata)
                sc.pp.scrublet(
                    self.adata,
                    batch_key=self.batch_key,
                    expected_doublet_rate=expected_doublet_rate,
                    random_state=42
                )
            else:
                self.adata = _run_scrublet(self.adata)
        
        return self.adata
    
    def filter_doublets(self):
        """Remove predicted doublets from the dataset"""
        if self.adata is None: return None
        
        if 'predicted_doublet' not in self.adata.obs:
            raise ValueError("Doublet detection has not been run. Run detect_doublets() first.")
        
        print(f"\n=== Filtering Doublets ===")
        
        def _remove_doublets(adata_obj):
            if 'predicted_doublet' in adata_obj.obs:
                return adata_obj[~adata_obj.obs['predicted_doublet'], :].copy()
            return adata_obj

        if self.adata_splits is not None:
            for batch_name, batch_adata in self.adata_splits.items():
                self.adata_splits[batch_name] = _remove_doublets(batch_adata)
            self._update_global_from_splits()
            print("✓ Filtered doublets from all batches")
            
        elif self.adata is not None:
            n_before = self.adata.n_obs
            self.adata = _remove_doublets(self.adata)
            n_after = self.adata.n_obs
            print(f"✓ Filtered doublets: {n_before} → {n_after} ({n_before - n_after} removed)")
        
        return self.adata
    
    def normalize_and_log1p(self, target_sum=1e4):
        """
        Normalize counts per cell and logarithmize.
        """
        print("\n=== Normalizing and Log-transforming ===")
        
        def _normalize(adata_obj):
            # Ensure CPU
            adata_obj = self._ensure_cpu(adata_obj)

            sc.pp.normalize_total(adata_obj, target_sum=target_sum)
            sc.pp.log1p(adata_obj)
            return adata_obj

        if self.adata_splits is not None:
            print("Normalizing each batch...")
            for batch_name, batch_adata in self.adata_splits.items():
                self.adata_splits[batch_name] = _normalize(batch_adata)
            self._update_global_from_splits()
            
        elif self.adata is not None:
            self.adata = _normalize(self.adata)
            
        print("✓ Data normalized (target_sum={}) and log1p transformed".format(target_sum))
        return self.adata

    def select_hvg(self, n_top_genes=2000, batch_key=None):
        """
        Select highly variable genes.
        """
        print(f"\n=== Selecting Highly Variable Genes ===")
        
        if self.adata is None:
            print("No data available.")
            return None

        target_batch_key = batch_key if batch_key else self.batch_key
        
        # Ensure CPU
        self.adata = self._ensure_cpu(self.adata)

        if target_batch_key and target_batch_key in self.adata.obs.columns:
            print(f"Using batch-aware HVG selection (batch_key='{target_batch_key}')")
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top_genes,
                batch_key=target_batch_key,
                subset=False
            )
        else:
            print("Using standard HVG selection")
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top_genes,
                subset=False
            )
            
        n_hvg = self.adata.var['highly_variable'].sum()
        print(f"✓ Identified {n_hvg} highly variable genes")
        
        return self.adata

    def _ensure_cpu(self, adata_obj):
        """
        Helper to ensure AnnData object is on CPU.
        """
        if adata_obj is None: return None
        
        # Check if X is likely on GPU
        is_gpu = False
        if hasattr(adata_obj.X, 'device'): # Torch
            is_gpu = True
        elif 'cupy' in str(type(adata_obj.X)): # Cupy
            is_gpu = True
            
        if is_gpu:
            # print("  Moving data to CPU for safe preprocessing...")
            try:
                if hasattr(adata_obj.X, 'get'):
                    adata_obj.X = adata_obj.X.get()
                elif hasattr(adata_obj.X, 'cpu'):
                    adata_obj.X = adata_obj.X.cpu().numpy()
            except Exception as e:
                print(f"  Warning: Failed to move data to CPU: {e}")
                
        return adata_obj
