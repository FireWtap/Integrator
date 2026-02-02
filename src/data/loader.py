"""
Data loader module for single-cell integration testing tool
Handles loading and initial batch splitting of single-cell datasets
"""


import scanpy as sc
import pandas as pd
import anndata as ad
from pathlib import Path
import numpy as np

class DatasetLoader:
    """
    Loader for single-cell RNA-seq datasets.
    Supports 10x MTX, 10x H5, H5AD formats, and direct AnnData input.
    """
    
    def __init__(self, dataset_path=None, adata=None, format='auto', output_dir=None, batch_key=None, **kwargs):
        """
        Initialize DatasetLoader
        
        Parameters:
        -----------
        dataset_path : str or Path, optional
            Path to dataset (if loading from file)
        adata : AnnData, optional
            Direct AnnData object input
        format : str
            Format of data ('10x_mtx', '10x_h5', 'h5ad', or 'auto')
        output_dir : str or Path, optional
            Directory for saving outputs
        batch_key : str, optional
            Column name in .obs for batch information
        **kwargs : dict
            Additional arguments passed to read functions
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.format = format
        self.adata = adata
        self.output_dir = Path(output_dir) if output_dir else None  
        self.batch_key = batch_key
        self.annotation_key = kwargs.get('annotation_key', None)
        self.kwargs = kwargs
        self.adata_splits = None
        
        # Load data if not provided directly
        if self.adata is None:
            if self.dataset_path is None:
                raise ValueError("Either dataset_path or adata must be provided")
            
            # Auto-detect format if not specified
            if format == 'auto':
                self.format = self._detect_format()
            
            self._load_data()
        else:
            print(f"✓ Using provided AnnData object: {self.adata.n_obs} cells × {self.adata.n_vars} genes")
        
        # Auto-split if batch_key provided
        if batch_key and batch_key in self.adata.obs.columns:
            self.split_by_obs(batch_key)
    
    def _detect_format(self):
        """Auto-detect data format based on path"""
        if self.dataset_path.is_file():
            if self.dataset_path.suffix == '.h5ad':
                return 'h5ad'
            elif self.dataset_path.suffix in ['.h5', '.hdf5']:
                return '10x_h5'
        elif self.dataset_path.is_dir():
            # Check for mtx files in directory
            if (self.dataset_path / 'matrix.mtx.gz').exists() or \
               (self.dataset_path / 'matrix.mtx').exists():
                return '10x_mtx'
        raise ValueError(f"Could not detect format for: {self.dataset_path}")
    
    def _load_data(self):
        """Load data based on detected/specified format"""
        if self.format == '10x_mtx':
            self.load_10x_mtx()
        elif self.format == '10x_h5':
            self.load_10x_h5()
        elif self.format == 'h5ad':
            self.load_h5ad()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def load_10x_mtx(self):
        """Load 10x Genomics data from MTX format"""
        print(f"Loading 10x MTX from {self.dataset_path}...")
        self.adata = sc.read_10x_mtx(
            self.dataset_path,
            var_names='gene_symbols',
            cache=True,
            **self.kwargs
        )
        print(f"✓ Loaded: {self.adata.n_obs} cells × {self.adata.n_vars} genes")
    
    def load_10x_h5(self):
        """Load 10x Genomics data from H5 format"""
        print(f"Loading 10x H5 from {self.dataset_path}...")
        self.adata = sc.read_10x_h5(
            self.dataset_path,
            **self.kwargs
        )
        print(f"✓ Loaded: {self.adata.n_obs} cells × {self.adata.n_vars} genes")
    
    def load_h5ad(self):
        """Load data from H5AD format"""
        print(f"Loading H5AD from {self.dataset_path}...")
        self.adata = sc.read_h5ad(
            self.dataset_path,
            **self.kwargs
        )
        print(f"✓ Loaded: {self.adata.n_obs} cells × {self.adata.n_vars} genes")
    
    def get_adata(self):
        """Return the loaded AnnData object"""
        return self.adata
    
    def split_by_obs(self, column_key, copy=True):
        """
        Split AnnData object into multiple objects based on unique values in an obs column
        
        Parameters:
        -----------
        column_key : str
            Name of the column in .obs to split by (e.g., 'batch', 'sample', 'condition')
        copy : bool
            Whether to return copies of the data (default True, recommended)
        
        Returns:
        --------
        dict : Dictionary of {value: AnnData} for each unique value in the column
        """
        if self.adata is None:
            raise ValueError("No data loaded")
        
        if column_key not in self.adata.obs.columns:
            raise ValueError(f"Column '{column_key}' not found in .obs. Available columns: {list(self.adata.obs.columns)}")
        
        print(f"\n=== Splitting by '{column_key}' ===")
        
        # Get unique values in the column
        unique_values = self.adata.obs[column_key].unique()
        
        # Create dictionary of split AnnData objects
        split_dict = {}
        skipped_count = 0
        
        for value in unique_values:
            # Skip NaN/None values
            if pd.isna(value):
                skipped_count += self.adata.obs[column_key].isna().sum()
                continue
            
            # Filter to cells with this value
            mask = self.adata.obs[column_key] == value
            subset = self.adata[mask, :]
            
            # Copy if requested (recommended to avoid reference issues)
            if copy:
                subset = subset.copy()
            
            split_dict[str(value)] = subset
            print(f"  {value}: {subset.n_obs} cells × {subset.n_vars} genes")
        
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} cells with NaN/missing values")

        print(f"✓ Split into {len(split_dict)} groups")
        self.adata_splits = split_dict 
        return split_dict

    
    def get_obs_summary(self, column_key=None):
        """
        Get summary of unique values in obs columns (useful before splitting)
        
        Parameters:
        -----------
        column_key : str, optional
            Specific column to summarize. If None, shows all columns
        
        Returns:
        --------
        dict or pd.Series : Summary of value counts
        """
        if self.adata is None:
            raise ValueError("No data loaded")
        
        if column_key is None:
            print("\n=== Available .obs columns ===")
            for col in self.adata.obs.columns:
                n_unique = self.adata.obs[col].nunique()
                print(f"{col}: {n_unique} unique values")
            return None
        else:
            if column_key not in self.adata.obs.columns:
                raise ValueError(f"Column '{column_key}' not found")
            
            print(f"\n=== Summary of '{column_key}' ===")
            counts = self.adata.obs[column_key].value_counts()
            print(counts)
            return counts

    # Static methods for multi-sample workflows
    
    @staticmethod
    def load_multiple_samples(sample_paths, batch_names=None, **kwargs):
        """
        Load multiple samples with batch labels
        
        Parameters:
        -----------
        sample_paths : dict or list
            Dictionary {batch_name: path} or list of paths
        batch_names : list, optional
            Batch names if sample_paths is a list
        
        Returns:
        --------
        dict : Dictionary of {batch_name: DatasetLoader}
        """
        if isinstance(sample_paths, dict):
            loaders = {}
            for batch_name, path in sample_paths.items():
                print(f"\n{'='*50}")
                print(f"Loading batch: {batch_name}")
                print(f"{'='*50}")
                loader = DatasetLoader(path, **kwargs)
                loader.adata.obs['batch'] = batch_name
                loaders[batch_name] = loader
            return loaders
        
        elif isinstance(sample_paths, list):
            if batch_names is None:
                batch_names = [f"batch_{i}" for i in range(len(sample_paths))]
            
            loaders = {}
            for batch_name, path in zip(batch_names, sample_paths):
                print(f"\n{'='*50}")
                print(f"Loading batch: {batch_name}")
                print(f"{'='*50}")
                loader = DatasetLoader(path, **kwargs)
                loader.adata.obs['batch'] = batch_name
                loaders[batch_name] = loader
            return loaders
        
        else:
            raise ValueError("sample_paths must be dict or list")
    
    @staticmethod
    def combine_samples(loaders_dict, batch_key='batch', join='outer'):
        """
        Combine multiple processed samples into one AnnData object
        
        Parameters:
        -----------
        loaders_dict : dict
            Dictionary of {batch_name: DatasetLoader}
        batch_key : str
            Name for batch column in combined data
        join : str
            How to combine gene sets ('outer' keeps all genes, 'inner' only common)
        
        Returns:
        --------
        AnnData : Combined dataset
        """
        print(f"\n{'='*60}")
        print("Combining processed samples")
        print(f"{'='*60}")
        
        adatas = []
        for batch_name, loader in loaders_dict.items():
            adata = loader.get_adata()
            print(f"{batch_name}: {adata.n_obs} cells")
            adatas.append(adata)
        
        # Concatenate
        combined = ad.concat(
            adatas,
            join=join,
            label=batch_key,
            keys=list(loaders_dict.keys()),
            index_unique='_'
        )
        
        print(f"\n✓ Combined dataset: {combined.n_obs} cells × {combined.n_vars} genes")
        print(f"  Batches: {combined.obs[batch_key].value_counts().to_dict()}")
        
        return combined
