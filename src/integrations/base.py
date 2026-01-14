from abc import ABC, abstractmethod
import anndata as ad
from typing import Union, Dict

class IntegrationMethod(ABC):
    """
    Abstract base class for single-cell integration methods.
    """
    
    def __init__(self, name: str, use_gpu: bool = True):
        self.name = name
        self.use_gpu = use_gpu

    @abstractmethod
    def check_dependencies(self) -> bool:
        """
        Check if necessary dependencies are installed.
        Returns True if available, raises ImportError or returns False otherwise.
        """
        pass

    def _prepare_input(self, data: Union[ad.AnnData, Dict[str, ad.AnnData]], batch_key: str) -> ad.AnnData:
        """
        Helper to ensure input is a single AnnData object.
        If a dictionary is provided, it concatenates them.
        """
        if isinstance(data, dict):
            print(f"Merging {len(data)} batches for integration...")
            try:
                # Concatenate
                # We use the keys as the batch labels
                adata = ad.concat(
                    list(data.values()),
                    label=batch_key,
                    keys=list(data.keys()),
                    join='outer',
                    index_unique='_'
                )
                print(f"✓ Merged: {adata.n_obs} cells")
                return adata
            except Exception as e:
                raise ValueError(f"Failed to merge dictionary input: {e}")
        elif isinstance(data, ad.AnnData):
            return data
        else:
            raise TypeError(f"Input must be AnnData or dict, got {type(data)}")

    @abstractmethod
    def run(self, adata: Union[ad.AnnData, Dict[str, ad.AnnData]], batch_key: str, **kwargs) -> ad.AnnData:
        """
        Run the integration method.
        
        Parameters:
        -----------
        adata : AnnData or Dict
            The annotated data matrix or dictionary of splits.
        batch_key : str
            The column in .obs containing batch information.
        **kwargs : dict
            Additional arguments for the specific integration method.
            
        Returns:
        --------
        AnnData : The annotated data matrix with integration results (usually in .obsm).
        """
        pass
    
    def __repr__(self):
        return f"IntegrationMethod(name='{self.name}')"
