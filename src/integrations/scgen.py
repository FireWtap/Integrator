import scanpy as sc
import anndata as ad
from typing import Union, Dict
from .base import IntegrationMethod
from utils.gpu import get_device

class SCGENIntegration(IntegrationMethod):
    """
    SCGEN integration wrapper.
    Uses scgen to predict batch-corrected expression profiles.
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__("SCGEN", use_gpu=use_gpu)

    def check_dependencies(self) -> bool:
        try:
            import scgen
            return True
        except ImportError:
            print("Warning: 'scgen' package not found. Please install it.")
            return False

    def run(self, adata: Union[ad.AnnData, Dict[str, ad.AnnData]], batch_key: str, cell_type_key: str = None, **kwargs) -> ad.AnnData:
        """
        Run SCGEN integration.
        
        Parameters:
        -----------
        adata : AnnData or Dict
            The annotated data matrix or dictionary of splits.
        batch_key : str
            The column in .obs containing batch information.
        cell_type_key : str, optional
            The column in .obs containing cell type labels (required for SCGEN).
            If None, tries to find 'cell_type' or 'annotation'.
        """
        # Ensure single AnnData
        adata = self._prepare_input(adata, batch_key)
        
        print(f"Running SCGEN integration on '{batch_key}'...")
        
        try:
            import scgen
        except ImportError:
            raise ImportError("SCGEN not installed. Run `pip install scgen`.")

        # SCGEN requires cell type labels
        if cell_type_key is None:
            # Try to guess
            for key in ['cell_type', 'annotation', 'predicted_labels', 'majority_voting']:
                if key in adata.obs.columns:
                    cell_type_key = key
                    break
        
        if cell_type_key is None or cell_type_key not in adata.obs.columns:
            raise ValueError("SCGEN requires a cell type column. Please provide 'cell_type_key' or run annotation first.")
            
        print(f"Using cell type key: '{cell_type_key}'")
        
        # Setup AnnData
        # SCGEN usually works on raw counts or normalized data. 
        # It's best to pass the object as is, but ensure it's setup correctly.
        
        # Create a copy to avoid modifying the original in unexpected ways during setup
        # But we want to return the corrected data on the original usually.
        # Let's work on a copy for training.
        
        # SCGEN setup
        scgen.SCGEN.setup_anndata(adata, batch_key=batch_key, labels_key=cell_type_key)
        
        # Train
        device = get_device(self.use_gpu)
        use_cuda = device == 'cuda'
        # SCGEN uses PyTorch Lightning under the hood usually or direct torch
        
        # Separate kwargs for init and train
        train_kwargs = {k: v for k, v in kwargs.items() if k in ['max_epochs', 'batch_size', 'early_stopping', 'early_stopping_patience']}
        init_kwargs = {k: v for k, v in kwargs.items() if k not in train_kwargs}
        
        print(f"Training SCGEN model (GPU={use_cuda})...")
        model = scgen.SCGEN(adata, **init_kwargs)
        # Determine accelerator
        accelerator = "auto"
        devices = "auto"
        if device == 'cuda':
            accelerator = 'gpu'
            devices = 1
        elif device == 'mps':
            accelerator = 'mps'
            devices = 1
        else:
            accelerator = 'cpu'

        model.train(
            max_epochs=train_kwargs.get('max_epochs', 100),
            batch_size=train_kwargs.get('batch_size', 32),
            early_stopping=train_kwargs.get('early_stopping', True),
            early_stopping_patience=train_kwargs.get('early_stopping_patience', 25),
            accelerator=accelerator,
            devices=devices
        )
        
        # Batch removal (correction)
        print("Applying batch removal...")
        corrected_adata = model.batch_removal()
        
        # corrected_adata is a new AnnData with corrected X
        # We can store this in the original adata as a layer or return the new one
        # The user asked for "batch corrected matrices".
        
        # Let's store in layers['scgen'] and also compute PCA on it for downstream
        adata.layers['scgen'] = corrected_adata.X
        
        # Also compute PCA on corrected data for visualization
        print("Computing PCA on corrected data...")
        X_scgen = corrected_adata.X
        
        # We can store this as a separate embedding or just return the corrected adata
        # Usually integration methods in this pipeline return the adata with .obsm updated
        # But SCGEN gives a full expression matrix.
        
        # Let's add X_pca_scgen
        temp_adata = ad.AnnData(X=X_scgen, obs=adata.obs)
        sc.tl.pca(temp_adata)
        adata.obsm['X_pca_scgen'] = temp_adata.obsm['X_pca']
        
        print("✓ SCGEN integration complete. Result in layers['scgen'] and obsm['X_pca_scgen'].")
        
        return adata
