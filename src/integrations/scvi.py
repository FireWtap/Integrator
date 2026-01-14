import scanpy as sc
import anndata as ad
from typing import Union, Dict
from .base import IntegrationMethod
from utils.gpu import get_device

class SCVIIntegration(IntegrationMethod):
    """
    scVI integration wrapper.
    Uses scvi-tools to learn a latent representation.
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__("scVI", use_gpu=use_gpu)

    def check_dependencies(self) -> bool:
        try:
            import scvi
            return True
        except ImportError:
            print("Warning: 'scvi-tools' package not found. Please install it.")
            return False

    def run(self, adata: Union[ad.AnnData, Dict[str, ad.AnnData]], batch_key: str, **kwargs) -> ad.AnnData:
        """
        Run scVI integration.
        
        Parameters:
        -----------
        adata : AnnData or Dict
            The annotated data matrix or dictionary of splits.
        batch_key : str
            The column in .obs containing batch information.
        """
        # Ensure single AnnData
        adata = self._prepare_input(adata, batch_key)
        
        print(f"Running scVI integration on '{batch_key}'...")
        
        try:
            import scvi
        except ImportError:
            raise ImportError("scvi-tools not installed. Run `pip install scvi-tools`.")

        # scVI requires raw counts usually
        # Check if we have 'counts' layer, otherwise assume X is counts if integers
        # But Preprocessor might have normalized X.
        
        if 'counts' in adata.layers:
            print("Using 'counts' layer for scVI...")
            # scvi setup expects the layer name or None for X
            layer = 'counts'
        else:
            print("Warning: 'counts' layer not found. Using X. Ensure X contains raw counts for best results.")
            layer = None

        # Setup
        scvi.model.SCVI.setup_anndata(adata, layer=layer, batch_key=batch_key)
        
        # Train
        device = get_device(self.use_gpu)
        use_cuda = device == 'cuda'
        # scvi-tools handles MPS automatically in recent versions if use_gpu=True/False logic matches
        # Actually scvi uses PyTorch Lightning.
        # We pass use_gpu to train() usually or accelerator='gpu'/'mps'
        
        # Separate kwargs for init and train
        # Common train args
        train_kwargs = {k: v for k, v in kwargs.items() if k in ['max_epochs', 'batch_size', 'early_stopping', 'early_stopping_patience', 'check_val_every_n_epoch']}
        # Remaining go to init
        init_kwargs = {k: v for k, v in kwargs.items() if k not in train_kwargs}
        
        print(f"Training scVI model (Target Device: {device})...")
        model = scvi.model.SCVI(adata, **init_kwargs)
        
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
            max_epochs=train_kwargs.get('max_epochs', None),
            batch_size=train_kwargs.get('batch_size', 128),
            early_stopping=train_kwargs.get('early_stopping', False),
            early_stopping_patience=train_kwargs.get('early_stopping_patience', 50),
            check_val_every_n_epoch=train_kwargs.get('check_val_every_n_epoch', None),
            accelerator=accelerator,
            devices=devices
        )
        
        # Get latent representation
        print("Extracting latent representation...")
        adata.obsm['X_scVI'] = model.get_latent_representation()
        
        # Also can get normalized expression
        # adata.layers['scvi_normalized'] = model.get_normalized_expression()
        
        print("✓ scVI integration complete. Result in obsm['X_scVI'].")
        
        return adata
