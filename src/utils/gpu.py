"""
GPU utility module for device detection (CUDA/MPS).
"""

import torch

def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or MPS) is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()

def is_rapids_available() -> bool:
    """Check if rapids_singlecell is available and we are on CUDA."""
    if not torch.cuda.is_available():
        return False
    try:
        import rapids_singlecell as rsc
        return True
    except ImportError:
        return False

def get_device(use_gpu: bool = True) -> str:
    """
    Get the device string ('cuda', 'mps', or 'cpu').
    
    Parameters:
    -----------
    use_gpu : bool
        Whether to request a GPU device if available.
        
    Returns:
    --------
    str : Device string.
    """
    if not use_gpu:
        return 'cpu'
        
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def print_device_info():
    """Print information about the available devices."""
    print("\n=== Device Information ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  RAPIDS Available: {is_rapids_available()}")
        
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    device = get_device()
    print(f"Selected Device: {device}")
