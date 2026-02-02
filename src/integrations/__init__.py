
from .base import IntegrationMethod
from .harmony import HarmonyIntegration
from .scanorama import ScanoramaIntegration

# Registry of available methods
INTEGRATION_METHODS = {
    'harmony': HarmonyIntegration,
    'scanorama': ScanoramaIntegration
}

def get_integration_method(name: str) -> IntegrationMethod:
    """
    Factory function to get an integration method instance.
    
    Parameters:
    -----------
    name : str
        Name of the method ('harmony', 'scanorama')
        
    Returns:
    --------
    IntegrationMethod : Instance of the requested method
    """
    name = name.lower()
    if name not in INTEGRATION_METHODS:
        raise ValueError(f"Unknown integration method: {name}. Available: {list(INTEGRATION_METHODS.keys())}")
    
    return INTEGRATION_METHODS[name]()
