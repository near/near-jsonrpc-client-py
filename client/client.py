from .base_client import NearBaseClient
from .api_methods import APIMixin


class NearClient(NearBaseClient, APIMixin):
    """NearClient with generated methods mixed-in."""
    pass
