"""Configures how to fetch state parts during state sync."""

from near_jsonrpc_models.external_storage_config import ExternalStorageConfig
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


"""Syncs state from the peers without reading anything from external storage."""
class SyncConfigPeers(RootModel[Literal['Peers']]):
    pass

class SyncConfigExternalStorage(StrictBaseModel):
    """Expects parts to be available in external storage.

Usually as a fallback after some number of attempts to use peers."""
    ExternalStorage: ExternalStorageConfig

class SyncConfig(RootModel[Union[SyncConfigPeers, SyncConfigExternalStorage]]):
    pass

