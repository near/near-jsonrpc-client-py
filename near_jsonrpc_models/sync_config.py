"""Configures how to fetch state parts during state sync.Peers: Syncs state from the peers without reading anything from external storage."""

from pydantic import RootModel
from typing import Literal


class SyncConfig(RootModel[Literal['Peers']]):
    pass

