"""This enum represents if a storage_get call will be performed through flat storage or trie"""

from pydantic import RootModel
from typing import Literal


class StorageGetMode(RootModel[Literal['FlatStorage', 'Trie']]):
    pass

