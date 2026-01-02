"""Errors which may occur during working with trie storages, storing
trie values (trie nodes and state values) by their hashes."""

from near_jsonrpc_models.missing_trie_value import MissingTrieValue
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


"""Key-value db internal failure"""
class StorageErrorStorageInternalError(RootModel[Literal['StorageInternalError']]):
    pass

class StorageErrorMissingTrieValue(StrictBaseModel):
    """Requested trie value by its hash which is missing in storage."""
    MissingTrieValue: MissingTrieValue

"""Found trie node which shouldn't be part of state. Raised during
validation of state sync parts where incorrect node was passed.
TODO (#8997): consider including hash of trie node."""
class StorageErrorUnexpectedTrieValue(RootModel[Literal['UnexpectedTrieValue']]):
    pass

class StorageErrorStorageInconsistentState(StrictBaseModel):
    """Either invalid state or key-value db is corrupted.
For PartialStorage it cannot be corrupted.
Error message is unreliable and for debugging purposes only. It's also probably ok to
panic in every place that produces this error.
We can check if db is corrupted by verifying everything in the state trie."""
    StorageInconsistentState: str

class StorageErrorFlatStorageBlockNotSupported(StrictBaseModel):
    """Flat storage error, meaning that it doesn't support some block anymore.
We guarantee that such block cannot become final, thus block processing
must resume normally."""
    FlatStorageBlockNotSupported: str

class StorageErrorMemTrieLoadingError(StrictBaseModel):
    """In-memory trie could not be loaded for some reason."""
    MemTrieLoadingError: str

class StorageError(RootModel[Union[StorageErrorStorageInternalError, StorageErrorMissingTrieValue, StorageErrorUnexpectedTrieValue, StorageErrorStorageInconsistentState, StorageErrorFlatStorageBlockNotSupported, StorageErrorMemTrieLoadingError]]):
    pass

