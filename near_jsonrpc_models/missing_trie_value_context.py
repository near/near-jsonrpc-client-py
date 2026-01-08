"""Contexts in which `StorageError::MissingTrieValue` error might occur.TrieIterator: Missing trie value when reading from TrieIterator.TriePrefetchingStorage: Missing trie value when reading from TriePrefetchingStorage.TrieMemoryPartialStorage: Missing trie value when reading from TrieMemoryPartialStorage.TrieStorage: Missing trie value when reading from TrieStorage."""

from pydantic import RootModel
from typing import Literal


class MissingTrieValueContext(RootModel[Literal['TrieIterator', 'TriePrefetchingStorage', 'TrieMemoryPartialStorage', 'TrieStorage']]):
    pass

