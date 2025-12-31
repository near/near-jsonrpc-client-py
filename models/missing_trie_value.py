from models.crypto_hash import CryptoHash
from models.missing_trie_value_context import MissingTrieValueContext
from pydantic import BaseModel


class MissingTrieValue(BaseModel):
    context: MissingTrieValueContext
    hash: CryptoHash
