from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.missing_trie_value_context import MissingTrieValueContext
from pydantic import BaseModel


class MissingTrieValue(BaseModel):
    context: MissingTrieValueContext
    hash: CryptoHash
