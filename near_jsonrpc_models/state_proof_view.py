"""A value read from a shard's state, with the trie nodes that prove it against the
chunk's `state_root`. An absent `value` is proved the same way."""

from near_jsonrpc_models.store_value import StoreValue
from pydantic import BaseModel
from typing import List


class StateProofView(BaseModel):
    nodes: List[str]
    value: StoreValue | None = None
