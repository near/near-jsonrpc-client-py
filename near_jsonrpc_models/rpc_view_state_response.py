"""Resulting state values for a view state query request"""

from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.state_item import StateItem
from pydantic import BaseModel
from pydantic import conint
from typing import List


class RpcViewStateResponse(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    proof: List[str] = None
    values: List[StateItem]
