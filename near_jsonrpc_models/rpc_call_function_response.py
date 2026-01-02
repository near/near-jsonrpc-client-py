"""A result returned by contract method"""

from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import conint
from typing import List


class RpcCallFunctionResponse(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    logs: List[str]
    result: List[conint(ge=0, le=255)]
