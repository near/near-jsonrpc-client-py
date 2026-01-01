"""A view of the contract code."""

from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import conint


class RpcViewCodeResponse(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    code_base64: str
    hash: CryptoHash
