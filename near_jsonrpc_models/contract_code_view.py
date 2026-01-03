"""A view of the contract code."""

from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class ContractCodeView(BaseModel):
    code_base64: str
    hash: CryptoHash
