from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel


class RpcIndexerBlockRequest(StrictBaseModel):
    block_hash: CryptoHash
