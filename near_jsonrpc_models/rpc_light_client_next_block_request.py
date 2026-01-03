from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class RpcLightClientNextBlockRequest(BaseModel):
    last_block_hash: CryptoHash
