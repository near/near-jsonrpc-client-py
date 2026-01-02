from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class RpcLightClientBlockProofRequest(BaseModel):
    block_hash: CryptoHash
    light_client_head: CryptoHash
