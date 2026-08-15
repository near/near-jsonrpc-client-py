from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.spice_chunk_id import SpiceChunkId
from pydantic import BaseModel


class RpcLightClientChunkExecutionProofRequest(BaseModel):
    chunk_id: SpiceChunkId
    light_client_head: CryptoHash
