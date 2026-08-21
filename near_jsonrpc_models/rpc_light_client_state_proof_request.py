from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.spice_chunk_id import SpiceChunkId
from near_jsonrpc_models.state_proof_target import StateProofTarget
from pydantic import BaseModel


class RpcLightClientStateProofRequest(BaseModel):
    chunk_id: SpiceChunkId
    light_client_head: CryptoHash
    target: StateProofTarget
