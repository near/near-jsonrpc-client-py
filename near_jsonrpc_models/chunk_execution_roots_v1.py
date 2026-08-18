from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.spice_chunk_id import SpiceChunkId
from pydantic import BaseModel


class ChunkExecutionRootsV1(BaseModel):
    chunk_id: SpiceChunkId
    outcome_root: CryptoHash
    outgoing_receipts_root: CryptoHash
    state_root: CryptoHash
