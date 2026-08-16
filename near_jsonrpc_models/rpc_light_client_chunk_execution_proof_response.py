from near_jsonrpc_models.chunk_execution_proof_view import ChunkExecutionProofView
from pydantic import BaseModel


class RpcLightClientChunkExecutionProofResponse(BaseModel):
    chunk_execution_proof: ChunkExecutionProofView
