from near_jsonrpc_models.chunk_execution_proof_view import ChunkExecutionProofView
from near_jsonrpc_models.state_proof_view import StateProofView
from pydantic import BaseModel


class RpcLightClientStateProofResponse(BaseModel):
    chunk_execution_proof: ChunkExecutionProofView
    state_proof: StateProofView
