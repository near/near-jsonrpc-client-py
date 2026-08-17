from near_jsonrpc_models.rpc_light_client_chunk_execution_proof_request import RpcLightClientChunkExecutionProofRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalLightClientChunkExecutionProof(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_light_client_chunk_execution_proof']
    params: RpcLightClientChunkExecutionProofRequest
