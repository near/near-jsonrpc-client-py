from near_jsonrpc_models.rpc_light_client_execution_proof_request import RpcLightClientExecutionProofRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForLightClientProof(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['light_client_proof']
    params: RpcLightClientExecutionProofRequest
