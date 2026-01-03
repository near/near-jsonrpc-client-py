from near_jsonrpc_models.rpc_light_client_execution_proof_request import RpcLightClientExecutionProofRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalLightClientProof(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_light_client_proof']
    params: RpcLightClientExecutionProofRequest
