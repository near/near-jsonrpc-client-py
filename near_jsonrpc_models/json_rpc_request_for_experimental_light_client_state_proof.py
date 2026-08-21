from near_jsonrpc_models.rpc_light_client_state_proof_request import RpcLightClientStateProofRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalLightClientStateProof(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_light_client_state_proof']
    params: RpcLightClientStateProofRequest
