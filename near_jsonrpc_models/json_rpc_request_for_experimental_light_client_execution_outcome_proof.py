from near_jsonrpc_models.rpc_light_client_execution_outcome_proof_request import RpcLightClientExecutionOutcomeProofRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalLightClientExecutionOutcomeProof(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_light_client_execution_outcome_proof']
    params: RpcLightClientExecutionOutcomeProofRequest
