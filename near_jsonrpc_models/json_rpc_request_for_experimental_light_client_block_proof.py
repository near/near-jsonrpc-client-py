from near_jsonrpc_models.rpc_light_client_block_proof_request import RpcLightClientBlockProofRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalLightClientBlockProof(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_light_client_block_proof']
    params: RpcLightClientBlockProofRequest
