from near_jsonrpc_models.execution_outcome_with_id_view import ExecutionOutcomeWithIdView
from near_jsonrpc_models.light_client_block_lite_view import LightClientBlockLiteView
from near_jsonrpc_models.merkle_path_item import MerklePathItem
from pydantic import BaseModel
from typing import List


class RpcLightClientExecutionProofResponse(BaseModel):
    block_header_lite: LightClientBlockLiteView
    block_proof: List[MerklePathItem]
    outcome_proof: ExecutionOutcomeWithIdView
    outcome_root_proof: List[MerklePathItem]
