from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.execution_outcome_view import ExecutionOutcomeView
from near_jsonrpc_models.merkle_path_item import MerklePathItem
from pydantic import BaseModel
from typing import List


class ExecutionOutcomeWithIdView(BaseModel):
    block_hash: CryptoHash
    id: CryptoHash
    outcome: ExecutionOutcomeView
    proof: List[MerklePathItem]
