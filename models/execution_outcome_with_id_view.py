from models.crypto_hash import CryptoHash
from models.execution_outcome_view import ExecutionOutcomeView
from models.merkle_path_item import MerklePathItem
from pydantic import BaseModel
from typing import List


class ExecutionOutcomeWithIdView(BaseModel):
    block_hash: CryptoHash
    id: CryptoHash
    outcome: ExecutionOutcomeView
    proof: List[MerklePathItem]
