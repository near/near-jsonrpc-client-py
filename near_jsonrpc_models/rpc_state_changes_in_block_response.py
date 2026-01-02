from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.state_change_with_cause_view import StateChangeWithCauseView
from pydantic import BaseModel
from typing import List


class RpcStateChangesInBlockResponse(BaseModel):
    block_hash: CryptoHash
    changes: List[StateChangeWithCauseView]
