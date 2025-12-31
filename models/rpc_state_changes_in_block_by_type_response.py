from models.crypto_hash import CryptoHash
from models.state_change_kind_view import StateChangeKindView
from pydantic import BaseModel
from typing import List


class RpcStateChangesInBlockByTypeResponse(BaseModel):
    block_hash: CryptoHash
    changes: List[StateChangeKindView]
