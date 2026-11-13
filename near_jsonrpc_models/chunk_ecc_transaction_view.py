"""An external call as a chunk holds it. Its unique id is unknown in a chunk that no block on the
canonical chain includes."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class ChunkEccTransactionView(BaseModel):
    contract_id: AccountId
    payload: str
    # The universal account state init the call carries.
    state_init: str | None = None
    # The unique id of this copy of the call, its outcome id. `None` when no block on the
    # canonical chain includes the chunk.
    transaction_id: CryptoHash | None = None
