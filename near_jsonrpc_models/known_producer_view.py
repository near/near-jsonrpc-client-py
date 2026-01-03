"""Information about a Producer: its account name, peer_id and a list of connected peers that
the node can use to send message for this producer."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel
from typing import List


class KnownProducerView(BaseModel):
    account_id: AccountId
    next_hops: List[PublicKey] | None = None
    peer_id: PublicKey
