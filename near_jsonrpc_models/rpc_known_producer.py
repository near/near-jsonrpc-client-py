from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.peer_id import PeerId
from pydantic import BaseModel


class RpcKnownProducer(BaseModel):
    account_id: AccountId
    addr: str | None = None
    peer_id: PeerId
