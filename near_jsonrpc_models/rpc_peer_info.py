from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.peer_id import PeerId
from pydantic import BaseModel


class RpcPeerInfo(BaseModel):
    account_id: AccountId | None = None
    addr: str | None = None
    id: PeerId
