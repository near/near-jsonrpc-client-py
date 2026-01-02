from near_jsonrpc_models.account_data_view import AccountDataView
from near_jsonrpc_models.known_producer_view import KnownProducerView
from near_jsonrpc_models.peer_info_view import PeerInfoView
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel
from pydantic import conint
from typing import List


class NetworkInfoView(BaseModel):
    connected_peers: List[PeerInfoView]
    known_producers: List[KnownProducerView]
    num_connected_peers: conint(ge=0, le=4294967295)
    peer_max_count: conint(ge=0, le=4294967295)
    tier1_accounts_data: List[AccountDataView]
    tier1_accounts_keys: List[PublicKey]
    tier1_connections: List[PeerInfoView]
