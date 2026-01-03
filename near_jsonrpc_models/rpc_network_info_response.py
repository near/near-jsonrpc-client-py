from near_jsonrpc_models.rpc_known_producer import RpcKnownProducer
from near_jsonrpc_models.rpc_peer_info import RpcPeerInfo
from pydantic import BaseModel
from pydantic import conint
from typing import List


class RpcNetworkInfoResponse(BaseModel):
    active_peers: List[RpcPeerInfo]
    # Accounts of known block and chunk producers from routing table.
    known_producers: List[RpcKnownProducer]
    num_active_peers: conint(ge=0, le=4294967295)
    peer_max_count: conint(ge=0, le=4294967295)
    received_bytes_per_sec: conint(ge=0, le=18446744073709551615)
    sent_bytes_per_sec: conint(ge=0, le=18446744073709551615)
