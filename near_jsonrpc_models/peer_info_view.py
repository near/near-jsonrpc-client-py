from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import conint
from typing import List


class PeerInfoView(BaseModel):
    account_id: AccountId | None = None
    addr: str
    archival: bool
    block_hash: CryptoHash | None = None
    connection_established_time_millis: conint(ge=0, le=18446744073709551615)
    height: conint(ge=0, le=18446744073709551615) | None = None
    is_highest_block_invalid: bool
    is_outbound_peer: bool
    last_time_peer_requested_millis: conint(ge=0, le=18446744073709551615)
    last_time_received_message_millis: conint(ge=0, le=18446744073709551615)
    # Connection nonce.
    nonce: conint(ge=0, le=18446744073709551615)
    peer_id: PublicKey
    received_bytes_per_sec: conint(ge=0, le=18446744073709551615)
    sent_bytes_per_sec: conint(ge=0, le=18446744073709551615)
    tracked_shards: List[ShardId]
