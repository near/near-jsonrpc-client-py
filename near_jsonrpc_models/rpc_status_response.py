from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.detailed_debug_status import DetailedDebugStatus
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.status_sync_info import StatusSyncInfo
from near_jsonrpc_models.validator_info import ValidatorInfo
from near_jsonrpc_models.version import Version
from pydantic import BaseModel
from pydantic import conint
from typing import List


class RpcStatusResponse(BaseModel):
    # Unique chain id.
    chain_id: str
    # Information about last blocks, network, epoch and chain & chunk info.
    detailed_debug_status: DetailedDebugStatus | None = None
    # Genesis hash of the chain.
    genesis_hash: CryptoHash
    # Latest protocol version that this client supports.
    latest_protocol_version: conint(ge=0, le=4294967295)
    # Deprecated; same as `validator_public_key` which you should use instead.
    node_key: PublicKey | None = None
    # Public key of the node.
    node_public_key: PublicKey
    # Currently active protocol version.
    protocol_version: conint(ge=0, le=4294967295)
    # Address for RPC server.  None if node doesn't have RPC endpoint enabled.
    rpc_addr: str | None = None
    # Sync status of the node.
    sync_info: StatusSyncInfo
    # Uptime of the node.
    uptime_sec: int
    # Validator id of the node
    validator_account_id: AccountId | None = None
    # Public key of the validator.
    validator_public_key: PublicKey | None = None
    # Current epoch validators.
    validators: List[ValidatorInfo]
    # Binary version.
    version: Version
