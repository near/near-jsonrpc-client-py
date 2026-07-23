"""Explains why a transaction status request returned a `RpcTransactionError::TimeoutError`:"""

from near_jsonrpc_models.rpc_transaction_response import RpcTransactionResponse
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class TimeoutErrorCauseCause(BaseModel):
    """The node never observed the transaction on chain."""
    cause: Literal['NOT_OBSERVED']

class TimeoutErrorCauseCauseStatus(BaseModel):
    """The transaction was observed but is still pending the requested finality. The
last-known status is included so the caller can re-poll for a higher finality."""
    cause: Literal['PENDING']
    status: RpcTransactionResponse

class TimeoutErrorCauseCauseShardId(BaseModel):
    """The node does not track the transaction's shard and could not get an answer from a
chunk producer that does before the timeout."""
    cause: Literal['DOES_NOT_TRACK_SHARD']
    shard_id: ShardId

class TimeoutErrorCauseCauseDebugInfo(BaseModel):
    """The node could not produce a usable transaction status before the timeout (for
example a repeated internal error, or no response at all)."""
    cause: Literal['ERROR']
    debug_info: str

class TimeoutErrorCause(RootModel[Union[TimeoutErrorCauseCause, TimeoutErrorCauseCauseStatus, TimeoutErrorCauseCauseShardId, TimeoutErrorCauseCauseDebugInfo]]):
    pass

