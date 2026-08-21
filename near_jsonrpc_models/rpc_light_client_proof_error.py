from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.shard_id import ShardId
from near_jsonrpc_models.spice_chunk_id import SpiceChunkId
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Any
from typing import Dict
from typing import Literal
from typing import Union


class RpcLightClientProofErrorUnknownBlock(BaseModel):
    info: Dict[str, Any]
    name: Literal['UNKNOWN_BLOCK']

class RpcLightClientProofErrorInconsistentStateInfo(BaseModel):
    execution_outcome_shard_id: ShardId
    number_or_shards: conint(ge=0, le=4294967295)

class RpcLightClientProofErrorInconsistentState(BaseModel):
    info: RpcLightClientProofErrorInconsistentStateInfo
    name: Literal['INCONSISTENT_STATE']

class RpcLightClientProofErrorNotConfirmedInfo(BaseModel):
    transaction_or_receipt_id: CryptoHash

class RpcLightClientProofErrorNotConfirmed(BaseModel):
    info: RpcLightClientProofErrorNotConfirmedInfo
    name: Literal['NOT_CONFIRMED']

class RpcLightClientProofErrorUnknownTransactionOrReceiptInfo(BaseModel):
    transaction_or_receipt_id: CryptoHash

class RpcLightClientProofErrorUnknownTransactionOrReceipt(BaseModel):
    info: RpcLightClientProofErrorUnknownTransactionOrReceiptInfo
    name: Literal['UNKNOWN_TRANSACTION_OR_RECEIPT']

class RpcLightClientProofErrorUnavailableShardInfo(BaseModel):
    shard_id: ShardId
    transaction_or_receipt_id: CryptoHash

class RpcLightClientProofErrorUnavailableShard(BaseModel):
    info: RpcLightClientProofErrorUnavailableShardInfo
    name: Literal['UNAVAILABLE_SHARD']

class RpcLightClientProofErrorShardNotTrackedInfo(BaseModel):
    shard_id: ShardId

class RpcLightClientProofErrorShardNotTracked(BaseModel):
    info: RpcLightClientProofErrorShardNotTrackedInfo
    name: Literal['SHARD_NOT_TRACKED']

class RpcLightClientProofErrorTargetShardMismatchInfo(BaseModel):
    account_id: AccountId
    account_shard_id: ShardId
    requested_shard_id: ShardId

class RpcLightClientProofErrorTargetShardMismatch(BaseModel):
    info: RpcLightClientProofErrorTargetShardMismatchInfo
    name: Literal['TARGET_SHARD_MISMATCH']

class RpcLightClientProofErrorStateNotAvailableInfo(BaseModel):
    chunk_id: SpiceChunkId

class RpcLightClientProofErrorStateNotAvailable(BaseModel):
    info: RpcLightClientProofErrorStateNotAvailableInfo
    name: Literal['STATE_NOT_AVAILABLE']

class RpcLightClientProofErrorChunkNotCertifiedInfo(BaseModel):
    chunk_id: SpiceChunkId

class RpcLightClientProofErrorChunkNotCertified(BaseModel):
    info: RpcLightClientProofErrorChunkNotCertifiedInfo
    name: Literal['CHUNK_NOT_CERTIFIED']

class RpcLightClientProofErrorLightClientHeadTooOldInfo(BaseModel):
    certifying_block_height: conint(ge=0, le=18446744073709551615)
    chunk_id: SpiceChunkId
    head_height: conint(ge=0, le=18446744073709551615)

class RpcLightClientProofErrorLightClientHeadTooOld(BaseModel):
    info: RpcLightClientProofErrorLightClientHeadTooOldInfo
    name: Literal['LIGHT_CLIENT_HEAD_TOO_OLD']

class RpcLightClientProofErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcLightClientProofErrorInternalError(BaseModel):
    info: RpcLightClientProofErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcLightClientProofError(RootModel[Union[RpcLightClientProofErrorUnknownBlock, RpcLightClientProofErrorInconsistentState, RpcLightClientProofErrorNotConfirmed, RpcLightClientProofErrorUnknownTransactionOrReceipt, RpcLightClientProofErrorUnavailableShard, RpcLightClientProofErrorShardNotTracked, RpcLightClientProofErrorTargetShardMismatch, RpcLightClientProofErrorStateNotAvailable, RpcLightClientProofErrorChunkNotCertified, RpcLightClientProofErrorLightClientHeadTooOld, RpcLightClientProofErrorInternalError]]):
    pass

