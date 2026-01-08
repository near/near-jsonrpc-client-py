from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.shard_id import ShardId
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

class RpcLightClientProofErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcLightClientProofErrorInternalError(BaseModel):
    info: RpcLightClientProofErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcLightClientProofError(RootModel[Union[RpcLightClientProofErrorUnknownBlock, RpcLightClientProofErrorInconsistentState, RpcLightClientProofErrorNotConfirmed, RpcLightClientProofErrorUnknownTransactionOrReceipt, RpcLightClientProofErrorUnavailableShard, RpcLightClientProofErrorInternalError]]):
    pass

