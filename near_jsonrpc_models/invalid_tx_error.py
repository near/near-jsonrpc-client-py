"""An error happened during TX execution"""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.actions_validation_error import ActionsValidationError
from near_jsonrpc_models.invalid_access_key_error import InvalidAccessKeyError
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.storage_error import StorageError
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class InvalidTxErrorInvalidAccessKeyError(StrictBaseModel):
    """Happens if a wrong AccessKey used or AccessKey has not enough permissions"""
    InvalidAccessKeyError: InvalidAccessKeyError

class InvalidTxErrorInvalidSignerIdPayload(BaseModel):
    signer_id: str

class InvalidTxErrorInvalidSignerId(StrictBaseModel):
    """TX signer_id is not a valid [`AccountId`]"""
    InvalidSignerId: InvalidTxErrorInvalidSignerIdPayload

class InvalidTxErrorSignerDoesNotExistPayload(BaseModel):
    signer_id: AccountId

class InvalidTxErrorSignerDoesNotExist(StrictBaseModel):
    """TX signer_id is not found in a storage"""
    SignerDoesNotExist: InvalidTxErrorSignerDoesNotExistPayload

class InvalidTxErrorInvalidNoncePayload(BaseModel):
    ak_nonce: conint(ge=0, le=18446744073709551615)
    tx_nonce: conint(ge=0, le=18446744073709551615)

class InvalidTxErrorInvalidNonce(StrictBaseModel):
    """Transaction nonce must be strictly greater than `account[access_key].nonce`."""
    InvalidNonce: InvalidTxErrorInvalidNoncePayload

class InvalidTxErrorNonceTooLargePayload(BaseModel):
    tx_nonce: conint(ge=0, le=18446744073709551615)
    upper_bound: conint(ge=0, le=18446744073709551615)

class InvalidTxErrorNonceTooLarge(StrictBaseModel):
    """Transaction nonce is larger than the upper bound given by the block height"""
    NonceTooLarge: InvalidTxErrorNonceTooLargePayload

class InvalidTxErrorInvalidReceiverIdPayload(BaseModel):
    receiver_id: str

class InvalidTxErrorInvalidReceiverId(StrictBaseModel):
    """TX receiver_id is not a valid AccountId"""
    InvalidReceiverId: InvalidTxErrorInvalidReceiverIdPayload

"""TX signature is not valid"""
class InvalidTxErrorInvalidSignature(RootModel[Literal['InvalidSignature']]):
    pass

class InvalidTxErrorNotEnoughBalancePayload(BaseModel):
    balance: NearToken
    cost: NearToken
    signer_id: AccountId

class InvalidTxErrorNotEnoughBalance(StrictBaseModel):
    """Account does not have enough balance to cover TX cost"""
    NotEnoughBalance: InvalidTxErrorNotEnoughBalancePayload

class InvalidTxErrorLackBalanceForStatePayload(BaseModel):
    # Required balance to cover the state.
    amount: NearToken
    # An account which doesn't have enough balance to cover storage.
    signer_id: AccountId

class InvalidTxErrorLackBalanceForState(StrictBaseModel):
    """Signer account doesn't have enough balance after transaction."""
    LackBalanceForState: InvalidTxErrorLackBalanceForStatePayload

"""An integer overflow occurred during transaction cost estimation."""
class InvalidTxErrorCostOverflow(RootModel[Literal['CostOverflow']]):
    pass

"""Transaction parent block hash doesn't belong to the current chain"""
class InvalidTxErrorInvalidChain(RootModel[Literal['InvalidChain']]):
    pass

"""Transaction has expired"""
class InvalidTxErrorExpired(RootModel[Literal['Expired']]):
    pass

class InvalidTxErrorActionsValidation(StrictBaseModel):
    """An error occurred while validating actions of a Transaction."""
    ActionsValidation: ActionsValidationError

class InvalidTxErrorTransactionSizeExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    size: conint(ge=0, le=18446744073709551615)

class InvalidTxErrorTransactionSizeExceeded(StrictBaseModel):
    """The size of serialized transaction exceeded the limit."""
    TransactionSizeExceeded: InvalidTxErrorTransactionSizeExceededPayload

"""Transaction version is invalid."""
class InvalidTxErrorInvalidTransactionVersion(RootModel[Literal['InvalidTransactionVersion']]):
    pass

class InvalidTxErrorStorageError(StrictBaseModel):
    StorageError: StorageError

class InvalidTxErrorShardCongestedPayload(BaseModel):
    # A value between 0 (no congestion) and 1 (max congestion).
    congestion_level: float
    # The congested shard.
    shard_id: conint(ge=0, le=4294967295)

class InvalidTxErrorShardCongested(StrictBaseModel):
    """The receiver shard of the transaction is too congested to accept new
transactions at the moment."""
    ShardCongested: InvalidTxErrorShardCongestedPayload

class InvalidTxErrorShardStuckPayload(BaseModel):
    # The number of blocks since the last included chunk of the shard.
    missed_chunks: conint(ge=0, le=18446744073709551615)
    # The shard that fails making progress.
    shard_id: conint(ge=0, le=4294967295)

class InvalidTxErrorShardStuck(StrictBaseModel):
    """The receiver shard of the transaction missed several chunks and rejects
new transaction until it can make progress again."""
    ShardStuck: InvalidTxErrorShardStuckPayload

class InvalidTxErrorInvalidNonceIndexPayload(BaseModel):
    # Number of nonces supported by the key. 0 means no nonce_index allowed (regular key).
    num_nonces: conint(ge=0, le=4294967295)
    # The nonce_index from the transaction (None if missing).
    tx_nonce_index: conint(ge=0, le=4294967295) | None = None

class InvalidTxErrorInvalidNonceIndex(StrictBaseModel):
    """Transaction is specifying an invalid nonce index. Gas key transactions
must have a nonce_index in valid range, regular transactions must not."""
    InvalidNonceIndex: InvalidTxErrorInvalidNonceIndexPayload

class InvalidTxError(RootModel[Union[InvalidTxErrorInvalidAccessKeyError, InvalidTxErrorInvalidSignerId, InvalidTxErrorSignerDoesNotExist, InvalidTxErrorInvalidNonce, InvalidTxErrorNonceTooLarge, InvalidTxErrorInvalidReceiverId, InvalidTxErrorInvalidSignature, InvalidTxErrorNotEnoughBalance, InvalidTxErrorLackBalanceForState, InvalidTxErrorCostOverflow, InvalidTxErrorInvalidChain, InvalidTxErrorExpired, InvalidTxErrorActionsValidation, InvalidTxErrorTransactionSizeExceeded, InvalidTxErrorInvalidTransactionVersion, InvalidTxErrorStorageError, InvalidTxErrorShardCongested, InvalidTxErrorShardStuck, InvalidTxErrorInvalidNonceIndex]]):
    pass

