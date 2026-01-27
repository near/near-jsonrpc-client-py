from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.function_call_error import FunctionCallError
from near_jsonrpc_models.global_contract_identifier import GlobalContractIdentifier
from near_jsonrpc_models.invalid_access_key_error import InvalidAccessKeyError
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.receipt_validation_error import ReceiptValidationError
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class ActionErrorKindAccountAlreadyExistsPayload(BaseModel):
    account_id: AccountId

class ActionErrorKindAccountAlreadyExists(StrictBaseModel):
    """Happens when CreateAccount action tries to create an account with account_id which is already exists in the storage"""
    AccountAlreadyExists: ActionErrorKindAccountAlreadyExistsPayload

class ActionErrorKindAccountDoesNotExistPayload(BaseModel):
    account_id: AccountId

class ActionErrorKindAccountDoesNotExist(StrictBaseModel):
    """Happens when TX receiver_id doesn't exist (but action is not Action::CreateAccount)"""
    AccountDoesNotExist: ActionErrorKindAccountDoesNotExistPayload

class ActionErrorKindCreateAccountOnlyByRegistrarPayload(BaseModel):
    account_id: AccountId
    predecessor_id: AccountId
    registrar_account_id: AccountId

class ActionErrorKindCreateAccountOnlyByRegistrar(StrictBaseModel):
    """A top-level account ID can only be created by registrar."""
    CreateAccountOnlyByRegistrar: ActionErrorKindCreateAccountOnlyByRegistrarPayload

class ActionErrorKindCreateAccountNotAllowedPayload(BaseModel):
    account_id: AccountId
    predecessor_id: AccountId

class ActionErrorKindCreateAccountNotAllowed(StrictBaseModel):
    """A newly created account must be under a namespace of the creator account"""
    CreateAccountNotAllowed: ActionErrorKindCreateAccountNotAllowedPayload

class ActionErrorKindActorNoPermissionPayload(BaseModel):
    account_id: AccountId
    actor_id: AccountId

class ActionErrorKindActorNoPermission(StrictBaseModel):
    """Administrative actions like `DeployContract`, `Stake`, `AddKey`, `DeleteKey`. can be proceed only if sender=receiver
or the first TX action is a `CreateAccount` action"""
    ActorNoPermission: ActionErrorKindActorNoPermissionPayload

class ActionErrorKindDeleteKeyDoesNotExistPayload(BaseModel):
    account_id: AccountId
    public_key: PublicKey

class ActionErrorKindDeleteKeyDoesNotExist(StrictBaseModel):
    """Account tries to remove an access key that doesn't exist"""
    DeleteKeyDoesNotExist: ActionErrorKindDeleteKeyDoesNotExistPayload

class ActionErrorKindAddKeyAlreadyExistsPayload(BaseModel):
    account_id: AccountId
    public_key: PublicKey

class ActionErrorKindAddKeyAlreadyExists(StrictBaseModel):
    """The public key is already used for an existing access key"""
    AddKeyAlreadyExists: ActionErrorKindAddKeyAlreadyExistsPayload

class ActionErrorKindDeleteAccountStakingPayload(BaseModel):
    account_id: AccountId

class ActionErrorKindDeleteAccountStaking(StrictBaseModel):
    """Account is staking and can not be deleted"""
    DeleteAccountStaking: ActionErrorKindDeleteAccountStakingPayload

class ActionErrorKindLackBalanceForStatePayload(BaseModel):
    # An account which needs balance
    account_id: AccountId
    # Balance required to complete an action.
    amount: NearToken

class ActionErrorKindLackBalanceForState(StrictBaseModel):
    """ActionReceipt can't be completed, because the remaining balance will not be enough to cover storage."""
    LackBalanceForState: ActionErrorKindLackBalanceForStatePayload

class ActionErrorKindTriesToUnstakePayload(BaseModel):
    account_id: AccountId

class ActionErrorKindTriesToUnstake(StrictBaseModel):
    """Account is not yet staked, but tries to unstake"""
    TriesToUnstake: ActionErrorKindTriesToUnstakePayload

class ActionErrorKindTriesToStakePayload(BaseModel):
    account_id: AccountId
    balance: NearToken
    locked: NearToken
    stake: NearToken

class ActionErrorKindTriesToStake(StrictBaseModel):
    """The account doesn't have enough balance to increase the stake."""
    TriesToStake: ActionErrorKindTriesToStakePayload

class ActionErrorKindInsufficientStakePayload(BaseModel):
    account_id: AccountId
    minimum_stake: NearToken
    stake: NearToken

class ActionErrorKindInsufficientStake(StrictBaseModel):
    InsufficientStake: ActionErrorKindInsufficientStakePayload

class ActionErrorKindFunctionCallError(StrictBaseModel):
    """An error occurred during a `FunctionCall` Action, parameter is debug message."""
    FunctionCallError: FunctionCallError

class ActionErrorKindNewReceiptValidationError(StrictBaseModel):
    """Error occurs when a new `ActionReceipt` created by the `FunctionCall` action fails
receipt validation."""
    NewReceiptValidationError: ReceiptValidationError

class ActionErrorKindOnlyImplicitAccountCreationAllowedPayload(BaseModel):
    account_id: AccountId

class ActionErrorKindOnlyImplicitAccountCreationAllowed(StrictBaseModel):
    """Error occurs when a `CreateAccount` action is called on a NEAR-implicit or ETH-implicit account.
See NEAR-implicit account creation NEP: <https://github.com/nearprotocol/NEPs/pull/71>.
Also, see ETH-implicit account creation NEP: <https://github.com/near/NEPs/issues/518>.

TODO(#8598): This error is named very poorly. A better name would be
`OnlyNamedAccountCreationAllowed`."""
    OnlyImplicitAccountCreationAllowed: ActionErrorKindOnlyImplicitAccountCreationAllowedPayload

class ActionErrorKindDeleteAccountWithLargeStatePayload(BaseModel):
    account_id: AccountId

class ActionErrorKindDeleteAccountWithLargeState(StrictBaseModel):
    """Delete account whose state is large is temporarily banned."""
    DeleteAccountWithLargeState: ActionErrorKindDeleteAccountWithLargeStatePayload

"""Signature does not match the provided actions and given signer public key."""
class ActionErrorKindDelegateActionInvalidSignature(RootModel[Literal['DelegateActionInvalidSignature']]):
    pass

class ActionErrorKindDelegateActionSenderDoesNotMatchTxReceiverPayload(BaseModel):
    receiver_id: AccountId
    sender_id: AccountId

class ActionErrorKindDelegateActionSenderDoesNotMatchTxReceiver(StrictBaseModel):
    """Receiver of the transaction doesn't match Sender of the delegate action"""
    DelegateActionSenderDoesNotMatchTxReceiver: ActionErrorKindDelegateActionSenderDoesNotMatchTxReceiverPayload

"""Delegate action has expired. `max_block_height` is less than actual block height."""
class ActionErrorKindDelegateActionExpired(RootModel[Literal['DelegateActionExpired']]):
    pass

class ActionErrorKindDelegateActionAccessKeyError(StrictBaseModel):
    """The given public key doesn't exist for Sender account"""
    DelegateActionAccessKeyError: InvalidAccessKeyError

class ActionErrorKindDelegateActionInvalidNoncePayload(BaseModel):
    ak_nonce: conint(ge=0, le=18446744073709551615)
    delegate_nonce: conint(ge=0, le=18446744073709551615)

class ActionErrorKindDelegateActionInvalidNonce(StrictBaseModel):
    """DelegateAction nonce must be greater sender[public_key].nonce"""
    DelegateActionInvalidNonce: ActionErrorKindDelegateActionInvalidNoncePayload

class ActionErrorKindDelegateActionNonceTooLargePayload(BaseModel):
    delegate_nonce: conint(ge=0, le=18446744073709551615)
    upper_bound: conint(ge=0, le=18446744073709551615)

class ActionErrorKindDelegateActionNonceTooLarge(StrictBaseModel):
    """DelegateAction nonce is larger than the upper bound given by the block height"""
    DelegateActionNonceTooLarge: ActionErrorKindDelegateActionNonceTooLargePayload

class ActionErrorKindGlobalContractDoesNotExistPayload(BaseModel):
    identifier: GlobalContractIdentifier

class ActionErrorKindGlobalContractDoesNotExist(StrictBaseModel):
    GlobalContractDoesNotExist: ActionErrorKindGlobalContractDoesNotExistPayload

class ActionErrorKindGasKeyDoesNotExistPayload(BaseModel):
    account_id: AccountId
    public_key: PublicKey

class ActionErrorKindGasKeyDoesNotExist(StrictBaseModel):
    """Gas key does not exist for the specified public key"""
    GasKeyDoesNotExist: ActionErrorKindGasKeyDoesNotExistPayload

class ActionErrorKindInsufficientGasKeyBalancePayload(BaseModel):
    account_id: AccountId
    balance: NearToken
    public_key: PublicKey
    required: NearToken

class ActionErrorKindInsufficientGasKeyBalance(StrictBaseModel):
    """Gas key does not have sufficient balance for the requested withdrawal"""
    InsufficientGasKeyBalance: ActionErrorKindInsufficientGasKeyBalancePayload

class ActionErrorKind(RootModel[Union[ActionErrorKindAccountAlreadyExists, ActionErrorKindAccountDoesNotExist, ActionErrorKindCreateAccountOnlyByRegistrar, ActionErrorKindCreateAccountNotAllowed, ActionErrorKindActorNoPermission, ActionErrorKindDeleteKeyDoesNotExist, ActionErrorKindAddKeyAlreadyExists, ActionErrorKindDeleteAccountStaking, ActionErrorKindLackBalanceForState, ActionErrorKindTriesToUnstake, ActionErrorKindTriesToStake, ActionErrorKindInsufficientStake, ActionErrorKindFunctionCallError, ActionErrorKindNewReceiptValidationError, ActionErrorKindOnlyImplicitAccountCreationAllowed, ActionErrorKindDeleteAccountWithLargeState, ActionErrorKindDelegateActionInvalidSignature, ActionErrorKindDelegateActionSenderDoesNotMatchTxReceiver, ActionErrorKindDelegateActionExpired, ActionErrorKindDelegateActionAccessKeyError, ActionErrorKindDelegateActionInvalidNonce, ActionErrorKindDelegateActionNonceTooLarge, ActionErrorKindGlobalContractDoesNotExist, ActionErrorKindGasKeyDoesNotExist, ActionErrorKindInsufficientGasKeyBalance]]):
    pass

