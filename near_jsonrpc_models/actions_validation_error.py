"""Describes the error for validating a list of actions."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.near_gas import NearGas
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


"""The delete action must be a final action in transaction"""
class ActionsValidationErrorDeleteActionMustBeFinal(RootModel[Literal['DeleteActionMustBeFinal']]):
    pass

class ActionsValidationErrorTotalPrepaidGasExceededPayload(BaseModel):
    limit: NearGas
    total_prepaid_gas: NearGas

class ActionsValidationErrorTotalPrepaidGasExceeded(StrictBaseModel):
    """The total prepaid gas (for all given actions) exceeded the limit."""
    TotalPrepaidGasExceeded: ActionsValidationErrorTotalPrepaidGasExceededPayload

class ActionsValidationErrorTotalNumberOfActionsExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    total_number_of_actions: conint(ge=0, le=18446744073709551615)

class ActionsValidationErrorTotalNumberOfActionsExceeded(StrictBaseModel):
    """The number of actions exceeded the given limit."""
    TotalNumberOfActionsExceeded: ActionsValidationErrorTotalNumberOfActionsExceededPayload

class ActionsValidationErrorAddKeyMethodNamesNumberOfBytesExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    total_number_of_bytes: conint(ge=0, le=18446744073709551615)

class ActionsValidationErrorAddKeyMethodNamesNumberOfBytesExceeded(StrictBaseModel):
    """The total number of bytes of the method names exceeded the limit in a Add Key action."""
    AddKeyMethodNamesNumberOfBytesExceeded: ActionsValidationErrorAddKeyMethodNamesNumberOfBytesExceededPayload

class ActionsValidationErrorAddKeyMethodNameLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class ActionsValidationErrorAddKeyMethodNameLengthExceeded(StrictBaseModel):
    """The length of some method name exceeded the limit in a Add Key action."""
    AddKeyMethodNameLengthExceeded: ActionsValidationErrorAddKeyMethodNameLengthExceededPayload

"""Integer overflow during a compute."""
class ActionsValidationErrorIntegerOverflow(RootModel[Literal['IntegerOverflow']]):
    pass

class ActionsValidationErrorInvalidAccountIdPayload(BaseModel):
    account_id: str

class ActionsValidationErrorInvalidAccountId(StrictBaseModel):
    """Invalid account ID."""
    InvalidAccountId: ActionsValidationErrorInvalidAccountIdPayload

class ActionsValidationErrorContractSizeExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    size: conint(ge=0, le=18446744073709551615)

class ActionsValidationErrorContractSizeExceeded(StrictBaseModel):
    """The size of the contract code exceeded the limit in a DeployContract action."""
    ContractSizeExceeded: ActionsValidationErrorContractSizeExceededPayload

class ActionsValidationErrorFunctionCallMethodNameLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class ActionsValidationErrorFunctionCallMethodNameLengthExceeded(StrictBaseModel):
    """The length of the method name exceeded the limit in a Function Call action."""
    FunctionCallMethodNameLengthExceeded: ActionsValidationErrorFunctionCallMethodNameLengthExceededPayload

class ActionsValidationErrorFunctionCallArgumentsLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class ActionsValidationErrorFunctionCallArgumentsLengthExceeded(StrictBaseModel):
    """The length of the arguments exceeded the limit in a Function Call action."""
    FunctionCallArgumentsLengthExceeded: ActionsValidationErrorFunctionCallArgumentsLengthExceededPayload

class ActionsValidationErrorUnsuitableStakingKeyPayload(BaseModel):
    public_key: PublicKey

class ActionsValidationErrorUnsuitableStakingKey(StrictBaseModel):
    """An attempt to stake with a public key that is not convertible to ristretto."""
    UnsuitableStakingKey: ActionsValidationErrorUnsuitableStakingKeyPayload

"""The attached amount of gas in a FunctionCall action has to be a positive number."""
class ActionsValidationErrorFunctionCallZeroAttachedGas(RootModel[Literal['FunctionCallZeroAttachedGas']]):
    pass

"""There should be the only one DelegateAction"""
class ActionsValidationErrorDelegateActionMustBeOnlyOne(RootModel[Literal['DelegateActionMustBeOnlyOne']]):
    pass

class ActionsValidationErrorUnsupportedProtocolFeaturePayload(BaseModel):
    protocol_feature: str
    version: conint(ge=0, le=4294967295)

class ActionsValidationErrorUnsupportedProtocolFeature(StrictBaseModel):
    """The transaction includes a feature that the current protocol version
does not support.

Note: we stringify the protocol feature name instead of using
`ProtocolFeature` here because we don't want to leak the internals of
that type into observable borsh serialization."""
    UnsupportedProtocolFeature: ActionsValidationErrorUnsupportedProtocolFeaturePayload

class ActionsValidationErrorInvalidDeterministicStateInitReceiverPayload(BaseModel):
    derived_id: AccountId
    receiver_id: AccountId

class ActionsValidationErrorInvalidDeterministicStateInitReceiver(StrictBaseModel):
    InvalidDeterministicStateInitReceiver: ActionsValidationErrorInvalidDeterministicStateInitReceiverPayload

class ActionsValidationErrorDeterministicStateInitKeyLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class ActionsValidationErrorDeterministicStateInitKeyLengthExceeded(StrictBaseModel):
    DeterministicStateInitKeyLengthExceeded: ActionsValidationErrorDeterministicStateInitKeyLengthExceededPayload

class ActionsValidationErrorDeterministicStateInitValueLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class ActionsValidationErrorDeterministicStateInitValueLengthExceeded(StrictBaseModel):
    DeterministicStateInitValueLengthExceeded: ActionsValidationErrorDeterministicStateInitValueLengthExceededPayload

class ActionsValidationErrorGasKeyInvalidNumNoncesPayload(BaseModel):
    limit: conint(ge=0, le=4294967295)
    requested_nonces: conint(ge=0, le=4294967295)

class ActionsValidationErrorGasKeyInvalidNumNonces(StrictBaseModel):
    GasKeyInvalidNumNonces: ActionsValidationErrorGasKeyInvalidNumNoncesPayload

class ActionsValidationErrorAddGasKeyWithNonZeroBalancePayload(BaseModel):
    balance: NearToken

class ActionsValidationErrorAddGasKeyWithNonZeroBalance(StrictBaseModel):
    AddGasKeyWithNonZeroBalance: ActionsValidationErrorAddGasKeyWithNonZeroBalancePayload

"""Gas keys with FunctionCall permission cannot have an allowance set."""
class ActionsValidationErrorGasKeyFunctionCallAllowanceNotAllowed(RootModel[Literal['GasKeyFunctionCallAllowanceNotAllowed']]):
    pass

class ActionsValidationError(RootModel[Union[ActionsValidationErrorDeleteActionMustBeFinal, ActionsValidationErrorTotalPrepaidGasExceeded, ActionsValidationErrorTotalNumberOfActionsExceeded, ActionsValidationErrorAddKeyMethodNamesNumberOfBytesExceeded, ActionsValidationErrorAddKeyMethodNameLengthExceeded, ActionsValidationErrorIntegerOverflow, ActionsValidationErrorInvalidAccountId, ActionsValidationErrorContractSizeExceeded, ActionsValidationErrorFunctionCallMethodNameLengthExceeded, ActionsValidationErrorFunctionCallArgumentsLengthExceeded, ActionsValidationErrorUnsuitableStakingKey, ActionsValidationErrorFunctionCallZeroAttachedGas, ActionsValidationErrorDelegateActionMustBeOnlyOne, ActionsValidationErrorUnsupportedProtocolFeature, ActionsValidationErrorInvalidDeterministicStateInitReceiver, ActionsValidationErrorDeterministicStateInitKeyLengthExceeded, ActionsValidationErrorDeterministicStateInitValueLengthExceeded, ActionsValidationErrorGasKeyInvalidNumNonces, ActionsValidationErrorAddGasKeyWithNonZeroBalance, ActionsValidationErrorGasKeyFunctionCallAllowanceNotAllowed]]):
    pass

