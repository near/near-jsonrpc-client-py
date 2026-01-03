from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


"""String encoding is bad UTF-16 sequence"""
class HostErrorBadUTF16(RootModel[Literal['BadUTF16']]):
    pass

"""String encoding is bad UTF-8 sequence"""
class HostErrorBadUTF8(RootModel[Literal['BadUTF8']]):
    pass

"""Exceeded the prepaid gas"""
class HostErrorGasExceeded(RootModel[Literal['GasExceeded']]):
    pass

"""Exceeded the maximum amount of gas allowed to burn per contract"""
class HostErrorGasLimitExceeded(RootModel[Literal['GasLimitExceeded']]):
    pass

"""Exceeded the account balance"""
class HostErrorBalanceExceeded(RootModel[Literal['BalanceExceeded']]):
    pass

"""Tried to call an empty method name"""
class HostErrorEmptyMethodName(RootModel[Literal['EmptyMethodName']]):
    pass

class HostErrorGuestPanicPayload(BaseModel):
    panic_msg: str

class HostErrorGuestPanic(StrictBaseModel):
    """Smart contract panicked"""
    GuestPanic: HostErrorGuestPanicPayload

"""IntegerOverflow happened during a contract execution"""
class HostErrorIntegerOverflow(RootModel[Literal['IntegerOverflow']]):
    pass

class HostErrorInvalidPromiseIndexPayload(BaseModel):
    promise_idx: conint(ge=0, le=18446744073709551615)

class HostErrorInvalidPromiseIndex(StrictBaseModel):
    """`promise_idx` does not correspond to existing promises"""
    InvalidPromiseIndex: HostErrorInvalidPromiseIndexPayload

"""Actions can only be appended to non-joint promise."""
class HostErrorCannotAppendActionToJointPromise(RootModel[Literal['CannotAppendActionToJointPromise']]):
    pass

"""Returning joint promise is currently prohibited"""
class HostErrorCannotReturnJointPromise(RootModel[Literal['CannotReturnJointPromise']]):
    pass

class HostErrorInvalidPromiseResultIndexPayload(BaseModel):
    result_idx: conint(ge=0, le=18446744073709551615)

class HostErrorInvalidPromiseResultIndex(StrictBaseModel):
    """Accessed invalid promise result index"""
    InvalidPromiseResultIndex: HostErrorInvalidPromiseResultIndexPayload

class HostErrorInvalidRegisterIdPayload(BaseModel):
    register_id: conint(ge=0, le=18446744073709551615)

class HostErrorInvalidRegisterId(StrictBaseModel):
    """Accessed invalid register id"""
    InvalidRegisterId: HostErrorInvalidRegisterIdPayload

class HostErrorIteratorWasInvalidatedPayload(BaseModel):
    iterator_index: conint(ge=0, le=18446744073709551615)

class HostErrorIteratorWasInvalidated(StrictBaseModel):
    """Iterator `iterator_index` was invalidated after its creation by performing a mutable operation on trie"""
    IteratorWasInvalidated: HostErrorIteratorWasInvalidatedPayload

"""Accessed memory outside the bounds"""
class HostErrorMemoryAccessViolation(RootModel[Literal['MemoryAccessViolation']]):
    pass

class HostErrorInvalidReceiptIndexPayload(BaseModel):
    receipt_index: conint(ge=0, le=18446744073709551615)

class HostErrorInvalidReceiptIndex(StrictBaseModel):
    """VM Logic returned an invalid receipt index"""
    InvalidReceiptIndex: HostErrorInvalidReceiptIndexPayload

class HostErrorInvalidIteratorIndexPayload(BaseModel):
    iterator_index: conint(ge=0, le=18446744073709551615)

class HostErrorInvalidIteratorIndex(StrictBaseModel):
    """Iterator index `iterator_index` does not exist"""
    InvalidIteratorIndex: HostErrorInvalidIteratorIndexPayload

"""VM Logic returned an invalid account id"""
class HostErrorInvalidAccountId(RootModel[Literal['InvalidAccountId']]):
    pass

"""VM Logic returned an invalid method name"""
class HostErrorInvalidMethodName(RootModel[Literal['InvalidMethodName']]):
    pass

"""VM Logic provided an invalid public key"""
class HostErrorInvalidPublicKey(RootModel[Literal['InvalidPublicKey']]):
    pass

class HostErrorProhibitedInViewPayload(BaseModel):
    method_name: str

class HostErrorProhibitedInView(StrictBaseModel):
    """`method_name` is not allowed in view calls"""
    ProhibitedInView: HostErrorProhibitedInViewPayload

class HostErrorNumberOfLogsExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)

class HostErrorNumberOfLogsExceeded(StrictBaseModel):
    """The total number of logs will exceed the limit."""
    NumberOfLogsExceeded: HostErrorNumberOfLogsExceededPayload

class HostErrorKeyLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class HostErrorKeyLengthExceeded(StrictBaseModel):
    """The storage key length exceeded the limit."""
    KeyLengthExceeded: HostErrorKeyLengthExceededPayload

class HostErrorValueLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class HostErrorValueLengthExceeded(StrictBaseModel):
    """The storage value length exceeded the limit."""
    ValueLengthExceeded: HostErrorValueLengthExceededPayload

class HostErrorTotalLogLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class HostErrorTotalLogLengthExceeded(StrictBaseModel):
    """The total log length exceeded the limit."""
    TotalLogLengthExceeded: HostErrorTotalLogLengthExceededPayload

class HostErrorNumberPromisesExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    number_of_promises: conint(ge=0, le=18446744073709551615)

class HostErrorNumberPromisesExceeded(StrictBaseModel):
    """The maximum number of promises within a FunctionCall exceeded the limit."""
    NumberPromisesExceeded: HostErrorNumberPromisesExceededPayload

class HostErrorNumberInputDataDependenciesExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    number_of_input_data_dependencies: conint(ge=0, le=18446744073709551615)

class HostErrorNumberInputDataDependenciesExceeded(StrictBaseModel):
    """The maximum number of input data dependencies exceeded the limit."""
    NumberInputDataDependenciesExceeded: HostErrorNumberInputDataDependenciesExceededPayload

class HostErrorReturnedValueLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class HostErrorReturnedValueLengthExceeded(StrictBaseModel):
    """The returned value length exceeded the limit."""
    ReturnedValueLengthExceeded: HostErrorReturnedValueLengthExceededPayload

class HostErrorContractSizeExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    size: conint(ge=0, le=18446744073709551615)

class HostErrorContractSizeExceeded(StrictBaseModel):
    """The contract size for DeployContract action exceeded the limit."""
    ContractSizeExceeded: HostErrorContractSizeExceededPayload

class HostErrorDeprecatedPayload(BaseModel):
    method_name: str

class HostErrorDeprecated(StrictBaseModel):
    """The host function was deprecated."""
    Deprecated: HostErrorDeprecatedPayload

class HostErrorECRecoverErrorPayload(BaseModel):
    msg: str

class HostErrorECRecoverError(StrictBaseModel):
    """General errors for ECDSA recover."""
    ECRecoverError: HostErrorECRecoverErrorPayload

class HostErrorAltBn128InvalidInputPayload(BaseModel):
    msg: str

class HostErrorAltBn128InvalidInput(StrictBaseModel):
    """Invalid input to alt_bn128 family of functions (e.g., point which isn't
on the curve)."""
    AltBn128InvalidInput: HostErrorAltBn128InvalidInputPayload

class HostErrorEd25519VerifyInvalidInputPayload(BaseModel):
    msg: str

class HostErrorEd25519VerifyInvalidInput(StrictBaseModel):
    """Invalid input to ed25519 signature verification function (e.g. signature cannot be
derived from bytes)."""
    Ed25519VerifyInvalidInput: HostErrorEd25519VerifyInvalidInputPayload

class HostError(RootModel[Union[HostErrorBadUTF16, HostErrorBadUTF8, HostErrorGasExceeded, HostErrorGasLimitExceeded, HostErrorBalanceExceeded, HostErrorEmptyMethodName, HostErrorGuestPanic, HostErrorIntegerOverflow, HostErrorInvalidPromiseIndex, HostErrorCannotAppendActionToJointPromise, HostErrorCannotReturnJointPromise, HostErrorInvalidPromiseResultIndex, HostErrorInvalidRegisterId, HostErrorIteratorWasInvalidated, HostErrorMemoryAccessViolation, HostErrorInvalidReceiptIndex, HostErrorInvalidIteratorIndex, HostErrorInvalidAccountId, HostErrorInvalidMethodName, HostErrorInvalidPublicKey, HostErrorProhibitedInView, HostErrorNumberOfLogsExceeded, HostErrorKeyLengthExceeded, HostErrorValueLengthExceeded, HostErrorTotalLogLengthExceeded, HostErrorNumberPromisesExceeded, HostErrorNumberInputDataDependenciesExceeded, HostErrorReturnedValueLengthExceeded, HostErrorContractSizeExceeded, HostErrorDeprecated, HostErrorECRecoverError, HostErrorAltBn128InvalidInput, HostErrorEd25519VerifyInvalidInput]]):
    pass

