"""Describes the error for validating a receipt."""

from near_jsonrpc_models.actions_validation_error import ActionsValidationError
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Union


class ReceiptValidationErrorInvalidPredecessorIdPayload(BaseModel):
    account_id: str

class ReceiptValidationErrorInvalidPredecessorId(StrictBaseModel):
    """The `predecessor_id` of a Receipt is not valid."""
    InvalidPredecessorId: ReceiptValidationErrorInvalidPredecessorIdPayload

class ReceiptValidationErrorInvalidReceiverIdPayload(BaseModel):
    account_id: str

class ReceiptValidationErrorInvalidReceiverId(StrictBaseModel):
    """The `receiver_id` of a Receipt is not valid."""
    InvalidReceiverId: ReceiptValidationErrorInvalidReceiverIdPayload

class ReceiptValidationErrorInvalidSignerIdPayload(BaseModel):
    account_id: str

class ReceiptValidationErrorInvalidSignerId(StrictBaseModel):
    """The `signer_id` of an ActionReceipt is not valid."""
    InvalidSignerId: ReceiptValidationErrorInvalidSignerIdPayload

class ReceiptValidationErrorInvalidDataReceiverIdPayload(BaseModel):
    account_id: str

class ReceiptValidationErrorInvalidDataReceiverId(StrictBaseModel):
    """The `receiver_id` of a DataReceiver within an ActionReceipt is not valid."""
    InvalidDataReceiverId: ReceiptValidationErrorInvalidDataReceiverIdPayload

class ReceiptValidationErrorReturnedValueLengthExceededPayload(BaseModel):
    length: conint(ge=0, le=18446744073709551615)
    limit: conint(ge=0, le=18446744073709551615)

class ReceiptValidationErrorReturnedValueLengthExceeded(StrictBaseModel):
    """The length of the returned data exceeded the limit in a DataReceipt."""
    ReturnedValueLengthExceeded: ReceiptValidationErrorReturnedValueLengthExceededPayload

class ReceiptValidationErrorNumberInputDataDependenciesExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    number_of_input_data_dependencies: conint(ge=0, le=18446744073709551615)

class ReceiptValidationErrorNumberInputDataDependenciesExceeded(StrictBaseModel):
    """The number of input data dependencies exceeds the limit in an ActionReceipt."""
    NumberInputDataDependenciesExceeded: ReceiptValidationErrorNumberInputDataDependenciesExceededPayload

class ReceiptValidationErrorActionsValidation(StrictBaseModel):
    """An error occurred while validating actions of an ActionReceipt."""
    ActionsValidation: ActionsValidationError

class ReceiptValidationErrorReceiptSizeExceededPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    size: conint(ge=0, le=18446744073709551615)

class ReceiptValidationErrorReceiptSizeExceeded(StrictBaseModel):
    """Receipt is bigger than the limit."""
    ReceiptSizeExceeded: ReceiptValidationErrorReceiptSizeExceededPayload

class ReceiptValidationErrorInvalidRefundToPayload(BaseModel):
    account_id: str

class ReceiptValidationErrorInvalidRefundTo(StrictBaseModel):
    """The `refund_to` of an ActionReceipt is not valid."""
    InvalidRefundTo: ReceiptValidationErrorInvalidRefundToPayload

class ReceiptValidationError(RootModel[Union[ReceiptValidationErrorInvalidPredecessorId, ReceiptValidationErrorInvalidReceiverId, ReceiptValidationErrorInvalidSignerId, ReceiptValidationErrorInvalidDataReceiverId, ReceiptValidationErrorReturnedValueLengthExceeded, ReceiptValidationErrorNumberInputDataDependenciesExceeded, ReceiptValidationErrorActionsValidation, ReceiptValidationErrorReceiptSizeExceeded, ReceiptValidationErrorInvalidRefundTo]]):
    pass

