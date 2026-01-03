from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class InvalidAccessKeyErrorAccessKeyNotFoundPayload(BaseModel):
    account_id: AccountId
    public_key: PublicKey

class InvalidAccessKeyErrorAccessKeyNotFound(StrictBaseModel):
    """The access key identified by the `public_key` doesn't exist for the account"""
    AccessKeyNotFound: InvalidAccessKeyErrorAccessKeyNotFoundPayload

class InvalidAccessKeyErrorReceiverMismatchPayload(BaseModel):
    ak_receiver: str
    tx_receiver: AccountId

class InvalidAccessKeyErrorReceiverMismatch(StrictBaseModel):
    """Transaction `receiver_id` doesn't match the access key receiver_id"""
    ReceiverMismatch: InvalidAccessKeyErrorReceiverMismatchPayload

class InvalidAccessKeyErrorMethodNameMismatchPayload(BaseModel):
    method_name: str

class InvalidAccessKeyErrorMethodNameMismatch(StrictBaseModel):
    """Transaction method name isn't allowed by the access key"""
    MethodNameMismatch: InvalidAccessKeyErrorMethodNameMismatchPayload

"""Transaction requires a full permission access key."""
class InvalidAccessKeyErrorRequiresFullAccess(RootModel[Literal['RequiresFullAccess']]):
    pass

class InvalidAccessKeyErrorNotEnoughAllowancePayload(BaseModel):
    account_id: AccountId
    allowance: NearToken
    cost: NearToken
    public_key: PublicKey

class InvalidAccessKeyErrorNotEnoughAllowance(StrictBaseModel):
    """Access Key does not have enough allowance to cover transaction cost"""
    NotEnoughAllowance: InvalidAccessKeyErrorNotEnoughAllowancePayload

"""Having a deposit with a function call action is not allowed with a function call access key."""
class InvalidAccessKeyErrorDepositWithFunctionCall(RootModel[Literal['DepositWithFunctionCall']]):
    pass

class InvalidAccessKeyError(RootModel[Union[InvalidAccessKeyErrorAccessKeyNotFound, InvalidAccessKeyErrorReceiverMismatch, InvalidAccessKeyErrorMethodNameMismatch, InvalidAccessKeyErrorRequiresFullAccess, InvalidAccessKeyErrorNotEnoughAllowance, InvalidAccessKeyErrorDepositWithFunctionCall]]):
    pass

