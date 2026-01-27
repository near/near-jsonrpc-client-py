"""Describes the permission scope for an access key. Whether it is a function call or a full access key."""

from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import List
from typing import Literal
from typing import Union


class AccessKeyPermissionViewFullAccess(RootModel[Literal['FullAccess']]):
    pass

class AccessKeyPermissionViewFunctionCallPayload(BaseModel):
    allowance: NearToken | None = None
    method_names: List[str]
    receiver_id: str

class AccessKeyPermissionViewFunctionCall(StrictBaseModel):
    FunctionCall: AccessKeyPermissionViewFunctionCallPayload

class AccessKeyPermissionViewGasKeyFunctionCallPayload(BaseModel):
    allowance: NearToken | None = None
    balance: NearToken
    method_names: List[str]
    num_nonces: conint(ge=0, le=4294967295)
    receiver_id: str

class AccessKeyPermissionViewGasKeyFunctionCall(StrictBaseModel):
    GasKeyFunctionCall: AccessKeyPermissionViewGasKeyFunctionCallPayload

class AccessKeyPermissionViewGasKeyFullAccessPayload(BaseModel):
    balance: NearToken
    num_nonces: conint(ge=0, le=4294967295)

class AccessKeyPermissionViewGasKeyFullAccess(StrictBaseModel):
    GasKeyFullAccess: AccessKeyPermissionViewGasKeyFullAccessPayload

class AccessKeyPermissionView(RootModel[Union[AccessKeyPermissionViewFullAccess, AccessKeyPermissionViewFunctionCall, AccessKeyPermissionViewGasKeyFunctionCall, AccessKeyPermissionViewGasKeyFullAccess]]):
    pass

