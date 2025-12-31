"""Describes the permission scope for an access key. Whether it is a function call or a full access key."""

from models.near_token import NearToken
from models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
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

class AccessKeyPermissionView(RootModel[Union[AccessKeyPermissionViewFullAccess, AccessKeyPermissionViewFunctionCall]]):
    pass

