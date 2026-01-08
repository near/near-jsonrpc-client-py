"""Defines permissions for AccessKey"""

from near_jsonrpc_models.function_call_permission import FunctionCallPermission
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class AccessKeyPermissionFunctionCall(StrictBaseModel):
    FunctionCall: FunctionCallPermission

"""Grants full access to the account.
NOTE: It's used to replace account-level public keys."""
class AccessKeyPermissionFullAccess(RootModel[Literal['FullAccess']]):
    pass

class AccessKeyPermission(RootModel[Union[AccessKeyPermissionFunctionCall, AccessKeyPermissionFullAccess]]):
    pass

