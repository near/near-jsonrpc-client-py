"""Defines permissions for AccessKey"""

from near_jsonrpc_models.function_call_permission import FunctionCallPermission
from near_jsonrpc_models.gas_key_info import GasKeyInfo
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union


class AccessKeyPermissionFunctionCall(StrictBaseModel):
    FunctionCall: FunctionCallPermission

"""Grants full access to the account.
NOTE: It's used to replace account-level public keys."""
class AccessKeyPermissionFullAccess(RootModel[Literal['FullAccess']]):
    pass

class AccessKeyPermissionGasKeyFunctionCall(StrictBaseModel):
    """Gas key with limited permission to make transactions with FunctionCallActions
Gas keys are a kind of access keys with a prepaid balance to pay for gas."""
    GasKeyFunctionCall: Tuple[GasKeyInfo, FunctionCallPermission]

class AccessKeyPermissionGasKeyFullAccess(StrictBaseModel):
    """Gas key with full access to the account.
Gas keys are a kind of access keys with a prepaid balance to pay for gas."""
    GasKeyFullAccess: GasKeyInfo

class AccessKeyPermission(RootModel[Union[AccessKeyPermissionFunctionCall, AccessKeyPermissionFullAccess, AccessKeyPermissionGasKeyFunctionCall, AccessKeyPermissionGasKeyFullAccess]]):
    pass

