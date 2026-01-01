"""Grants limited permission to make transactions with FunctionCallActions
The permission can limit the allowed balance to be spent on the prepaid gas.
It also restrict the account ID of the receiver for this function call.
It also can restrict the method name for the allowed function calls."""

from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel
from typing import List


class FunctionCallPermission(BaseModel):
    # Allowance is a balance limit to use by this access key to pay for function call gas and
    # transaction fees. When this access key is used, both account balance and the allowance is
    # decreased by the same value.
    # `None` means unlimited allowance.
    # NOTE: To change or increase the allowance, the old access key needs to be deleted and a new
    # access key should be created.
    allowance: NearToken | None = None
    # A list of method names that can be used. The access key only allows transactions with the
    # function call of one of the given method names.
    # Empty list means any method name can be used.
    method_names: List[str]
    # The access key only allows transactions with the given receiver's account id.
    receiver_id: str
