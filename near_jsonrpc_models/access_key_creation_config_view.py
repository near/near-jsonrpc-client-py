"""Describes the cost of creating an access key."""

from near_jsonrpc_models.fee import Fee
from pydantic import BaseModel


class AccessKeyCreationConfigView(BaseModel):
    # Base cost of creating a full access access-key.
    full_access_cost: Fee
    # Base cost of creating an access-key restricted to specific functions.
    function_call_cost: Fee
    # Cost per byte of method_names of creating a restricted access-key.
    function_call_cost_per_byte: Fee
