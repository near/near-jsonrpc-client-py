"""Describes different fees for the runtime"""

from near_jsonrpc_models.action_creation_config_view import ActionCreationConfigView
from near_jsonrpc_models.data_receipt_creation_config_view import DataReceiptCreationConfigView
from near_jsonrpc_models.fee import Fee
from near_jsonrpc_models.storage_usage_config_view import StorageUsageConfigView
from pydantic import BaseModel
from pydantic import conint
from pydantic import conlist
from typing import List


class RuntimeFeesConfigView(BaseModel):
    # Describes the cost of creating a certain action, `Action`. Includes all variants.
    action_creation_config: ActionCreationConfigView = None
    # Describes the cost of creating an action receipt, `ActionReceipt`, excluding the actual cost
    # of actions.
    # - `send` cost is burned when a receipt is created using `promise_create` or
    #     `promise_batch_create`
    # - `exec` cost is burned when the receipt is being executed.
    action_receipt_creation_config: Fee = None
    # Fraction of the burnt gas to reward to the contract account for execution.
    burnt_gas_reward: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Describes the cost of creating a data receipt, `DataReceipt`.
    data_receipt_creation_config: DataReceiptCreationConfigView = None
    # Pessimistic gas price inflation ratio.
    pessimistic_gas_price_inflation_ratio: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Describes fees for storage.
    storage_usage_config: StorageUsageConfigView = None
