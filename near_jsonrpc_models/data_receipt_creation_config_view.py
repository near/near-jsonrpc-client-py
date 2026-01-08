"""The fees settings for a data receipt creation"""

from near_jsonrpc_models.fee import Fee
from pydantic import BaseModel


class DataReceiptCreationConfigView(BaseModel):
    # Base cost of creating a data receipt.
    # Both `send` and `exec` costs are burned when a new receipt has input dependencies. The gas
    # is charged for each input dependency. The dependencies are specified when a receipt is
    # created using `promise_then` and `promise_batch_then`.
    # NOTE: Any receipt with output dependencies will produce data receipts. Even if it fails.
    # Even if the last action is not a function call (in case of success it will return empty
    # value).
    base_cost: Fee = None
    # Additional cost per byte sent.
    # Both `send` and `exec` costs are burned when a function call finishes execution and returns
    # `N` bytes of data to every output dependency. For each output dependency the cost is
    # `(send(sir) + exec()) * N`.
    cost_per_byte: Fee = None
