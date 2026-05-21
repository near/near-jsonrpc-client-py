"""Resulting state values for a view state query request"""

from near_jsonrpc_models.state_item import StateItem
from near_jsonrpc_models.store_key import StoreKey
from pydantic import BaseModel
from typing import List


class ViewStateResult(BaseModel):
    last_key: StoreKey | None = None
    proof: List[str] = None
    values: List[StateItem]
