"""Resulting state values for a view state query request"""

from near_jsonrpc_models.state_item import StateItem
from pydantic import BaseModel
from typing import List


class ViewStateResult(BaseModel):
    proof: List[str] = None
    values: List[StateItem]
