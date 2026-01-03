"""Item of the state, key and value are serialized in base64 and proof for inclusion of given state item."""

from near_jsonrpc_models.store_key import StoreKey
from near_jsonrpc_models.store_value import StoreValue
from pydantic import BaseModel


class StateItem(BaseModel):
    key: StoreKey
    value: StoreValue
