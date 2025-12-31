"""Item of the state, key and value are serialized in base64 and proof for inclusion of given state item."""

from models.store_key import StoreKey
from models.store_value import StoreValue
from pydantic import BaseModel


class StateItem(BaseModel):
    key: StoreKey
    value: StoreValue
