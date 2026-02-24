"""Gas key nonces view returned by the `view_gas_key_nonces` RPC query."""

from pydantic import BaseModel
from pydantic import conint
from typing import List


class GasKeyNoncesView(BaseModel):
    nonces: List[conint(ge=0, le=18446744073709551615)]
