"""Stores the congestion level of a shard. More info about congestion [here](https://near.github.io/nearcore/architecture/how/receipt-congestion.html?highlight=congestion#receipt-congestion)"""

from pydantic import BaseModel
from pydantic import conint


class CongestionInfoView(BaseModel):
    allowed_shard: conint(ge=0, le=65535)
    buffered_receipts_gas: str
    delayed_receipts_gas: str
    receipt_bytes: conint(ge=0, le=18446744073709551615)
