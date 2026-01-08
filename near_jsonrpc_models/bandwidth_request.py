"""`BandwidthRequest` describes the size of receipts that a shard would like to send to another shard.
When a shard wants to send a lot of receipts to another shard, it needs to create a request and wait
for a bandwidth grant from the bandwidth scheduler."""

from near_jsonrpc_models.bandwidth_request_bitmap import BandwidthRequestBitmap
from pydantic import BaseModel
from pydantic import conint


class BandwidthRequest(BaseModel):
    # Bitmap which describes what values of bandwidth are requested.
    requested_values_bitmap: BandwidthRequestBitmap
    # Requesting bandwidth to this shard.
    to_shard: conint(ge=0, le=65535)
