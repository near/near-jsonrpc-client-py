"""Version 1 of [`BandwidthRequest`]."""

from near_jsonrpc_models.bandwidth_request import BandwidthRequest
from pydantic import BaseModel
from typing import List


class BandwidthRequestsV1(BaseModel):
    requests: List[BandwidthRequest]
