"""Version 1 of [`BandwidthRequest`]."""

from models.bandwidth_request import BandwidthRequest
from pydantic import BaseModel
from typing import List


class BandwidthRequestsV1(BaseModel):
    requests: List[BandwidthRequest]
