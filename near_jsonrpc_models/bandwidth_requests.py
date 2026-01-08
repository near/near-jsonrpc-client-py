"""A list of shard's bandwidth requests.
Describes how much the shard would like to send to other shards."""

from near_jsonrpc_models.bandwidth_requests_v1 import BandwidthRequestsV1
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class BandwidthRequestsV1Option(StrictBaseModel):
    V1: BandwidthRequestsV1

class BandwidthRequests(RootModel[Union[BandwidthRequestsV1Option]]):
    pass

