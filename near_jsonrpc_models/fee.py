"""Costs associated with an object that can only be sent over the network (and executed
by the receiver).
NOTE: `send_sir` or `send_not_sir` fees are usually burned when the item is being created.
And `execution` fee is burned when the item is being executed."""

from near_jsonrpc_models.near_gas import NearGas
from pydantic import BaseModel


class Fee(BaseModel):
    # Fee for executing the object.
    execution: NearGas
    # Fee for sending an object potentially across the shards.
    send_not_sir: NearGas
    # Fee for sending an object from the sender to itself, guaranteeing that it does not leave
    # the shard.
    send_sir: NearGas
