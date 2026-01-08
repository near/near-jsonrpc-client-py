"""Config for the Chunk Distribution Network feature.
This allows nodes to push and pull chunks from a central stream.
The two benefits of this approach are: (1) less request/response traffic
on the peer-to-peer network and (2) lower latency for RPC nodes indexing the chain."""

from near_jsonrpc_models.chunk_distribution_uris import ChunkDistributionUris
from pydantic import BaseModel


class ChunkDistributionNetworkConfig(BaseModel):
    enabled: bool = None
    uris: ChunkDistributionUris = None
