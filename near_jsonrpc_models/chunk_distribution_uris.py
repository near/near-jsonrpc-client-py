"""URIs for the Chunk Distribution Network feature."""

from pydantic import BaseModel


class ChunkDistributionUris(BaseModel):
    # URI for pulling chunks from the stream.
    get: str = None
    # URI for publishing chunks to the stream.
    set: str = None
