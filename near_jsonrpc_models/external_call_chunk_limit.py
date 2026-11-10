"""A share of a chunk's limits that its external calls used up, with what the chunk's earlier
external calls used and the share."""

from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Union


class ExternalCallChunkLimitComputePayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    used: conint(ge=0, le=18446744073709551615)

class ExternalCallChunkLimitCompute(StrictBaseModel):
    """The compute used by the chunk's earlier external calls reached their share of the chunk's
compute limit."""
    Compute: ExternalCallChunkLimitComputePayload

class ExternalCallChunkLimitStorageProofPayload(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    used: conint(ge=0, le=18446744073709551615)

class ExternalCallChunkLimitStorageProof(StrictBaseModel):
    """The storage proof recorded by the chunk's earlier external calls reached their share of
the chunk's storage proof limit."""
    StorageProof: ExternalCallChunkLimitStorageProofPayload

class ExternalCallChunkLimit(RootModel[Union[ExternalCallChunkLimitCompute, ExternalCallChunkLimitStorageProof]]):
    pass

