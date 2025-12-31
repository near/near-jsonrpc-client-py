"""Contains the split storage information."""

from pydantic import BaseModel
from pydantic import conint


class RpcSplitStorageInfoResponse(BaseModel):
    cold_head_height: conint(ge=0, le=18446744073709551615) | None = None
    final_head_height: conint(ge=0, le=18446744073709551615) | None = None
    head_height: conint(ge=0, le=18446744073709551615) | None = None
    hot_db_kind: str | None = None
