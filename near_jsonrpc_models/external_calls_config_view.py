"""Limits on external contract calls, per call and per chunk."""

from pydantic import BaseModel
from pydantic import conint
from pydantic import conlist
from typing import List


class ExternalCallsConfigView(BaseModel):
    # The share of the chunk's compute limit that its external calls may use.
    chunk_compute_share: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2)
    # The share of `main_storage_proof_size_soft_limit` that a chunk's external calls may
    # record.
    chunk_storage_proof_share: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2)
    # The share of `combined_transactions_size_limit` that a chunk's external calls may take.
    chunk_transactions_size_share: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2)
    # Max storage proof an external call's run may record, in bytes.
    max_call_recorded_size: conint(ge=0, le=4294967295)
    # Max size of an external contract call, in bytes.
    max_call_size: conint(ge=0, le=18446744073709551615)
    # Max number of external calls in a chunk.
    max_calls_per_chunk: conint(ge=0, le=18446744073709551615)
