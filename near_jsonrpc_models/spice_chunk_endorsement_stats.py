"""Per-validator chunk endorsement stats accumulated over a spice epoch,
indexed by the current epoch's validator id. Carried on the last block of
the epoch (see `BlockHeaderInnerRestV7`) and consumed by reward and kickout."""

from pydantic import BaseModel
from pydantic import conint


class SpiceChunkEndorsementStats(BaseModel):
    expected: conint(ge=0, le=4294967295)
    produced: conint(ge=0, le=4294967295)
