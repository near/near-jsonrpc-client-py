from pydantic import BaseModel
from pydantic import conint


class RangeOfUint64(BaseModel):
    end: conint(ge=0, le=18446744073709551615)
    start: conint(ge=0, le=18446744073709551615)
