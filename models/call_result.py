"""A result returned by contract method"""

from pydantic import BaseModel
from pydantic import conint
from typing import List


class CallResult(BaseModel):
    logs: List[str]
    result: List[conint(ge=0, le=255)]
