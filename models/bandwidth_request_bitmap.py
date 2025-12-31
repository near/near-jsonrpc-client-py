"""Bitmap which describes which values from the predefined list are being requested.
The nth bit is set to 1 when the nth value from the list is being requested."""

from pydantic import BaseModel
from pydantic import conint
from pydantic import conlist
from typing import List


class BandwidthRequestBitmap(BaseModel):
    data: conlist(conint(ge=0, le=255), min_length=5, max_length=5)
