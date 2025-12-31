from models.gas_key_info_view import GasKeyInfoView
from pydantic import BaseModel
from typing import List


class GasKeyList(BaseModel):
    keys: List[GasKeyInfoView]
