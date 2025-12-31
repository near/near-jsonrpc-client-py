"""Lists access keys"""

from models.access_key_info_view import AccessKeyInfoView
from pydantic import BaseModel
from typing import List


class AccessKeyList(BaseModel):
    keys: List[AccessKeyInfoView]
