"""Lists access keys"""

from near_jsonrpc_models.access_key_info_view import AccessKeyInfoView
from pydantic import BaseModel
from typing import List


class AccessKeyList(BaseModel):
    keys: List[AccessKeyInfoView]
