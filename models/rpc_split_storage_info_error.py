from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcSplitStorageInfoErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcSplitStorageInfoErrorInternalError(BaseModel):
    info: RpcSplitStorageInfoErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcSplitStorageInfoError(RootModel[Union[RpcSplitStorageInfoErrorInternalError]]):
    pass

