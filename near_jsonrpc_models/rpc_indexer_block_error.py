from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcIndexerBlockErrorDataUnavailableInfo(BaseModel):
    error_message: str

class RpcIndexerBlockErrorDataUnavailable(BaseModel):
    info: RpcIndexerBlockErrorDataUnavailableInfo
    name: Literal['DATA_UNAVAILABLE']

class RpcIndexerBlockErrorIncompleteDataInfo(BaseModel):
    error_message: str

class RpcIndexerBlockErrorIncompleteData(BaseModel):
    info: RpcIndexerBlockErrorIncompleteDataInfo
    name: Literal['INCOMPLETE_DATA']

class RpcIndexerBlockErrorUnsupportedInfo(BaseModel):
    error_message: str

class RpcIndexerBlockErrorUnsupported(BaseModel):
    info: RpcIndexerBlockErrorUnsupportedInfo
    name: Literal['UNSUPPORTED']

class RpcIndexerBlockErrorLimitExceeded(BaseModel):
    name: Literal['LIMIT_EXCEEDED']

class RpcIndexerBlockErrorBusy(BaseModel):
    name: Literal['BUSY']

class RpcIndexerBlockErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcIndexerBlockErrorInternalError(BaseModel):
    info: RpcIndexerBlockErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcIndexerBlockError(RootModel[Union[RpcIndexerBlockErrorDataUnavailable, RpcIndexerBlockErrorIncompleteData, RpcIndexerBlockErrorUnsupported, RpcIndexerBlockErrorLimitExceeded, RpcIndexerBlockErrorBusy, RpcIndexerBlockErrorInternalError]]):
    pass

