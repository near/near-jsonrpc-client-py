from models.genesis_config_error import GenesisConfigError
from models.internal_error import InternalError
from models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForGenesisConfigErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForGenesisConfigErrorHandlerError(BaseModel):
    cause: GenesisConfigError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForGenesisConfigErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForGenesisConfigError(RootModel[Union[ErrorWrapperForGenesisConfigErrorRequestValidationError, ErrorWrapperForGenesisConfigErrorHandlerError, ErrorWrapperForGenesisConfigErrorInternalError]]):
    pass

