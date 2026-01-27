from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_light_client_next_block_error import RpcLightClientNextBlockError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcLightClientNextBlockErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcLightClientNextBlockErrorHandlerError(BaseModel):
    cause: RpcLightClientNextBlockError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcLightClientNextBlockErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcLightClientNextBlockError(RootModel[Union[ErrorWrapperForRpcLightClientNextBlockErrorRequestValidationError, ErrorWrapperForRpcLightClientNextBlockErrorHandlerError, ErrorWrapperForRpcLightClientNextBlockErrorInternalError]]):
    pass

