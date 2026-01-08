from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_light_client_proof_error import RpcLightClientProofError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcLightClientProofErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcLightClientProofErrorHandlerError(BaseModel):
    cause: RpcLightClientProofError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcLightClientProofErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcLightClientProofError(RootModel[Union[ErrorWrapperForRpcLightClientProofErrorRequestValidationError, ErrorWrapperForRpcLightClientProofErrorHandlerError, ErrorWrapperForRpcLightClientProofErrorInternalError]]):
    pass

