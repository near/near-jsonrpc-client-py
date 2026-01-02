from near_jsonrpc_models.error_wrapper_for_rpc_validator_error import ErrorWrapperForRpcValidatorError
from near_jsonrpc_models.rpc_validator_response import RpcValidatorResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcValidatorResponseAndRpcValidatorErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcValidatorResponse

class JsonRpcResponseForRpcValidatorResponseAndRpcValidatorErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcValidatorError

class JsonRpcResponseForRpcValidatorResponseAndRpcValidatorError(RootModel[Union[JsonRpcResponseForRpcValidatorResponseAndRpcValidatorErrorResult, JsonRpcResponseForRpcValidatorResponseAndRpcValidatorErrorError]]):
    pass

