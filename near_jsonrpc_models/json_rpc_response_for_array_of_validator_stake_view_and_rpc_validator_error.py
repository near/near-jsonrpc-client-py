from near_jsonrpc_models.error_wrapper_for_rpc_validator_error import ErrorWrapperForRpcValidatorError
from near_jsonrpc_models.validator_stake_view import ValidatorStakeView
from pydantic import BaseModel
from pydantic import RootModel
from typing import List
from typing import Union


class JsonRpcResponseForArrayOfValidatorStakeViewAndRpcValidatorErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: List[ValidatorStakeView]

class JsonRpcResponseForArrayOfValidatorStakeViewAndRpcValidatorErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcValidatorError

class JsonRpcResponseForArrayOfValidatorStakeViewAndRpcValidatorError(RootModel[Union[JsonRpcResponseForArrayOfValidatorStakeViewAndRpcValidatorErrorResult, JsonRpcResponseForArrayOfValidatorStakeViewAndRpcValidatorErrorError]]):
    pass

