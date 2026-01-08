from near_jsonrpc_models.strict_model import StrictBaseModel
from near_jsonrpc_models.tx_execution_error import TxExecutionError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


"""The execution has not yet started."""
class FinalExecutionStatusNotStarted(RootModel[Literal['NotStarted']]):
    pass

"""The execution has started and still going."""
class FinalExecutionStatusStarted(RootModel[Literal['Started']]):
    pass

class FinalExecutionStatusFailure(StrictBaseModel):
    """The execution has failed with the given error."""
    Failure: TxExecutionError

class FinalExecutionStatusSuccessValue(StrictBaseModel):
    """The execution has succeeded and returned some value or an empty vec encoded in base64."""
    SuccessValue: str

class FinalExecutionStatus(RootModel[Union[FinalExecutionStatusNotStarted, FinalExecutionStatusStarted, FinalExecutionStatusFailure, FinalExecutionStatusSuccessValue]]):
    pass

