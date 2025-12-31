"""Error returned in the ExecutionOutcome in case of failure"""

from models.action_error import ActionError
from models.invalid_tx_error import InvalidTxError
from models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class TxExecutionErrorActionError(StrictBaseModel):
    """An error happened during Action execution"""
    ActionError: ActionError

class TxExecutionErrorInvalidTxError(StrictBaseModel):
    """An error happened during Transaction execution"""
    InvalidTxError: InvalidTxError

class TxExecutionError(RootModel[Union[TxExecutionErrorActionError, TxExecutionErrorInvalidTxError]]):
    pass

