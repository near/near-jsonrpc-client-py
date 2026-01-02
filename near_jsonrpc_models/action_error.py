"""An error happened during Action execution"""

from near_jsonrpc_models.action_error_kind import ActionErrorKind
from pydantic import BaseModel
from pydantic import conint


class ActionError(BaseModel):
    # Index of the failed action in the transaction.
    # Action index is not defined if ActionError.kind is `ActionErrorKind::LackBalanceForState`
    index: conint(ge=0, le=18446744073709551615) | None = None
    # The kind of ActionError happened
    kind: ActionErrorKind
